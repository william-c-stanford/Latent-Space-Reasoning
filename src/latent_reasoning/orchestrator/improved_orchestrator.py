"""Improved orchestrator with diversity-based evolution and heuristic re-ranking.

Key improvements over base orchestrator:
1. Forces more evolution generations (no early convergence on similarity)
2. Uses diversity scoring to encourage latent exploration
3. Re-ranks final survivors using text heuristic quality scores
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch
from torch import Tensor

from latent_reasoning.config import Config, EvolutionConfig
from latent_reasoning.core.encoder import Encoder, LLMEncoder
from latent_reasoning.core.judge import ScorerJudge
from latent_reasoning.core.panel import JudgePanel
from latent_reasoning.core.chain import ChainState
from latent_reasoning.core.heuristic_scorer import HeuristicScorer, DiversityBonusScorer
from latent_reasoning.evolution.loop import EvolutionLoop, EvolutionResult
from latent_reasoning.orchestrator.budget import ComputeBudget
from latent_reasoning.utils.logging import log_event, print_header, print_result, LogLevel, set_verbosity


@dataclass
class ImprovedOrchestrationResult:
    """Result of the improved orchestration process."""

    # Final outputs
    final_latent: Tensor
    decoded_outputs: List[str]
    best_score: float

    # Heuristic scores for outputs
    heuristic_scores: List[float]

    # Survivors
    survivors: List[ChainState]

    # Stats
    generations: int
    total_evaluations: int
    stop_reason: str

    # Comparison info
    diversity_from_seed: float = 0.0


class ImprovedOrchestrator:
    """
    Improved orchestrator that uses diversity-based evolution.

    Key changes from base orchestrator:
    - Raises convergence threshold to 1.1 (effectively disabling early stop)
    - Uses diversity bonus in scoring to encourage exploration
    - Re-ranks decoded outputs using text heuristics
    """

    def __init__(
        self,
        config: Config,
        encoder: Encoder | None = None,
        judge_panel: JudgePanel | None = None,
        diversity_weight: float = 0.3,
    ):
        """
        Initialize the improved orchestrator.

        Args:
            config: Full configuration
            encoder: Encoder instance (optional)
            judge_panel: Judge panel instance (optional)
            diversity_weight: Weight of diversity in scoring (0-1)
        """
        self.config = config
        self.diversity_weight = diversity_weight

        # Modify config to prevent early convergence
        self._modify_config()

        set_verbosity(config.output.verbosity)

        # Initialize encoder
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = self._create_encoder()

        # Initialize judge panel
        if judge_panel is not None:
            self.judge_panel = judge_panel
        else:
            self.judge_panel = self._create_judge_panel()

        # Initialize budget
        self.budget = ComputeBudget(
            max_generations=config.evolution.generations,
            max_evaluations=config.budget.max_evaluations,
            max_time=config.budget.max_time,
        )

        # Initialize evolution loop with modified config
        self.evolution_loop = EvolutionLoop(
            judge_panel=self.judge_panel,
            config=config.evolution,
        )

        # Initialize heuristic scorer for text quality
        self.heuristic_scorer = HeuristicScorer()

        # Initialize diversity scorer
        self.diversity_scorer = DiversityBonusScorer(
            diversity_weight=diversity_weight,
        )

    def _modify_config(self) -> None:
        """Modify config to encourage more exploration."""
        # Raise convergence threshold to prevent early stopping
        self.config.evolution.convergence.threshold = 1.1  # Impossible to reach

        # Increase patience
        self.config.evolution.convergence.patience = 100

        # Increase temperature for more diversity
        self.config.evolution.temperature = 0.8

        # Slower decay for sustained exploration
        self.config.evolution.temperature_decay = 0.98

    def _create_encoder(self) -> Encoder:
        """Create encoder from config."""
        return LLMEncoder(
            model_name=self.config.encoder.model,
            extraction_layer=self.config.encoder.layer,
            pooling=self.config.encoder.pooling,
            device_preference=self.config.encoder.device,
        )

    def _create_judge_panel(self) -> JudgePanel:
        """Create judge panel from config."""
        scorers = []
        for scorer_config in self.config.judges.scorers:
            scorer = ScorerJudge(
                model_name=scorer_config.model,
                extraction_layer=scorer_config.layer,
                head_weights=scorer_config.head,
                canonical_dim=self.encoder.latent_dim,
                device_preference=self.config.encoder.device,
            )
            scorers.append(scorer)

        return JudgePanel(
            scorers=scorers,
            modifiers=[],  # Skip modifiers for now
            aggregation=self.config.judges.aggregation,
            calibrate=self.config.judges.calibrate,
        )

    def run(self, query: str) -> ImprovedOrchestrationResult:
        """
        Run the improved reasoning pipeline.

        Args:
            query: Input query to reason about

        Returns:
            ImprovedOrchestrationResult with ranked outputs
        """
        print_header("Improved Latent Space Reasoning Engine")

        log_event("START", query=query[:50] + "..." if len(query) > 50 else query)

        self.budget.start()

        # Encode query
        log_event("ENCODE", level=LogLevel.VERBOSE)
        seed = self.encoder.encode(query)
        log_event(
            "ENCODED",
            level=LogLevel.VERBOSE,
            shape=tuple(seed.shape),
            norm=f"{seed.norm().item():.2f}",
        )

        # Set scorer reference to query latent
        for scorer in self.judge_panel.scorers:
            scorer.set_reference(embedding=seed)

        # Set diversity scorer seed
        self.diversity_scorer.set_seed(seed)

        # Run evolution
        log_event("EVOLVE", level=LogLevel.NORMAL)
        evolution_result = self.evolution_loop.run(
            seed=seed,
            max_evaluations=self.budget.max_evaluations - self.budget.evaluations_used,
        )

        self.budget.evaluations_used += evolution_result.total_evaluations
        self.budget.generations_used = evolution_result.generations

        # Decode ALL survivors (more options to choose from)
        log_event("DECODE", level=LogLevel.VERBOSE)
        decoded_outputs = []
        for survivor in evolution_result.survivors[:self.config.synthesis.max_survivors]:
            decoded = self.encoder.decode(survivor.latent, query=query)
            decoded_outputs.append(decoded)

        # Score decoded outputs using text heuristics
        log_event("HEURISTIC_SCORE", level=LogLevel.VERBOSE)
        heuristic_scores = []
        for output in decoded_outputs:
            score = self.heuristic_scorer.score(output)
            heuristic_scores.append(score.overall_score)

        # Re-rank by heuristic score
        if decoded_outputs:
            sorted_indices = sorted(
                range(len(heuristic_scores)),
                key=lambda i: heuristic_scores[i],
                reverse=True,
            )
            decoded_outputs = [decoded_outputs[i] for i in sorted_indices]
            heuristic_scores = [heuristic_scores[i] for i in sorted_indices]

            # Update survivors order
            survivors_reranked = [evolution_result.survivors[i] for i in sorted_indices]
        else:
            survivors_reranked = evolution_result.survivors

        # Calculate diversity of best survivor from seed
        if survivors_reranked:
            best_latent = survivors_reranked[0].latent
            diversity = self.diversity_scorer.score_diversity(best_latent)
        else:
            best_latent = seed
            diversity = 0.0

        log_event(
            "DONE",
            level=LogLevel.NORMAL,
            score=f"{evolution_result.best_score:.3f}",
            heuristic=f"{heuristic_scores[0]:.3f}" if heuristic_scores else "N/A",
            generations=evolution_result.generations,
            reason=evolution_result.stop_reason,
        )

        result = ImprovedOrchestrationResult(
            final_latent=best_latent,
            decoded_outputs=decoded_outputs,
            best_score=evolution_result.best_score,
            heuristic_scores=heuristic_scores,
            survivors=survivors_reranked,
            generations=evolution_result.generations,
            total_evaluations=evolution_result.total_evaluations,
            stop_reason=evolution_result.stop_reason,
            diversity_from_seed=diversity,
        )

        if decoded_outputs:
            print_result(
                decoded_outputs[0],
                heuristic_scores[0] if heuristic_scores else 0.0,
                generations=evolution_result.generations,
                evaluations=evolution_result.total_evaluations,
            )

        return result

    def run_baseline(self, query: str) -> str:
        """Run baseline generation without latent space reasoning."""
        seed = self.encoder.encode(query)
        return self.encoder.decode(seed, query=query)

    def compare(self, query: str) -> dict:
        """Compare baseline vs improved latent space reasoning."""
        # Run baseline
        baseline_output = self.run_baseline(query)
        baseline_heuristic = self.heuristic_scorer.score(baseline_output)

        # Run improved LR
        result = self.run(query)

        return {
            "query": query,
            "baseline": baseline_output,
            "baseline_heuristic": baseline_heuristic.overall_score,
            "latent_reasoning": result.decoded_outputs[0] if result.decoded_outputs else "",
            "lr_heuristic": result.heuristic_scores[0] if result.heuristic_scores else 0.0,
            "lr_score": result.best_score,
            "generations": result.generations,
            "evaluations": result.total_evaluations,
            "diversity": result.diversity_from_seed,
        }

    def reset(self) -> None:
        """Reset the orchestrator state."""
        self.budget.reset()
        self.evolution_loop.reset()
