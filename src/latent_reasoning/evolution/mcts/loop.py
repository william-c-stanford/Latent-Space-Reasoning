"""
MCTS Evolution Loop - Drop-in replacement for standard EvolutionLoop.

This module provides MCTSEvolutionLoop, which has the same interface as
EvolutionLoop but uses Monte Carlo Tree Search instead of population-based
evolutionary algorithms.

=============================================================================
WHY A DROP-IN REPLACEMENT?
=============================================================================

The existing orchestrator expects an evolution loop with this interface:

    result = evolution_loop.run(seed, max_evaluations)

By matching this interface, we can use MCTS without changing any other code.
Just swap out the evolution loop and everything else stays the same.

This is important because:
1. No changes to Engine, Orchestrator, or other components
2. Easy A/B testing between evolution and MCTS
3. Can switch strategies at runtime
4. Existing configs still work

=============================================================================
HOW IT INTEGRATES
=============================================================================

The MCTSEvolutionLoop wraps LatentMCTS and translates:
- EvolutionConfig -> MCTSConfig (reuses relevant parameters)
- MCTSResult -> EvolutionResult (same return type as EvolutionLoop)

This means you can do:

    from latent_reasoning.evolution.mcts import MCTSEvolutionLoop
    
    # Replace evolution loop in orchestrator
    orchestrator.evolution_loop = MCTSEvolutionLoop(
        judge_panel=orchestrator.judge_panel,
        config=config.evolution,
    )
    
    # Use orchestrator normally
    result = orchestrator.run("Your query")

=============================================================================
MAPPING PARAMETERS
=============================================================================

EvolutionConfig -> MCTSConfig mapping:
- generations -> n_iterations (MCTS iterations ≈ generations * chains)
- chains -> affects n_iterations multiplier
- temperature -> base temperature for expansion
- temperature_decay -> temperature_decay
- convergence.threshold -> early_stop_threshold
- convergence.patience -> early_stop_patience
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
from torch import Tensor

from latent_reasoning.config import EvolutionConfig
from latent_reasoning.core.chain import ChainState
from latent_reasoning.core.panel import JudgePanel
from latent_reasoning.evolution.loop import EvolutionResult
from latent_reasoning.evolution.mcts.search import LatentMCTS, MCTSConfig, MCTSResult
from latent_reasoning.utils.logging import log_event, log_generation, LogLevel


class MCTSEvolutionLoop:
    """
    MCTS-based evolution loop - drop-in replacement for EvolutionLoop.
    
    This class has the same interface as EvolutionLoop, so you can swap
    it in without changing any other code. It uses Monte Carlo Tree Search
    instead of population-based evolutionary algorithms.
    
    Key Differences from EvolutionLoop:
    - Tree-structured search instead of population
    - UCB-based selection instead of fitness-proportional
    - Backpropagation of scores instead of pure replacement
    - More sample-efficient (fewer evaluations for same quality)
    
    Attributes:
        judge_panel: Panel of judges for evaluation (same as EvolutionLoop)
        config: Evolution configuration (automatically mapped to MCTSConfig)
        
    Example:
        >>> from latent_reasoning.evolution.mcts import MCTSEvolutionLoop
        >>> 
        >>> # Create with same config as EvolutionLoop
        >>> mcts_loop = MCTSEvolutionLoop(
        ...     judge_panel=judge_panel,
        ...     config=config.evolution,
        ... )
        >>> 
        >>> # Use exactly like EvolutionLoop
        >>> result = mcts_loop.run(seed, max_evaluations=100)
        >>> print(f"Best score: {result.best_score:.3f}")
    """
    
    def __init__(
        self,
        judge_panel: JudgePanel,
        config: EvolutionConfig,
        encoder=None,
    ):
        """
        Initialize MCTS Evolution Loop.
        
        Args:
            judge_panel: Panel of judges for evaluation
            config: EvolutionConfig (will be mapped to MCTSConfig)
            encoder: Optional encoder reference
        """
        self.judge_panel = judge_panel
        self.evolution_config = config
        self.encoder = encoder
        
        # Map EvolutionConfig to MCTSConfig
        self.mcts_config = self._map_config(config)
        
        # Create MCTS instance
        self.mcts = LatentMCTS(
            judge_panel=judge_panel,
            encoder=encoder,
            config=self.mcts_config,
        )
        
        # Track state
        self.total_evaluations = 0
    
    def _map_config(self, config: EvolutionConfig) -> MCTSConfig:
        """
        Map EvolutionConfig parameters to MCTSConfig.
        
        The key insight: generations * chains ≈ total work in evolution.
        We map this to MCTS iterations, which is the equivalent unit of work.
        
        Args:
            config: Evolution configuration
            
        Returns:
            Equivalent MCTS configuration
        """
        # MCTS iterations ≈ generations * chains (similar total evaluations)
        n_iterations = config.generations * config.chains
        
        # Map exploration: lower temperature = more exploitation
        # In MCTS, this is the exploration constant
        # Default sqrt(2) ≈ 1.41 is balanced
        exploration_constant = 1.41
        
        return MCTSConfig(
            n_iterations=n_iterations,
            exploration_constant=exploration_constant,
            max_depth=config.generations,  # Depth ≈ generations
            temperature_decay=config.temperature_decay,
            min_temperature=0.1,
            use_modifier=len(self.judge_panel.modifiers) > 0,
            early_stop_threshold=config.convergence.threshold,
            early_stop_patience=config.convergence.patience,
        )
    
    def run(
        self,
        seed: Tensor,
        max_evaluations: int | None = None,
    ) -> EvolutionResult:
        """
        Run MCTS search - same interface as EvolutionLoop.run().
        
        This is the main method. It has the exact same signature and return
        type as EvolutionLoop.run(), so you can swap them without changes.
        
        Args:
            seed: Seed latent vector (from encoder.encode())
            max_evaluations: Optional limit on judge evaluations
            
        Returns:
            EvolutionResult with best latent, score, survivors, and stats
            
        Note:
            The 'survivors' in the result are the top-k nodes from the
            MCTS tree, converted to ChainState objects for compatibility.
        """
        log_event("MCTS_START", level=LogLevel.VERBOSE, 
                  iterations=self.mcts_config.n_iterations)
        
        # Run MCTS search
        mcts_result = self.mcts.search(seed, max_evaluations)
        
        # Log progress
        for i, h in enumerate(mcts_result.history[::10]):  # Every 10th iteration
            log_generation(
                gen=h['iteration'],
                chains=h['tree_size'],
                best_score=h['best_score'],
                mean_score=h['current_score'],
            )
        
        # Convert MCTS result to EvolutionResult format
        evolution_result = self._convert_result(mcts_result, seed)
        
        log_event("MCTS_DONE", level=LogLevel.VERBOSE,
                  best_score=f"{mcts_result.best_score:.3f}",
                  iterations=mcts_result.iterations,
                  evaluations=mcts_result.total_evaluations)
        
        self.total_evaluations = mcts_result.total_evaluations
        
        return evolution_result
    
    def _convert_result(self, mcts_result: MCTSResult, seed: Tensor) -> EvolutionResult:
        """
        Convert MCTSResult to EvolutionResult.
        
        This ensures compatibility with the rest of the system, which
        expects EvolutionResult from the evolution loop.
        """
        # Extract top-k latents as "survivors"
        top_k_latents = self.mcts.get_top_k_latents(
            mcts_result.root, 
            k=self.evolution_config.selection.survivors
        )
        
        # Convert to ChainState objects (what EvolutionResult expects)
        survivors = []
        for i, latent in enumerate(top_k_latents):
            survivors.append(ChainState(
                latent=latent,
                score=mcts_result.root.best_score if i == 0 else 0.0,
                generation=mcts_result.iterations,
            ))
        
        # Build history in same format as EvolutionLoop
        history = []
        for h in mcts_result.history:
            history.append({
                'generation': h['iteration'],
                'best_score': h['best_score'],
                'mean_score': h['current_score'],
                'num_chains': h['tree_size'],
                'diversity': 0.0,  # Not applicable to tree search
            })
        
        # Determine stop reason
        if mcts_result.best_score >= self.mcts_config.early_stop_threshold:
            stop_reason = "score_threshold"
        elif mcts_result.iterations >= self.mcts_config.n_iterations:
            stop_reason = "max_iterations"
        else:
            stop_reason = "budget_exhausted"
        
        return EvolutionResult(
            best_latent=mcts_result.best_latent,
            best_score=mcts_result.best_score,
            survivors=survivors,
            generations=mcts_result.iterations,
            total_evaluations=mcts_result.total_evaluations,
            history=history,
            converged=mcts_result.best_score >= self.mcts_config.early_stop_threshold,
            stop_reason=stop_reason,
        )
    
    def reset(self) -> None:
        """Reset the evolution loop state."""
        self.total_evaluations = 0
        # MCTS doesn't have persistent state to reset
        # (tree is built fresh each run)


def create_mcts_orchestrator(config, encoder=None, judge_panel=None):
    """
    Create an orchestrator that uses MCTS instead of evolution.
    
    This is a convenience function for quickly setting up an MCTS-based
    orchestrator without manually swapping components.
    
    Args:
        config: Full Config object
        encoder: Optional pre-created encoder
        judge_panel: Optional pre-created judge panel
        
    Returns:
        Orchestrator with MCTSEvolutionLoop
        
    Example:
        >>> from latent_reasoning import Config
        >>> from latent_reasoning.evolution.mcts import create_mcts_orchestrator
        >>> 
        >>> config = Config()
        >>> orchestrator = create_mcts_orchestrator(config)
        >>> result = orchestrator.run("Your query")
    """
    from latent_reasoning.orchestrator.orchestrator import Orchestrator
    
    # Create standard orchestrator
    orchestrator = Orchestrator(
        config=config,
        encoder=encoder,
        judge_panel=judge_panel,
    )
    
    # Replace evolution loop with MCTS
    orchestrator.evolution_loop = MCTSEvolutionLoop(
        judge_panel=orchestrator.judge_panel,
        config=config.evolution,
        encoder=orchestrator.encoder,
    )
    
    return orchestrator

