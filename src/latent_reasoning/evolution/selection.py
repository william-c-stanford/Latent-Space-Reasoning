"""
Selection strategies for evolutionary optimization in latent space.

This module implements various selection strategies that determine which candidates
survive from one generation to the next. Selection is crucial for balancing
exploitation (keeping the best solutions) with exploration (maintaining diversity).

Available strategies:
- ElitistSelection: Always keeps top performers, adds randomness for diversity
- TournamentSelection: Runs competitions between random subsets
- RouletteSelection: Probability proportional to fitness scores

Each strategy has different characteristics in terms of selection pressure,
diversity maintenance, and computational efficiency.
"""

from __future__ import annotations

import random
from typing import List, Tuple

import torch
from torch import Tensor

from latent_reasoning.evolution.operators import SelectionStrategy


class ElitistSelection(SelectionStrategy):
    """
    Elitist selection strategy with fitness-proportional selection for remaining slots.

    This strategy guarantees that the best candidates always survive (elitism) while
    filling remaining slots through weighted random selection. This balances strong
    exploitation of good solutions with exploration through diversity.

    Algorithm:
    1. Sort all candidates by fitness score (descending)
    2. Always select the top elite_k candidates
    3. For remaining slots, use fitness-proportional random selection
    4. Ensure no duplicates in the final selection

    Characteristics:
    - **High exploitation**: Best solutions always survive
    - **Moderate exploration**: Random selection maintains some diversity
    - **Stable convergence**: Prevents loss of good solutions
    - **Configurable pressure**: elite_k controls selection pressure

    Best for:
    - Problems where preserving the best solutions is critical
    - Situations requiring steady progress without backtracking
    - Balancing convergence speed with diversity maintenance
    """

    def __init__(self, elite_k: int = 2):
        """
        Initialize elitist selection strategy.

        Args:
            elite_k: Number of elite candidates to always keep. Should be
                less than the typical population size. Higher values increase
                selection pressure but may reduce diversity.
                - 1-2: Moderate elitism (recommended)
                - 3-5: Strong elitism (faster convergence, less diversity)
                - 0: No elitism (pure fitness-proportional selection)

        Example:
            >>> # Moderate elitism - keeps top 2, selects rest randomly
            >>> strategy = ElitistSelection(elite_k=2)
            >>>
            >>> # Strong elitism - keeps top 4 for faster convergence
            >>> strategy = ElitistSelection(elite_k=4)
        """
        self.elite_k = elite_k

    def select(
        self,
        population: List[Tensor],
        scores: List[float],
        n_survivors: int,
    ) -> Tuple[List[Tensor], List[float]]:
        if len(population) <= n_survivors:
            return population.copy(), scores.copy()

        # Sort by score
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        # Always keep elite
        elite_count = min(self.elite_k, n_survivors)
        selected_indices = sorted_indices[:elite_count]

        # Select remaining from non-elite, weighted by score
        remaining_needed = n_survivors - elite_count
        if remaining_needed > 0:
            non_elite_indices = sorted_indices[elite_count:]
            non_elite_scores = [scores[i] for i in non_elite_indices]

            # Convert scores to selection probabilities
            # Shift scores to be positive
            min_score = min(non_elite_scores) if non_elite_scores else 0
            shifted_scores = [s - min_score + 0.1 for s in non_elite_scores]
            total = sum(shifted_scores)

            if total > 0:
                probs = [s / total for s in shifted_scores]
                additional = random.choices(
                    non_elite_indices,
                    weights=probs,
                    k=min(remaining_needed, len(non_elite_indices)),
                )
                selected_indices.extend(additional)

        selected_latents = [population[i].clone() for i in selected_indices]
        selected_scores = [scores[i] for i in selected_indices]

        return selected_latents, selected_scores


class TournamentSelection(SelectionStrategy):
    """
    Tournament selection strategy.

    Randomly selects groups and picks the best from each group.
    """

    def __init__(self, tournament_size: int = 3):
        """
        Initialize tournament selection.

        Args:
            tournament_size: Number of candidates in each tournament
        """
        self.tournament_size = tournament_size

    def select(
        self,
        population: List[Tensor],
        scores: List[float],
        n_survivors: int,
    ) -> Tuple[List[Tensor], List[float]]:
        if len(population) <= n_survivors:
            return population.copy(), scores.copy()

        selected_indices = []
        used_indices = set()

        while len(selected_indices) < n_survivors:
            # Random tournament
            available = [i for i in range(len(population)) if i not in used_indices]
            if not available:
                # Reset if we've used all
                used_indices.clear()
                available = list(range(len(population)))

            tournament = random.sample(available, min(self.tournament_size, len(available)))

            # Pick winner
            winner = max(tournament, key=lambda i: scores[i])
            selected_indices.append(winner)
            used_indices.add(winner)

        selected_latents = [population[i].clone() for i in selected_indices]
        selected_scores = [scores[i] for i in selected_indices]

        return selected_latents, selected_scores


class RankSelection(SelectionStrategy):
    """
    Rank selection strategy.

    Simply selects the top N by score.
    """

    def select(
        self,
        population: List[Tensor],
        scores: List[float],
        n_survivors: int,
    ) -> Tuple[List[Tensor], List[float]]:
        if len(population) <= n_survivors:
            return population.copy(), scores.copy()

        # Sort by score descending
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_indices = sorted_indices[:n_survivors]

        selected_latents = [population[i].clone() for i in top_indices]
        selected_scores = [scores[i] for i in top_indices]

        return selected_latents, selected_scores


class RouletteSelection(SelectionStrategy):
    """
    Roulette wheel selection strategy.

    Selection probability proportional to fitness score.
    """

    def select(
        self,
        population: List[Tensor],
        scores: List[float],
        n_survivors: int,
    ) -> Tuple[List[Tensor], List[float]]:
        if len(population) <= n_survivors:
            return population.copy(), scores.copy()

        # Shift scores to be positive
        min_score = min(scores)
        shifted_scores = [s - min_score + 0.1 for s in scores]
        total = sum(shifted_scores)

        probs = [s / total for s in shifted_scores]

        # Select with replacement allowed
        selected_indices = random.choices(
            range(len(population)),
            weights=probs,
            k=n_survivors,
        )

        selected_latents = [population[i].clone() for i in selected_indices]
        selected_scores = [scores[i] for i in selected_indices]

        return selected_latents, selected_scores


def get_selection_strategy(name: str, **kwargs) -> SelectionStrategy:
    """Factory function to get a selection strategy by name."""
    strategies = {
        "elitist": ElitistSelection,
        "tournament": TournamentSelection,
        "rank": RankSelection,
        "roulette": RouletteSelection,
    }

    if name not in strategies:
        raise ValueError(f"Unknown selection strategy: {name}")

    return strategies[name](**kwargs)
