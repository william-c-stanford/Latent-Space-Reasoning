"""Crossover strategies for evolutionary optimization."""

from __future__ import annotations

import random

import torch
from torch import Tensor

from latent_reasoning.evolution.operators import CrossoverStrategy


class MeanCrossover(CrossoverStrategy):
    """
    Mean crossover strategy.

    Simply averages the two parents.
    """

    def crossover(
        self,
        parent_a: Tensor,
        parent_b: Tensor,
        score_a: float = 0.0,
        score_b: float = 0.0,
    ) -> Tensor:
        return (parent_a + parent_b) / 2


class WeightedCrossover(CrossoverStrategy):
    """
    Weighted crossover strategy.

    Weights parents by their scores.
    """

    def crossover(
        self,
        parent_a: Tensor,
        parent_b: Tensor,
        score_a: float = 0.0,
        score_b: float = 0.0,
    ) -> Tensor:
        # Shift scores to be positive
        min_score = min(score_a, score_b)
        shifted_a = score_a - min_score + 0.1
        shifted_b = score_b - min_score + 0.1

        total = shifted_a + shifted_b
        weight_a = shifted_a / total
        weight_b = shifted_b / total

        return weight_a * parent_a + weight_b * parent_b


class InterpolationCrossover(CrossoverStrategy):
    """
    Interpolation crossover strategy.

    Picks a random point on the line between the two parents.
    """

    def crossover(
        self,
        parent_a: Tensor,
        parent_b: Tensor,
        score_a: float = 0.0,
        score_b: float = 0.0,
    ) -> Tensor:
        # Random interpolation factor
        alpha = random.random()

        return alpha * parent_a + (1 - alpha) * parent_b


class SliceCrossover(CrossoverStrategy):
    """
    Slice crossover strategy.

    Takes the first half of one parent and second half of the other.
    """

    def crossover(
        self,
        parent_a: Tensor,
        parent_b: Tensor,
        score_a: float = 0.0,
        score_b: float = 0.0,
    ) -> Tensor:
        # Find midpoint
        mid = parent_a.shape[-1] // 2

        # Combine slices
        child = torch.cat([parent_a[..., :mid], parent_b[..., mid:]], dim=-1)

        return child


class BlendCrossover(CrossoverStrategy):
    """
    Blend crossover strategy.

    Creates offspring by blending parent genes with some randomness.
    """

    def __init__(self, alpha: float = 0.5):
        """
        Initialize blend crossover.

        Args:
            alpha: Blend factor (higher = more exploration beyond parents)
        """
        self.alpha = alpha

    def crossover(
        self,
        parent_a: Tensor,
        parent_b: Tensor,
        score_a: float = 0.0,
        score_b: float = 0.0,
    ) -> Tensor:
        # For each dimension, sample uniformly from extended range
        diff = parent_b - parent_a
        min_val = torch.minimum(parent_a, parent_b) - self.alpha * torch.abs(diff)
        max_val = torch.maximum(parent_a, parent_b) + self.alpha * torch.abs(diff)

        # Sample uniformly
        child = min_val + torch.rand_like(parent_a) * (max_val - min_val)

        return child


def get_crossover_strategy(name: str, **kwargs) -> CrossoverStrategy:
    """Factory function to get a crossover strategy by name."""
    strategies = {
        "mean": MeanCrossover,
        "weighted": WeightedCrossover,
        "interpolation": InterpolationCrossover,
        "slice": SliceCrossover,
        "blend": BlendCrossover,
    }

    if name not in strategies:
        raise ValueError(f"Unknown crossover strategy: {name}")

    return strategies[name](**kwargs)


def select_crossover_pairs(
    population: list[Tensor],
    scores: list[float],
    n_pairs: int,
    diversity_threshold: float = 0.3,
) -> list[tuple[int, int]]:
    """
    Select pairs of candidates for crossover.

    Prefers diverse pairs (not too similar) with good scores.

    Args:
        population: List of latent vectors
        scores: Corresponding scores
        n_pairs: Number of pairs to select
        diversity_threshold: Minimum cosine distance for pair selection

    Returns:
        List of (index_a, index_b) tuples
    """
    if len(population) < 2:
        return []

    pairs = []
    used = set()

    # Sort by score to prefer good candidates
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    for i in sorted_indices:
        if len(pairs) >= n_pairs:
            break

        for j in sorted_indices:
            if i >= j or (i, j) in used:
                continue

            # Check diversity
            cos_sim = torch.nn.functional.cosine_similarity(
                population[i].flatten().unsqueeze(0).float(),
                population[j].flatten().unsqueeze(0).float(),
            ).item()

            if cos_sim < (1 - diversity_threshold):
                pairs.append((i, j))
                used.add((i, j))
                used.add((j, i))
                break

    return pairs


def population_diversity(population: list[Tensor]) -> float:
    """
    Compute the diversity of a population.

    Returns:
        Float between 0 (all identical) and 1 (maximally diverse)
    """
    if len(population) < 2:
        return 0.0

    # Compute average pairwise cosine distance
    distances = []
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            cos_sim = torch.nn.functional.cosine_similarity(
                population[i].flatten().unsqueeze(0).float(),
                population[j].flatten().unsqueeze(0).float(),
            ).item()
            distances.append(1 - cos_sim)  # Convert similarity to distance

    return sum(distances) / len(distances)
