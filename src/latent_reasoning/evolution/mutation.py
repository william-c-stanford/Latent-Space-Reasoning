"""
Mutation strategies for evolutionary optimization in latent space.

This module implements various mutation strategies that create variations of
existing candidates by introducing controlled random changes. Mutation is the
primary source of exploration in evolutionary algorithms, allowing discovery
of new regions in the latent space.

Available strategies:
- GaussianMutation: Adds random Gaussian noise for general exploration
- DirectedMutation: Follows hints from modifier judges for guided search
- AdaptiveMutation: Adjusts mutation strength based on population diversity

Each strategy balances exploration (finding new areas) with exploitation
(refining existing solutions) through different approaches to randomness
and guidance utilization.
"""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from latent_reasoning.evolution.operators import MutationStrategy


class GaussianMutation(MutationStrategy):
    """
    Gaussian mutation strategy for general-purpose exploration.

    This strategy adds random Gaussian (normal) noise to latent vectors,
    providing unbiased exploration around the current candidate. It's the
    most common and well-understood mutation approach, suitable for most
    optimization problems.

    Algorithm:
    1. Generate random noise from normal distribution N(0, noise_scale²)
    2. Scale noise by temperature parameter
    3. Add scaled noise to the candidate vector
    4. Return the perturbed vector

    Characteristics:
    - **Unbiased exploration**: Equal probability in all directions
    - **Scalable intensity**: Temperature controls mutation strength
    - **Gaussian distribution**: Most changes are small, few are large
    - **Simple and robust**: Works well across different problem types

    Best for:
    - General-purpose optimization when no specific guidance is available
    - Early exploration phases when discovering the landscape
    - Problems where all dimensions are equally important
    - Baseline comparison with more sophisticated strategies
    """

    def __init__(self, noise_scale: float = 1.0):
        """
        Initialize Gaussian mutation strategy.

        Args:
            noise_scale: Base scale factor for the Gaussian noise. This controls
                the standard deviation of the noise distribution:
                - 0.1-0.5: Conservative mutation (small changes)
                - 1.0: Standard mutation (recommended default)
                - 2.0-5.0: Aggressive mutation (large changes)

                The actual mutation strength is: noise_scale * temperature

        Example:
            >>> # Conservative mutation for fine-tuning
            >>> strategy = GaussianMutation(noise_scale=0.3)
            >>>
            >>> # Standard mutation for general use
            >>> strategy = GaussianMutation(noise_scale=1.0)
            >>>
            >>> # Aggressive mutation for exploration
            >>> strategy = GaussianMutation(noise_scale=2.0)
        """
        self.noise_scale = noise_scale

    def mutate(
        self,
        candidate: Tensor,
        modification_hint: Tensor | None,
        temperature: float,
    ) -> Tensor:
        # Generate Gaussian noise
        noise = torch.randn_like(candidate) * self.noise_scale

        # Scale by temperature
        scaled_noise = temperature * noise

        return candidate + scaled_noise


class DirectedMutation(MutationStrategy):
    """
    Directed mutation strategy that follows guidance from modifier judges.

    This strategy uses hints from modifier judges to guide mutation in promising
    directions while still maintaining some random exploration. It's more
    sophisticated than pure random mutation and can lead to faster convergence
    when good guidance is available.

    Algorithm:
    1. If modification hint is available:
       a. Use hint as primary direction (scaled by temperature)
       b. Add small amount of random noise for exploration
    2. If no hint available:
       a. Fall back to Gaussian mutation
    3. Blend directed and random components based on hint_weight

    Characteristics:
    - **Guided exploration**: Uses judge feedback to improve search direction
    - **Adaptive behavior**: Falls back to random when no guidance available
    - **Balanced approach**: Combines exploitation (hints) with exploration (noise)
    - **Judge-dependent**: Effectiveness depends on quality of modifier judges

    Best for:
    - Problems where modifier judges can provide meaningful guidance
    - Later stages of evolution when refinement is needed
    - Situations requiring faster convergence than pure random search
    - Complex latent spaces where random exploration is inefficient
    """

    def __init__(self, trust: float = 0.7, noise_scale: float = 0.3):
        """
        Initialize directed mutation.

        Args:
            trust: How much to trust the modification hint (0-1)
            noise_scale: Scale factor for random exploration
        """
        self.trust = trust
        self.noise_scale = noise_scale

    def mutate(
        self,
        candidate: Tensor,
        modification_hint: Tensor | None,
        temperature: float,
    ) -> Tensor:
        device = candidate.device
        dtype = candidate.dtype

        # Generate random direction
        random_direction = torch.randn_like(candidate)
        random_direction = F.normalize(random_direction.flatten(), dim=0).reshape(candidate.shape)

        if modification_hint is not None:
            # Normalize the hint
            hint = modification_hint.to(device=device, dtype=dtype)
            hint_norm = hint.norm()
            if hint_norm > 1e-6:
                hint_normalized = hint / hint_norm
            else:
                hint_normalized = hint

            # Blend hint with random direction
            direction = (
                self.trust * hint_normalized +
                (1 - self.trust) * random_direction * self.noise_scale
            )
        else:
            # No hint, use random direction
            direction = random_direction * self.noise_scale

        # Apply mutation
        step = temperature * direction

        return candidate + step


class AdaptiveMutation(MutationStrategy):
    """
    Adaptive mutation strategy.

    Adjusts mutation strength based on historical performance.
    Uses a stronger mutation when stuck, gentler when progressing.
    """

    def __init__(
        self,
        base_trust: float = 0.7,
        noise_scale: float = 0.3,
        adaptation_rate: float = 0.1,
    ):
        """
        Initialize adaptive mutation.

        Args:
            base_trust: Base trust level for modification hints
            noise_scale: Base scale for random noise
            adaptation_rate: How quickly to adapt
        """
        self.base_trust = base_trust
        self.noise_scale = noise_scale
        self.adaptation_rate = adaptation_rate

        # Tracking
        self.current_trust = base_trust
        self.current_noise = noise_scale
        self.last_scores: list[float] = []

    def mutate(
        self,
        candidate: Tensor,
        modification_hint: Tensor | None,
        temperature: float,
    ) -> Tensor:
        device = candidate.device
        dtype = candidate.dtype

        # Generate random direction
        random_direction = torch.randn_like(candidate)
        random_direction = F.normalize(random_direction.flatten(), dim=0).reshape(candidate.shape)

        if modification_hint is not None:
            hint = modification_hint.to(device=device, dtype=dtype)
            hint_norm = hint.norm()
            if hint_norm > 1e-6:
                hint_normalized = hint / hint_norm
            else:
                hint_normalized = hint

            # Use adaptive trust level
            direction = (
                self.current_trust * hint_normalized +
                (1 - self.current_trust) * random_direction * self.current_noise
            )
        else:
            direction = random_direction * self.current_noise

        step = temperature * direction

        return candidate + step

    def update_adaptation(self, score: float) -> None:
        """
        Update adaptation parameters based on new score.

        Args:
            score: Latest score from evaluation
        """
        self.last_scores.append(score)

        # Keep only recent scores
        if len(self.last_scores) > 5:
            self.last_scores = self.last_scores[-5:]

        if len(self.last_scores) < 2:
            return

        # Check if improving
        recent_improvement = self.last_scores[-1] - self.last_scores[-2]

        if recent_improvement > 0:
            # Improving: increase trust, decrease noise
            self.current_trust = min(0.95, self.current_trust + self.adaptation_rate)
            self.current_noise = max(0.1, self.current_noise - self.adaptation_rate)
        else:
            # Not improving: decrease trust, increase noise (explore more)
            self.current_trust = max(0.3, self.current_trust - self.adaptation_rate)
            self.current_noise = min(0.8, self.current_noise + self.adaptation_rate)

    def reset(self) -> None:
        """Reset adaptation state."""
        self.current_trust = self.base_trust
        self.current_noise = self.noise_scale
        self.last_scores.clear()


def get_mutation_strategy(name: str, **kwargs) -> MutationStrategy:
    """Factory function to get a mutation strategy by name."""
    strategies = {
        "gaussian": GaussianMutation,
        "directed": DirectedMutation,
        "adaptive": AdaptiveMutation,
    }

    if name not in strategies:
        raise ValueError(f"Unknown mutation strategy: {name}")

    return strategies[name](**kwargs)
