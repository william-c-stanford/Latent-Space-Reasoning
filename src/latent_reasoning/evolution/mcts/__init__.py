"""
Latent Monte Carlo Tree Search (L-MCTS).

A geometry-aware exploration strategy for latent space reasoning.
This module provides an alternative to the standard evolutionary loop,
using tree-structured search instead of population-based evolution.

=============================================================================
QUICK START
=============================================================================

Basic usage:

    >>> from latent_reasoning import Engine
    >>> from latent_reasoning.evolution.mcts import LatentMCTS, MCTSConfig
    >>> 
    >>> # Get the engine's components
    >>> engine = Engine(encoder="Qwen/Qwen3-0.6B")
    >>> orchestrator = engine._get_orchestrator()
    >>> 
    >>> # Create MCTS
    >>> mcts = LatentMCTS(
    ...     judge_panel=orchestrator.judge_panel,
    ...     encoder=orchestrator.encoder,
    ... )
    >>> 
    >>> # Search
    >>> seed = orchestrator.encoder.encode("How to implement caching?")
    >>> result = mcts.search(seed)
    >>> 
    >>> # Decode best latent
    >>> response = orchestrator.encoder.decode(result.best_latent, query="How to implement caching?")

Using with MCTSEvolutionLoop (drop-in replacement for EvolutionLoop):

    >>> from latent_reasoning.evolution.mcts import MCTSEvolutionLoop
    >>> 
    >>> # This plugs directly into the existing system
    >>> mcts_loop = MCTSEvolutionLoop(
    ...     judge_panel=orchestrator.judge_panel,
    ...     config=orchestrator.config.evolution,
    ... )
    >>> result = mcts_loop.run(seed)

=============================================================================
MODULE CONTENTS
=============================================================================

Classes:
    LatentNode: A node in the latent search tree
    LatentMCTS: The main MCTS search class
    MCTSConfig: Configuration for MCTS search
    MCTSResult: Result from MCTS search
    MCTSEvolutionLoop: Drop-in replacement for EvolutionLoop

Functions:
    mcts_search: Quick search function
    create_root_from_latent: Create root node from encoded query

=============================================================================
WHEN TO USE MCTS vs EVOLUTION
=============================================================================

Use MCTS when:
- You want structured exploration (paths, not just points)
- The latent space has local structure you want to exploit
- You care about sample efficiency (fewer evaluations)
- You want to understand the search process (tree is inspectable)

Use Evolution when:
- You want maximum diversity in candidates
- The scoring function is very noisy
- You have plenty of compute budget
- You want multiple independent solutions
"""

from latent_reasoning.evolution.mcts.node import (
    LatentNode,
    create_root_from_latent,
)

from latent_reasoning.evolution.mcts.search import (
    LatentMCTS,
    MCTSConfig,
    MCTSResult,
    mcts_search,
)

from latent_reasoning.evolution.mcts.loop import (
    MCTSEvolutionLoop,
)

__all__ = [
    # Node
    "LatentNode",
    "create_root_from_latent",
    # Search
    "LatentMCTS",
    "MCTSConfig", 
    "MCTSResult",
    "mcts_search",
    # Integration
    "MCTSEvolutionLoop",
]

