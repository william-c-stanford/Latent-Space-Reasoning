#!/usr/bin/env python3
"""
L-MCTS Quick Start - See MCTS working in under 2 minutes.

This script demonstrates the basic usage of Latent Monte Carlo Tree Search.
Run it to see MCTS explore the latent space and find improved reasoning.

Usage:
    python examples/mcts/quick_start.py
    
    # With custom query:
    python examples/mcts/quick_start.py --query "Your question here"
    
    # With different model:
    python examples/mcts/quick_start.py --model Qwen/Qwen3-1.7B
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from latent_reasoning import Engine, Config
from latent_reasoning.evolution.mcts import LatentMCTS, MCTSConfig


def main():
    parser = argparse.ArgumentParser(description="L-MCTS Quick Start")
    parser.add_argument("--query", default="How do I implement a thread-safe cache?",
                        help="Query to process")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B",
                        help="Model to use")
    parser.add_argument("--iterations", type=int, default=30,
                        help="MCTS iterations")
    args = parser.parse_args()
    
    print("="*60)
    print("L-MCTS Quick Start")
    print("="*60)
    print(f"\nQuery: {args.query}")
    print(f"Model: {args.model}")
    print(f"Iterations: {args.iterations}")
    
    # =========================================================================
    # STEP 1: Create Engine and get components
    # =========================================================================
    print("\n[1/4] Loading model...")
    
    config = Config()
    config.encoder.model = args.model
    config.output.verbosity = "minimal"
    
    engine = Engine(config=config)
    orchestrator = engine._get_orchestrator()
    
    print(f"      Latent dimension: {orchestrator.encoder.latent_dim}")
    
    # =========================================================================
    # STEP 2: Encode the query
    # =========================================================================
    print("\n[2/4] Encoding query...")
    
    seed_latent = orchestrator.encoder.encode(args.query)
    print(f"      Latent shape: {seed_latent.shape}")
    print(f"      Latent norm: {seed_latent.norm().item():.2f}")
    
    # Set reference for judges (they need to know what we're optimizing for)
    for scorer in orchestrator.judge_panel.scorers:
        scorer.set_reference(embedding=seed_latent)
    
    # =========================================================================
    # STEP 3: Run MCTS
    # =========================================================================
    print("\n[3/4] Running MCTS...")
    
    mcts_config = MCTSConfig(
        n_iterations=args.iterations,
        exploration_constant=1.41,  # sqrt(2) - balanced exploration
        max_depth=10,
        use_modifier=len(orchestrator.judge_panel.modifiers) > 0,
    )
    
    mcts = LatentMCTS(
        judge_panel=orchestrator.judge_panel,
        encoder=orchestrator.encoder,
        config=mcts_config,
    )
    
    result = mcts.search(seed_latent)
    
    print(f"\n      Best score: {result.best_score:.3f}")
    print(f"      Iterations: {result.iterations}")
    print(f"      Evaluations: {result.total_evaluations}")
    print(f"      Tree size: {mcts._count_nodes(result.root)} nodes")
    print(f"      Path length: {len(result.best_trajectory)} steps")
    
    # =========================================================================
    # STEP 4: Decode the best latent
    # =========================================================================
    print("\n[4/4] Decoding best latent...")
    
    output = orchestrator.encoder.decode(
        result.best_latent,
        query=args.query,
        max_new_tokens=1024,
        temperature=0.7,
    )
    
    print("\n" + "="*60)
    print("MCTS OUTPUT")
    print("="*60)
    print(output)
    print("="*60)
    
    # =========================================================================
    # BONUS: Show tree structure
    # =========================================================================
    print("\n" + "-"*60)
    print("TREE STRUCTURE (first 3 levels)")
    print("-"*60)
    
    def print_tree(node, prefix="", is_last=True, max_depth=3):
        if node.depth > max_depth:
            return
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}Node(depth={node.depth}, visits={node.visits}, value={node.value:.3f})")
        
        children = node.children
        for i, child in enumerate(children[:5]):  # Limit children shown
            is_last_child = (i == len(children) - 1) or (i == 4)
            new_prefix = prefix + ("    " if is_last else "│   ")
            print_tree(child, new_prefix, is_last_child, max_depth)
        
        if len(children) > 5:
            new_prefix = prefix + ("    " if is_last else "│   ")
            print(f"{new_prefix}... and {len(children) - 5} more children")
    
    print_tree(result.root)
    
    print("\n" + "-"*60)
    print("SEARCH PROGRESS")
    print("-"*60)
    
    # Show score progression
    if result.history:
        for h in result.history[::max(1, len(result.history)//5)]:  # Show ~5 points
            bar_len = int(h['best_score'] * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"  Iter {h['iteration']:3d}: [{bar}] {h['best_score']:.3f}")
    
    print("\nDone! ✓")


if __name__ == "__main__":
    main()

