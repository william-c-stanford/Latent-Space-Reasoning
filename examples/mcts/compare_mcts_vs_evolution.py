#!/usr/bin/env python3
"""
Compare L-MCTS vs Standard Evolution on the same queries.

This script runs both approaches on test queries and produces a detailed
comparison showing:
- Final scores
- Number of evaluations used
- Runtime
- Quality of decoded outputs

=============================================================================
USAGE
=============================================================================

Basic comparison (5 queries, default model):

    python examples/mcts/compare_mcts_vs_evolution.py

Full comparison on all test queries:

    python examples/mcts/compare_mcts_vs_evolution.py --full

With specific model:

    python examples/mcts/compare_mcts_vs_evolution.py --model Qwen/Qwen3-1.7B

Save results to file:

    python examples/mcts/compare_mcts_vs_evolution.py --output results.json

=============================================================================
WHAT TO EXPECT
=============================================================================

L-MCTS should show:
- Similar or better final scores
- Fewer total evaluations (more sample efficient)
- More structured exploration (visible in the tree)

The key insight from the article:
> "If you can extract better reasoning from a frozen model using a lossy 
> projection, simple mutation, and one cheap judge… then most systems today 
> are bottlenecked not by knowledge, but by access."

MCTS is a smarter way to access that intelligence.

=============================================================================
OUTPUT
=============================================================================

The script produces:
1. Per-query comparison table
2. Aggregate statistics
3. Optional JSON export for further analysis

Example output:

    === COMPARISON RESULTS ===
    
    Query: "How to implement caching?"
    +-------------+-------+-------+----------+
    |   Method    | Score | Evals | Time (s) |
    +-------------+-------+-------+----------+
    | Evolution   | 0.634 |  50   |   12.3   |
    | MCTS        | 0.712 |  35   |   10.1   |
    +-------------+-------+-------+----------+
    Winner: MCTS (+12.3%)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import torch

from latent_reasoning import Engine, Config
from latent_reasoning.evolution.mcts import MCTSEvolutionLoop, MCTSConfig


# =============================================================================
# TEST QUERIES
# =============================================================================

QUICK_QUERIES = [
    {
        "id": "cache",
        "query": "How do I implement a thread-safe LRU cache with O(1) operations?",
        "category": "system_design",
    },
    {
        "id": "auth",
        "query": "Design a secure user authentication system with JWT tokens.",
        "category": "system_design",
    },
    {
        "id": "knights",
        "query": "On an island, Knights always tell truth, Knaves always lie. A says 'I am a Knave'. What is A?",
        "category": "logic",
    },
    {
        "id": "math",
        "query": "Prove that the sum of first n odd numbers equals n squared.",
        "category": "math",
    },
    {
        "id": "code",
        "query": "Write a Python function to find the longest increasing subsequence in O(n log n).",
        "category": "code",
    },
]

FULL_QUERIES = QUICK_QUERIES + [
    {
        "id": "distributed",
        "query": "Design a distributed rate limiting system for a global API gateway handling 10M requests/sec.",
        "category": "system_design",
    },
    {
        "id": "hanoi",
        "query": "Explain the Frame-Stewart algorithm for 4-peg Tower of Hanoi and derive the recurrence relation.",
        "category": "algorithms",
    },
    {
        "id": "river",
        "query": "A farmer must cross a river with a wolf, goat, and cabbage. The boat holds farmer + one item. Solve this.",
        "category": "logic",
    },
    {
        "id": "sequence",
        "query": "Find the next number: 2, 6, 12, 20, 30, ?",
        "category": "math",
    },
    {
        "id": "debug",
        "query": "Debug this sieve: `for i in range(2, n): if sieve[i]: for j in range(i, n, i): sieve[j] = False`",
        "category": "code",
    },
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class RunResult:
    """Result from a single run."""
    query_id: str
    method: str  # "evolution" or "mcts"
    score: float
    evaluations: int
    generations: int
    runtime: float
    output_length: int
    output_preview: str


@dataclass
class ComparisonResult:
    """Comparison between methods for one query."""
    query_id: str
    query: str
    category: str
    evolution: RunResult
    mcts: RunResult
    winner: str
    improvement: float  # Percentage improvement of winner


# =============================================================================
# COMPARISON LOGIC
# =============================================================================

def run_evolution(engine, query: str, query_id: str) -> RunResult:
    """Run standard evolution approach."""
    start = time.time()
    result = engine.run(query)
    runtime = time.time() - start
    
    return RunResult(
        query_id=query_id,
        method="evolution",
        score=result.confidence,
        evaluations=result.evaluations,
        generations=result.generations,
        runtime=runtime,
        output_length=len(result.plan),
        output_preview=result.plan[:200] + "..." if len(result.plan) > 200 else result.plan,
    )


def run_mcts(engine, query: str, query_id: str) -> RunResult:
    """Run MCTS approach."""
    orchestrator = engine._get_orchestrator()
    
    # Create MCTS loop
    mcts_config = MCTSConfig(
        n_iterations=engine.config.evolution.generations * engine.config.evolution.chains,
        exploration_constant=1.41,
        max_depth=engine.config.evolution.generations,
        use_modifier=len(orchestrator.judge_panel.modifiers) > 0,
    )
    
    mcts_loop = MCTSEvolutionLoop(
        judge_panel=orchestrator.judge_panel,
        config=engine.config.evolution,
        encoder=orchestrator.encoder,
    )
    mcts_loop.mcts_config = mcts_config  # Override with our config
    
    # Run MCTS
    start = time.time()
    seed = orchestrator.encoder.encode(query)
    
    # Set reference for scorers
    for scorer in orchestrator.judge_panel.scorers:
        scorer.set_reference(embedding=seed)
    
    evolution_result = mcts_loop.run(seed)
    
    # Decode
    output = orchestrator.encoder.decode(
        evolution_result.best_latent,
        query=query,
        max_new_tokens=engine.config.synthesis.max_tokens,
        temperature=engine.config.synthesis.temperature,
    )
    runtime = time.time() - start
    
    return RunResult(
        query_id=query_id,
        method="mcts",
        score=evolution_result.best_score,
        evaluations=evolution_result.total_evaluations,
        generations=evolution_result.generations,
        runtime=runtime,
        output_length=len(output),
        output_preview=output[:200] + "..." if len(output) > 200 else output,
    )


def compare_query(engine, query_data: dict) -> ComparisonResult:
    """Compare both methods on one query."""
    query = query_data["query"]
    query_id = query_data["id"]
    category = query_data["category"]
    
    print(f"\n{'='*60}")
    print(f"Query: {query[:50]}...")
    print(f"{'='*60}")
    
    # Run evolution
    print("  Running Evolution...", end=" ", flush=True)
    evo_result = run_evolution(engine, query, query_id)
    print(f"Score: {evo_result.score:.3f}")
    
    # Reset engine state
    engine.reset()
    
    # Run MCTS
    print("  Running MCTS...", end=" ", flush=True)
    mcts_result = run_mcts(engine, query, query_id)
    print(f"Score: {mcts_result.score:.3f}")
    
    # Determine winner
    if mcts_result.score > evo_result.score:
        winner = "mcts"
        improvement = (mcts_result.score - evo_result.score) / max(abs(evo_result.score), 0.01) * 100
    elif evo_result.score > mcts_result.score:
        winner = "evolution"
        improvement = (evo_result.score - mcts_result.score) / max(abs(mcts_result.score), 0.01) * 100
    else:
        winner = "tie"
        improvement = 0.0
    
    return ComparisonResult(
        query_id=query_id,
        query=query,
        category=category,
        evolution=evo_result,
        mcts=mcts_result,
        winner=winner,
        improvement=improvement,
    )


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def print_comparison(result: ComparisonResult):
    """Print formatted comparison for one query."""
    print(f"\n┌{'─'*58}┐")
    print(f"│ Query: {result.query[:48]:<48} │")
    print(f"├{'─'*58}┤")
    print(f"│ {'Method':<12} │ {'Score':>7} │ {'Evals':>5} │ {'Time':>6} │ {'Len':>6} │")
    print(f"├{'─'*58}┤")
    
    evo = result.evolution
    mcts = result.mcts
    
    evo_marker = " ✓" if result.winner == "evolution" else ""
    mcts_marker = " ✓" if result.winner == "mcts" else ""
    
    print(f"│ {'Evolution':<12} │ {evo.score:>7.3f} │ {evo.evaluations:>5} │ {evo.runtime:>5.1f}s │ {evo.output_length:>6} │{evo_marker}")
    print(f"│ {'MCTS':<12} │ {mcts.score:>7.3f} │ {mcts.evaluations:>5} │ {mcts.runtime:>5.1f}s │ {mcts.output_length:>6} │{mcts_marker}")
    
    print(f"└{'─'*58}┘")
    
    if result.winner != "tie":
        print(f"  Winner: {result.winner.upper()} (+{result.improvement:.1f}%)")


def print_summary(results: List[ComparisonResult]):
    """Print aggregate summary."""
    print(f"\n{'='*60}")
    print("AGGREGATE SUMMARY")
    print(f"{'='*60}\n")
    
    mcts_wins = sum(1 for r in results if r.winner == "mcts")
    evo_wins = sum(1 for r in results if r.winner == "evolution")
    ties = sum(1 for r in results if r.winner == "tie")
    
    print(f"MCTS Wins:      {mcts_wins}/{len(results)} ({100*mcts_wins/len(results):.0f}%)")
    print(f"Evolution Wins: {evo_wins}/{len(results)} ({100*evo_wins/len(results):.0f}%)")
    print(f"Ties:           {ties}/{len(results)} ({100*ties/len(results):.0f}%)")
    
    # Score comparison
    avg_evo_score = sum(r.evolution.score for r in results) / len(results)
    avg_mcts_score = sum(r.mcts.score for r in results) / len(results)
    
    print(f"\nAverage Scores:")
    print(f"  Evolution: {avg_evo_score:.3f}")
    print(f"  MCTS:      {avg_mcts_score:.3f}")
    
    # Efficiency comparison
    avg_evo_evals = sum(r.evolution.evaluations for r in results) / len(results)
    avg_mcts_evals = sum(r.mcts.evaluations for r in results) / len(results)
    
    print(f"\nAverage Evaluations:")
    print(f"  Evolution: {avg_evo_evals:.0f}")
    print(f"  MCTS:      {avg_mcts_evals:.0f}")
    
    efficiency_gain = (avg_evo_evals - avg_mcts_evals) / avg_evo_evals * 100
    print(f"  MCTS efficiency gain: {efficiency_gain:.1f}%")
    
    # By category
    categories = set(r.category for r in results)
    if len(categories) > 1:
        print(f"\nResults by Category:")
        for cat in sorted(categories):
            cat_results = [r for r in results if r.category == cat]
            mcts_wins_cat = sum(1 for r in cat_results if r.winner == "mcts")
            print(f"  {cat}: MCTS wins {mcts_wins_cat}/{len(cat_results)}")


def save_results(results: List[ComparisonResult], path: str):
    """Save results to JSON file."""
    data = {
        "results": [
            {
                "query_id": r.query_id,
                "query": r.query,
                "category": r.category,
                "winner": r.winner,
                "improvement": r.improvement,
                "evolution": asdict(r.evolution),
                "mcts": asdict(r.mcts),
            }
            for r in results
        ],
        "summary": {
            "mcts_wins": sum(1 for r in results if r.winner == "mcts"),
            "evolution_wins": sum(1 for r in results if r.winner == "evolution"),
            "ties": sum(1 for r in results if r.winner == "tie"),
            "avg_evolution_score": sum(r.evolution.score for r in results) / len(results),
            "avg_mcts_score": sum(r.mcts.score for r in results) / len(results),
        }
    }
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nResults saved to {path}")


def generate_markdown_report(results: List[ComparisonResult], model: str) -> str:
    """Generate a detailed Markdown comparison report."""
    from datetime import datetime
    
    lines = [
        "# L-MCTS vs Evolution Comparison Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Model:** `{model}`",
        f"**Queries Tested:** {len(results)}",
        "",
        "---",
        "",
        "## Summary",
        "",
    ]
    
    # Calculate summary stats
    mcts_wins = sum(1 for r in results if r.winner == "mcts")
    evo_wins = sum(1 for r in results if r.winner == "evolution")
    ties = sum(1 for r in results if r.winner == "tie")
    
    avg_evo_score = sum(r.evolution.score for r in results) / len(results)
    avg_mcts_score = sum(r.mcts.score for r in results) / len(results)
    
    avg_evo_evals = sum(r.evolution.evaluations for r in results) / len(results)
    avg_mcts_evals = sum(r.mcts.evaluations for r in results) / len(results)
    
    avg_evo_time = sum(r.evolution.runtime for r in results) / len(results)
    avg_mcts_time = sum(r.mcts.runtime for r in results) / len(results)
    
    # Summary table
    lines.extend([
        "| Metric | Evolution | MCTS | Winner |",
        "|--------|-----------|------|--------|",
        f"| **Wins** | {evo_wins} ({100*evo_wins/len(results):.0f}%) | {mcts_wins} ({100*mcts_wins/len(results):.0f}%) | {'Evolution' if evo_wins > mcts_wins else 'MCTS' if mcts_wins > evo_wins else 'Tie'} |",
        f"| **Avg Score** | {avg_evo_score:.3f} | {avg_mcts_score:.3f} | {'Evolution' if avg_evo_score > avg_mcts_score else 'MCTS' if avg_mcts_score > avg_evo_score else 'Tie'} |",
        f"| **Avg Evaluations** | {avg_evo_evals:.0f} | {avg_mcts_evals:.0f} | {'MCTS' if avg_mcts_evals < avg_evo_evals else 'Evolution'} (fewer is better) |",
        f"| **Avg Runtime** | {avg_evo_time:.1f}s | {avg_mcts_time:.1f}s | {'MCTS' if avg_mcts_time < avg_evo_time else 'Evolution'} (faster is better) |",
        "",
    ])
    
    # Winner determination
    score_winner = "Evolution" if avg_evo_score > avg_mcts_score else "MCTS" if avg_mcts_score > avg_evo_score else "Tie"
    lines.extend([
        f"### Overall Winner by Score: **{score_winner}**",
        "",
        f"Score difference: {abs(avg_evo_score - avg_mcts_score)*100:.1f}%",
        "",
        "---",
        "",
        "## Detailed Results",
        "",
    ])
    
    # Per-query results
    for r in results:
        winner_emoji = "🏆" if r.winner == "mcts" else "📈" if r.winner == "evolution" else "🤝"
        lines.extend([
            f"### {r.query_id}: {r.query[:60]}{'...' if len(r.query) > 60 else ''}",
            "",
            f"**Category:** {r.category} | **Winner:** {r.winner.upper()} {winner_emoji}",
            "",
            "| Method | Score | Evaluations | Runtime | Output Length |",
            "|--------|-------|-------------|---------|---------------|",
            f"| Evolution | {r.evolution.score:.3f} | {r.evolution.evaluations} | {r.evolution.runtime:.1f}s | {r.evolution.output_length} chars |",
            f"| MCTS | {r.mcts.score:.3f} | {r.mcts.evaluations} | {r.mcts.runtime:.1f}s | {r.mcts.output_length} chars |",
            "",
        ])
        
        if r.improvement != 0:
            lines.append(f"**Improvement:** +{r.improvement:.1f}% for {r.winner}")
        lines.append("")
    
    # Category breakdown
    categories = set(r.category for r in results)
    if len(categories) > 1:
        lines.extend([
            "---",
            "",
            "## Results by Category",
            "",
            "| Category | Evolution Wins | MCTS Wins | Ties |",
            "|----------|----------------|-----------|------|",
        ])
        
        for cat in sorted(categories):
            cat_results = [r for r in results if r.category == cat]
            evo_cat = sum(1 for r in cat_results if r.winner == "evolution")
            mcts_cat = sum(1 for r in cat_results if r.winner == "mcts")
            ties_cat = sum(1 for r in cat_results if r.winner == "tie")
            lines.append(f"| {cat} | {evo_cat} | {mcts_cat} | {ties_cat} |")
        
        lines.append("")
    
    # Analysis section
    lines.extend([
        "---",
        "",
        "## Analysis",
        "",
        "### Key Observations",
        "",
    ])
    
    if avg_mcts_score >= avg_evo_score:
        lines.append("- ✅ **MCTS achieved equal or better scores** on average")
    else:
        lines.append(f"- ⚠️ **Evolution outperformed MCTS** by {(avg_evo_score - avg_mcts_score)*100:.1f}%")
    
    if avg_mcts_evals < avg_evo_evals:
        efficiency = (avg_evo_evals - avg_mcts_evals) / avg_evo_evals * 100
        lines.append(f"- ✅ **MCTS was {efficiency:.0f}% more sample efficient** (fewer evaluations)")
    else:
        lines.append(f"- ⚠️ **MCTS used more evaluations** than Evolution")
    
    if avg_mcts_time < avg_evo_time:
        speedup = (avg_evo_time - avg_mcts_time) / avg_evo_time * 100
        lines.append(f"- ✅ **MCTS was {speedup:.0f}% faster**")
    else:
        lines.append(f"- ⚠️ **Evolution was faster** than MCTS")
    
    lines.extend([
        "",
        "### Recommendations",
        "",
    ])
    
    if mcts_wins >= evo_wins:
        lines.extend([
            "Based on these results, **MCTS is recommended** for:",
            "- Queries requiring structured exploration",
            "- When sample efficiency matters",
            "- When you want inspectable search paths",
        ])
    else:
        lines.extend([
            "Based on these results, **Evolution may be preferable** for:",
            "- Maximum diversity in solutions",
            "- When compute budget is not a concern",
            "- The current query types tested",
            "",
            "Consider tuning MCTS parameters:",
            "- Increase `exploration_constant` for more exploration",
            "- Increase `n_iterations` for longer search",
            "- Try different `temperature_decay` values",
        ])
    
    lines.append("")
    
    return "\n".join(lines)


def save_markdown_report(results: List[ComparisonResult], path: str, model: str):
    """Save results as a Markdown report."""
    report = generate_markdown_report(results, model)
    
    with open(path, 'w') as f:
        f.write(report)
    
    print(f"\n📊 Markdown report saved to {path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare L-MCTS vs Standard Evolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B",
                        help="Model to use (default: Qwen/Qwen3-0.6B)")
    parser.add_argument("--full", action="store_true",
                        help="Run full query set (10 queries)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    parser.add_argument("--chains", type=int, default=5,
                        help="Number of evolution chains (default: 5)")
    parser.add_argument("--generations", type=int, default=10,
                        help="Number of generations (default: 10)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show verbose output")
    
    args = parser.parse_args()
    
    # Select queries
    queries = FULL_QUERIES if args.full else QUICK_QUERIES
    
    print(f"\n{'='*60}")
    print("L-MCTS vs Evolution Comparison")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Queries: {len(queries)}")
    print(f"Chains: {args.chains}")
    print(f"Generations: {args.generations}")
    
    # Create engine
    config = Config()
    config.encoder.model = args.model
    config.evolution.chains = args.chains
    config.evolution.generations = args.generations
    config.output.verbosity = "verbose" if args.verbose else "minimal"
    
    print("\nLoading model...")
    engine = Engine(config=config)
    
    # Run comparisons
    results = []
    for query_data in queries:
        try:
            result = compare_query(engine, query_data)
            results.append(result)
            print_comparison(result)
        except Exception as e:
            print(f"\nError on query {query_data['id']}: {e}")
            continue
        
        # Reset between queries
        engine.reset()
    
    # Print summary
    if results:
        print_summary(results)
        
        if args.output:
            save_results(results, args.output)
        
        # Always save markdown report
        report_path = args.output.replace('.json', '.md') if args.output else 'mcts_comparison_report.md'
        save_markdown_report(results, report_path, args.model)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

