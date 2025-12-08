#!/usr/bin/env python3
"""
Latent Space Reasoning Engine - Interactive Demo

This script demonstrates the key features of the Latent Space Reasoning Engine
through practical examples. It shows how evolutionary optimization in latent
space produces higher-quality, more specific responses than standard LLM generation.

🔬 How it works:
1. Encode your query into a latent vector using an LLM's hidden states
2. Evolve that vector through selection, mutation, and crossover
3. Score evolved latents using a trained neural network judge
4. Decode the best result back to structured, high-quality text

💡 Why it's better:
- Produces specific, actionable content instead of generic templates
- Includes technical details like variable names, functions, code patterns
- Generates structured multi-step plans instead of vague advice
- Optimizes for quality through iterative refinement

🚀 Requirements:
- Python 3.10+ with PyTorch
- GPU recommended (2GB+ VRAM), CPU also supported
- Install: pip install -e .

Run with: python demo.py [--simple|--engine|--compare|--custom-encoder|--all]
"""

import argparse
import sys


def demo_simple():
    """Demonstrate the simplest usage - one function call."""
    print("\n" + "=" * 70)
    print("SIMPLE API DEMO - One-liner usage")
    print("=" * 70)

    print("\nThis demo shows the easiest way to use latent space reasoning.")
    print("Just import and call reason() - that's it!")

    query = "What steps should I take to implement user authentication in a web application?"

    print(f"\n📝 Query: {query}")
    print("\n⚡ Running latent space reasoning...")
    print("   (This will encode → evolve → decode your query)")

    try:
        from latent_reasoning import reason
        result = reason(query, verbosity="normal")

        print("\n" + "=" * 50)
        print("✅ RESULT:")
        print("=" * 50)
        print(result.plan)

        print(f"\n📊 Quality Metrics:")
        print(f"   • Confidence Score: {result.confidence:.3f} (0-1 scale, higher = better)")
        print(f"   • Evolution Generations: {result.generations}")
        print(f"   • Total Evaluations: {result.evaluations}")
        print(f"   • Stop Reason: {result.stop_reason}")

        if result.confidence > 0.8:
            print("   🎉 High quality result!")
        elif result.confidence > 0.6:
            print("   👍 Good quality result!")
        else:
            print("   💡 Consider using a larger model or more generations for better quality")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Make sure you have:")
        print("   1. Installed the package: pip install -e .")
        print("   2. PyTorch with CUDA support (or CPU fallback)")
        print("   3. Sufficient memory (2GB+ VRAM recommended)")
        return False

    return True


def demo_engine():
    """Demonstrate the Engine class for more control and customization."""
    print("\n" + "=" * 70)
    print("🔧 ENGINE API DEMO - Full control and customization")
    print("=" * 70)

    print("\nThis demo shows how to use the Engine class for maximum control.")
    print("You can customize models, evolution parameters, and run multiple queries.")

    try:
        from latent_reasoning import Engine
        from latent_reasoning.config import get_default_config

        # Get default config and customize it
        config = get_default_config()
        config.evolution.chains = 6          # More parallel chains for better exploration
        config.evolution.generations = 12    # More generations for better refinement
        config.evolution.temperature = 0.4   # Lower temperature for more focused search

        print(f"\n⚙️  Custom Configuration:")
        print(f"   • Encoder Model: {config.encoder.model}")
        print(f"   • Evolution Chains: {config.evolution.chains}")
        print(f"   • Max Generations: {config.evolution.generations}")
        print(f"   • Mutation Temperature: {config.evolution.temperature}")

        # Create engine with custom config
        engine = Engine(config=config, verbosity="normal")

        query = "How do I optimize database queries for a high-traffic e-commerce site?"

        print(f"\n📝 Query: {query}")
        print("\n⚡ Running with custom configuration...")

        result = engine.run(query)

        print("\n" + "=" * 50)
        print("✅ RESULT:")
        print("=" * 50)
        print(result.plan)

        print(f"\n📊 Evolution Statistics:")
        print(f"   • Final Confidence: {result.confidence:.3f}")
        print(f"   • Generations Run: {result.generations}")
        print(f"   • Total Evaluations: {result.evaluations}")
        print(f"   • Survivor Plans: {len(result.all_plans)}")

        # Show evolution history if available
        if result.history:
            print(f"\n📈 Evolution Progress:")
            for i, gen_stats in enumerate(result.history[:5]):  # Show first 5 generations
                best_score = gen_stats.get('best_score', 0)
                mean_score = gen_stats.get('mean_score', 0)
                print(f"   Generation {i+1:2d}: best={best_score:.3f}, mean={mean_score:.3f}")
            if len(result.history) > 5:
                print(f"   ... and {len(result.history) - 5} more generations")

        # Demonstrate multiple queries with same engine
        print(f"\n🔄 Running additional queries with the same engine...")
        quick_queries = [
            "How to implement caching?",
            "Best practices for API design?",
        ]

        for i, q in enumerate(quick_queries, 1):
            print(f"\n   Query {i}: {q}")
            result = engine.run(q)
            print(f"   Result: {result.plan[:100]}..." if len(result.plan) > 100 else f"   Result: {result.plan}")
            print(f"   Confidence: {result.confidence:.3f}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

    return True


def demo_compare():
    """Demonstrate side-by-side comparison of baseline vs latent reasoning."""
    print("\n" + "=" * 70)
    print("⚖️  COMPARISON DEMO - See the difference!")
    print("=" * 70)

    print("\nThis demo compares standard LLM generation vs latent space reasoning.")
    print("You'll see how evolution produces more specific, actionable content.")

    try:
        from latent_reasoning import compare

        query = "What's the best approach to implement a recommendation system?"

        print(f"\n📝 Query: {query}")
        print("\n⚡ Running both methods...")
        print("   • Baseline: Direct LLM generation")
        print("   • Latent Reasoning: Evolutionary optimization")

        result = compare(query)

        print("\n" + "=" * 60)
        print("📊 COMPARISON RESULTS")
        print("=" * 60)

        print("\n🔸 BASELINE (Direct Generation):")
        print("-" * 50)
        baseline = result["baseline"]
        if len(baseline) > 400:
            print(baseline[:400] + "\n... [truncated for display]")
        else:
            print(baseline)

        print("\n🔹 LATENT REASONING (Evolved):")
        print("-" * 50)
        lr_plan = result["latent_reasoning"]
        if len(lr_plan) > 400:
            print(lr_plan[:400] + "\n... [truncated for display]")
        else:
            print(lr_plan)

        print(f"\n📈 Quality Metrics:")
        print(f"   • Latent Reasoning Score: {result['latent_score']:.3f}")
        print(f"   • Evolution Generations: {result.get('generations', 'N/A')}")
        print(f"   • Total Evaluations: {result.get('evaluations', 'N/A')}")

        # Analysis
        print(f"\n🔍 Analysis:")
        baseline_words = len(baseline.split())
        lr_words = len(lr_plan.split())
        print(f"   • Baseline length: {baseline_words} words")
        print(f"   • Latent reasoning length: {lr_words} words")

        if result['latent_score'] > 0.7:
            print("   ✅ Latent reasoning shows significant improvement!")
        elif result['latent_score'] > 0.5:
            print("   👍 Latent reasoning shows moderate improvement")
        else:
            print("   💡 Results are similar - try a more complex query or larger model")

        print(f"\n💡 Key Differences to Look For:")
        print(f"   • Baseline often gives generic templates or asks clarifying questions")
        print(f"   • Latent reasoning typically provides specific, actionable steps")
        print(f"   • Latent reasoning includes more technical details and concrete examples")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

    return True


def demo_custom_encoder():
    """Use a specific encoder model."""
    from latent_reasoning import Engine

    print("\n" + "=" * 60)
    print("CUSTOM ENCODER DEMO")
    print("=" * 60)

    # Use a smaller/faster model
    engine = Engine(
        encoder="Qwen/Qwen3-0.6B",  # Smallest Qwen3 model
        verbosity="minimal",
    )

    query = "How do I set up CI/CD for a Python project?"

    print(f"\nQuery: {query}")
    print("Using Qwen3-0.6B encoder (fast, lower VRAM)...")

    result = engine.run(query)

    print("\n" + "-" * 40)
    print("RESULT:")
    print("-" * 40)
    print(result.plan)


def main():
    parser = argparse.ArgumentParser(
        description="Latent Space Reasoning Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py                  # Run simple demo
  python demo.py --engine         # Run engine demo
  python demo.py --compare        # Run comparison demo
  python demo.py --all            # Run all demos
  python demo.py --custom-encoder # Run with custom encoder
        """,
    )

    parser.add_argument("--simple", action="store_true", help="Run simple API demo")
    parser.add_argument("--engine", action="store_true", help="Run Engine API demo")
    parser.add_argument("--compare", action="store_true", help="Run comparison demo")
    parser.add_argument("--custom-encoder", action="store_true", help="Run custom encoder demo")
    parser.add_argument("--all", action="store_true", help="Run all demos")

    args = parser.parse_args()

    # Default to simple demo if no args
    if not any([args.simple, args.engine, args.compare, args.custom_encoder, args.all]):
        args.simple = True

    print("\n" + "#" * 60)
    print("#  LATENT SPACE REASONING DEMO")
    print("#" * 60)

    try:
        if args.all or args.simple:
            demo_simple()

        if args.all or args.engine:
            demo_engine()

        if args.all or args.compare:
            demo_compare()

        if args.all or args.custom_encoder:
            demo_custom_encoder()

        print("\n" + "=" * 60)
        print("Demo complete!")
        print("=" * 60 + "\n")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have:")
        print("  1. Installed the package: pip install -e .")
        print("  2. A GPU with sufficient VRAM (8GB+ recommended)")
        print("  3. Downloaded the required models")
        raise


if __name__ == "__main__":
    main()
