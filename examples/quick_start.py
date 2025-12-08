#!/usr/bin/env python3
"""
Quick Start Examples for Latent Space Reasoning Engine

This script shows the most common usage patterns to help you get started quickly.
Run this file directly or copy the examples into your own code.

Usage: python examples/quick_start.py
"""

def example_1_simple_usage():
    """Example 1: Simplest possible usage - one function call"""
    print("=" * 60)
    print("EXAMPLE 1: Simple Usage")
    print("=" * 60)
    
    from latent_reasoning import reason
    
    # Just call reason() with your query - that's it!
    result = reason("How do I implement user authentication?")
    
    print("Query: How do I implement user authentication?")
    print(f"Result: {result.plan}")
    print(f"Confidence: {result.confidence:.3f}")
    print()


def example_2_compare_methods():
    """Example 2: Compare baseline vs latent reasoning"""
    print("=" * 60)
    print("EXAMPLE 2: Compare Methods")
    print("=" * 60)
    
    from latent_reasoning import compare
    
    # Compare both methods on the same query
    result = compare("How to optimize database performance?")
    
    print("Query: How to optimize database performance?")
    print("\nBaseline:")
    print(result["baseline"][:200] + "..." if len(result["baseline"]) > 200 else result["baseline"])
    print("\nLatent Reasoning:")
    print(result["latent_reasoning"][:200] + "..." if len(result["latent_reasoning"]) > 200 else result["latent_reasoning"])
    print(f"\nLatent Score: {result['latent_score']:.3f}")
    print()


def example_3_custom_model():
    """Example 3: Using a specific model"""
    print("=" * 60)
    print("EXAMPLE 3: Custom Model")
    print("=" * 60)
    
    from latent_reasoning import Engine
    
    # Use a specific model (larger = better quality, more VRAM)
    engine = Engine(encoder="Qwen/Qwen3-1.7B", verbosity="minimal")
    
    result = engine.run("Design a REST API for a todo app")
    
    print("Query: Design a REST API for a todo app")
    print(f"Model: Qwen/Qwen3-1.7B")
    print(f"Result: {result.plan}")
    print(f"Confidence: {result.confidence:.3f}")
    print()


def example_4_multiple_queries():
    """Example 4: Running multiple queries efficiently"""
    print("=" * 60)
    print("EXAMPLE 4: Multiple Queries")
    print("=" * 60)
    
    from latent_reasoning import Engine
    
    # Create engine once, use for multiple queries
    engine = Engine(encoder="Qwen/Qwen3-0.6B", verbosity="silent")
    
    queries = [
        "How to implement caching?",
        "Best practices for API security?",
        "How to handle database migrations?",
    ]
    
    print("Running multiple queries with the same engine:")
    for i, query in enumerate(queries, 1):
        result = engine.run(query)
        print(f"\n{i}. {query}")
        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   Preview: {result.plan[:100]}...")


def example_5_error_handling():
    """Example 5: Proper error handling"""
    print("=" * 60)
    print("EXAMPLE 5: Error Handling")
    print("=" * 60)
    
    try:
        from latent_reasoning import reason
        
        result = reason("How to deploy a web application?", verbosity="silent")
        
        # Check result quality
        if result.confidence > 0.8:
            print("✅ High quality result!")
        elif result.confidence > 0.6:
            print("👍 Good result")
        else:
            print("⚠️  Lower quality result - consider using a larger model")
            
        print(f"Result: {result.plan}")
        
    except ImportError:
        print("❌ Package not installed. Run: pip install -e .")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Check your GPU/CUDA setup or try with CPU")


if __name__ == "__main__":
    print("🚀 Latent Space Reasoning - Quick Start Examples")
    print("=" * 60)
    print()
    
    try:
        example_1_simple_usage()
        example_2_compare_methods()
        example_3_custom_model()
        example_4_multiple_queries()
        example_5_error_handling()
        
        print("=" * 60)
        print("✅ All examples completed!")
        print("\n💡 Next steps:")
        print("   • Try your own queries")
        print("   • Experiment with different models")
        print("   • Check out demo.py for more advanced examples")
        print("   • Read the README.md for full documentation")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("\n💡 Make sure you have:")
        print("   1. Installed the package: pip install -e .")
        print("   2. PyTorch with CUDA support")
        print("   3. Sufficient memory (2GB+ VRAM recommended)")
