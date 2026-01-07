# Latent Monte Carlo Tree Search (L-MCTS)

> A geometry-aware exploration strategy for latent space reasoning.

This document explains everything you need to know about L-MCTS: what it is, why it works, how to use it, and how to extend it.

---

## Table of Contents

1. [What is L-MCTS?](#what-is-l-mcts)
2. [Why MCTS for Latent Reasoning?](#why-mcts-for-latent-reasoning)
3. [Quick Start](#quick-start)
4. [How It Works](#how-it-works)
5. [The UCB Formula](#the-ucb-formula)
6. [Integration Guide](#integration-guide)
7. [Configuration](#configuration)
8. [Comparison with Evolution](#comparison-with-evolution)
9. [Advanced Usage](#advanced-usage)
10. [FAQ](#faq)

---

## What is L-MCTS?

L-MCTS adapts Monte Carlo Tree Search—the algorithm behind AlphaGo—for exploring the latent space of language models.

### The Core Idea

Instead of searching game states, we search the model's internal representation space:

```
Traditional MCTS:           L-MCTS:
Game State → Move           Latent Vector → Mutation
    ↓                           ↓
Next State                  Next Latent
    ↓                           ↓
   Win/Lose                 Judge Score
```

### Why Trees?

The current evolution approach in this repo uses a **population**:
- Generate N random mutations
- Score all of them
- Keep the best, repeat

This is inefficient because:
1. **No memory**: Forgets which directions were promising
2. **Random jumps**: Mutations often leave the useful latent manifold
3. **Flat structure**: Can't represent refinement paths

Trees fix this:
1. **Memory**: Visit statistics remember good directions
2. **Structure**: Children stay near parents (on the manifold)
3. **Refinement**: Deep paths represent iterative improvement

---

## Why MCTS for Latent Reasoning?

From the original article:

> "Random perturbation means most of your candidates are useless. The evolutionary process is robust enough to survive this, but it's wasteful."

And:

> "Geometry-aware evolution would understand the manifold. Mutations would stay on the surface of useful representations."

**MCTS addresses both problems:**

1. **UCB selection** naturally focuses on promising regions
2. **Tree structure** keeps children near parents (staying on manifold)
3. **Backpropagation** learns which directions are good

### Sample Efficiency

Evolution evaluates many random mutations. MCTS evaluates fewer, more targeted mutations:

```
Evolution:  [random] [random] [random] [random] [random]
            Most are useless, we keep the lucky ones

MCTS:       [guided] ──> [refined] ──> [refined] ──> [best]
            Each step builds on what worked
```

---

## Quick Start

### Minimal Example

```python
from latent_reasoning import Engine
from latent_reasoning.evolution.mcts import LatentMCTS, MCTSConfig

# Setup
engine = Engine(encoder="Qwen/Qwen3-0.6B")
orchestrator = engine._get_orchestrator()

# Create MCTS
mcts = LatentMCTS(
    judge_panel=orchestrator.judge_panel,
    encoder=orchestrator.encoder,
)

# Search
seed = orchestrator.encoder.encode("How to implement caching?")
for scorer in orchestrator.judge_panel.scorers:
    scorer.set_reference(embedding=seed)

result = mcts.search(seed)

# Use result
output = orchestrator.encoder.decode(result.best_latent, query="How to implement caching?")
print(output)
```

### As Drop-in Replacement

```python
from latent_reasoning.evolution.mcts import MCTSEvolutionLoop

# Swap evolution loop
orchestrator.evolution_loop = MCTSEvolutionLoop(
    judge_panel=orchestrator.judge_panel,
    config=config.evolution,
)

# Use normally - everything else stays the same
result = orchestrator.run("Your query")
```

### Run Comparison Script

```bash
# Quick comparison (5 queries)
python examples/mcts/compare_mcts_vs_evolution.py

# Full comparison (10 queries)
python examples/mcts/compare_mcts_vs_evolution.py --full

# Save results
python examples/mcts/compare_mcts_vs_evolution.py --output results.json
```

---

## How It Works

### The Four Phases

Every MCTS iteration has four phases:

```
           SELECTION
              │
    ┌─────────┴─────────┐
    │   Walk down tree  │
    │   using UCB       │
    └─────────┬─────────┘
              │
           EXPANSION
              │
    ┌─────────┴─────────┐
    │   Add new child   │
    │   (mutation)      │
    └─────────┬─────────┘
              │
           EVALUATION
              │
    ┌─────────┴─────────┐
    │   Score with      │
    │   JudgePanel      │
    └─────────┬─────────┘
              │
         BACKPROPAGATION
              │
    ┌─────────┴─────────┐
    │   Update stats    │
    │   up to root      │
    └─────────┘
```

### Phase 1: Selection

Starting from root, walk down the tree by repeatedly picking the child with highest UCB value:

```python
def _select(self, node):
    while not node.is_leaf:
        if not node.is_fully_expanded:
            return node  # Expand this one
        node = self._best_ucb_child(node)
    return node
```

**Key insight**: UCB balances exploration (try new things) vs exploitation (use what works).

### Phase 2: Expansion

Create a new child by mutating the parent's latent:

```python
def _expand(self, node):
    # Get direction from ModifierJudge (if available)
    modification = self.judge_panel.get_modification(node.latent)
    
    if modification is not None:
        # Directed mutation: follow the suggestion
        direction = modification / modification.norm()
        child_latent = node.latent + temperature * direction
    else:
        # Random mutation: small step in random direction
        noise = torch.randn_like(node.latent)
        child_latent = node.latent + temperature * 0.3 * noise
    
    return node.add_child(child_latent)
```

**Key insight**: Using ModifierJudge for direction keeps mutations on the manifold.

### Phase 3: Evaluation

Score the new node using the existing JudgePanel:

```python
def _evaluate_node(self, node):
    verdict = self.judge_panel.evaluate(node.latent)
    self.total_evaluations += 1
    return verdict
```

**Key insight**: This is cheap—we score latents directly, no decoding needed.

### Phase 4: Backpropagation

Update statistics from the new node up to root:

```python
def _backpropagate(self, node, score):
    while node is not None:
        node.visits += 1
        node.total_score += score
        node.best_score = max(node.best_score, score)
        node = node.parent
```

**Key insight**: This is how MCTS learns which paths are promising.

---

## The UCB Formula

UCB (Upper Confidence Bound) is the secret sauce of MCTS:

```
UCB(node) = value(node) + c × √(ln(parent.visits) / node.visits)
            └─────┬─────┘   └──────────────┬─────────────────┘
            Exploitation              Exploration
```

### Exploitation Term

`value(node) = total_score / visits`

This is the average score of this node. High value = we should probably keep exploring here.

### Exploration Term

`c × √(ln(parent.visits) / node.visits)`

This grows when:
- Parent has been visited many times (numerator increases)
- This node has been visited few times (denominator decreases)

Effect: Even low-value nodes get explored if we haven't tried them much.

### The Balance

- When `c` is high: More exploration (try many different things)
- When `c` is low: More exploitation (focus on what works)
- Default `c = √2 ≈ 1.41`: Balanced (theoretical optimum for some problems)

### Intuition

Imagine picking a restaurant:
- **Exploitation**: Go to your favorite place (you know it's good)
- **Exploration**: Try that new place (might be amazing, might be terrible)

UCB is like: "Go to favorite place usually, but occasionally try new places, especially ones you've never been to."

---

## Integration Guide

### Option 1: Direct Use

Use MCTS directly when you want full control:

```python
from latent_reasoning.evolution.mcts import LatentMCTS, MCTSConfig

config = MCTSConfig(
    n_iterations=100,
    exploration_constant=1.41,
    max_depth=10,
)

mcts = LatentMCTS(judge_panel, encoder, config)
result = mcts.search(seed_latent)
```

### Option 2: Drop-in Replacement

Replace EvolutionLoop for seamless integration:

```python
from latent_reasoning.evolution.mcts import MCTSEvolutionLoop

# Create orchestrator normally
orchestrator = Orchestrator(config)

# Swap in MCTS
orchestrator.evolution_loop = MCTSEvolutionLoop(
    judge_panel=orchestrator.judge_panel,
    config=config.evolution,
)

# Use normally
result = orchestrator.run("Your query")
```

### Option 3: Engine-level

Use the helper function:

```python
from latent_reasoning.evolution.mcts import create_mcts_orchestrator

orchestrator = create_mcts_orchestrator(config)
result = orchestrator.run("Your query")
```

---

## Configuration

### MCTSConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_iterations` | 100 | Total MCTS iterations |
| `exploration_constant` | 1.41 | UCB exploration weight (c) |
| `max_depth` | 10 | Maximum tree depth |
| `temperature_decay` | 0.85 | Temperature reduction per depth |
| `min_temperature` | 0.1 | Minimum mutation temperature |
| `use_modifier` | True | Use ModifierJudge for directed mutations |
| `early_stop_threshold` | 0.95 | Stop if score exceeds this |
| `early_stop_patience` | 20 | Stop after N iterations without improvement |

### Tuning Guide

**For more exploration:**
```python
config = MCTSConfig(
    exploration_constant=2.0,  # Higher = more exploration
    n_iterations=150,          # More iterations
    max_depth=15,              # Allow deeper search
)
```

**For faster convergence:**
```python
config = MCTSConfig(
    exploration_constant=0.7,  # Lower = more exploitation
    n_iterations=50,           # Fewer iterations
    early_stop_patience=10,    # Stop earlier
)
```

**For complex queries:**
```python
config = MCTSConfig(
    n_iterations=200,
    max_depth=15,
    temperature_decay=0.9,     # Slower decay = maintain exploration
)
```

---

## Comparison with Evolution

### When to Use MCTS

| MCTS | Evolution |
|------|-----------|
| You want structured paths | You want maximum diversity |
| Sample efficiency matters | Compute budget is large |
| Local structure exists | Scoring is very noisy |
| You want inspectable search | You need many independent solutions |

### Expected Differences

Based on the repo's existing benchmarks:

| Metric | Evolution | MCTS |
|--------|-----------|------|
| Evaluations | More | Fewer |
| Diversity | Higher | Lower |
| Path quality | Random | Structured |
| Inspectability | Low | High (tree) |
| Best for | Open-ended | Targeted |

### Running the Comparison

```bash
# See the difference yourself
python examples/mcts/compare_mcts_vs_evolution.py
```

---

## Advanced Usage

### Inspecting the Tree

After search, you can inspect what MCTS explored:

```python
result = mcts.search(seed)

# Access the root
root = result.root

# Walk the tree
def print_tree(node, depth=0):
    indent = "  " * depth
    print(f"{indent}Node: visits={node.visits}, value={node.value:.3f}")
    for child in node.children:
        print_tree(child, depth + 1)

print_tree(root)
```

### Extracting Multiple Candidates

Get top-k latents for aggregation:

```python
top_latents = mcts.get_top_k_latents(result.root, k=5)

# These can be aggregated like evolution survivors
combined = torch.stack([l.float() for l in top_latents]).mean(dim=0)
```

### Custom Expansion Strategy

Override expansion to use your own mutation logic:

```python
class CustomMCTS(LatentMCTS):
    def _expand(self, node):
        # Your custom expansion logic
        child_latent = your_mutation_function(node.latent)
        return node.add_child(child_latent)
```

### Analyzing Search Progress

```python
result = mcts.search(seed)

# Plot progress
import matplotlib.pyplot as plt

iterations = [h['iteration'] for h in result.history]
scores = [h['best_score'] for h in result.history]

plt.plot(iterations, scores)
plt.xlabel('Iteration')
plt.ylabel('Best Score')
plt.title('MCTS Search Progress')
plt.show()
```

---

## FAQ

### Q: How many iterations should I use?

**A**: Start with `n_iterations = generations × chains` from your evolution config. This gives roughly equivalent compute. Then tune up for better quality or down for speed.

### Q: Is MCTS always better than evolution?

**A**: No. MCTS is better when:
- The latent space has local structure
- You want sample efficiency
- You need inspectable search

Evolution might be better when:
- You want many diverse solutions
- The scoring is very noisy
- You have unlimited compute

### Q: Why use ModifierJudge for expansion?

**A**: Random mutations often leave the "useful" region of latent space. The ModifierJudge suggests directions that stay on the manifold, making expansion more efficient.

From the article:
> "Mutations would stay on the surface of useful representations."

### Q: Can I use MCTS with my own judges?

**A**: Yes! MCTS uses the same JudgePanel as evolution. Just configure your judges normally and pass the panel to MCTS.

### Q: How does backpropagation help?

**A**: It creates "credit assignment" - when a deep node gets a good score, all its ancestors get credit. This helps MCTS learn which early decisions lead to good outcomes.

### Q: What's the computational cost?

**A**: Each iteration = 1 judge evaluation (same as evolution). The tree structure is negligible overhead. MCTS is often more sample-efficient, meaning better scores with fewer evaluations.

---

## Next Steps

1. **Run the quick start**: `python examples/mcts/quick_start.py`
2. **Compare methods**: `python examples/mcts/compare_mcts_vs_evolution.py`
3. **Read the code**: Start with `node.py`, then `search.py`
4. **Experiment**: Try different `exploration_constant` values
5. **Extend**: Add your own expansion strategies or judges

---

## References

- [Monte Carlo Tree Search - Wikipedia](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)
- [AlphaGo Paper](https://www.nature.com/articles/nature16961)
- [UCB1 Algorithm](https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf)
- [Original Latent Reasoning Article](https://artificialintelligencemadesimple.substack.com/)

