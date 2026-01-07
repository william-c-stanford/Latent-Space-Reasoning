# L-MCTS vs Evolution Comparison Report

**Generated:** 2026-01-07 13:27:20
**Model:** `Qwen/Qwen3-0.6B`
**Queries Tested:** 5

---

## Summary

| Metric | Evolution | MCTS | Winner |
|--------|-----------|------|--------|
| **Wins** | 5 (100%) | 0 (0%) | Evolution |
| **Avg Score** | 0.576 | 0.569 | Evolution |
| **Avg Evaluations** | 7 | 13 | Evolution (fewer is better) |
| **Avg Runtime** | 44.3s | 43.7s | MCTS (faster is better) |

### Overall Winner by Score: **Evolution**

Score difference: 0.7%

---

## Detailed Results

### cache: How do I implement a thread-safe LRU cache with O(1) operati...

**Category:** system_design | **Winner:** EVOLUTION 📈

| Method | Score | Evaluations | Runtime | Output Length |
|--------|-------|-------------|---------|---------------|
| Evolution | 0.545 | 7 | 46.7s | 2267 chars |
| MCTS | 0.539 | 13 | 52.3s | 3524 chars |

**Improvement:** +1.1% for evolution

### auth: Design a secure user authentication system with JWT tokens.

**Category:** system_design | **Winner:** EVOLUTION 📈

| Method | Score | Evaluations | Runtime | Output Length |
|--------|-------|-------------|---------|---------------|
| Evolution | 0.583 | 7 | 46.3s | 3081 chars |
| MCTS | 0.578 | 12 | 42.4s | 2574 chars |

**Improvement:** +0.7% for evolution

### knights: On an island, Knights always tell truth, Knaves always lie. ...

**Category:** logic | **Winner:** EVOLUTION 📈

| Method | Score | Evaluations | Runtime | Output Length |
|--------|-------|-------------|---------|---------------|
| Evolution | 0.561 | 7 | 19.7s | 412 chars |
| MCTS | 0.547 | 9 | 16.9s | 349 chars |

**Improvement:** +2.6% for evolution

### math: Prove that the sum of first n odd numbers equals n squared.

**Category:** math | **Winner:** EVOLUTION 📈

| Method | Score | Evaluations | Runtime | Output Length |
|--------|-------|-------------|---------|---------------|
| Evolution | 0.602 | 7 | 52.4s | 1062 chars |
| MCTS | 0.597 | 16 | 46.1s | 1029 chars |

**Improvement:** +0.9% for evolution

### code: Write a Python function to find the longest increasing subse...

**Category:** code | **Winner:** EVOLUTION 📈

| Method | Score | Evaluations | Runtime | Output Length |
|--------|-------|-------------|---------|---------------|
| Evolution | 0.587 | 7 | 56.2s | 1851 chars |
| MCTS | 0.583 | 16 | 61.1s | 1937 chars |

**Improvement:** +0.8% for evolution

---

## Results by Category

| Category | Evolution Wins | MCTS Wins | Ties |
|----------|----------------|-----------|------|
| code | 1 | 0 | 0 |
| logic | 1 | 0 | 0 |
| math | 1 | 0 | 0 |
| system_design | 2 | 0 | 0 |

---

## Analysis

### Key Observations

- ⚠️ **Evolution outperformed MCTS** by 0.7%
- ⚠️ **MCTS used more evaluations** than Evolution
- ✅ **MCTS was 1% faster**

### Recommendations

Based on these results, **Evolution may be preferable** for:
- Maximum diversity in solutions
- When compute budget is not a concern
- The current query types tested

Consider tuning MCTS parameters:
- Increase `exploration_constant` for more exploration
- Increase `n_iterations` for longer search
- Try different `temperature_decay` values
