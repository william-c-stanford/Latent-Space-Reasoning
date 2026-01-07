# L-MCTS vs Evolution Comparison Report

**Generated:** 2026-01-07 15:17:37
**Model:** `Qwen/Qwen3-0.6B`
**Queries Tested:** 5

---

## Summary

| Metric | Evolution | MCTS | Winner |
|--------|-----------|------|--------|
| **Wins** | 0 (0%) | 5 (100%) | MCTS |
| **Avg Score** | 0.571 | 0.671 | MCTS |
| **Avg Evaluations** | 7 | 16 | Evolution (fewer is better) |
| **Avg Runtime** | 45.2s | 42.4s | MCTS (faster is better) |

### Overall Winner by Score: **MCTS**

Score difference: 10.0%

---

## Detailed Results

### cache: How do I implement a thread-safe LRU cache with O(1) operati...

**Category:** system_design | **Winner:** MCTS 🏆

| Method | Score | Evaluations | Runtime | Output Length |
|--------|-------|-------------|---------|---------------|
| Evolution | 0.531 | 7 | 63.1s | 3875 chars |
| MCTS | 0.662 | 16 | 75.0s | 5780 chars |

**Improvement:** +24.5% for mcts

### auth: Design a secure user authentication system with JWT tokens.

**Category:** system_design | **Winner:** MCTS 🏆

| Method | Score | Evaluations | Runtime | Output Length |
|--------|-------|-------------|---------|---------------|
| Evolution | 0.583 | 7 | 46.9s | 3043 chars |
| MCTS | 0.674 | 16 | 45.2s | 2778 chars |

**Improvement:** +15.6% for mcts

### knights: On an island, Knights always tell truth, Knaves always lie. ...

**Category:** logic | **Winner:** MCTS 🏆

| Method | Score | Evaluations | Runtime | Output Length |
|--------|-------|-------------|---------|---------------|
| Evolution | 0.559 | 7 | 18.9s | 344 chars |
| MCTS | 0.664 | 16 | 21.7s | 519 chars |

**Improvement:** +18.8% for mcts

### math: Prove that the sum of first n odd numbers equals n squared.

**Category:** math | **Winner:** MCTS 🏆

| Method | Score | Evaluations | Runtime | Output Length |
|--------|-------|-------------|---------|---------------|
| Evolution | 0.602 | 7 | 46.3s | 1125 chars |
| MCTS | 0.679 | 16 | 37.5s | 939 chars |

**Improvement:** +12.9% for mcts

### code: Write a Python function to find the longest increasing subse...

**Category:** code | **Winner:** MCTS 🏆

| Method | Score | Evaluations | Runtime | Output Length |
|--------|-------|-------------|---------|---------------|
| Evolution | 0.579 | 7 | 50.8s | 2204 chars |
| MCTS | 0.675 | 16 | 32.5s | 758 chars |

**Improvement:** +16.5% for mcts

---

## Results by Category

| Category | Evolution Wins | MCTS Wins | Ties |
|----------|----------------|-----------|------|
| code | 0 | 1 | 0 |
| logic | 0 | 1 | 0 |
| math | 0 | 1 | 0 |
| system_design | 0 | 2 | 0 |

---

## Analysis

### Key Observations

- ✅ **MCTS achieved equal or better scores** on average
- ⚠️ **MCTS used more evaluations** than Evolution
- ✅ **MCTS was 6% faster**

### Recommendations

Based on these results, **MCTS is recommended** for:
- Queries requiring structured exploration
- When sample efficiency matters
- When you want inspectable search paths
