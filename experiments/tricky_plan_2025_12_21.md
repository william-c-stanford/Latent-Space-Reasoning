# Tricky Plan Evaluation (2025-12-21)

Purpose: Evaluate plan quality on 10 tricky logic queries (tests 34-43) with a
1024 token decode budget across quantization and decode strategies.

Setup:
- Model: Qwen/Qwen3-0.6B
- Evolution: chains=3, generations=3, survivors=2
- Quantization: none, auto, 4bit
- Decode strategy: best, combined

Findings (qualitative):
- Plans were verbose and step-like but usually truncated before a final answer.
- No meaningful plan-quality differences across quantization or decode strategy.
- test42_logic_grid_houses failed across all configs with error:
  "cannot convert float infinity to integer".
- Only one config emitted a final answer-like line (test38 auto_combined),
  and it was incorrect.

Artifacts (gitignored):
- eval_results/latent_matrix_tricky_1024
