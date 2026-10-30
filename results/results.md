# Experimental results

All numbers are copied from the thesis (Chapter 5). Preference accuracy is computed on the
held-out, prompt-level test split of each dataset, so absolute values are **not comparable
across experiments** - compare each base model with its own fine-tuned version.
External benchmarks are zero-shot `lm-eval` runs on at most 250 examples per task
(standard error of about +-3 points), so differences of a few points are within noise.

## Preference accuracy (AutoBench test split)

| Exp. | Model | Dataset | Checkpoint | Raw acc. (%) | Length-normalised acc. (%) |
|---|---|---|---|---|---|
| E1 | Gemma 2 2B | Best-vs-Worst | Base | 54.84 | 67.74 |
| E1 | Gemma 2 2B | Best-vs-Worst | DPO | 54.84 | 67.74 |
| E2 | Qwen 2.5 0.5B | Best-vs-Worst | Base | 29.03 | 38.71 |
| E2 | Qwen 2.5 0.5B | Best-vs-Worst | DPO | 29.03 | 38.71 |
| E3 | Qwen 2.5 0.5B | Pairwise (delta>=0.3) | Base | 32.55 | 40.94 |
| E3 | Qwen 2.5 0.5B | Pairwise (delta>=0.3) | DPO | 32.72 | 40.94 |
| E4 | Qwen 2.5 0.5B | Pairwise (delta>=0.5) | Base | 29.55 | 41.56 |
| E4 | Qwen 2.5 0.5B | Pairwise (delta>=0.5) | 1 epoch | 29.55 | 41.88 |
| E4 | Qwen 2.5 0.5B | Pairwise (delta>=0.5) | 2 epochs | 29.87 | 42.52 |

## External benchmarks (accuracy %, IFEval = prompt-level strict)

### E1 - Gemma 2 2B + Best-vs-Worst

| Benchmark | Base | DPO | Delta |
|---|---|---|---|
| ARC-Challenge | 49.20 | 48.80 | -0.40 |
| HellaSwag | 66.00 | 64.40 | -1.60 |
| Winogrande | 70.80 | 71.20 | +0.40 |
| TruthfulQA MC2 | 55.06 | 55.08 | +0.02 |
| MMLU | 57.63 | 57.67 | +0.04 |
| IFEval | 39.60 | 44.00 | +4.40 |

### E2 - Qwen 2.5 0.5B + Best-vs-Worst

| Benchmark | Base | DPO | Delta |
|---|---|---|---|
| ARC-Challenge | 36.40 | 36.40 | 0.00 |
| HellaSwag | 48.00 | 47.20 | -0.80 |
| Winogrande | 59.20 | 58.40 | -0.80 |
| TruthfulQA MC2 | 42.37 | 42.31 | -0.06 |
| MMLU | 45.92 | 46.11 | +0.19 |
| IFEval | 20.00 | 16.40 | -3.60 |

### E3 - Qwen 2.5 0.5B + Pairwise (delta >= 0.3)

| Benchmark | Base | DPO | Delta |
|---|---|---|---|
| ARC-Challenge | 36.40 | 36.40 | 0.00 |
| HellaSwag | 48.00 | 47.20 | -0.80 |
| Winogrande | 59.20 | 59.20 | 0.00 |
| TruthfulQA MC2 | 42.37 | 42.47 | +0.10 |
| MMLU | 45.92 | 46.24 | +0.32 |
| IFEval | 20.00 | 18.40 | -1.60 |

### E4 - Qwen 2.5 0.5B + Pairwise (delta >= 0.5), per-epoch checkpoints

| Benchmark | Base | 1 epoch | 2 epochs |
|---|---|---|---|
| ARC-Challenge | 36.40 | 32.00 | 33.00 |
| HellaSwag | 48.00 | 49.00 | 49.00 |
| Winogrande | 59.20 | 58.00 | 58.00 |
| TruthfulQA MC2 | 42.37 | 39.76 | 40.02 |
| MMLU | 45.92 | 45.84 | 46.07 |
| IFEval | 20.00 | 15.00 | 16.00 |
