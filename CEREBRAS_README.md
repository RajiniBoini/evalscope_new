# Cerebras Benchmark Pruning Extension

**evalscope fork** · Pinned commit: `47a272d9e787dae8bda07e7ec9d504f9f48a6f11`

This fork adds a discriminative benchmark pruning extension to [modelscope/evalscope](https://github.com/modelscope/evalscope).

---

## What's added

| Path | What it is |
|---|---|
| `evalscope_ext/pruners/discriminative_pruner.py` | Core pruning algorithm — universal, works across benchmarks |
| `evalscope/benchmarks/live_code_bench_pruned/` | Pruned LCB adapter (315 → 47 samples, Kendall-τ = 1.0) |
| `evalscope/benchmarks/aa_lcr_pruned/` | Pruned AA-LCR adapter (100 → 30 samples, Kendall-τ = 1.0) |
| `evalscope/benchmarks/mmmu_probe/` | MMMU image-encoder degradation probe (Part B) |
| `evalscope_ext/tools/compare_runs.py` | CLI to compare full vs pruned evaluation results |
| `docs/handout_a_technical.md` | Technical writeup — pruning methodology and trade-offs |
| `docs/handout_b_business.md` | Business writeup — customer value and usage guide |

---

## Install

```bash
git clone https://github.com/RajiniBoini/evalscope-fork
cd evalscope-fork
pip install -e .
```

---

## Usage

### Part A — Run pruned evaluations

```bash
# Full run (baseline, run once per benchmark)
evalscope eval --model <model> --datasets live_code_bench \
    --output ./results_full/

# Pruned run (fast, for every new model)
evalscope eval --model <model> --datasets live_code_bench_pruned \
    --dataset-args '{"pruning_strategy": "discriminative", "prune_ratio": 0.15, \
                    "reviews_dir": "./Evals/Part 1/reviews"}' \
    --output ./results_pruned/

# AA-LCR pruned
evalscope eval --model <model> --datasets aa_lcr_pruned \
    --dataset-args '{"prune_ratio": 0.3, "reviews_dir": "./Evals/Part 1/reviews"}' \
    --output ./results_pruned/

# Compare full vs pruned — verify rank preservation
python -m evalscope_ext.tools.compare_runs \
    --full ./results_full/ --pruned ./results_pruned/
```

### Part B — MMMU image-encoder probe

```bash
evalscope eval --model <model> --datasets mmmu_probe \
    --dataset-args '{"n_probe": 100, "n_control": 50}' \
    --output ./results_mmmu_probe/
```

Interpret: `probe_acc << control_acc` → encoder degradation detected.

---

## Pruning algorithm

**Stratified Rank-Directed Discriminative Selection:**

1. Build score matrix from seed model reviews (N samples × M models)
2. Compute per-sample combined score:
   - Difficulty = mean score across models
   - Raw discriminability = inter-model variance
   - Rank-direction alignment = fraction of model pairs this sample correctly orders (relative to full-set ranking), normalised to [−1, 1]
   - **Combined = discriminability × (1 + rank_alignment) / 2**
3. Stratify into 3 difficulty tiers, allocate budget proportionally
4. Select top combined-score samples per tier

Validates ranking preservation via Kendall-τ ≥ 0.8.

### Results

| Benchmark | Full | Pruned | Reduction | Kendall-τ |
|---|---|---|---|---|
| LiveCodeBench v5 | 315 | 47 | **85.1%** | **1.000** |
| AA-LCR | 100 | 30 | **70.0%** | **1.000** |

---

## Handouts

- [Handout A — Technical](docs/handout_a_technical.md)
- [Handout B — Business](docs/handout_b_business.md)
