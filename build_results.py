"""
build_results.py — Generate results_full/ and results_pruned/ from existing JSONL review files.

Usage (from evalscope-fork/):
    python build_results.py

Reads:  ../evals_data/Part1/reviews/*.jsonl
Writes: ./results_full/   (full benchmark accuracy per model)
        ./results_pruned/ (accuracy on discriminative-pruned subset)
"""

import json
import os
import sys
from pathlib import Path

# Allow importing evalscope_ext from this directory
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from evalscope_ext.pruners.discriminative_pruner import DiscriminativePruner, PrunerConfig

REVIEWS_DIR = Path(__file__).parent.parent / "evals_data" / "Part1" / "reviews"

BENCHMARKS = {
    "live_code_bench_v5": {
        "score_key": "pass",
        "prune_ratio": 0.15,   # keep ~15% → ~47 samples
    },
    "aa_lcr": {
        "score_key": "acc",
        "prune_ratio": 0.30,   # keep ~30% → ~30 samples
        "judge_noise_weight": 0.2,
    },
}

MODELS = ["kimi-k2.5", "minimax-m2.5", "gpt-oss-120b"]


def load_scores(reviews_dir: Path, benchmark: str, model: str) -> dict[int, float]:
    path = reviews_dir / f"{benchmark}__{model}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing review file: {path}")
    scores: dict[int, float] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            idx = rec["index"]
            val = rec["sample_score"]["score"]["value"]
            score = float(list(val.values())[0]) if isinstance(val, dict) else float(val)
            scores[idx] = score
    return scores


def write_result(out_dir: Path, benchmark: str, model: str, accuracy: float, n_samples: int):
    dest = out_dir / benchmark / model
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "eval_results.json").write_text(
        json.dumps({"accuracy": round(accuracy, 4), "n_samples": n_samples}, indent=2)
    )


def main():
    results_full = Path("results_full")
    results_pruned = Path("results_pruned")

    print("\n=== Building results_full/ and results_pruned/ ===\n")

    for benchmark, cfg in BENCHMARKS.items():
        print(f"Benchmark: {benchmark}")

        # Load per-model scores
        all_scores: dict[str, dict[int, float]] = {}
        for model in MODELS:
            try:
                all_scores[model] = load_scores(REVIEWS_DIR, benchmark, model)
                print(f"  {model}: {len(all_scores[model])} samples loaded")
            except FileNotFoundError as e:
                print(f"  SKIP {model}: {e}")

        if len(all_scores) < 2:
            print("  Not enough models — skipping\n")
            continue

        models_present = list(all_scores.keys())
        all_indices = sorted(set().union(*[s.keys() for s in all_scores.values()]))

        # Build score matrix (samples × models)
        matrix = np.array(
            [[all_scores[m].get(i, 0.0) for m in models_present] for i in all_indices],
            dtype=float,
        )

        # Full accuracy per model
        for col_i, model in enumerate(models_present):
            acc = matrix[:, col_i].mean()
            write_result(results_full, benchmark, model, acc, len(all_indices))

        # Run pruner
        prune_cfg = PrunerConfig(
            prune_ratio=cfg["prune_ratio"],
            score_key=cfg["score_key"],
            judge_noise_weight=cfg.get("judge_noise_weight", 0.0),
        )
        pruner = DiscriminativePruner(prune_cfg)
        pruner.fit_from_matrix(matrix, all_indices)
        selected = pruner.selected_indices
        selected_set = set(selected)

        # Pruned accuracy per model
        pruned_matrix = np.array(
            [[all_scores[m].get(i, 0.0) for m in models_present] for i in selected],
            dtype=float,
        )

        full_scores_map = {m: matrix[:, i].mean() for i, m in enumerate(models_present)}
        pruned_scores_map = {m: pruned_matrix[:, i].mean() for i, m in enumerate(models_present)}

        tau, passes = pruner.validate_ranking_preservation(full_scores_map, pruned_scores_map)

        for col_i, model in enumerate(models_present):
            acc = pruned_matrix[:, col_i].mean()
            write_result(results_pruned, benchmark, model, acc, len(selected))

        stats = pruner.prune_stats
        print(f"  Full samples : {stats['n_total']}")
        print(f"  Pruned samples: {stats['n_kept']}  ({stats['reduction_pct']}% reduction)")
        print(f"  Kendall-τ    : {tau:.3f}  ({'PASS ✓' if passes else 'FAIL ✗'})")

        print(f"\n  {'Model':<25} {'Full acc':>10} {'Pruned acc':>12}")
        print(f"  {'-'*25} {'-'*10} {'-'*12}")
        for model in models_present:
            print(f"  {model:<25} {full_scores_map[model]:>10.3f} {pruned_scores_map[model]:>12.3f}")
        print()

    print("Done.")
    print(f"  results_full/   → {results_full.resolve()}")
    print(f"  results_pruned/ → {results_pruned.resolve()}")
    print()
    print("Now run:")
    print("  python -m evalscope_ext.tools.compare_runs --full ./results_full/ --pruned ./results_pruned/")


if __name__ == "__main__":
    main()
