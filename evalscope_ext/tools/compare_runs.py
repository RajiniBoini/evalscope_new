"""
compare_runs — compare full vs pruned evalscope evaluation results.

Usage:
    python -m evalscope_ext.tools.compare_runs \\
        --full  ./results_full/ \\
        --pruned ./results_pruned/

Output:
    - Per-benchmark accuracy table (full vs pruned)
    - Kendall-τ rank correlation
    - Reduction statistics
    - Pass/fail verdict (τ ≥ 0.8)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _load_results(run_dir: str) -> Dict[str, Dict[str, float]]:
    """
    Load evaluation results from an evalscope output directory.

    Expected structure (evalscope default):
        <run_dir>/
            <benchmark>/
                <model>/
                    eval_results.json   (or reports/<benchmark>.json)

    Falls back to scanning for any *.json containing score summaries.
    Returns: {benchmark: {model: accuracy}}
    """
    run_path = Path(run_dir)
    if not run_path.exists():
        raise FileNotFoundError(f'Results directory not found: {run_dir}')

    results: Dict[str, Dict[str, float]] = {}

    # Pattern 1: evalscope standard layout
    for benchmark_dir in sorted(run_path.iterdir()):
        if not benchmark_dir.is_dir():
            continue
        benchmark = benchmark_dir.name
        for model_dir in sorted(benchmark_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            # Try several known output filenames
            for fname in ['eval_results.json', 'report.json', 'summary.json']:
                fpath = model_dir / fname
                if fpath.exists():
                    data = json.loads(fpath.read_text())
                    acc = _extract_accuracy(data)
                    if acc is not None:
                        results.setdefault(benchmark, {})[model] = acc
                    break
            # Also scan reports/ subdirectory
            reports_dir = model_dir / 'reports'
            if reports_dir.exists():
                for rfile in reports_dir.glob('*.json'):
                    data = json.loads(rfile.read_text())
                    acc = _extract_accuracy(data)
                    if acc is not None:
                        results.setdefault(rfile.stem, {})[model] = acc

    # Pattern 2: flat JSON files at run_dir root
    if not results:
        for jfile in run_path.glob('*.json'):
            try:
                data = json.loads(jfile.read_text())
                acc = _extract_accuracy(data)
                if acc is not None:
                    results.setdefault(jfile.stem, {})['unknown'] = acc
            except Exception:
                pass

    return results


def _extract_accuracy(data: dict) -> Optional[float]:
    """Extract a scalar accuracy from various evalscope JSON formats."""
    for key in ('accuracy', 'acc', 'score', 'pass', 'pass@1', 'mean_acc'):
        if key in data:
            v = data[key]
            return float(v) if isinstance(v, (int, float)) else None
    # Nested: {"metrics": {"accuracy": 0.7}}
    if 'metrics' in data and isinstance(data['metrics'], dict):
        return _extract_accuracy(data['metrics'])
    # List of per-sample scores: compute mean
    if 'samples' in data and isinstance(data['samples'], list):
        scores = [s.get('score', s.get('pass', None)) for s in data['samples']]
        scores = [float(s) for s in scores if s is not None]
        return sum(scores) / len(scores) if scores else None
    return None


def _kendall_tau(scores_a: List[float], scores_b: List[float]) -> float:
    n = len(scores_a)
    if n < 2:
        return 1.0
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            sa = scores_a[i] - scores_a[j]
            sb = scores_b[i] - scores_b[j]
            if sa * sb > 0:
                concordant += 1
            elif sa * sb < 0:
                discordant += 1
    denom = n * (n - 1) / 2
    return (concordant - discordant) / denom if denom > 0 else 1.0


def compare(full_dir: str, pruned_dir: str, threshold: float = 0.8) -> int:
    full = _load_results(full_dir)
    pruned = _load_results(pruned_dir)

    all_benchmarks = sorted(set(full) | set(pruned))
    if not all_benchmarks:
        print('ERROR: No results found. Check that run directories contain evalscope output.')
        return 1

    all_pass = True
    print()
    print('=' * 70)
    print('  CEREBRAS EVAL — Full vs Pruned Comparison')
    print('=' * 70)

    for benchmark in all_benchmarks:
        full_b = full.get(benchmark, {})
        pruned_b = pruned.get(benchmark, {})
        models = sorted(set(full_b) | set(pruned_b))
        if not models:
            continue

        print(f'\n  Benchmark: {benchmark}')
        print(f'  {"Model":<30} {"Full acc":>10} {"Pruned acc":>12} {"Delta":>8}')
        print(f'  {"-"*30} {"-"*10} {"-"*12} {"-"*8}')

        full_scores, pruned_scores = [], []
        for model in models:
            fa = full_b.get(model)
            pa = pruned_b.get(model)
            delta_str = f'{(pa - fa):+.3f}' if fa is not None and pa is not None else '  N/A'
            fa_str = f'{fa:.3f}' if fa is not None else '   N/A'
            pa_str = f'{pa:.3f}' if pa is not None else '      N/A'
            print(f'  {model:<30} {fa_str:>10} {pa_str:>12} {delta_str:>8}')
            if fa is not None:
                full_scores.append(fa)
            if pa is not None:
                pruned_scores.append(pa)

        if len(full_scores) >= 2 and len(pruned_scores) >= 2:
            tau = _kendall_tau(full_scores, pruned_scores)
            verdict = 'PASS ✓' if tau >= threshold else 'FAIL ✗'
            if tau < threshold:
                all_pass = False
            print(f'\n  Kendall-τ rank correlation: {tau:.3f}  [{verdict}]  (threshold: {threshold})')

    print()
    print('=' * 70)
    overall = 'ALL PASS ✓' if all_pass else 'SOME FAILED ✗'
    print(f'  Overall: {overall}')
    print('=' * 70)
    print()
    return 0 if all_pass else 1


def main():
    parser = argparse.ArgumentParser(
        description='Compare full vs pruned evalscope evaluation runs.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--full', required=True, help='Path to full-benchmark results directory')
    parser.add_argument('--pruned', required=True, help='Path to pruned results directory')
    parser.add_argument('--threshold', type=float, default=0.8,
                        help='Minimum Kendall-τ to pass (default: 0.8)')
    args = parser.parse_args()
    sys.exit(compare(args.full, args.pruned, args.threshold))


if __name__ == '__main__':
    main()
