# flake8: noqa: E501
"""
LiveCodeBench Pruned Adapter — Cerebras eval extension.

Wraps the upstream LiveCodeBench adapter and filters samples to the
discriminatively-selected subset computed by DiscriminativePruner.

CLI usage:
    evalscope eval --model <model> --datasets live_code_bench_pruned \\
        --dataset-args '{"pruning_strategy": "discriminative", "prune_ratio": 0.15}' \\
        --output ./results_pruned/

Pruning args (all optional, passed via --dataset-args JSON):
    pruning_strategy  : "discriminative" (default) — only strategy currently supported
    prune_ratio       : float 0-1, fraction of samples to keep (default 0.15)
    reviews_dir       : path to pre-computed review JSONL files for seeding the pruner
    models            : list of model names whose reviews to use for seeding
    min_samples       : minimum pruned set size (default 10)
    kendall_threshold : Kendall-τ threshold for validation warning (default 0.8)
    index_list        : explicit list of indices to keep (bypasses algorithm, for reproducibility)
"""

from typing import Any, Dict, List, Optional

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.live_code_bench.live_code_bench_adapter import LiveCodeBenchAdapter
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

# Default seed models and reviews directory (relative to CWD or absolute)
DEFAULT_MODELS = ['gpt-oss-120b', 'kimi-k2.5', 'minimax-m2.5']
DEFAULT_BENCHMARK_NAME = 'live_code_bench_v5'


@register_benchmark(
    BenchmarkMeta(
        name='live_code_bench_pruned',
        pretty_name='Live-Code-Bench (Pruned)',
        tags=[Tags.CODING],
        description="""
## Overview

A discriminatively-pruned version of LiveCodeBench v5 (315 → ~47 samples at 15% ratio).

Samples are selected using **Stratified Discriminative Selection**: samples are grouped
into difficulty tiers (easy / medium / hard) and ranked within each tier by inter-model
score variance. The highest-variance samples per tier are retained, preserving the
difficulty distribution while maximising signal about relative model rankings.

Validated by Kendall-τ rank correlation (≥ 0.8 required) against the full benchmark.

## Usage

```bash
evalscope eval --model <model> --datasets live_code_bench_pruned \\
    --dataset-args '{"prune_ratio": 0.15, "reviews_dir": "./Evals/Part 1/reviews"}' \\
    --output ./results_pruned/
```
""",
        dataset_id='evalscope/livecodebench_code_generation_lite_parquet',
        subset_list=['v5'],
        metric_list=['acc'],
    )
)
class LiveCodeBenchPrunedAdapter(LiveCodeBenchAdapter):
    """
    Pruned LiveCodeBench adapter. Inherits all evaluation logic from the upstream
    adapter; only overrides dataset loading to filter to the selected subset.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._selected_indices: Optional[List[int]] = None

    def _resolve_selected_indices(self) -> List[int]:
        """Compute or load the pruned index list."""
        # Explicit index list takes priority (reproducibility / CI use)
        if 'index_list' in self.extra_params:
            return list(self.extra_params['index_list'])

        prune_ratio = float(self.extra_params.get('prune_ratio', 0.15))
        reviews_dir = self.extra_params.get('reviews_dir')
        models = self.extra_params.get('models', DEFAULT_MODELS)
        min_samples = int(self.extra_params.get('min_samples', 10))
        kendall_threshold = float(self.extra_params.get('kendall_threshold', 0.8))

        from evalscope_ext.pruners.discriminative_pruner import DiscriminativePruner, PrunerConfig

        config = PrunerConfig(
            prune_ratio=prune_ratio,
            min_samples=min_samples,
            kendall_threshold=kendall_threshold,
            score_key='pass',
        )
        pruner = DiscriminativePruner(config)

        if reviews_dir:
            logger.info(f'[LCBPruned] Fitting pruner from {reviews_dir} with models {models}')
            pruner.fit(
                reviews_dir=reviews_dir,
                models=models,
                benchmark=DEFAULT_BENCHMARK_NAME,
            )
        else:
            logger.warning(
                '[LCBPruned] No reviews_dir provided — using built-in index list '
                '(pre-computed at prune_ratio=0.15 from shipped reviews). '
                'Pass reviews_dir to recompute.'
            )
            # Pre-computed fallback indices (prune_ratio=0.15, seed models)
            pruner._selected_indices = _PRECOMPUTED_INDICES_015

        stats = pruner.prune_stats
        logger.info(
            f'[LCBPruned] Pruned {stats["n_total"]} → {stats["n_kept"]} samples '
            f'({stats["reduction_pct"]}% reduction). '
            f'Avg discriminability: {stats["avg_discriminability_kept"]:.4f} '
            f'(vs {stats["avg_discriminability_all"]:.4f} overall)'
        )
        return pruner.selected_indices

    def load_dataset(self) -> Any:
        """Load the full dataset then filter to selected indices."""
        if self._selected_indices is None:
            self._selected_indices = self._resolve_selected_indices()

        dataset = super().load_dataset()
        index_set = set(self._selected_indices)

        # Filter: evalscope datasets are iterable; we wrap with index-aware filtering
        return _FilteredDataset(dataset, index_set)


class _FilteredDataset:
    """Lightweight dataset wrapper that filters samples by index field."""

    def __init__(self, dataset: Any, index_set: set):
        self._dataset = dataset
        self._index_set = index_set
        self._items: Optional[List[Any]] = None

    def _materialize(self) -> List[Any]:
        if self._items is None:
            self._items = [
                item for item in self._dataset
                if _get_index(item) in self._index_set
            ]
        return self._items

    def __iter__(self):
        return iter(self._materialize())

    def __len__(self):
        return len(self._materialize())

    def __getitem__(self, idx):
        return self._materialize()[idx]


def _get_index(item: Any) -> int:
    """Extract sample index from various container types."""
    if isinstance(item, dict):
        return int(item.get('index', item.get('id', -1)))
    if hasattr(item, 'index'):
        return int(item.index)
    if hasattr(item, 'id'):
        return int(item.id)
    return -1


# Pre-computed indices at prune_ratio=0.15 from the three shipped models.
# Regenerated by: DiscriminativePruner(PrunerConfig(prune_ratio=0.15)).fit(reviews_dir, models, 'live_code_bench_v5')
_PRECOMPUTED_INDICES_015: List[int] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 28, 29, 41, 57, 58, 60, 61, 62, 63, 65, 67, 69,
    72, 73, 74, 77, 83, 88, 92, 105, 108,
]
