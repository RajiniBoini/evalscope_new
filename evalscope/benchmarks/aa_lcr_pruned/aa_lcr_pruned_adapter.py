# flake8: noqa: E501
"""
AA-LCR Pruned Adapter — Cerebras eval extension.

Wraps the upstream AA-LCR adapter with discriminative pruning.

AA-LCR note: scored by an LLM judge which is non-deterministic. The pruner
accounts for this by applying a judge_noise_weight that down-weights samples
where the judge disagrees with itself across repeated calls. In the shipped
data we approximate noise by using score variance across models as a proxy,
capped to avoid discarding genuinely discriminative hard samples.

CLI usage:
    evalscope eval --model <model> --datasets aa_lcr_pruned \\
        --dataset-args '{"prune_ratio": 0.3, "reviews_dir": "./Evals/Part 1/reviews"}' \\
        --output ./results_pruned/
"""

from typing import Any, List, Optional

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.aa_lcr.aa_lcr_adapter import AALCRAdapter
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

DEFAULT_MODELS = ['gpt-oss-120b', 'kimi-k2.5', 'minimax-m2.5']
DEFAULT_BENCHMARK_NAME = 'aa_lcr'


@register_benchmark(
    BenchmarkMeta(
        name='aa_lcr_pruned',
        pretty_name='AA-LCR (Pruned)',
        tags=[Tags.KNOWLEDGE, Tags.REASONING, Tags.LONG_CONTEXT],
        description="""
## Overview

A discriminatively-pruned version of AA-LCR (100 → ~30 samples at 30% ratio).

Uses Stratified Discriminative Selection with LLM-judge noise awareness:
samples are stratified by difficulty tier; within each tier, samples are
ranked by inter-model score variance. AA-LCR is graded by an LLM judge
whose non-determinism adds noise — this is accounted for by treating
moderate-difficulty samples (difficulty 0.3–0.7) as more reliable signal
than extreme-difficulty samples where judge noise is harder to separate
from genuine model variance.

## Usage

```bash
evalscope eval --model <model> --datasets aa_lcr_pruned \\
    --dataset-args '{"prune_ratio": 0.3, "reviews_dir": "./Evals/Part 1/reviews"}' \\
    --output ./results_pruned/
```
""",
        dataset_id='',  # AA-LCR loads from local cache, same as upstream
        subset_list=['default'],
        metric_list=['acc'],
    )
)
class AALCRPrunedAdapter(AALCRAdapter):
    """
    Pruned AA-LCR adapter. Inherits all prompt/judge logic from upstream;
    only overrides dataset loading to filter to the selected subset.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._selected_indices: Optional[List[int]] = None

    def _resolve_selected_indices(self) -> List[int]:
        if 'index_list' in self.extra_params:
            return list(self.extra_params['index_list'])

        prune_ratio = float(self.extra_params.get('prune_ratio', 0.30))
        reviews_dir = self.extra_params.get('reviews_dir')
        models = self.extra_params.get('models', DEFAULT_MODELS)
        min_samples = int(self.extra_params.get('min_samples', 10))
        kendall_threshold = float(self.extra_params.get('kendall_threshold', 0.8))

        from evalscope_ext.pruners.discriminative_pruner import DiscriminativePruner, PrunerConfig

        config = PrunerConfig(
            prune_ratio=prune_ratio,
            min_samples=min_samples,
            kendall_threshold=kendall_threshold,
            score_key='acc',
            # AA-LCR: moderate noise weight since LLM judge is non-deterministic
            judge_noise_weight=0.2,
        )
        pruner = DiscriminativePruner(config)

        if reviews_dir:
            logger.info(f'[AALCRPruned] Fitting pruner from {reviews_dir} with models {models}')
            pruner.fit(reviews_dir=reviews_dir, models=models, benchmark=DEFAULT_BENCHMARK_NAME)
        else:
            logger.warning('[AALCRPruned] No reviews_dir — using pre-computed index list.')
            pruner._selected_indices = _PRECOMPUTED_INDICES_030

        stats = pruner.prune_stats
        logger.info(
            f'[AALCRPruned] Pruned {stats["n_total"]} → {stats["n_kept"]} samples '
            f'({stats["reduction_pct"]}% reduction).'
        )
        return pruner.selected_indices

    def load_dataset(self) -> Any:
        if self._selected_indices is None:
            self._selected_indices = self._resolve_selected_indices()

        dataset = super().load_dataset()
        index_set = set(self._selected_indices)
        return _FilteredDataset(dataset, index_set)


class _FilteredDataset:
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
    if isinstance(item, dict):
        return int(item.get('index', item.get('id', -1)))
    if hasattr(item, 'index'):
        return int(item.index)
    if hasattr(item, 'id'):
        return int(item.id)
    return -1


# Pre-computed indices at prune_ratio=0.30
_PRECOMPUTED_INDICES_030: List[int] = [
    0, 1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14, 15, 19, 20,
    21, 22, 25, 26, 27, 29, 30, 31, 33, 35, 39, 43, 47, 48,
]
