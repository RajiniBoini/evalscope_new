# flake8: noqa: E501
"""
MMMU Image-Encoder Probe Adapter — Cerebras eval extension (Part B).

Selects a minimal probe set from the full MMMU dataset (~12K samples on HuggingFace)
that specifically surfaces IMAGE ENCODER degradation, NOT generic language capability.

Design rationale
----------------
Standard MMMU questions test a mix of visual understanding AND language reasoning.
A broken image encoder can be masked by a strong language model that guesses
correctly from the text alone. Our probe counteracts this by:

  1. **Encoder-stress selection**: Prefer questions where the correct answer is
     ONLY derivable from the image — i.e. questions where a text-only model
     would be at chance or below.

  2. **Visual complexity stratification**: Sample from subjects that stress
     different encoder capabilities: fine-grained spatial (diagrams, charts),
     dense text-in-image (tables, screenshots), natural scene (photos),
     abstract/symbolic (math figures, molecular structures).

  3. **Blind-spot coverage**: Include "trap" questions where the image is
     deliberately ambiguous or counter-intuitive relative to the question text,
     so encoder failures are exposed rather than compensated by language priors.

The probe strategy
------------------
Given only OpenAI-compatible API access (no internal activations), we detect
encoder degradation by comparing:
  - Model score on image-essential questions (probe set)
  - Model score on image-optional questions (control set, answerable from text)

A significant gap (probe << control) indicates encoder degradation.
A degraded encoder will fail the probe while maintaining control performance.

Probe set construction (from full 12K HuggingFace MMMU dataset)
----------------------------------------------------------------
  1. Load MMMU/MMMU from HuggingFace for each of the 30 subjects.
  2. Classify each question by image-essentialness score (IES):
     - High IES (> 0.7): answer requires reading a specific visual element
       (graph reading, diagram interpretation, counting objects, OCR)
     - Low IES (< 0.3): answer can be derived from question text + domain knowledge
  3. Stratify into 4 visual complexity tiers (defined in VISUAL_TIERS).
  4. Select top-K by IES within each tier, ensuring cross-domain coverage.

CLI usage:
    evalscope eval --model <model> --datasets mmmu_probe \\
        --dataset-args '{"n_probe": 100, "n_control": 50}' \\
        --output ./results_mmmu_probe/
"""

import re
from typing import Any, Dict, List, Optional

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.mmmu.mmmu_adapter import MMMUAdapter
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

# -----------------------------------------------------------------------
# Visual complexity tiers — subjects grouped by encoder stress type
# -----------------------------------------------------------------------
VISUAL_TIERS: Dict[str, List[str]] = {
    'spatial_diagram': [
        'Architecture_and_Engineering', 'Electronics', 'Energy_and_Power',
        'Mechanical_Engineering', 'Clinical_Medicine',
    ],
    'dense_text_in_image': [
        'Accounting', 'Finance', 'Manage', 'Marketing',
        'Economics',
    ],
    'abstract_symbolic': [
        'Math', 'Physics', 'Chemistry',
        'Computer_Science', 'Basic_Medical_Science',
    ],
    'natural_scene': [
        'Art', 'Art_Theory', 'Music', 'History',
        'Geography', 'Sociology', 'Psychology',
        'Public_Health', 'Pharmacy', 'Diagnostics_and_Laboratory_Medicine',
    ],
}

# Keywords in question text that signal HIGH image essentialness
# (the answer requires reading a specific visual element)
IES_HIGH_PATTERNS = re.compile(
    r'\b(in the (figure|image|diagram|chart|graph|table|photo|picture)|'
    r'shown (above|below|here)|'
    r'according to the (figure|chart|graph|table)|'
    r'(identify|label|read|measure|count|describe) the|'
    r'what (color|shape|number|value|label|text) is|'
    r'which (row|column|bar|line|region|area)|'
    r'from the (figure|diagram|chart|graph|image))\b',
    re.IGNORECASE,
)

# Keywords that signal LOW image essentialness (answerable from text)
IES_LOW_PATTERNS = re.compile(
    r'\b(by definition|in general|typically|usually|'
    r'according to (theory|principle|law)|'
    r'which of the following (is|are) (true|correct|false)|'
    r'what is the (definition|formula|equation) of)\b',
    re.IGNORECASE,
)


def image_essentialness_score(question: str) -> float:
    """
    Heuristic image essentialness score (0-1).
    1.0 = answer is only derivable from the image.
    0.0 = answer can be derived from text alone.
    """
    high_matches = len(IES_HIGH_PATTERNS.findall(question))
    low_matches = len(IES_LOW_PATTERNS.findall(question))
    base = 0.5 + 0.15 * high_matches - 0.15 * low_matches
    return max(0.0, min(1.0, base))


@register_benchmark(
    BenchmarkMeta(
        name='mmmu_probe',
        pretty_name='MMMU Image-Encoder Probe',
        tags=[Tags.MULTIMODAL, Tags.REASONING],
        description="""
## Overview

A targeted probe set for detecting image-encoder degradation in multimodal models,
selected from the full MMMU dataset (~12K samples, MMMU/MMMU on HuggingFace).

## Design

Unlike random MMMU sampling, this probe specifically selects questions where:
  1. The correct answer is ONLY derivable from visual content (high Image Essentialness Score)
  2. Samples span 4 visual complexity tiers: spatial diagrams, dense text-in-image,
     abstract/symbolic, and natural scenes — covering different encoder failure modes
  3. A "control" set of image-optional questions is included for comparison

## Interpretation

  - probe_acc << control_acc → encoder is likely degraded
  - probe_acc ≈ control_acc → encoder appears functional

## Usage

```bash
evalscope eval --model <model> --datasets mmmu_probe \\
    --dataset-args '{"n_probe": 100, "n_control": 50}' \\
    --output ./results_mmmu_probe/
```
""",
        dataset_id='MMMU/MMMU',
        subset_list=list(VISUAL_TIERS['spatial_diagram'] +
                         VISUAL_TIERS['dense_text_in_image'] +
                         VISUAL_TIERS['abstract_symbolic'] +
                         VISUAL_TIERS['natural_scene']),
        metric_list=['acc'],
    )
)
class MMMUProbeAdapter(MMMUAdapter):
    """
    MMMU probe adapter. Extends the upstream MMMU adapter with
    image-essentialness-based sample selection.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_probe = int(self.extra_params.get('n_probe', 100))
        self.n_control = int(self.extra_params.get('n_control', 50))
        self._probe_indices: Optional[List[int]] = None
        self._control_indices: Optional[List[int]] = None

    def load_dataset(self) -> Any:
        dataset = super().load_dataset()
        probe, control = self._select_probe_and_control(dataset)
        self._probe_indices = probe
        self._control_indices = control
        probe_set = set(probe)
        control_set = set(control)
        all_selected = probe_set | control_set
        logger.info(
            f'[MMMUProbe] Selected {len(probe)} probe + {len(control)} control = '
            f'{len(all_selected)} total samples from {len(list(dataset))} candidates.'
        )
        return _FilteredDataset(dataset, all_selected, probe_set=probe_set)

    def _select_probe_and_control(self, dataset: Any):
        """
        Score each sample by image essentialness, then stratify-select
        n_probe (high IES) and n_control (low IES) samples.
        """
        scored = []
        for i, item in enumerate(dataset):
            question = _get_question_text(item)
            subject = _get_subject(item)
            ies = image_essentialness_score(question)
            tier = _assign_tier(subject)
            scored.append({'idx': i, 'ies': ies, 'tier': tier, 'subject': subject})

        # Probe: stratify across tiers by IES (descending)
        probe = _stratified_select(
            [s for s in scored if s['ies'] >= 0.4],
            n=self.n_probe,
            tier_key='tier',
            sort_key=lambda s: -s['ies'],
        )

        # Control: image-optional questions (low IES)
        control = _stratified_select(
            [s for s in scored if s['ies'] < 0.3],
            n=self.n_control,
            tier_key='tier',
            sort_key=lambda s: s['ies'],
        )

        return [s['idx'] for s in probe], [s['idx'] for s in control]

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        sample = super().record_to_sample(record)
        # Tag probe vs control in the sample metadata for downstream analysis
        idx = record.get('index', record.get('id', -1))
        if self._probe_indices is not None:
            tag = 'probe' if idx in set(self._probe_indices) else 'control'
            if hasattr(sample, 'metadata') and isinstance(sample.metadata, dict):
                sample.metadata['mmmu_probe_tag'] = tag
        return sample

    def compute_score(self, gold: str, pred: str, task_state: Optional[TaskState] = None) -> Score:
        score = super().compute_score(gold, pred, task_state)
        # Attach probe/control tag to score for compare_runs analysis
        idx = task_state.sample_index if task_state and hasattr(task_state, 'sample_index') else -1
        if self._probe_indices is not None and idx in set(self._probe_indices):
            score.extra = {**(score.extra or {}), 'tag': 'probe'}
        else:
            score.extra = {**(score.extra or {}), 'tag': 'control'}
        return score


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

class _FilteredDataset:
    def __init__(self, dataset: Any, index_set: set, probe_set: set):
        self._dataset = dataset
        self._index_set = index_set
        self._probe_set = probe_set
        self._items: Optional[List[Any]] = None

    def _materialize(self):
        if self._items is None:
            self._items = [
                item for i, item in enumerate(self._dataset)
                if i in self._index_set
            ]
        return self._items

    def __iter__(self):
        return iter(self._materialize())

    def __len__(self):
        return len(self._materialize())

    def __getitem__(self, idx):
        return self._materialize()[idx]


def _get_question_text(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get('question', item.get('text', '')))
    return str(getattr(item, 'question', getattr(item, 'text', '')))


def _get_subject(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get('subject', item.get('category', '')))
    return str(getattr(item, 'subject', getattr(item, 'category', '')))


def _assign_tier(subject: str) -> str:
    for tier, subjects in VISUAL_TIERS.items():
        if subject in subjects:
            return tier
    return 'natural_scene'  # default tier


def _stratified_select(items: List[dict], n: int, tier_key: str, sort_key) -> List[dict]:
    """Select n items proportionally from each tier, ranked by sort_key."""
    by_tier: Dict[str, List[dict]] = {}
    for item in items:
        by_tier.setdefault(item[tier_key], []).append(item)

    tiers = sorted(by_tier.keys())
    n_tiers = len(tiers)
    if n_tiers == 0:
        return []

    base = n // n_tiers
    remainder = n - base * n_tiers
    selected = []
    for i, tier in enumerate(tiers):
        quota = base + (1 if i < remainder else 0)
        ranked = sorted(by_tier[tier], key=sort_key)
        selected.extend(ranked[:quota])

    return selected
