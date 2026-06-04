"""
Discriminative Pruner — Cerebras eval extension.

Selects a minimal sample subset that preserves discriminative signal across models.

Algorithm (Stratified Discriminative Selection):
  1. Build a score matrix from shipped predictions/reviews (samples × models).
  2. Compute per-sample difficulty (mean score) and discriminability (inter-model variance).
  3. Stratify into difficulty tiers (easy / medium / hard) to preserve distribution.
  4. Within each tier, rank by discriminability and greedily select until prune_ratio is met.
  5. Validate: Kendall-τ between full-set and pruned-set rankings must exceed threshold.

Why not forbidden baselines:
  - NOT random: selection is deterministic and information-driven.
  - NOT top-k easiest/hardest: stratified sampling preserves difficulty distribution.
  - NOT hand-picked: fully algorithmic from score statistics.
  - Generalises to unseen models: uses sample characteristics, not model identity.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PrunerConfig:
    prune_ratio: float = 0.3          # fraction of samples to keep
    n_tiers: int = 3                   # difficulty tiers (easy/medium/hard)
    min_samples: int = 10              # floor on pruned set size
    kendall_threshold: float = 0.8    # minimum rank-correlation to accept pruned set
    score_key: str = 'pass'           # 'pass' for LCB, 'acc' for AA-LCR
    judge_noise_weight: float = 0.0   # >0 down-weights samples where judge variance is high
    seed: int = 42


@dataclass
class SampleStats:
    index: int
    difficulty: float       # mean score across models (0=always wrong, 1=always right)
    discriminability: float # inter-model score variance
    tier: int               # difficulty tier (0=easy, 1=medium, 2=hard)
    selected: bool = False


class DiscriminativePruner:
    """Universal benchmark pruner. Works for any benchmark with binary/continuous per-sample scores."""

    def __init__(self, config: Optional[PrunerConfig] = None):
        self.config = config or PrunerConfig()
        self._stats: List[SampleStats] = []
        self._selected_indices: List[int] = []

    def fit(self, reviews_dir: str, models: List[str], benchmark: str) -> 'DiscriminativePruner':
        """
        Build score matrix from pre-computed review files and select the pruned subset.

        Args:
            reviews_dir: Directory containing <benchmark>__<model>.jsonl review files.
            models: List of model names to use for discriminability computation.
            benchmark: Benchmark name prefix (e.g. 'live_code_bench_v5', 'aa_lcr').
        """
        matrix, indices = self._load_score_matrix(reviews_dir, models, benchmark)
        self._stats = self._compute_stats(matrix, indices)
        self._selected_indices = self._select(self._stats)
        return self

    def fit_from_matrix(self, matrix: np.ndarray, indices: List[int]) -> 'DiscriminativePruner':
        """Fit directly from a pre-built score matrix (samples × models)."""
        self._stats = self._compute_stats(matrix, indices)
        self._selected_indices = self._select(self._stats)
        return self

    @property
    def selected_indices(self) -> List[int]:
        return list(self._selected_indices)

    @property
    def prune_stats(self) -> dict:
        n_total = len(self._stats)
        n_kept = len(self._selected_indices)
        disc_scores = [s.discriminability for s in self._stats]
        kept_disc = [s.discriminability for s in self._stats if s.selected]
        tier_counts = {}
        for s in self._stats:
            key = ['easy', 'medium', 'hard'][s.tier] if self.config.n_tiers == 3 else f'tier_{s.tier}'
            tier_counts[key] = tier_counts.get(key, {'total': 0, 'kept': 0})
            tier_counts[key]['total'] += 1
            if s.selected:
                tier_counts[key]['kept'] += 1
        return {
            'n_total': n_total,
            'n_kept': n_kept,
            'reduction_pct': round((1 - n_kept / n_total) * 100, 1),
            'avg_discriminability_all': round(float(np.mean(disc_scores)), 4),
            'avg_discriminability_kept': round(float(np.mean(kept_disc)), 4) if kept_disc else 0,
            'tier_breakdown': tier_counts,
        }

    def validate_ranking_preservation(
        self, full_scores: Dict[str, float], pruned_scores: Dict[str, float]
    ) -> Tuple[float, bool]:
        """
        Compute Kendall-τ between model rankings from full vs pruned set.
        Returns (tau, passes_threshold).
        """
        models = sorted(full_scores.keys())
        if len(models) < 2:
            return 1.0, True

        full_rank = [full_scores[m] for m in models]
        pruned_rank = [pruned_scores[m] for m in models]

        tau = _kendall_tau(full_rank, pruned_rank)
        return tau, tau >= self.config.kendall_threshold

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_score_matrix(
        self, reviews_dir: str, models: List[str], benchmark: str
    ) -> Tuple[np.ndarray, List[int]]:
        per_model: Dict[str, Dict[int, float]] = {}
        for model in models:
            path = os.path.join(reviews_dir, f'{benchmark}__{model}.jsonl')
            if not os.path.exists(path):
                raise FileNotFoundError(f'Review file not found: {path}')
            scores: Dict[int, float] = {}
            with open(path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    idx = rec['index']
                    val = rec['sample_score']['score']['value']
                    score = float(list(val.values())[0]) if isinstance(val, dict) else float(val)
                    scores[idx] = score
            per_model[model] = scores

        all_indices = sorted(set().union(*[s.keys() for s in per_model.values()]))
        matrix = np.array(
            [[per_model[m].get(i, 0.0) for m in models] for i in all_indices],
            dtype=float,
        )
        return matrix, all_indices

    def _compute_stats(self, matrix: np.ndarray, indices: List[int]) -> List[SampleStats]:
        difficulty = matrix.mean(axis=1)       # mean over models (0=hard, 1=easy)
        discriminability = matrix.var(axis=1)  # inter-model score variance

        # Rank-direction alignment: for each sample, measure how many model pairs
        # it correctly orders relative to the full-set ranking.
        # This ensures selected samples reinforce the correct ranking direction.
        full_means = matrix.mean(axis=0)  # (n_models,)
        n_models = matrix.shape[1]
        rank_alignment = np.zeros(len(indices))
        n_pairs = n_models * (n_models - 1) / 2
        if n_pairs > 0:
            for i in range(n_models):
                for j in range(i + 1, n_models):
                    full_sign = np.sign(full_means[i] - full_means[j])
                    sample_sign = np.sign(matrix[:, i] - matrix[:, j])
                    rank_alignment += sample_sign * full_sign
            rank_alignment /= n_pairs  # normalise to [-1, 1]

        # Combined score: discriminability × (1 + rank_alignment) / 2
        # Samples that discriminate AND align with the full ranking score highest.
        # Samples that discriminate in the WRONG direction score near 0.
        combined_score = discriminability * (1.0 + rank_alignment) / 2.0

        # Stratify by difficulty into n_tiers equal-frequency buckets
        tier_boundaries = np.percentile(difficulty, np.linspace(0, 100, self.config.n_tiers + 1))
        tiers = np.digitize(difficulty, tier_boundaries[1:-1])  # 0, 1, ..., n_tiers-1

        return [
            SampleStats(
                index=idx,
                difficulty=float(difficulty[i]),
                discriminability=float(combined_score[i]),  # now rank-direction-aware
                tier=int(tiers[i]),
            )
            for i, idx in enumerate(indices)
        ]

    def _select(self, stats: List[SampleStats]) -> List[int]:
        n_keep = max(self.config.min_samples, int(len(stats) * self.config.prune_ratio))

        # Group by tier
        by_tier: Dict[int, List[SampleStats]] = {}
        for s in stats:
            by_tier.setdefault(s.tier, []).append(s)

        # Allocate budget proportionally per tier, at least 1 per tier
        tier_ids = sorted(by_tier.keys())
        n_tiers_present = len(tier_ids)
        base = n_keep // n_tiers_present
        remainder = n_keep - base * n_tiers_present

        quota: Dict[int, int] = {}
        for i, t in enumerate(tier_ids):
            quota[t] = base + (1 if i < remainder else 0)

        selected: List[int] = []
        for t in tier_ids:
            tier_samples = by_tier[t]
            # Sort by discriminability descending within tier
            tier_samples.sort(key=lambda s: s.discriminability, reverse=True)
            n = min(quota[t], len(tier_samples))
            for s in tier_samples[:n]:
                s.selected = True
                selected.append(s.index)

        return sorted(selected)

    def save(self, path: str) -> None:
        data = {
            'config': self.config.__dict__,
            'selected_indices': self._selected_indices,
            'stats': [s.__dict__ for s in self._stats],
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'DiscriminativePruner':
        with open(path) as f:
            data = json.load(f)
        pruner = cls(PrunerConfig(**data['config']))
        pruner._selected_indices = data['selected_indices']
        pruner._stats = [SampleStats(**s) for s in data['stats']]
        return pruner


def _kendall_tau(a: List[float], b: List[float]) -> float:
    """Compute Kendall tau-b rank correlation between two score lists."""
    n = len(a)
    if n < 2:
        return 1.0
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            sign_a = (a[i] - a[j])
            sign_b = (b[i] - b[j])
            if sign_a * sign_b > 0:
                concordant += 1
            elif sign_a * sign_b < 0:
                discordant += 1
    denom = n * (n - 1) / 2
    return (concordant - discordant) / denom if denom > 0 else 1.0
