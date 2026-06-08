"""
MMMU Image-Encoder Degradation Probe — Cerebras eval extension.

Builds a two-set probe from MMMU to detect whether a model's image encoder
is degraded, without running the full 12K-question benchmark.

Design (see handout_a_technical.md §Part B):
  Probe set  (n=100): high-IES samples — correct answer *requires* reading the image.
  Control set (n=50): low-IES samples  — answerable from question text alone.

Detection signal:
  A functioning encoder  → probe_acc ≈ control_acc
  A degraded encoder     → probe_acc << control_acc

Visual complexity tiers (each stresses a different encoder failure mode):
  SPATIAL      — circuit/engineering diagrams; fails when positional detail is lost
  DENSE_TEXT   — OCR tasks; fails first as encoder resolution degrades
  ABSTRACT     — math figures, molecular structures; fails on symbol confusion
  NATURAL      — natural scenes; last to degrade, used as baseline
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Visual complexity tiers
# ---------------------------------------------------------------------------

class VisualTier(str, Enum):
    SPATIAL = "spatial"          # Engineering/circuit diagrams, floor plans
    DENSE_TEXT = "dense_text"    # Text-in-image, tables, OCR
    ABSTRACT = "abstract"        # Math figures, chemical structures, symbols
    NATURAL = "natural"          # Photographs, natural scenes (baseline)


# Keyword heuristics that suggest a given tier from the question text.
_TIER_PATTERNS: Dict[VisualTier, List[str]] = {
    VisualTier.SPATIAL: [
        r"\b(diagram|circuit|schematic|layout|floor.?plan|cross.?section|exploded.?view"
        r"|blueprint|wiring|topology|network.?diagram)\b",
    ],
    VisualTier.DENSE_TEXT: [
        r"\b(table|chart|graph|plot|legend|axis|label|caption|text in (the )?image"
        r"|written|inscription|sign|poster|document|ocr|read(ing)? (the )?text)\b",
    ],
    VisualTier.ABSTRACT: [
        r"\b(formula|equation|symbol|molecule|chemical|structure|reaction"
        r"|mathematical|geometric|matrix|vector|coordinate|function graph)\b",
    ],
    VisualTier.NATURAL: [
        r"\b(photo(graph)?|picture|scene|image shows|depicted|illustrated)\b",
    ],
}

_COMPILED: Dict[VisualTier, re.Pattern] = {
    tier: re.compile("|".join(pats), re.IGNORECASE)
    for tier, pats in _TIER_PATTERNS.items()
}


# ---------------------------------------------------------------------------
# Image Essentialness Score (IES)
# ---------------------------------------------------------------------------

# Phrases that strongly indicate the image *must* be read to answer.
_HIGH_IES_PATTERNS = re.compile(
    r"\b(count(ing)?|how many|locate|identify (in|on|from) (the )?(image|figure|diagram)"
    r"|read (the )?(value|label|text|axis|scale|table)"
    r"|according to (the )?(figure|image|chart|graph|diagram|table)"
    r"|from (the )?(figure|image|chart|graph|diagram|table)"
    r"|shown in (the )?(figure|image|chart|graph|diagram)"
    r"|what (is|are) (labeled|marked|indicated|shown|depicted)"
    r"|measure|coordinates|position of|angle of|distance between"
    r"|which (arrow|box|region|area|color|section))\b",
    re.IGNORECASE,
)

# Phrases that suggest the question is answerable from language alone.
_LOW_IES_PATTERNS = re.compile(
    r"\b(definition of|what is meant by|explain|describe the concept"
    r"|which of the following (best )?describes|generally speaking"
    r"|in general|theoretical|principle of|formula for"
    r"|historically|by definition)\b",
    re.IGNORECASE,
)


class ImageEssentialnessScore:
    """
    Heuristic scorer (0–1) estimating how much the correct answer depends on
    reading the image vs. recalling domain knowledge from text alone.

    Score interpretation:
      0.0 – 0.3 : low   — likely answerable without the image (control-set candidate)
      0.3 – 0.6 : medium — ambiguous
      0.6 – 1.0 : high  — image must be read to answer correctly (probe-set candidate)

    This is a heuristic, not ground truth. Accuracy improves when a text-only
    baseline model run is used to calibrate (see handout §Assumptions).
    """

    def score(self, question: str, options: Optional[List[str]] = None) -> float:
        """
        Return IES in [0, 1] for a single MMMU question.

        Args:
            question: The question text (may include stem + answer choices).
            options:  Optional list of answer option strings for extra signal.
        """
        text = question
        if options:
            text = text + " " + " ".join(options)

        high_hits = len(_HIGH_IES_PATTERNS.findall(text))
        low_hits = len(_LOW_IES_PATTERNS.findall(text))

        # Short questions with no strong signal default to medium-low.
        raw = (high_hits * 0.25) - (low_hits * 0.15)
        # Clamp to [0, 1] and add a small baseline so truly empty questions
        # don't collapse to 0 (they may still need the image).
        return float(min(1.0, max(0.05, 0.4 + raw)))

    def classify_tier(self, question: str, subject: Optional[str] = None) -> VisualTier:
        """
        Assign a VisualTier from question text (and optional MMMU subject string).
        Falls back to NATURAL if no specific tier is detected.
        """
        text = question if not subject else f"{subject} {question}"
        for tier in (VisualTier.SPATIAL, VisualTier.DENSE_TEXT, VisualTier.ABSTRACT):
            if _COMPILED[tier].search(text):
                return tier
        return VisualTier.NATURAL


# ---------------------------------------------------------------------------
# Probe builder
# ---------------------------------------------------------------------------

@dataclass
class MMSample:
    index: int
    question: str
    options: List[str]
    subject: Optional[str] = None
    ies: float = 0.0
    tier: VisualTier = VisualTier.NATURAL
    in_probe: bool = False
    in_control: bool = False


@dataclass
class ProbeConfig:
    probe_n: int = 100    # total probe-set size (high-IES, spread across tiers)
    control_n: int = 50   # total control-set size (low-IES)
    ies_probe_min: float = 0.6    # minimum IES to be eligible for probe set
    ies_control_max: float = 0.3  # maximum IES to be eligible for control set
    seed: int = 42


class MMMUProbeBuilder:
    """
    Selects a probe set and control set from a collection of MMMU samples.

    Usage::

        scorer = ImageEssentialnessScore()
        builder = MMMUProbeBuilder()
        probe_idx, control_idx = builder.build(samples)
    """

    def __init__(self, config: Optional[ProbeConfig] = None):
        self.config = config or ProbeConfig()
        self._scorer = ImageEssentialnessScore()

    def annotate(self, samples: List[MMSample]) -> List[MMSample]:
        """Score and tier-classify all samples in-place, return them."""
        for s in samples:
            s.ies = self._scorer.score(s.question, s.options)
            s.tier = self._scorer.classify_tier(s.question, s.subject)
        return samples

    def build(
        self, samples: List[MMSample]
    ) -> Tuple[List[int], List[int]]:
        """
        Build probe and control sets.

        Returns:
            (probe_indices, control_indices) — lists of sample .index values.
        """
        self.annotate(samples)

        # --- Probe set: top-IES per tier, proportional budget ---
        probe_candidates: Dict[VisualTier, List[MMSample]] = {t: [] for t in VisualTier}
        for s in samples:
            if s.ies >= self.config.ies_probe_min:
                probe_candidates[s.tier].append(s)

        tiers_with_data = [t for t in VisualTier if probe_candidates[t]]
        per_tier = max(1, self.config.probe_n // max(1, len(tiers_with_data)))
        probe_indices: List[int] = []
        for tier in VisualTier:
            bucket = sorted(probe_candidates[tier], key=lambda x: x.ies, reverse=True)
            for s in bucket[:per_tier]:
                s.in_probe = True
                probe_indices.append(s.index)

        # --- Control set: low-IES samples ---
        control_candidates = sorted(
            [s for s in samples if s.ies <= self.config.ies_control_max],
            key=lambda x: x.ies,
        )
        control_indices: List[int] = []
        for s in control_candidates[: self.config.control_n]:
            s.in_control = True
            control_indices.append(s.index)

        return sorted(probe_indices), sorted(control_indices)

    def detection_signal(
        self, probe_acc: float, control_acc: float
    ) -> Dict[str, object]:
        """
        Interpret the two accuracy values as an encoder health signal.

        Returns a dict with 'gap', 'degraded' flag, and a human-readable
        'verdict' string suitable for a sales-engineer report.
        """
        gap = control_acc - probe_acc
        degraded = gap > 0.10  # >10 pp gap is considered a degradation signal
        if gap > 0.25:
            verdict = "SEVERE degradation — encoder likely non-functional for vision-critical tasks"
        elif gap > 0.10:
            verdict = "MODERATE degradation — encoder loses accuracy on image-required questions"
        elif gap > 0.03:
            verdict = "MARGINAL — within noise; run with more samples to confirm"
        else:
            verdict = "HEALTHY — probe and control accuracy are comparable"
        return {
            "probe_acc": round(probe_acc, 4),
            "control_acc": round(control_acc, 4),
            "gap_pp": round(gap * 100, 2),
            "degraded": degraded,
            "verdict": verdict,
        }
