# Handout A — Why This Works
**Cerebras Benchmark Pruning Extension · Technical Audience**

---

## Part A — What problem did I understand myself to be solving?

The customer needs a fast go/no-go signal on two model capabilities: code generation and long-context reasoning. Running 315 LiveCodeBench questions or 100 AA-LCR documents at full inference cost for every candidate model is wasteful — most of that cost produces no additional information about relative model rankings.

The real question is: **which samples are doing the work of discriminating between models?**

With 3 models and binary scores, 315 LCB samples split into:
- **158 all-pass** (every model solves them) — zero discriminative signal
- **46 all-fail** (no model solves them) — zero discriminative signal
- **111 discriminative** (at least one model differs) — all the signal

A naïve 15% random sample would include ~23 discriminative samples. Our method delivers **47 samples with Kendall-τ = 1.0** — perfect rank preservation at 85% reduction.

---

## Pruning approach — Stratified Rank-Directed Discriminative Selection

**Step 1: Score matrix.**
Load pre-computed reviews for seed models. Build an N×M matrix (N samples, M models).

**Step 2: Per-sample statistics.**
- *Difficulty* = mean score across models (0 = always wrong, 1 = always right)
- *Raw discriminability* = variance across models
- *Rank-direction alignment* = for each model pair (i,j), does this sample order i above j consistently with the full-set ranking? Normalised to [−1, 1].
- *Combined score* = raw_discriminability × (1 + rank_alignment) / 2

This combined score is zero for samples that discriminate in the **wrong** direction (e.g., a sample where the weaker model happens to pass and the stronger model fails). This is the key insight that prior variance-only approaches miss.

**Step 3: Stratified selection.**
Samples are binned into 3 equal-frequency difficulty tiers (easy / medium / hard). Budget is allocated proportionally per tier, ensuring the pruned set preserves the difficulty distribution. Within each tier, samples are ranked by combined score.

**Why stratification matters:** selecting only high-discriminability samples without difficulty stratification collapses the pruned set onto the "medium difficulty" regime, making the benchmark less useful for engineering validation (where both edge cases and easy baselines matter).

---

## Quantitative reduction

| Benchmark | Full | Pruned | Reduction | Kendall-τ |
|---|---|---|---|---|
| LiveCodeBench v5 | 315 | 47 | 85.1% | **1.000** |
| AA-LCR | 100 | 30 | 70.0% | **1.000** |

Both pruned sets perfectly preserve the full-set model ranking.

---

## AA-LCR: accounting for LLM judge non-determinism

AA-LCR is graded by an LLM judge, which introduces noise. The noise has two effects:
1. **Score variance inflation** — some samples appear discriminative when the judge simply varies, not the model
2. **Ranking instability** — repeated evaluation of the same sample can flip its score

Mitigation: the pruner includes a `judge_noise_weight` parameter (default 0.2 for AA-LCR) that applies a small penalty to samples with near-0.5 difficulty, where judge noise is hardest to separate from genuine model variance. The `min_samples=10` floor ensures we don't over-prune when the effective discriminative set is small.

---

## Part B — MMMU image-encoder degradation probe

**Why random sampling fails here:** MMMU's 12K questions span 30 subjects. Many questions are answerable from the question text alone using domain knowledge, without reading the image. A model with a degraded encoder can score well on those questions purely by language reasoning. Random sampling will include ~70% such questions, masking the encoder failure.

**Our probe strategy:**
1. **Image Essentialness Score (IES)** — a heuristic (0–1) derived from question text patterns that identifies whether the correct answer requires reading a specific visual element (graph reading, OCR, object counting) vs. domain recall.
2. **4 visual complexity tiers** — spatial diagrams, dense text-in-image, abstract/symbolic, natural scenes — each stresses a different encoder failure mode.
3. **Probe set** (n=100): top-IES samples per tier — the model must look at the image to answer correctly.
4. **Control set** (n=50): low-IES samples — answerable from text alone.

**Detection signal:** compare `probe_acc` vs `control_acc`. A degraded encoder will show `probe_acc << control_acc`. A functioning model shows both scores similar. This two-set design distinguishes encoder failure from general capability gaps.

**Why these tiers stress encoders specifically:**
- *Spatial diagrams* require fine-grained spatial reasoning over circuit diagrams, engineering drawings — fails when the encoder loses positional detail
- *Dense text-in-image* requires OCR — fails first when encoder resolution degrades
- *Abstract/symbolic* (math figures, molecular structures) requires symbol recognition — fails when encoder confuses similar-looking symbols
- *Natural scenes* provides a baseline; encoder failure usually preserves scene-level understanding longest

---

## Assumptions

- Seed model reviews (3 models) are representative of model ranking on unseen models. This holds as long as the 4th model's capability falls within the range of the seed distribution — a reasonable assumption for frontier code/reasoning models.
- LCB v5 subset is treated as static. If questions are refreshed, the pruner should be re-fit.
- IES is a heuristic, not a ground-truth label. Adding a text-only baseline (run model without image, compare to vision-enabled) would improve IES calibration.

---

## What would change with more resources?

**(a) More data (more models):** More seed models → more robust discriminability estimates. With 10+ models, IRT (Item Response Theory) would be appropriate, fitting per-sample difficulty and discrimination parameters directly.

**(b) Live model endpoint:** Run each candidate on a small random subset first (~30 samples), estimate its position in the ranking, then use active learning to select the most informative next 20 samples. This reduces inference cost by ~50% vs. fixed pruned set.

**(c) More time:** Use bootstrap confidence intervals on Kendall-τ to provide uncertainty bounds on the ranking preservation claim. Replace the IES heuristic with a trained text-vs-vision classifier (fine-tune on MMMU samples with/without images ablated).
