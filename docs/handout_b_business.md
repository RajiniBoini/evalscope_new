# Handout B — Why This Matters and How to Use It
**Cerebras Benchmark Pruning Extension · Developers, Test Engineers, Product, Customer Team**

---

## What changes for the customer conversation?

Today, answering "is this model good enough for our workload?" requires running hundreds of benchmark questions — a process that takes hours and costs real inference budget. By the time you have an answer, the customer conversation has moved on.

With this pruner, you get the **same ranking answer in 15% of the time and cost.**

- LiveCodeBench: 315 questions → **47 questions**, same model ranking
- AA-LCR: 100 documents → **30 documents**, same model ranking

That means a same-day answer instead of overnight, and a confident "yes/no" in a live customer meeting instead of "we'll follow up."

---

## How to actually run this tomorrow

You need three things: the evalscope fork, a model endpoint, and the benchmark data.

**Step 1 — Set up**
```bash
git clone https://github.com/RajiniBoini/evalscope-fork
cd evalscope-fork
pip install -e .
```

**Step 2 — Run full benchmark (once, for any new benchmark)**
```bash
evalscope eval --model <model-endpoint> \
    --datasets live_code_bench \
    --output ./results_full/
```

**Step 3 — Run pruned benchmark (every new model from now on)**
```bash
evalscope eval --model <model-endpoint> \
    --datasets live_code_bench_pruned \
    --dataset-args '{"prune_ratio": 0.15, "reviews_dir": "./Evals/Part 1/reviews"}' \
    --output ./results_pruned/
```

**Step 4 — Compare and verify**
```bash
python -m evalscope_ext.tools.compare_runs \
    --full ./results_full/ \
    --pruned ./results_pruned/
```

This prints a table showing full vs pruned accuracy per model, the Kendall-τ rank correlation, and a PASS/FAIL verdict. If it says PASS, you can trust the pruned ranking for your customer conversation.

---

## What does the multimodal probe tell you that random sampling can't?

Most MMMU questions can be answered from the question text alone — without actually looking at the image. A model with a broken visual system can score 60%+ on random MMMU samples just by using language reasoning.

The probe specifically selects questions where the only path to the right answer is **reading a specific thing in the image** — a chart value, a diagram label, a table entry. If the model's image encoder is degraded, it will fail these questions even if its language reasoning is intact.

**The practical impact:** When a customer asks "will this model work for our document Q&A pipeline?" (which involves reading images, tables, and figures), a probe score gap is a concrete early warning before they discover the issue in production.

---

## Why should a customer-facing PM care about any of this?

**Shorter sales cycle.** You can answer capability questions in the same meeting instead of scheduling a follow-up. "Yes, this model passes our coding benchmark — here are the 47 questions we ran and the ranking" is more credible than "we'll run the full suite and get back to you."

**Credible differentiation.** The pruned benchmark is reproducible and explainable. You can tell a customer exactly which question types were tested, why they were chosen, and what the result means — rather than citing an opaque aggregate score.

**Risk reduction before commitment.** The multimodal probe catches image-encoder issues before a customer builds a pipeline on a model that looks fine on text but fails on their PDFs and charts. That's a saved escalation, not just a benchmark score.
