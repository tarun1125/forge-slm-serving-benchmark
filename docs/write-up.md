# FORGE: what it actually costs to serve a fine-tuned 1.5B model yourself

A LoRA fine-tune of Qwen2.5-Coder-1.5B-Instruct, trained to turn natural-language
questions into PyMongo queries, measured across three real serving stacks —
`mlx_lm`, Ollama, and vLLM on Metal — three quantization levels each, and a hosted
API, on an Apple M5 Pro. 2,511 real requests. Zero invented numbers. This is what
came out of it.

## The headline finding: prefill and decode don't scale the same way

Every "latency" number in this project is reported as two separate numbers —
time-to-first-token (TTFT) and inter-token latency (ITL) — because collapsing them
into one hides the actual story. Averaged across all three arms and all three
quantization levels, at concurrency 1:

| Prompt bucket | avg prompt tokens | TTFT p50 | ITL p50 |
|---|---|---|---|
| short | 128 | 99 ms | 8.23 ms |
| medium | 1,411 | 135 ms | 8.55 ms |
| long | 4,773 | 455 ms | 9.15 ms |

TTFT rises **4.6x** from the short bucket to the long bucket. ITL moves **11%**
over the same range. Prefill is compute-bound — more prompt tokens means more
matrix-multiply work before the first token can appear, and it shows immediately.
Decode is memory-bandwidth-bound — generating the *next* token costs roughly the
same regardless of how long the prompt was, because the bottleneck is repeatedly
reading the model's weights from memory, not the prompt. That's why quantization
(a smaller model resident in memory) and concurrency (batching those memory reads
across requests) are the two levers that actually move decode throughput — prompt
length isn't one of them. See `notebooks/forge_benchmark_analysis.ipynb` for the
full reproducible chart.

## Where batching wins — and where it doesn't look how you'd expect

The plan behind this project expected to find a concurrency threshold where vLLM
"overtakes" `mlx_lm`. The real data doesn't have a crossover: at the medium
bucket, each arm's fastest quantization variant, `vllm_metal` is faster than
`mlx_lm` and Ollama **at every concurrency level tested, including concurrency 1**
(191.6 tok/s vs. 116.2 and 103.4). What batching buys instead is a widening gap:
from concurrency 1 to 16, `vllm_metal`'s continuous batching scales throughput
**~3.0x** (191.6 → 581.0 tok/s), while `mlx_lm` scales **~1.4x** and Ollama **~1.6x**
over the same range. The ranking never changes; the cost of picking the slower
arm at high concurrency does. That's a more useful thing to know for capacity
planning than a crossover point would have been, and it's not the answer the plan
went in expecting.

## The accuracy cliff that has nothing to do with quantization

Every one of the 9 (arm × quantization) combinations scores **exactly 0%
execution accuracy on the short prompt bucket** — a reduced, single-collection
schema. Medium and long buckets score 15–50% on the same arms and variants.
Quantization barely moves the number within a bucket (4-bit vs. bf16 accuracy
differs by at most ~3 percentage points, and not always in the direction you'd
guess — Ollama's q4 actually scored *higher* than its f16 baseline on this sample,
most plausibly sampling noise at n≈10 per cell). Which bucket a question falls
into dominates completely. Root cause: the fine-tune was trained exclusively on
the full, real multi-database batch system prompt — a reduced schema is
*objectively simpler* but a shape the model never saw in training. Out of
distribution, not under-capable. This is why the deployed demo's schema is fixed,
not free-form editable — and why one of the demo's own example questions produces
a genuinely malformed (not just wrong) PyMongo query on a harder aggregation
shape (`$size` inside `$in`). Both are documented in
[docs/failure-gallery.md](failure-gallery.md) with the real, unedited output.

**Given that, was 4-bit quantization worth it?** Yes, clearly: `vllm_metal` at
4-bit gets a 3.1x throughput gain over bf16 (148.9 → 459.8 tok/s) for an accuracy
difference within noise. There's no real accuracy cost being traded away here.

## What self-hosting actually costs, and where it stops making sense

At the representative volume this project settled on (10,000 queries/month,
medium bucket), **cost per query is within about 9 paise of itself across all
nine local (arm × quantization) configurations** — ₹0.6942–0.6943. That's not a
rounding artifact: at this volume, every configuration is running at well under
10% of the machine's lifetime throughput capacity, so cost is dominated entirely
by fixed hardware amortization (a 3-year-amortized M5 Pro), not by how fast the
model actually generates tokens. Quantization's real payoff in this cost model
isn't a lower cost-per-query at typical volumes — it's a higher **sustainable**
volume before another machine is needed.

Self-hosting only beats Groq's hosted API (`openai/gpt-oss-120b`, ₹0.0382/query
flat, no utilization dependence) at **300,000 queries/month** — for every local
variant, since their near-identical per-query cost means they all cross the same
flat hosted line at the same point. That number is most sensitive to the
amortization-window assumption: a 5-year window instead of 3 would roughly halve
every local cost figure and pull the break-even point down substantially. It's a
stated assumption, not a measurement — see `docs/cost-model.md` for the full
citation.

Worth being honest about what didn't make it into this comparison: a real
vLLM-on-CUDA arm was planned and never measured (no CUDA hardware available). The
cloud-GPU line in the cost model is a documented estimate — a bandwidth-ratio
scaling factor (A100 HBM2e vs. M5 Pro unified memory) applied to a real measured
`vllm_metal` number — labeled as an estimate everywhere it appears, not presented
as a fifth measured arm.

And on raw correctness alone: Groq's much larger, untuned `gpt-oss-120b` scored
**66.7%** accuracy on a real 15-case sample — meaningfully higher than any local
variant's medium-bucket accuracy — at a real cost and latency penalty, and with
far less predictable per-request latency (TTFT on the same workload ranged from
under a second to 15 seconds, driven by the reasoning model's hidden "thinking"
tokens before anything visible appears). "Bigger and hosted" beats "small and
fine-tuned" on correctness here. The fine-tune's case is throughput, cost at
volume, and running entirely on hardware you already own — not raw accuracy.

## Contributing back

Verifying `mlx-community/Qwen2.5-Coder-1.5B-Instruct-bf16` (plus 8-bit and 4-bit
MLX conversions) on `vllm-metal` — 135 real requests across concurrency 1–16 and
three prompt lengths, zero failures, real throughput numbers — turned out to be
exactly the kind of report that project's own tracking issue asks the community
for. Reported it as
[vllm-project/vllm-metal#689](https://github.com/vllm-project/vllm-metal/pull/689),
following that repo's own documented convention (a prose note citing a PR,
matching their existing AWQ/GGUF sections) rather than adding a redundant table
row for an already-✅-listed model family. See
[docs/vllm-metal-contribution.md](vllm-metal-contribution.md) for the full
reasoning and what was and wasn't tested.

## What's honestly missing

- No live hosted demo — Hugging Face now requires a PRO subscription to host a
  Gradio Space on free CPU, and this project isn't paying for one. The model is
  public on HF Hub and the demo (`space/app.py`) runs locally with one `pip
  install` — see `space/README.md`.
- No vLLM-on-CUDA measurement, as above — an estimate everywhere it appears, not
  a sixth measured arm.
- Accuracy numbers per cell come from small samples (n≈10–20) — real enough to
  show the short-bucket cliff cleanly (100% consistent across 9 combinations
  isn't sampling noise), but not enough to distinguish, say, a 3-percentage-point
  quantization difference from noise with confidence.

## The five questions this project should be able to answer cold

1. **TTFT at ~1k vs. ~128 prompt tokens, and why they differ:** 135 ms vs. 99 ms
   (medium vs. short bucket, averaged across arms) — prefill is compute-bound, so
   more prompt tokens directly costs more matmul work before the first token.
2. **At what concurrency does vLLM overtake mlx_lm, and what causes it:** it
   doesn't need to — `vllm_metal` is faster at every concurrency level tested,
   including 1. Continuous batching widens that lead (~3.0x scaling 1→16) rather
   than creating a crossover; `mlx_lm` and Ollama scale far less (~1.4x, ~1.6x).
3. **What 4-bit quantization cost in accuracy, and was it worth it:** at most
   ~3 percentage points, not consistently in the "worse" direction, against a
   3.1x throughput gain on `vllm_metal`. Worth it, clearly.
4. **Break-even volume for self-hosting, and its most sensitive assumption:**
   300,000 queries/month against Groq's hosted price, dominated by (and most
   sensitive to) the 3-year hardware amortization window.
5. **What broke, and what was learned:** a 100%-reproducible accuracy cliff
   driven by prompt shape rather than quantization, a genuinely malformed
   generation caught in the deployed demo, and real bugs at nearly every phase —
   Ollama's silent 4096-token context truncation, a dead hosted-API model ID, a
   config-loading bug that left an API key silently `None`, three stacked thermal-
   monitoring bugs, and a KV-cache reuse bug that made the demo's output
   order-dependent. Full detail in `docs/failure-gallery.md`.
