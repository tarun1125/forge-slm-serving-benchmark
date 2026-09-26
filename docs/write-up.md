# FORGE: what it actually costs to serve a fine-tuned 1.5B model yourself

A LoRA fine-tune of Qwen2.5-Coder-1.5B-Instruct, trained to turn natural-language
questions into PyMongo queries, measured across three real serving stacks —
`mlx_lm`, Ollama, and vLLM on Metal — three quantization levels each, and a hosted
API, on an Apple M5 Pro — then the same Ollama stack on a rented Azure Cobalt 100
CPU, and every local arm again with the prompt cache defeated. 2,511 requests in
the original sweep, plus 135 cached and 135 cache-busted on Azure and 1,215
cache-busted locally. Zero invented numbers. This is what came out of it —
including the part where the cloud run showed some of the original latency
numbers were measuring the wrong thing.

## The headline finding: prefill and decode don't scale the same way

Every "latency" number in this project is reported as two separate numbers —
time-to-first-token (TTFT) and inter-token latency (ITL) — because collapsing them
into one hides the actual story. Averaged across all three arms and all three
quantization levels, at concurrency 1:

| Prompt bucket | avg prompt tokens | TTFT p50, as first published | TTFT p50, prompt cache defeated | ITL p50 (either) |
|---|---|---|---|---|
| short | 179 | 99 ms | 131 ms | 8.2 ms |
| medium | 1,424 | 135 ms | 322 ms | 8.6 ms |
| long | 4,647 | 455 ms | 931 ms | 9.2 ms |

With every request forced to prefill from scratch, TTFT rises **7.1x** from the
short bucket to the long bucket; ITL moves **12%** over the same range. The
first published version of this table said 4.6x — the right shape, understated,
because every serving stack here reuses the KV cache for a prompt prefix it has
already seen, and this sweep's prompts share long schema prefixes and repeat
across rounds. Some of those "prefills" were cache hits (Ollama q4's ~5,000-token
prompts at 0.1 s is not a prefill rate an M5 Pro can reach). The Azure run is what
exposed it: on a CPU a cache miss costs tens of seconds, so the two populations
could no longer hide inside one percentile. The cached numbers still describe a
real deployment — every request sharing one fixed system prompt, which is what
the demo does — but they aren't a measurement of prefill. Both are reported now,
labelled; see failure #5 in [docs/failure-gallery.md](failure-gallery.md). Prefill is compute-bound — more prompt tokens means more
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
arm at high concurrency does. With the prompt cache defeated the gap is smaller
but the ranking identical — at concurrency 8, medium bucket: `vllm_metal` 4-bit
184.9 tok/s, `mlx_lm` 4-bit 107.8, Ollama q4 76.2 (against 459.8 / 154.9 / 151.7
cached). Some of vLLM's apparent batching win was its prefix cache. (`mlx_lm`
also dropped 3 of those 24 requests at concurrency 8 — a connection-backlog
limit that the original sweep's automatic retries had been hiding; failure #5.) That's a more useful thing to know for capacity
planning than a crossover point would have been, and it's not the answer the plan
went in expecting.

## The accuracy cliff that has nothing to do with quantization

Every one of the 9 (arm × quantization) combinations scores **exactly 0%
execution accuracy on the short prompt bucket** — a reduced, single-collection
schema. Medium and long buckets score 19–50% on the same arms and variants.
Quantization doesn't move the number in any consistent direction within a bucket:
4-bit vs. full precision differs by at most 10 percentage points — one case, at
10 cases per cell — and 4-bit scored *higher* in 4 of the 6 (arm × bucket)
comparisons, equal in one, lower in one. That is sampling noise, not a
quantization effect. (These figures are re-scored: an earlier version averaged
over distinct generations rather than cases, double-counting any case whose
output varied with concurrency. Fifteen groups moved; failure #5.) Which bucket a question falls
into dominates completely. Root cause: the fine-tune was trained exclusively on
the full, real multi-database batch system prompt — a reduced schema is
*objectively simpler* but a shape the model never saw in training. Out of
distribution, not under-capable. This is why the deployed demo's schema is fixed,
not free-form editable — and why one of the demo's own example questions produces
a genuinely malformed (not just wrong) PyMongo query on a harder aggregation
shape (`$size` inside `$in`). Both are documented in
[docs/failure-gallery.md](failure-gallery.md) with the real, unedited output.

**Given that, was 4-bit quantization worth it?** Yes, clearly: `vllm_metal` at
4-bit gets a 3.1x throughput gain over bf16 (148.9 → 459.8 tok/s, cached) for an
accuracy difference within noise — and in the wrong direction for a cost (43.3%
vs. 35.6% on the medium bucket). There's no real accuracy cost being traded away
here.

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

**Renting instead of owning, measured.** The same Ollama version, the same
sha256-verified GGUF and the same daemon settings on an Azure `Standard_D4ps_v6`
(Cobalt 100, 4 Neoverse-N2 vCPUs, $0.0924/h, Central India):

| Ollama q4, medium bucket | M5 Pro | Azure 4× Neoverse-N2 |
|---|---|---|
| ITL p50 | 5.7 ms | 36.9 ms |
| TTFT p50, concurrency 1, cache defeated | 0.42 s | 13.1 s |
| Throughput, concurrency 8, cached / cache defeated | 151.7 / 76.2 tok/s | 18.2 / 3.6 tok/s |
| Execution accuracy | 50.0% | 50.0% |
| Cost per query at 10k/month (cached throughput) | ₹0.694 | **₹0.594** |

Decode is 6–8x slower on the VM and cold prefill (~70–120 tok/s) is the real
wall — a 5,000-token prompt waits most of a minute. Yet at 10k queries a month
the VM is *cheaper* per query than the Mac, because below ~300k queries/month
both are paying for idle hardware, and $67/month of always-on VM costs less than
a 3-year-amortized M5 Pro. The VM stops winning when throughput matters: at 1M
queries/month the cost model needs two VMs, and priced at the cache-defeated
throughput (3.6 tok/s) a single VM's capacity is ~5x lower still. Which of those
throughputs applies depends on whether production requests share a system
prompt — a deployment decision, not a hardware one. Same weights, same version,
temperature 0, and the long bucket still scored 22% on the VM against 30% on
the Mac: llama.cpp's CPU and Metal kernels accumulate differently, and greedy
decoding is deterministic per machine, not across machines.

Still missing from this comparison: a real vLLM-on-CUDA arm. The harness and a
full Azure runbook exist (`docs/cloud-arm.md`); the run is waiting on GPU quota.
Until then the cloud-GPU line in the cost model is a documented estimate — a
bandwidth-ratio scaling factor applied to a real measured `vllm_metal` number —
labeled as an estimate everywhere it appears.

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
- No vLLM-on-CUDA measurement yet, as above — an estimate everywhere it appears,
  pending GPU quota. (The estimate's bandwidth figure is the A100 SXM's 2,039 GB/s
  while the price it's paired with is for a PCIe A100, 1,935 GB/s — ~5% optimistic.)
- The cost table prices every arm at its *cached* concurrency-8 throughput. That is
  right for a single shared system prompt and optimistic for unique prompts; the
  cache-defeated numbers above are the other bound.
- One cache-defeated `vllm_metal/8bit` group had to be re-run: its first cells
  after startup took 17–58 s to first token. It did not reproduce; the likeliest
  cause is memory contention (Ollama's q8 model was still resident on the 24 GB
  machine from the previous group), but that is inference, not measurement.
- Accuracy numbers per cell come from small samples (n≈10–20) — real enough to
  show the short-bucket cliff cleanly (100% consistent across 9 combinations
  isn't sampling noise), but not enough to distinguish, say, a 3-percentage-point
  quantization difference from noise with confidence.
- **No thermal data.** The sweep ran with `--skip-thermal` (the sampler needs
  `sudo powermetrics`, impractical to keep authenticated across a long
  unattended run), so `thermal_pressure_level` and the power columns are null
  on every row. This project therefore cannot say whether the machine throttled
  during the sweep — only that no request failed. A logging bug made this
  easy to misread as "no throttling occurred"; both the bug and the correction
  are in `docs/failure-gallery.md`.

## The five questions this project should be able to answer cold

1. **TTFT at ~1.4k vs. ~180 prompt tokens, and why they differ:** 322 ms vs. 131 ms
   with the prompt cache defeated (medium vs. short, averaged across arms) — prefill
   is compute-bound, so more prompt tokens directly costs more matmul work before
   the first token. The first published answer (135 vs. 99 ms) was partly measuring
   KV-cache hits, which is its own lesson: check the prefill rate a TTFT implies
   before believing it.
2. **At what concurrency does vLLM overtake mlx_lm, and what causes it:** it
   doesn't need to — `vllm_metal` is faster at every concurrency level tested,
   including 1. Continuous batching widens that lead (~3.0x scaling 1→16) rather
   than creating a crossover; `mlx_lm` and Ollama scale far less (~1.4x, ~1.6x).
3. **What 4-bit quantization cost in accuracy, and was it worth it:** nothing
   measurable — within one case (≤10 points) at n=10, higher in 4 of 6
   comparisons — against a 3.1x throughput gain on `vllm_metal`. Worth it, clearly.
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
