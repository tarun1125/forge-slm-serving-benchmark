# Cost model

Every number below either traces to a real measurement from Phase 2's sweep
(`results/sweep/`, `results/accuracy_by_group.json`, `results/groq_sample.jsonl`)
or is cited from a real, dated external source in the assumptions table. Nothing
here is invented — see `src/forge/phase3/cost_model.py` for the arithmetic and
`src/forge/phase3/build_report.py` for exactly where each input comes from.

## Assumptions

| Assumption | Value | Source |
|---|---|---|
| Mac hardware cost | ₹249,900 | Apple India official price, 14" MacBook Pro M5 Pro, 24GB/1TB (Sept 2026) |
| Amortization window | 3 years | Stated assumption, not measured — dominates the result, see note below |
| Mac power draw (active inference) | 75 W | Estimated: 62W CPU sustained load (Notebookcheck, 16" M5 Pro) + system/display overhead |
| Indian electricity tariff | ₹5.50/kWh | India residential average, FY 2025-26 |
| Cloud GPU hourly rate | $1.39/hr | RunPod, A100 80GB PCIe, on-demand (Sept 2026) |
| Cloud GPU throughput scaling | 6.64x | A100 HBM2e (2039 GB/s) / M5 Pro unified memory (307 GB/s) — decode is memory-bandwidth-bound (Phase 2's own headline finding), applied to a REAL measured vllm_metal number. **Estimated, not measured** — the vLLM-on-CUDA arm was deferred. The least certain number in this model. |
| Groq input pricing | $0.15/1M tokens | Groq, `openai/gpt-oss-120b`, confirmed live against the real account (Sept 2026) |
| Groq output pricing | $0.60/1M tokens | Same source |
| USD→INR | ₹88 | Approximate, Sept 2026 — a daily-fluctuating assumption, flagged as such |

**The amortization window and utilization assumption dominate the result, exactly as
the plan warns.** A 3-year window was chosen as a conservative, realistic laptop
replacement cycle — a 5-year window would roughly halve every local cost-per-query
figure below without changing anything about measured throughput.

**A deliberate simplification:** electricity cost is computed only for active
inference time, not idle baseline power. This slightly understates local cost at
low utilization (an idle laptop still draws some power) but doesn't change the
qualitative story, since idle-power draw is a small fraction of active-inference draw.

## Real measured results (representative point: concurrency=8, medium prompt bucket)

| Arm | Variant | Throughput (tok/s) | Accuracy | Cost/query @ 10k/mo | Break-even (queries/mo) | Cost per accuracy point |
|---|---|---|---|---|---|---|
| mlx_lm | bf16 | 69.8 | 36.4% | ₹0.6943 | 300,000 | ₹1.9093 |
| mlx_lm | 8bit | 110.6 | 25.0% | ₹0.6942 | 300,000 | ₹2.7770 |
| mlx_lm | 4bit | 154.9 | 33.3% | ₹0.6942 | 300,000 | ₹2.0827 |
| ollama | f16 | 76.8 | 40.0% | ₹0.6943 | 300,000 | ₹1.7357 |
| ollama | q8 | 113.4 | 40.0% | ₹0.6942 | 300,000 | ₹1.7356 |
| ollama | q4 | 151.7 | 50.0% | ₹0.6942 | 300,000 | ₹1.3884 |
| vllm_metal | bf16 | 148.9 | 36.4% | ₹0.6942 | 300,000 | ₹1.9091 |
| vllm_metal | 8bit | 330.6 | 30.8% | ₹0.6942 | 300,000 | ₹2.2561 |
| vllm_metal | 4bit | 459.8 | 35.7% | ₹0.6942 | 300,000 | ₹1.9437 |

**Hosted API (Groq, gpt-oss-120b):** ₹0.0382/query flat (no utilization dependence), 66.7% accuracy on a real 15-case sample (avg 1469 prompt + 356 completion tokens — completion includes hidden reasoning-model tokens, confirmed live: at max_tokens=300 the model silently burned its entire budget on invisible reasoning on 4 of 15 real requests and returned no visible output at all).

## Break-even

See `break-even-curve.png` for the full curve. The break-even volume in the table above is per-variant, not a single portfolio number — exactly the plan's own point: the crossover moves with which arm/quantization you pick, and with utilization.

## What surprised the numbers

- **The short prompt bucket has 0% execution accuracy across every single local arm and quantization level** — not a bug, a real out-of-distribution effect (the model was fine-tuned exclusively on full multi-database batch prompts; a reduced single-collection schema, despite being objectively simpler, is a shape it never saw in training). Cost per accuracy point is infinite for every short-bucket cell.
- **Groq's untuned, much larger, much more expensive generalist model (gpt-oss-120b) scored 66.7% accuracy — meaningfully higher than any local variant's medium-bucket accuracy.** "Bigger and hosted" beat "small and fine-tuned" on raw correctness here, at a real cost and latency penalty, and with far less predictable per-request latency (real TTFT on the same workload ranged from under a second to 15 seconds — the reasoning model's hidden "thinking" time before the first visible token, not network latency).
- `vllm_metal` at 4-bit was the fastest local configuration at the representative concurrency level, by a wide margin over both `mlx_lm` and `ollama` at their own 4-bit settings — continuous batching doing exactly what it's supposed to.
- **At 10k queries/month, cost per query is nearly identical across every local variant regardless of throughput** (₹0.6942-0.6943 across all nine rows above) — counter-intuitive if you'd expect the faster quantization levels to show up as cheaper immediately. They don't, because at this volume every variant is running at well under 10% of the machine's actual lifetime throughput capacity (see the utilization column `local_cost_per_query` computes), so cost is dominated entirely by fixed hardware amortization — throughput only starts moving the cost needle once volume approaches the machine's real capacity ceiling. Quantization's real payoff in this cost model is enabling higher SUSTAINABLE volume before hitting that ceiling, not a lower per-query cost at typical modest volumes.
