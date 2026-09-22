# FORGE — interview study guide

**Current tag: 🔴 NOT-DEFENSIBLE (unstudied). Target: `D1`, then `D2`.**
FORGE is off every resume until this is done. It unblocks role family **F6 — Inference,
Serving & Performance**, and it partly unblocks the serving half of F2/F3 JDs (HDFC Life
names vLLM explicitly).

**`docs/write-up.md` is the study text.** It already contains the honest version of every
claim including what was never measured. Reading it properly is most of the way to `D1`.

---

## 0. Start with the cost model, not the benchmarks

Deliberate ordering. The cost model is **an argument** — you can reason through it from first
principles in an afternoon. The serving benchmarks are **measurements** that require you to
understand prefill vs decode and memory bandwidth, which is a different order of work.

So: §1 first, then §2, then §3. Don't start where the impressive numbers are.

---

## 1. The cost model — `D1`, one afternoon

**The numbers:**
- At 10,000 queries/month, **all nine local (arm × quantization) configurations cost
  ₹0.6942–0.6943 per query.** That's ₹694/1k, and they're within ₹0.0001 of each other.
- Hosted (Groq `openai/gpt-oss-120b`) is **₹0.0382/query flat**, no utilisation dependence.
- **Self-hosting only breaks even at 300,000 queries/month.**

**The insight, which is the actual content:** why do nine configurations with very different
throughput cost the *same*? Because at 10k/month every one runs at well under 10% of the
machine's lifetime throughput capacity, so **cost is dominated entirely by fixed hardware
amortisation, not by how fast the model generates tokens.**

Which gives you the non-obvious conclusion: **quantization's payoff in this cost model is not
a lower cost per query. It's a higher *sustainable* volume before you need another machine.**

**The assumption that carries it:** a 3-year amortisation window on an M5 Pro. *"A 5-year
window instead of 3 would roughly halve every local cost figure and pull the break-even point
down substantially."* Say this before you're asked — a cost model whose author can't name its
most sensitive assumption isn't a cost model.

**And the conclusion argues against the project's own premise.** The project set out to
benchmark self-hosting; the cost model says the hosted API is the rational default below 300k
queries/month. You published that. It's one of the strongest honesty signals in the portfolio.

### Defend this
- *"Why 3 years?"* — stated assumption, cited in `docs/cost-model.md`, not a measurement.
- *"What would change the answer?"* — amortisation window; hosted pricing; sustained volume
  above 300k; needing data to stay on-prem, which is a constraint the cost model doesn't price.
- *"Isn't the M5 Pro doing other work?"* — yes, and full attribution to this workload is
  conservative in the wrong direction. Know that you know it.

---

## 2. The accuracy cliff — `D1`, and it's the best finding here

**This was missing from the resume entirely and it's better than what was on it.**

> All **9 (arm × quantization) combinations score exactly 0% execution accuracy on the short
> prompt bucket** — a reduced, single-collection schema. Medium and long buckets score 15–50%
> on the same arms and variants.

**Root cause:** the fine-tune was trained exclusively on the full, real multi-database batch
system prompt. A reduced schema is *objectively simpler* but a shape the model never saw.
**Out of distribution, not under-capable.**

Two consequences you shipped because of it: the deployed demo's schema is fixed rather than
free-form editable, and `docs/failure-gallery.md` documents a genuinely malformed PyMongo
query (`$size` inside `$in`) from one of the demo's own example questions, unedited.

**Why this is a strong interview answer:** it's specific, surprising, diagnosed to a cause,
and it changed a product decision. "Which bucket a question falls into dominates completely"
is a sentence about your system that most people can't produce about theirs.

---

## 3. Quantization — `D1` claim, `D2` mechanism

🔴 **The retired claim: "4-bit quantization produced no statistically measurable accuracy cost
on the 1,517-case held-out set."** Three faults — there is no such run; accuracy is scored per
cell at **n ≈ 10**; and "no statistically measurable cost" implies a powered test that failed
to reject, when at n≈10 nothing short of a huge effect is detectable.

**The correct claim:** *4-bit quantization bought vllm-metal a **3.1× throughput gain**
(148.9 → 459.8 tok/s) for an accuracy difference within noise.* Quantization moves accuracy by
at most ~3pp within a bucket and **not always in the expected direction** — Ollama's q4 scored
*higher* than its f16 baseline, which at n≈10 per cell is sampling noise, and you called it
that rather than claiming a win.

**"Within noise" ≠ "no difference."** If asked what effect you could have detected: you
couldn't detect much, and the honest framing is that the throughput gain was large and
obvious while any accuracy cost was below what this sample could see.

`D2` — the mechanism: 4-bit stores weights at reduced precision with per-group scales. You
lose precision in the *weights*, not the activations. Why it barely hurts here is worth
understanding rather than asserting.

---

## 4. Serving internals — `D2`, the hardest part

This is the block that takes real time. Don't claim any of it until it's done.

### Prefill vs decode
| Phase | Metric | Bound by | Measured |
|---|---|---|---|
| **Prefill** — whole prompt in parallel | TTFT | **Compute** | 99 → 455 ms (**4.6×**) |
| **Decode** — one token at a time | inter-token latency | **Memory bandwidth** | 8.2 → 9.2 ms (**11%**) |

Prefill scales with prompt length because it's a large matrix multiply over all input tokens.
Decode barely moves because each step re-reads the model weights to emit one token — bandwidth
bound, not compute bound. **That asymmetry is why the two diverged by a factor of ~40**, and
it's why continuous batching helps decode so much: one weight read amortised across many
sequences.

### Continuous batching
Static batching waits for the slowest sequence in the batch. Continuous batching lets a
finished sequence leave and a queued one take its slot mid-flight, so the accelerator stops
idling on stragglers.

**The number to lead with is the comparison, not the raw figure.** Concurrency 1 → 16:

| Arm | Scaling | Throughput |
|---|---|---|
| **vllm-metal** | **~3.0×** | 191.6 → **581.0 tok/s** |
| mlx_lm | ~1.4× | |
| Ollama | ~1.6× | |

*"The ranking never changes; the cost of picking the slower arm at high concurrency does.
That's more useful for capacity planning than a crossover point would have been — and it
isn't the answer the plan went in expecting."*

---

## 5. What was never measured — say it before you're asked

- **No vLLM-on-CUDA arm.** Planned, never run, no CUDA hardware available.
- **The cloud-GPU line in the cost model is an estimate** — a bandwidth-ratio scaling factor
  (A100 HBM2e vs M5 Pro unified memory) applied to a real measured vllm-metal number. Labelled
  as an estimate everywhere it appears, **not presented as a fifth measured arm.**

Volunteering this is worth more than any measurement in the project.

---

## 6. Break it on purpose

| Break | Expect | Teaches |
|---|---|---|
| Run the short-prompt bucket against the base (non-fine-tuned) model | Accuracy should *not* be 0% | Confirms the cliff is the fine-tune's distribution, not the task |
| Set the amortisation window to 5 years | Local cost roughly halves, break-even drops a lot | Which assumption the conclusion rests on |
| Benchmark at concurrency 1 only | The arms look nearly equivalent | Why the sweep exists at all |
| Compare TTFT at a fixed prompt length across arms | Little movement | That TTFT scales with *prompt*, not with the serving stack |

---

## 7. Two-week plan

| Days | Focus | Done when |
|---|---|---|
| 1–2 | **Cost model.** Read `docs/cost-model.md`, re-derive the break-even by hand, redo it at 5 years. | You explain why nine configs cost the same, and name the load-bearing assumption unprompted. |
| 3–4 | **The accuracy cliff.** Read the write-up and `docs/failure-gallery.md`. | You explain out-of-distribution vs under-capable, and why the demo schema is fixed. |
| 5–6 | **Quantization.** The corrected claim, per-group scales, why n≈10 limits what you can say. | You never say "no measurable cost" again. |
| 7–9 | **Prefill vs decode.** Compute-bound vs bandwidth-bound from first principles — arithmetic intensity, why decode re-reads weights. | You derive why TTFT scales and ITL doesn't, on paper. |
| 10–11 | **Continuous batching.** Static vs continuous, why the scaling differs by arm. | You explain the 3.0× / 1.4× / 1.6× gap mechanically. |
| 12 | **Break it** (§6). | |
| 13–14 | **Rehearse**, and re-read `docs/write-up.md` end to end. | No notes. |

---

## 8. Do not say, until studied

- ❌ **"No statistically measurable accuracy cost."** Retired. Use "within noise at n≈10".
- ❌ Any TTFT / inter-token-latency figure — they require §4 to defend.
- ❌ "581 tok/s" bare. Use the comparison: ~3.0× vs 1.4× and 1.6×.
- ❌ vLLM as a skill on a resume. It's in this project and this project is unstudied.
- ❌ Any claim about CUDA or cloud GPU performance. Never measured.

**And if the Azure deployment lands:** *"I deployed the fine-tuned model as an Azure inference
endpoint"* is defensible the moment you've done it — it's the **capstone's** model (the LoRA
work is capstone, `D3`), so frame it that way. It does **not** make anything in §3 or §4
defensible. Deployment and serving-performance analysis are different claims.
