# Failure gallery

The project plan this repo started from expected a failure gallery built
from hardware failure modes: OOM at high concurrency, thermal throttling
mid-sweep, a model vllm-metal refuses to load. Across the full real sweep —
2,511 requests, 3 arms, 3 quantization levels each, concurrency 1–16, three
prompt-length buckets — there are **zero request failures**. No OOM, and
nothing vllm-metal refused to load.

**On thermal throttling, this project cannot make a claim either way, and an
earlier draft of this document wrongly implied it could.** `thermal_pressure_level`
is `null` on all 2,511 rows, and it would be easy to read that as "no
throttling occurred." It isn't — *no thermal telemetry was ever captured*.
`cpu_power_mw`, `gpu_power_mw` and `peak_memory_mb` are null on every row too,
which is the tell: that's missing instrumentation, not a quiet machine.

The evidence says the sweep ran with thermal monitoring disabled
(`--skip-thermal` exists precisely because the sampler shells out to `sudo
powermetrics`, impractical to keep authenticated across a long unattended run).
Two facts pin it down: had the monitor been running and reporting `Nominal`,
that string would have been stamped onto the rows instead of `null`; and had it
been running but returning nothing, `wait_for_cooldown()` would have timed out
and logged `cooled_down_before_run=False`, whereas MLflow records `True` for
all 150 cells. The only configuration consistent with both is a monitor that
was never started.

Either way the conclusion is the same: whether the M5 Pro throttled during this
sweep is unmeasured. See failure #3 below for the logging bug that made this
much harder to notice than it should have been.

What actually broke is more interesting anyway: a 100%-reproducible
accuracy cliff, a genuinely malformed model output caught in the deployed
demo, and a real bug at nearly every phase of building this. All of it is
below, unedited.

## 1. The short-prompt-bucket accuracy cliff (0% — every arm, every quant level)

From `results/accuracy_by_group.json`: every one of the 9 (arm ×
quantization) combinations scores **exactly 0.0 execution accuracy on the
short prompt bucket** — `mlx_lm`/bf16/8bit/4bit, `ollama`/f16/q8/q4,
`vllm_metal`/bf16/8bit/4bit, all nine, no exceptions. Medium and long
buckets score 19–50% on the same arms and variants. Quantization barely
moves the number within a bucket (at most one case in ten, in no consistent
direction); which bucket a question falls into
determines almost everything.

Root cause, not guessed: the fine-tune was trained exclusively on the full,
real multi-database batch system prompt (4–6k tokens, every collection in a
batch's schema spelled out — see `space/system_prompt.txt` for the actual
text). The "short" bucket uses a deliberately reduced, single-collection
schema. Objectively *simpler* for a human, but a shape the model never saw
in training — it's out of distribution, not under-capable. This is why
`space/app.py`'s demo has a fixed schema rather than a free-form editable
one: shipping a demo that lets a visitor accidentally trigger this failure
mode would look like a broken model, when it's actually a documented,
understood scope boundary.

## 2. A malformed, syntactically invalid generation (caught in the deployed demo)

Testing the Space demo's `generate_query()` end to end on "Find the ids of
the departments where any manager is managing 4 or more employees" produced:

```
db.employees.aggregate([{"$group": {"_id": "$MANAGER_ID"}}, {"$match": {"_id": {"$in": [{"$expr": {"$gte": [{"$size": "$employees"}, 4]}}, {"$expr": {"$gte": [{"$size": "$employees"}, 3]}}, {"$expr": {"$gte": [{"$size": "$employees"}, 2]}}, {"$expr": {"$gte": [{"$size": "$employees"}, 1]}}, {"$expr": {"$gte": [{"$size": "$employees"}, 0]}]}]}}, {"$project": {"_id": 0, "MANAGER_ID": "$_id"}}])
```

This isn't just semantically wrong — the bracket nesting near the end is
malformed and it isn't valid PyMongo. Nesting `$expr` dicts inside a `$in`
array's list is invalid syntax outright. Verified reproducible (5/5 runs,
deterministic once a separate KV-cache bug below was fixed) rather than a
one-off sampling fluke: this specific question, on this specific model,
reliably produces broken output. Consistent with the ~25–50% medium-bucket
accuracy already measured — an incorrect answer on a harder aggregation
question isn't surprising, but *malformed* (not just wrong) output is a
qualitatively different failure than the clean-but-incorrect answers Phase
2's accuracy scoring otherwise found. Swapped out of the demo's default
example questions (a first-time visitor's first impression shouldn't be a
syntax error) but kept here rather than quietly dropped.

## 3. Telemetry that reported a check it never ran

Found while reviewing this repo after the benchmark was "finished," and the
reason the thermal gap above went unnoticed for so long.

`sweep.py` initialised `cooled_down = True` before deciding whether thermal
monitoring was even enabled, and `mlflow_tracking.log_thermal_flag()` was
called unconditionally — outside the `if thermal_monitor is not None` guard
that everything else thermal-related sat behind. The result: a `--skip-thermal`
run logged `cooled_down_before_run=True` to MLflow for **all 150 cells**
(135 sweep cells plus the 15 Ollama long-bucket re-runs), which reads as
"a pre-run cooldown was verified" when in fact no cooldown check ever ran.

The failure mode is the dangerous kind: not a crash, not a wrong number, but
telemetry that makes an experiment look more controlled than it was, and that
positively asserts a safety property nobody measured. It also actively
disguised the missing thermal data — MLflow said the runs were cooled, so the
null `thermal_pressure_level` column looked like "nothing to report" rather
than "nothing was recorded."

Fixed by defaulting the flag to `None` and logging it as the literal
`"not_monitored"` when thermal monitoring is off, so "never checked" can no
longer be mistaken for "checked, and it was fine."

## 4. Real bugs found and fixed, by phase

Selected from the commit history — not the only bugs found (see `git log`
for the rest), but the ones with the clearest "here's what broke and why"
story:

**Ollama's default 4096-token context window silently rejected the entire
long-prompt bucket**
([b8264b9](https://github.com/tarun1125/forge-slm-serving-benchmark/commit/b8264b9)).
Found by pattern-matching real sweep failures: every one of 15 failing
cells was `ollama` + long bucket — 100% of failures, 0% elsewhere, across
every quantization level and concurrency setting tested. Ollama defaults
`num_ctx` to 4096 regardless of what the underlying model actually
supports (Qwen2.5-Coder's own `max_position_embeddings` is 32768) — the
server's own error message confirmed it outright: `request (4153 tokens)
exceeds the available context size (4096 tokens)`. `mlx_lm` and
`vllm_metal` never hit this because their context length is set explicitly
at launch; Ollama needed `PARAMETER num_ctx 8192` added to its Modelfiles
and all three tags re-registered.

**The hosted-API arm had three independent real bugs, not one**
([25a3484](https://github.com/tarun1125/forge-slm-serving-benchmark/commit/25a3484)).
(1) The model ID inherited from the project plan's own resume text,
`llama-3.3-70b-versatile`, was dead against the real Groq account — a
plain 404, confirmed via `client.models.list()` returning no Llama models
at all on this key. (2) The arm read `os.environ.get("GROQ_API_KEY")`
directly, which this project's own convention explicitly says not to do —
`pydantic-settings` loads `.env` into `Settings`' own fields, not into the
process environment, so the key was silently `None` at runtime the entire
time this went unnoticed. (3) The replacement model, `openai/gpt-oss-120b`,
is a reasoning model: at `max_tokens=300` (tuned for this project's own
small, non-reasoning fine-tune), 4 of 15 real requests silently burned the
whole token budget on invisible reasoning tokens and returned empty content
with no error at all.

**Thermal monitoring had three live bugs stacked on top of each other**
([c2d5638](https://github.com/tarun1125/forge-slm-serving-benchmark/commit/c2d5638)).
The `--samplers smc` powermetrics flag doesn't exist on this macOS
version. Fixing the sampler name alone still produced all-`None` readings,
because `powermetrics` fully buffers its own stdout when not attached to a
tty — nothing reached the pipe during a short test window — compounded by
a shutdown-ordering bug that would have discarded a final flushed burst
even if one had arrived. Fixed by abandoning continuous streaming for
one-shot `-n1` polls on a background timer, which flush unconditionally.

**A `run_id` collision crashed the very first real (non-smoke-test) sweep
launch**
([d6afea3](https://github.com/tarun1125/forge-slm-serving-benchmark/commit/d6afea3)).
`sweep.py` pre-generated its own `run_id` and also passed it into
`start_run()`'s kwargs — which mints and returns its own `run_id` — so the
two collided at the first `logger.info()` call: `TypeError: got multiple
values for keyword argument 'run_id'`.

**The deployed demo's model gave different answers to the same question
depending on what was asked right before it**
([c15ba95](https://github.com/tarun1125/forge-slm-serving-benchmark/commit/c15ba95)).
`llama_cpp.Llama` keeps KV-cache and token-history state across calls, and
the Space reuses one global model instance across every visitor's request.
Without an explicit `.reset()` before each generation, one question's
context could silently bleed into the next generation's output — at
`temperature=0`, which should be deterministic. Verified directly: the same
question, same settings, produced different output purely depending on
which other questions had been asked earlier in the same process. This is
also how failure #2 above was correctly isolated as the model's real,
order-independent output rather than an artifact of test ordering.

## 5. What the cloud run exposed about the local numbers

Running the same Ollama, the same GGUF (sha256-verified on the VM) and the same
daemon settings on a 4-vCPU Azure Cobalt 100 VM was meant to add one row to the
cost table. It mostly found problems in the rows that were already there.

**Time-to-first-token on long prompts was measuring the prompt cache, not
prefill.** Ollama (llama.cpp), vLLM and `mlx_lm.server` all reuse the KV cache
for the longest prefix a new prompt shares with an earlier one, and this
sweep's prompts share long schema prefixes and repeat across rounds. The
published local `ollama/q4` long-bucket TTFT at concurrency 1 was **0.09–0.10 s
for ~5,000-token prompts** — a rate no M5 Pro prefill reaches. On the Mac the
cache made this invisible. On the CPU VM a miss costs 37–160 s, so the same
effect split the long bucket into two populations (0.3–1 s hits, 37–160 s
misses) and produced a nonsense p50 of 77.7 s at concurrency 2 next to 1.05 s at
concurrency 1.

The fix is a `--bust-prompt-cache` sweep flag: a random per-request tag at the
*start* of the system prompt, so no two requests share a prefix (placed
anywhere later, the shared schema before it stays cacheable). Each row records
its nonce, and the sweep refuses to write a busted run into `results/sweep/`.
Re-measured cold, `ollama/q4` at concurrency 1:

| | Cached (published) | Cold, Mac | Cold, Azure 4× Neoverse-N2 |
|---|---|---|---|
| long TTFT p50 | 0.10 s | 1.29 s | 54.96 s |
| medium TTFT p50 | 0.12 s | 0.42 s | 13.11 s |
| ITL p50, long | 6.0 ms | 6.4 ms | 47.9 ms |

ITL does not move — decode never touches the cache question — which is the
cleanest confirmation that the cache, not noise, is what changed. The cached
numbers aren't wrong so much as mislabelled: they measure a deployment whose
requests share one fixed system prompt (the demo is exactly that), not the
cost of prefill. Both are now reported, labelled.

**`mlx_lm` at concurrency 8 was dropping connections, and retries hid it.**
The cold re-run used `--max-retries 0` (see below) and `mlx_lm` failed 23 of 216
measured requests — every one `Connection error.`, every one at concurrency 8,
all three variants, never at concurrency ≤ 4 and never on vLLM or Ollama. The
server's own log shows no error: the connections never reached it.
`mlx_lm.server` is built on the standard library's `ThreadingHTTPServer`,
whose listen backlog is `socketserver.TCPServer.request_queue_size = 5`;
eight simultaneous connects while the one Python thread is busy prefilling
overflow it, and macOS refuses the rest. The published sweep ran the same cells
with the OpenAI SDK's default 2 retries and reported **zero** failures — so
those cells most likely contain retried requests whose TTFT silently includes
the failed attempt and its backoff. Likely, not proven: retries were never
recorded, which is the other half of the bug.

**Retries were on by default, and they falsify latency.** `run_request` stamps
the start time before the SDK's first attempt, so a request that failed and
succeeded on retry reports the retry's delay as TTFT, and the retry re-sends load
to a server that was already saturated. Retries are now a sweep flag, and every
cloud and cold run uses `--max-retries 0`: a failed request is an honest
`n_failed`; a retried one is a wrong number.

**Accuracy averaged over generations, not cases.** `score_accuracy` deduplicated
`(case_id, generated_text)` pairs and averaged over them, so a case that
produced two different outputs across concurrency levels counted twice and a
stable case once — `mlx_lm/4bit/long` scored 13 generations drawn from 10
cases. Now each case counts once (generations weighted by how many requests
produced them). Fifteen local groups moved, some a lot: `vllm_metal/4bit/long`
went from 15.8% to 27.9%. A second bug in the same file: it overwrote its output
wholesale, so scoring a cloud-only directory would have deleted every local
group. It now merges.

**The same GGUF gives different answers on different hardware.** Byte-identical
weights (hash-checked), same Ollama version, temperature 0: medium-bucket
accuracy matched exactly (50.0% both), but the long bucket scored 22.0% on the
Azure CPU against 30.0% on the Mac. llama.cpp's CPU and Metal kernels accumulate
in different orders, and at 4–6k-token prompts the drift is enough to flip some
greedy decodes. "Deterministic at temperature 0" holds per machine, not across
machines.

**A dropped SSH tunnel used to look like a finished run.** Remote arms launch
nothing, so nothing checked the endpoint: a dead tunnel produced a sweep that
completed, writing rows whose `error` read `Connection error.`. It happened for
real between the two Azure runs (the idle tunnel died). A preflight now requires
`/models` to answer and list the model, or the sweep refuses to start.

**On a fast remote server, inter-token latency measures the network, not the
model.** The T4 run reported an ITL p50 of 0.43 ms on the medium bucket — 2,300
tok/s, several times what a T4 can decode a 1.5B model at. The server wasn't
the problem: it sent ~1 chunk per token (0.95 chunks/token). The client timed
each chunk's *arrival* after an SSH tunnel, and at ~14 ms per token the tunnel
delivered them in bursts — 42–50% of gaps under 1 ms. `client.py`'s own
docstring had predicted exactly this ("if a specific arm's ITL numbers look
implausibly smooth… check this assumption first"). Per-request decode rate —
tokens after the first divided by first-to-last-token time — is immune, and
gives 73 tok/s. The CPU VM escaped it because at 30+ ms per token each chunk
arrives on its own; the local arms never cross a network. Lesson: the ITL
percentile is only valid when the per-token time comfortably exceeds the
transport's batching window, so a remote row's ITL has to be checked against
its per-request rate before it is believed.

**vLLM wouldn't start: no C compiler.** The first `vllm serve` on the T4 died
in its memory-profiling step with `InductorError: Failed to find C compiler` —
vLLM compiles kernels at startup (Triton, `torch.compile`), and Azure's Ubuntu
image ships without `gcc`. The log line directly above it, `FA2 is only supported
on devices with compute capability >= 8`, looks like the cause and isn't: vLLM
falls back to Triton attention on a T4 by itself. `build-essential` is now in
the runbook's driver step.

**Requests lost in transit at high concurrency.** Both remote GPU sweeps lost a
handful of requests at concurrency 32–64 only (6 cached, 3 cache-busted, of
~2,300), all `Connection error.`. vLLM's access log shows every request it
received returned HTTP 200 and the count matches sent-minus-lost exactly — so
they died between the Mac and the server, most likely in the SSH tunnel under
that many simultaneous connections. Four isolated 64-request bursts through the
same tunnel all succeeded, so the mechanism is unproven. They stay in the data
as `n_failed`; retrying them would have hidden exactly the kind of thing this
section is about.

**A mid-sweep upload contaminated two cells.** While the first GPU sweep was
running, a 3 GB `scp` to the same VM saturated the Mac's uplink (11.8 MB/s)
for four minutes. The Mac is the benchmark client, so its network link is part
of the measurement. File timestamps on the VM placed the upload at
14:07:31–14:11:40 UTC; cell-finish timestamps showed exactly two cells,
`c32 long` and `c64 long`, overlapping it. Both were re-run clean and
overwritten. (It also cleared the upload of blame for the lost requests above:
11 of the first 19 happened before it started, and the clean re-run lost 6 more.)
A benchmark client has to be treated like a lab bench — nothing else runs on it
during a sweep.

## What this gallery is not

It is not a claim that this project found every bug or that the ones above
are the only interesting ones — `git log` has the complete list, including
smaller process bugs (a missing DCO sign-off on the vllm-metal PR, a README
that claimed a "vLLM on CUDA" arm was measured when it had actually been
deferred). It's a record of what was verified broken, how it was diagnosed,
and what the fix actually was — the same standard the rest of this repo
holds every other number to.
