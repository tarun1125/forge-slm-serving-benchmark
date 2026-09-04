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
buckets score 15–50% on the same arms and variants. Quantization barely
moves the number within a bucket; which bucket a question falls into
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

## What this gallery is not

It is not a claim that this project found every bug or that the ones above
are the only interesting ones — `git log` has the complete list, including
smaller process bugs (a missing DCO sign-off on the vllm-metal PR, a README
that claimed a "vLLM on CUDA" arm was measured when it had actually been
deferred). It's a record of what was verified broken, how it was diagnosed,
and what the fix actually was — the same standard the rest of this repo
holds every other number to.
