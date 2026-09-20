# The cloud arm (`vllm_cuda`) — design notes and dry-run procedure

The plan's fifth arm — vLLM on a CUDA GPU — was deferred for the obvious
reason: there is no CUDA hardware here. Phase 3 filled the hole with an
estimate (`cloud_gpu_throughput_scaling_factor = 2039/307`, the A100/M5 Pro
memory-bandwidth ratio) and flagged it, everywhere it appears, as the least
certain number in the cost model.

This document covers the harness work that makes measuring it possible, the
decisions taken along the way, and how the whole thing was verified without
renting anything. **No cloud numbers exist yet.** Nothing in `results/` or
`docs/cost-model.md` has changed; the arm is wired and tested, not run.

## What it costs to find out

The full sweep for one variant is 1,524 requests (7 concurrency levels × 3
buckets × 4 rounds, summed over concurrency), of which 1,143 are measured.
That is under ten minutes of GPU time. At September 2026 list prices a
two-hour booking — enough for the driver check, the model download, vLLM's
first-start compile and one thing going wrong — is roughly $1.16 on an Azure
`Standard_NC4as_T4_v3` or $7.35 on an `NC24ads_A100_v4`. The cost of the
fifth arm is not the constraint; GPU quota approval is.

## Decisions

### Configuration comes from `Settings`, not `os.environ`

`arms.py`'s own docstring already records why: an earlier version of
`hosted_api_arm()` read `os.environ.get("GROQ_API_KEY")` directly and
silently never saw the key, because pydantic-settings loads `.env` into
`Settings` fields rather than into the process environment. `vllm_cuda_arm()`
takes the same path through `Settings` as a result. Ten new fields is more
`.env` surface than any other arm needs, and that is the point — see
provenance below.

### `launch_command=None`, and what that costs

The server runs on a rented VM; this process only holds a URL. `ManagedServer`
therefore treats it exactly as it treats Ollama and the hosted API: a no-op
start and stop.

The consequence is easy to miss and was confirmed deliberately during the dry
run (layer 3 below): **nothing health-checks the endpoint before the sweep
starts.** Point the arm at a dead port and the sweep does not error — it runs
to completion, writing rows whose `error` field reads `Connection error.` and
whose cells report `n_failed` instead of `n_succeeded`. That is correct
behaviour for a harness that must not abort a two-hour sweep over one dropped
request, but it means a dead SSH tunnel looks like a finished run. Curl
`$VLLM_CUDA_BASE_URL/models` before starting, every time.

### `bf16` only, enforced in the builder

`models/fused-4bit/config.json` and `fused-8bit/config.json` both carry
`"quantization": {"group_size": 64, "bits": 4, "mode": "affine"}` — MLX's
affine quantisation, which vLLM on CUDA cannot load. The GGUF exports are a
third format again, and vLLM's GGUF support would measure that path rather
than the model.

So `vllm_cuda_arm()` raises `ValueError` on any variant but `bf16` rather than
accepting the argument and failing later against a server that is billing by
the second. A CUDA-native quantisation (FP8, AWQ) is possible but produces new
weights, which under this repo's rules means a new parity check and a new
`MANIFEST.json` entry — a separate phase, not a quiet widening of this one.

### `server_hardware` is mandatory, and `hardware` is left alone

`hardware.py` says an unlabelled number is worthless. The sweep runs on the
Mac, so `client.py` stamps `HardwareInfo` — correctly — with `Apple M5 Pro`
onto every row, including rows an A100 produced. Overwriting that field would
be a different lie: the Mac really did measure the request.

`ServerHardware` is the second fingerprint, `None` for exactly those arms
where the client is the server. `accelerator` has no default, and
`vllm_cuda_arm()` raises when `VLLM_CUDA_GPU_NAME` is unset, because the
failure mode otherwise is silent and unrecoverable: a cloud row and a local
row become indistinguishable in the saved data, and the machine that could
have told you the difference has been deleted. `memory_bandwidth_gb_s` is
captured per-row specifically because it is the independent variable in the
scaling factor this arm exists to test.

MLflow gets the same treatment — `forge.chip` stays the client's chip and
`forge.accelerator` is tagged alongside it, rather than one being quietly
reinterpreted as the other.

### The client stays on the Mac

`sweep.py` calls `assert_native_arm64()`, which hard-fails on x86-64 Linux, and
`hardware.py` is macOS-only (`sysctl`, `sw_vers`, `system_profiler`). Running
the client on the VM would mean relaxing both, plus putting the capstone repo
and Atlas credentials on a rented machine. Running it here instead costs one
network round trip inside every TTFT measurement.

That trade is acceptable because of which metric it damages. **TTFT carries
the RTT; ITL does not.** A constant network delay shifts the whole token
stream without changing the gaps between tokens, and the gaps are where the
memory-bandwidth hypothesis lives. Measure ITL at concurrency 1 for the
cleanest test of the scaling factor, record an RTT baseline before and after
the sweep, and report TTFT both raw and RTT-corrected.

This also makes region a measurement decision rather than a price one: a
nearby region costs 10–20% more per hour and saves ~200 ms of RTT, which on a
two-hour booking is under fifty cents.

### Run this arm with `--skip-thermal`

`ThermalMonitor` reads *this* machine's thermal pressure, and for a remote arm
this machine is a client sending JSON. Recording its thermal state against a
cloud row would be meaningless. The harness already handles the distinction
properly — `cooled_down` is written as `None`, not `True`, when monitoring is
off, so "never checked" stays distinguishable from "checked and cool".

## Dry run — verifying all of this without a cloud account

The arm contains no CUDA. It is transport and configuration, so every line of
it can be exercised locally against any OpenAI-compatible server. Run all
three layers before provisioning anything: debugging your own wiring while a
GPU bills by the second is the expensive way to find a typo.

### Layer 1 — unit

`tests/test_arms.py`, 16 tests, no network. Covers construction from
`Settings`, `launch_command is None`, the full `ServerHardware` round trip,
both refusal paths (missing base URL, missing GPU name), rejection of every
unloadable variant, the local arms carrying `None`, and the `sweep.py` wiring.

```bash
uv run pytest tests/test_arms.py
```

### Layer 2 — a real sweep against a local stand-in

`mlx_lm.server` is the best stand-in because it serves *the same weights*, so
the generations are meaningful too.

```bash
# terminal 1
uv run python -m mlx_lm.server --model models/fused-bf16 --port 8099 \
  --decode-concurrency 1 --prompt-concurrency 1

# terminal 2
MLFLOW_TRACKING_URI="sqlite:////tmp/dryrun-mlflow.db" \
VLLM_CUDA_BASE_URL=http://127.0.0.1:8099/v1 \
VLLM_CUDA_MODEL_ID=models/fused-bf16 \
VLLM_CUDA_GPU_NAME="DRY RUN - mlx_lm.server stand-in, not a GPU" \
uv run python -m forge.phase2.sweep --arms vllm_cuda --skip-thermal \
  --concurrency-levels 1 2 --n-repeats 1 --n-warmup 1 --n-per-bucket 2 \
  --output-dir /tmp/_dryrun
```

**Use `--output-dir`, and point `MLFLOW_TRACKING_URI` somewhere disposable.**
A dry-run file written into `results/sweep/` is indistinguishable from a real
one to `score_accuracy.py`, which globs the whole directory, and to
`build_report.py`, which opens cell files by name. A laptop-generated
`vllm_cuda_bf16_c8_medium.jsonl` would silently become a row in a published
cost table.

Verified output, 20 September 2026: 6 cell files, 9 rows, 0 failures, real
MongoDB aggregations in `generated_text`, and every row carrying both
fingerprints —

```
client hw : Apple M5 Pro
server hw : {"accelerator": "DRY RUN - mlx_lm.server stand-in, not a GPU",
             "provider": "local", "memory_bandwidth_gb_s": 307.0, ...}
```

### Layer 3 — the failure path

Kill the stand-in and re-run one small cell. Verified: the sweep completes,
each cell reports `n_failed: 1`, rows carry `error: "Connection error."`, and
`server_hardware` survives on the failed rows. This is the dropped-SSH-tunnel
scenario, and confirming what it looks like in the data is the point —
otherwise you cannot tell it apart from a finished run.

## Getting the weights to the VM

`models/fused-bf16` is 3.1 GB (2.9 GiB) and gitignored, and it is the only artifact in
`models/` that vLLM on CUDA can load. `forge.phase4.upload_fused_bf16` moves it
to a **private** Hugging Face repo, after which every VM you ever build pulls it
at datacentre speed. Cloud ingress is free on both Azure and AWS, so it goes up
your home connection exactly once.

```bash
python -m forge.phase4.upload_fused_bf16 --dry-run   # both guards, no network, no token
python -m forge.phase4.upload_fused_bf16
```

Private by default, unlike `upload_model.py`. That module publishes the demo's
GGUF deliberately; this one exists to move weights onto a machine you rented,
and a 3 GB fine-tune should not become public because a default was convenient.
`--public` overrides.

Two guards run first, and both refuse rather than warn:

**Every file vLLM needs must be present**, `chat_template.jinja` above all.
`models/fused-bf16/tokenizer_config.json` has no `chat_template` key — checked,
not assumed — so the template exists only in that sibling file. A repo missing
it serves a model that formats prompts differently from every other arm, and the
failure is silent: the server starts, answers, and scores near zero for reasons
that look like a model problem. This runs first because it is the cheap check
and because it gives the actionable error; a missing file also trips the drift
guard below, and "missing chat_template.jinja" sends you somewhere more useful
than "the hash doesn't match".

**The directory must still be the artifact Phase 1 verified.** It is hashed with
`artifact_hash.hash_directory` and compared to `models/MANIFEST.json`, and the
manifest's parity check must have passed. This is the same promise
`sweep.py`'s `load_verified_variants()` makes locally, applied at the moment the
artifact leaves this machine — uploading a drifted checkpoint would put
unverified weights on a GPU and produce numbers that look real. The hash is
path-sensitive, so a stray file counts as drift; that is why the generated model
card is committed from memory as bytes and never written into the directory it
describes.

**An existing repo's visibility must match what was asked for.**
`create_repo(exist_ok=True)` does not change the settings of a repo that already
exists, so without this check a private upload into a repo created public
earlier would succeed, log `private=True`, and publish the weights anyway. It
refuses rather than flipping the setting: changing the visibility of a repo
someone may already be consuming is not this script's call.

All files go up in a **single `create_commit`**, not one call per file. A
six-call upload that fails after `model.safetensors` lands leaves a repo missing
its tokenizer — which still starts and serves, and is exactly the partial state
the completeness guard exists to prevent.

The generated model card records the adapter hash, the parity-check result, the
directory hash, and a per-file SHA-256 table, so the published repo can be
verified file-by-file rather than trusted. It also documents `--chat-template`
and the Turing `--dtype float16` caveat, because the person who needs those
warnings is whoever is reading the repo on the VM at the time.

Then, on the VM:

```bash
hf auth login                                                  # a read token; the repo is private
hf download <you>/forge-qwen2.5-coder-1.5b-mongodb-bf16 --local-dir ~/models/fused-bf16
```

## After a real run, these stop being true

Six places currently state that this arm was never measured. Measuring it and
leaving them is the one thing the write-up is built not to do.

| Where | What says it |
|---|---|
| `docs/cost-model.md` | assumptions row: "**Estimated, not measured** — the vLLM-on-CUDA arm was deferred" |
| `build_report.py:57–58` | `cloud_gpu_hourly_usd=1.39` (RunPod) and the `2039/307` factor |
| `build_report.py:43` | `LOCAL_VARIANTS` is hardcoded to three arms |
| `cost_model.py:130` | `cloud_gpu_cost_per_query()`'s docstring opens "Estimated, not measured" |
| `README.md` | "That fifth arm was deferred (no CUDA hardware) and is never presented as measured" |
| `docs/write-up.md`, `docs/failure-gallery.md` | the deferral narrative, and whatever broke on the way |

Set the scaling factor to `1.0` and pass the measured throughput directly —
`cloud_gpu_cost_per_query()` needs no rewrite, only honest inputs.
