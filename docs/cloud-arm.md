# The cloud arms (`ollama_cloud`, `vllm_cuda`) — design notes and dry-run procedure

The plan's fifth arm — vLLM on a CUDA GPU — was deferred for the obvious
reason: there is no CUDA hardware here. Phase 3 filled the hole with an
estimate (`cloud_gpu_throughput_scaling_factor = 2039/307`, the A100/M5 Pro
memory-bandwidth ratio) and flagged it, everywhere it appears, as the least
certain number in the cost model.

A second cloud arm turned out to matter more in practice. `ollama_cloud` runs
the *same Ollama daemon, Modelfile and registered tag* as the local `ollama`
arm on a rented commodity CPU — no GPU quota, no pay-as-you-go upgrade, and it
adds the price point the cost model is actually missing: commodity cloud CPU
against unified-memory Apple Silicon against a per-token hosted API.

This document covers the harness work that makes measuring both possible, the
decisions taken along the way, and how they were verified without renting
anything. **No cloud numbers exist yet.** Nothing in `results/` or
`docs/cost-model.md` has changed; the arms are wired and tested, not run.

## Which arm first

`ollama_cloud`. It needs no quota, so it can run on day one while a GPU quota
request sits in a queue for up to two days — and on Azure it runs while the
free trial's hard spending limit is still switched on, since that limit only
disappears when you upgrade to pay-as-you-go to *ask* for GPU quota. Do the CPU
arm, then upgrade, then the GPU arm. If quota never lands, a four-arm
comparison with a real cloud CPU curve stands on its own.

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

This used to mean **nothing health-checked the endpoint before the sweep
started**: pointed at a dead port, the sweep ran to completion writing rows
whose `error` read `Connection error.` (layer 3 below). `ManagedServer` now runs
a one-shot preflight for every arm with `server_hardware` set: `GET
$BASE_URL/models` must answer *and* list `model_id`, or the sweep refuses to
start. A tunnel that drops *mid*-sweep still produces failed rows rather than an
abort — the right trade for a two-hour sweep — so read `n_failed` in the
`sweep.cell_finish` log lines as the run goes.

### One arm name per machine, not per stack

`ollama_cloud` could have been `ollama_arm` with a different `base_url`. It
isn't, and the reason is not stylistic: `sweep.py` writes results to
`f"{arm}_{variant}_c{n}_{bucket}.jsonl"`, so a cloud run reusing the name
`ollama` would **overwrite the local result files in `results/sweep/`**, and
`score_accuracy.py` groups by `(arm, model_variant, prompt_bucket)` and would
score the two populations as one. A distinct name is what keeps them separable
on disk and in the accuracy report. `tests/test_arms.py` asserts both
properties directly.

Everything else about that arm is deliberately identical to the local one —
same daemon, same `ollama/Modelfile.q4`, same `forge-qwen-coder-ft:q4` tag, so
`model_id` defaults to the local arm's. Holding the serving software constant
means the only variable between the two rows is the hardware, which is a much
cleaner comparison than swapping stack and machine at once.

One practical note that belongs in the write-up rather than the code: run this
arm with `--concurrency-levels 1 2 4 8`. Firing 64 concurrent requests at 8
vCPUs measures queueing, not throughput, and an honest truncated sweep with one
sentence explaining the ceiling reads far better than a full one whose tail is
noise.

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

### `processor`, not `accelerator`

The field on `ServerHardware` that names the serving hardware is called
`processor`. It was `accelerator` until the CPU arm landed and made that a
small lie — a Graviton4 is not an accelerator, and this is the one field whose
entire job is to label honestly. `processor_memory_gb` follows the same logic:
VRAM for a GPU, system RAM for a CPU VM.

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

Now 33 tests across both arms, including the two that lock in the name
separation: arm names are disjoint from the local ones, and the same variant
produces different result filenames.

### Layer 2 for the CPU arm

Identical shape — any OpenAI-compatible server stands in, and
`OLLAMA_CLOUD_MODEL_ID` overrides the default tag so the stand-in doesn't need
the model registered under Ollama's name:

```bash
OLLAMA_CLOUD_BASE_URL=http://127.0.0.1:8099/v1 \
OLLAMA_CLOUD_MODEL_ID="$PWD/models/fused-bf16" \
OLLAMA_CLOUD_CPU_NAME="DRY RUN - mlx_lm stand-in, not a cloud CPU" \
uv run python -m forge.phase2.sweep --arms ollama_cloud --skip-thermal \
  --concurrency-levels 1 --n-repeats 1 --n-warmup 1 --n-per-bucket 1 \
  --output-dir /tmp/_dryrun_cpu
```

Verified 20 September 2026: 3 cells, 3 rows, 0 failures, files written as
`ollama_cloud_q4_c1_*.jsonl` — distinct from `ollama_q4_*`, which is the
collision the separate arm name exists to prevent.

### Layer 2 — a real sweep against a local stand-in

`mlx_lm.server` is the best stand-in because it serves *the same weights*, so
the generations are meaningful too.

```bash
# terminal 1
uv run python -m mlx_lm.server --model "$PWD/models/fused-bf16" --port 8099 \
  --decode-concurrency 1 --prompt-concurrency 1

# terminal 2
MLFLOW_TRACKING_URI="sqlite:////tmp/dryrun-mlflow.db" \
VLLM_CUDA_BASE_URL=http://127.0.0.1:8099/v1 \
VLLM_CUDA_MODEL_ID="$PWD/models/fused-bf16" \
VLLM_CUDA_GPU_NAME="DRY RUN - mlx_lm.server stand-in, not a GPU" \
uv run python -m forge.phase2.sweep --arms vllm_cuda --skip-thermal \
  --concurrency-levels 1 2 --n-repeats 1 --n-warmup 1 --n-per-bucket 2 \
  --output-dir /tmp/_dryrun
```

Absolute paths on both sides, because `mlx_lm.server` lists its model under
the *resolved* path and the preflight compares that string exactly — a
relative `models/fused-bf16` is refused as "does not serve".

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

Kill the stand-in and re-run: the preflight now refuses to start
(`remote endpoint ... is not answering`). To see the mid-sweep failure shape,
kill the stand-in *after* the first cell finishes. Verified before the preflight
existed: the sweep completes, each cell reports `n_failed: 1`, rows carry `error: "Connection error."`, and
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

## Provisioning on Azure — shared by both arms

Run from the Mac. Needs the Azure CLI (`brew install azure-cli`) and `jq`.
**Written against the CLI's documented flags but not yet executed end to end**
— where a flag is known to vary by size or subscription, the fallback is
given inline. Fix this section, not your shell history, when something differs.

**0. Sign in and pick a region.**

```bash
az login
az account show --query '{subscription:name, id:id}' -o table

RG=forge-bench
LOC=centralindia            # any region that offers the sizes below
MYIP=$(curl -s https://api.ipify.org)
```

A subscription that has never created a VM may not have the compute resource
provider registered — then `az vm list-usage` returns `[]` rather than an
error, and every quota check below silently shows nothing. Found on a real
subscription whose only prior workload was Container Apps. Registration is free
and takes a minute or two:

```bash
az provider show -n Microsoft.Compute --query registrationState -o tsv   # want "Registered"
az provider register -n Microsoft.Compute --wait
```

On pay-as-you-go there is no spending limit, so set a budget alert before the
first VM exists (the portal's Cost Management → Budgets is the simplest route;
alerts only notify, they never stop anything — auto-shutdown and tear-down do).

**1. Price, availability, quota — before creating anything.** Record the rate
now: it goes into `.env` and from there onto every result row, which is what
`build_report` prices the run with. The public Retail Prices API needs no
login:

```bash
az_price() {  # $1 = size, $2 = region -> Linux pay-as-you-go USD/hour
  curl -s -G https://prices.azure.com/api/retail/prices \
    --data-urlencode "\$filter=serviceName eq 'Virtual Machines' and armSkuName eq '$1' and armRegionName eq '$2' and priceType eq 'Consumption'" |
  jq -r '[.Items[] | select((.productName|test("Windows")|not) and (.skuName|test("Spot|Low Priority")|not))][0].retailPrice'
}
az_price Standard_D4ps_v6 $LOC
```

Checked 23 September 2026 for `centralindia`: `Standard_D4ps_v6` $0.0924/h,
`Standard_NC4as_T4_v3` $0.579/h, `Standard_NC24ads_A100_v4` $5.142/h. Rates
differ by region; always record the one you are billed.

```bash
# Offered here, and not restricted for this subscription? (Restrictions column must be "None")
az vm list-skus --location $LOC --size Standard_D4ps_v6 --all -o table
# vCPU quota: current usage + the size's vCPUs must fit under Limit, for both
# the family and "Total Regional vCPUs"
az vm list-usage --location $LOC -o table | grep -Ei 'total regional|dpsv6|ncast4|a100'
```

GPU families (`Standard NCASv3_T4 Family`, `Standard NCADS_A100_v4 Family`)
start at 0 and a free trial cannot raise them. Upgrade to pay-as-you-go, then
request quota in the portal (Quotas → Compute → filter by family): 4 vCPUs for
`NC4as_T4_v3`, 24 for `NC24ads_A100_v4`. Approval can take up to two days,
which is why the CPU arm runs first.

**2. Resource group and a locked-down NSG.** One group for everything, so tear
down is one command. The only inbound rule is SSH from your current IP —
neither Ollama nor vLLM is ever exposed; both are reached through the tunnel.

```bash
az group create -n $RG -l $LOC
az network nsg create -g $RG -n forge-nsg
az network nsg rule create -g $RG --nsg-name forge-nsg -n ssh-from-me \
  --priority 1000 --direction Inbound --access Allow --protocol Tcp \
  --source-address-prefixes "$MYIP/32" --destination-port-ranges 22
```

If your IP changes (new network, VPN), update the rule rather than widening it:
`az network nsg rule update -g $RG --nsg-name forge-nsg -n ssh-from-me --source-address-prefixes "$(curl -s https://api.ipify.org)/32"`.

**3. Create the VM** — the CPU one here, the GPU one in its runbook below.

```bash
az vm create -g $RG -n forge-cpu \
  --size Standard_D4ps_v6 \
  --image Canonical:ubuntu-24_04-lts:server-arm64:latest \
  --admin-username azureuser --generate-ssh-keys \
  --nsg forge-nsg --public-ip-sku Standard \
  --os-disk-size-gb 64
```

The image must be **Arm64** (`server-arm64`) — an x64 image is refused for an
Arm size, and that is the most common first failure. If creation is refused
over the disk controller, add `--disk-controller-type NVMe`; if it is refused
over the security type, add `--security-type Standard`.

**4. Guard against the forgotten VM, then connect.**

```bash
az vm auto-shutdown -g $RG -n forge-cpu --time 1830     # daily, UTC — stops compute billing
VM_IP=$(az vm show -d -g $RG -n forge-cpu --query publicIps -o tsv)
ssh azureuser@$VM_IP 'uname -m; lscpu | grep "Model name"; free -g | head -2'
```

`uname -m` must print `aarch64`. The `lscpu` and `free` lines are
`OLLAMA_CLOUD_CPU_NAME` and `OLLAMA_CLOUD_MEMORY_GB`.

**5. Tear down.** Between sessions, `az vm deallocate -g $RG -n forge-cpu`
stops compute billing (the disk and public IP still bill a little). When the
numbers are saved and copied off the VM:

```bash
az group delete -n $RG --yes --no-wait
```

## Runbook — `ollama_cloud` on an Azure Arm VM

The client stays on the Mac (see above); only Ollama runs on the VM. Every step
here exists because skipping it produces rows that look real and aren't.

**1. Provision.** Steps 0–4 of the provisioning section above: an Arm64 size —
Cobalt 100 (`Dpsv6`, e.g. `Standard_D4ps_v6`) or Ampere Altra (`Dpsv5`) — on
an Arm64 image, SSH-only NSG, auto-shutdown on.

**2. Tunnel.** Never open 11434 — Ollama has no authentication. Keep this
running in its own terminal for the whole session:

```bash
ssh -N -L 11435:127.0.0.1:11434 azureuser@$VM_IP
```

**3. Ollama — same version, same daemon settings as the Mac.** "Only the
hardware differs" is false if the Ollama version or its server env differs.
Every result row now carries a `server_software` record (stack, version,
settings — see `hardware.ServerSoftware`), so a mismatch is visible in the
data rather than only in your notes:

```bash
curl -fsSL https://ollama.com/install.sh | sh     # on the VM; supports arm64
ollama --version                                  # on the Mac AND the VM — should match
```

The **version** is detected from the live daemon (`/api/version`) when the
sweep starts; nothing to type. The **settings** can't be detected — neither
daemon reports its own env — so set `OLLAMA_NUM_PARALLEL`,
`OLLAMA_FLASH_ATTENTION` and `OLLAMA_KV_CACHE_TYPE` on the VM (`sudo systemctl
edit ollama`) to whatever the local daemon runs with, and record both ends in
`.env` as JSON:

```bash
OLLAMA_SERVER_ENV={"OLLAMA_NUM_PARALLEL": 4}        # the Mac's daemon (`ollama` arm)
OLLAMA_CLOUD_SERVER_ENV={"OLLAMA_NUM_PARALLEL": 4}  # the VM's daemon
```

The sweep logs `arms.server_settings_unrecorded` if the cloud one is blank.
`OLLAMA_NUM_PARALLEL` in particular decides whether concurrency 8 is batching
or queueing.

**4. The weights — byte-identical, and proven so.** Keep the repo's relative
layout so `ollama/Modelfile.q4` is used unmodified (Ollama resolves `FROM`
against the Modelfile's own directory — see `ollama_register.py`):

```bash
# on the VM
mkdir -p ~/forge/ollama ~/forge/models/fused-gguf
hf download tarun-11/forge-qwen2.5-coder-1.5b-mongodb-gguf model-Q4_K_M.gguf \
  --local-dir ~/forge/models/fused-gguf
sha256sum ~/forge/models/fused-gguf/model-Q4_K_M.gguf
# must equal models/MANIFEST.json's model-Q4_K_M hash:
#   31c8a27c12ace4937272da05070e200b8cec00c488a877068534fe91f79e1bb3

# from the Mac
scp ollama/Modelfile.q4 azureuser@<vm-ip>:~/forge/ollama/

# on the VM
cd ~/forge && ollama create forge-qwen-coder-ft:q4 -f ollama/Modelfile.q4
```

The sweep does not gate `ollama_cloud` on the manifest (it cannot see the VM's
disk), so the `sha256sum` is the only verification this arm gets. Do not skip it.

**5. `.env` on the Mac.** `OLLAMA_CLOUD_BASE_URL=http://127.0.0.1:11435/v1`,
`OLLAMA_CLOUD_CPU_NAME` from `lscpu | grep 'Model name'` on the VM (a Cobalt
100 reports its Neoverse core, not "Cobalt" — record what it says, and put the
product name in `INSTANCE_TYPE`), `OLLAMA_CLOUD_PROVIDER=azure`, the size,
region, RAM and the hourly rate you are actually billed. Blank values are fine;
they read as unset.

**6. Run.**

```bash
curl -s -o /dev/null -w 'rtt %{time_total}s\n' http://127.0.0.1:11435/v1/models   # before
uv run python -m forge.phase2.sweep --arms ollama_cloud --skip-thermal \
  --concurrency-levels 1 2 4 8 --max-retries 0 --request-timeout-s 900
curl -s -o /dev/null -w 'rtt %{time_total}s\n' http://127.0.0.1:11435/v1/models   # after
```

`--max-retries 0` and the long timeout are not optional on a CPU. Local Ollama
on the M5 Pro already shows a long-bucket p95 TTFT of ~5.5 s at c4 (see
`results/sweep_summary.json`); a few vCPUs are an order of magnitude slower at
prefill, so queued requests at c4–c8 can pass the default 120 s before the first
byte. With retries on, the SDK then silently re-sends them: TTFT absorbs the
failed attempt and the server gets extra load. A timed-out row with retries off
is an honest `n_failed`; a retried one is a wrong number.

Then score and price it — both are safe to run with the local results
already present:

```bash
uv run python -m forge.phase2.score_accuracy   # merges into accuracy_by_group.json
uv run python -m forge.phase3.build_report     # fills the "Cloud VMs" section and curve
```

`score_accuracy` merges into its output instead of overwriting it, so scoring a
separate directory no longer deletes the other groups. `build_report` prices
the cell as an always-on VM from the rows' own measured throughput and
`server_hardware.hourly_usd` — set `OLLAMA_CLOUD_HOURLY_USD` *before* the
sweep so the rate is recorded with the numbers.

**7. Tear down.** Step 5 of the provisioning section.

## Runbook — `vllm_cuda` on an Azure GPU VM

Same shape as the CPU runbook: the client stays on the Mac, the server is
reached only through a tunnel, and every row records what served it. Do the
dry run (layer 2 above) first — debugging your own wiring while a GPU bills by
the second is the expensive way to find a typo.

**1. Choose the GPU — this is a measurement decision, not just a price.**

| Size | GPU | Memory bandwidth | `--dtype` | Note |
|---|---|---|---|---|
| `Standard_NC4as_T4_v3` | Tesla T4 16 GB | 320 GB/s | `float16` | Turing has no bf16 — the checkpoint is cast to fp16 at load |
| `Standard_NC24ads_A100_v4` | A100 80 GB **PCIe** | 1,935 GB/s | `bfloat16` (default) | the one the cost model's scaling factor is about |

The T4 serves the bf16 artifact *cast to fp16*, so its numbers are not quite
the verified artifact's. Record the dtype (step 7), and compare its accuracy
against `mlx_lm/bf16` before trusting its throughput.

The A100 in `NC A100 v4` is the **PCIe** part, 1,935 GB/s — not the SXM
part's 2,039 GB/s that `build_report.ASSUMPTIONS` currently uses in its
estimated scaling factor. Record the PCIe figure; see "After a real run".

**2. Provision.** Provisioning steps 0–2 and 4–5 as for the CPU arm, with the
GPU family's quota. Step 3 differs — x64, Ubuntu 22.04, and a bigger disk for
vLLM plus the CUDA wheels plus the 3.1 GB checkpoint:

```bash
az vm create -g $RG -n forge-gpu \
  --size Standard_NC4as_T4_v3 \
  --image Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest \
  --security-type Standard \
  --admin-username azureuser --generate-ssh-keys \
  --nsg forge-nsg --public-ip-sku Standard \
  --os-disk-size-gb 128
az vm auto-shutdown -g $RG -n forge-gpu --time 1830
VM_IP=$(az vm show -d -g $RG -n forge-gpu --query publicIps -o tsv)
```

`--security-type Standard` turns off Trusted Launch: with Secure Boot on, an
NVIDIA kernel module that isn't signed for it silently fails to load and
`nvidia-smi` finds no device.

**3. Driver.** vLLM's wheels bring their own CUDA runtime; the VM only needs
the kernel driver.

```bash
# on the VM
sudo apt-get update && sudo apt-get install -y ubuntu-drivers-common tmux
sudo ubuntu-drivers install
sudo reboot
# reconnect, then:
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
```

That line is `VLLM_CUDA_GPU_NAME` and `VLLM_CUDA_GPU_MEMORY_GB`. If
`ubuntu-drivers` picks nothing, Azure's extension does the same job (from the
Mac; it reboots the VM itself):
`az vm extension set -g $RG --vm-name forge-gpu --name NvidiaGpuDriverLinux --publisher Microsoft.HpcCompute`.

**4. vLLM.**

```bash
# on the VM
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
uv venv ~/vllm --python 3.12 && source ~/vllm/bin/activate
uv pip install vllm        # pin vllm==X.Y.Z to repeat a run; the version is recorded either way
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The last line must print `True` and the GPU name — otherwise vLLM will fail
later with a less direct error.

**5. Weights, verified.** From the private repo `upload_fused_bf16` created
(see "Getting the weights to the VM"):

```bash
# on the VM
hf auth login              # a READ token — the repo is private
hf download <you>/forge-qwen2.5-coder-1.5b-mongodb-bf16 --local-dir ~/models/fused-bf16
hf auth logout             # the token doesn't need to outlive the download
cd ~/models/fused-bf16 && sha256sum $(ls | grep -v README.md)
```

Compare each hash with the per-file SHA-256 table in the repo's model card
(`README.md`). The `vllm_cuda` arm is gated on the local manifest, but that
gate proves what is on the *Mac*; this is the only check of what is on the VM.

**6. Serve** — inside `tmux`, so a dropped SSH session doesn't kill the server
mid-sweep:

```bash
# on the VM
tmux new -s vllm
source ~/vllm/bin/activate
vllm serve ~/models/fused-bf16 \
  --served-model-name forge-bf16 \
  --host 127.0.0.1 --port 8000 \
  --chat-template ~/models/fused-bf16/chat_template.jinja \
  --max-num-seqs 256 --gpu-memory-utilization 0.9 \
  --dtype float16 \
  2>&1 | tee ~/vllm-serve.log
# A100: drop --dtype float16 (bf16 is native)
```

- `--host 127.0.0.1`: only reachable through the tunnel, so no `--api-key` is
  needed. Add one (and `VLLM_CUDA_API_KEY`) if you ever bind wider.
- `--chat-template` is explicit because `tokenizer_config.json` has no
  template (see "Getting the weights to the VM"). Without it the server starts,
  answers, and scores near zero.
- `--max-num-seqs 256 --gpu-memory-utilization 0.9` are `vllm_metal_arm()`'s
  values, so the two vLLM arms differ in hardware, not scheduler settings.

Smoke-test on the VM before tunnelling:

```bash
curl -s 127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "forge-bf16", "temperature": 0, "max_tokens": 60,
  "messages": [{"role": "user", "content": "How many singers are there?"}]}' | jq -r '.choices[0].message.content'
curl -s 127.0.0.1:8000/version
```

Expect a PyMongo-shaped answer (`db.singer...`). Chat prose, or `<|im_end|>`
repeated, means the chat template isn't being applied — stop and fix it.

**7. Tunnel and `.env` on the Mac.**

```bash
ssh -N -L 8001:127.0.0.1:8000 azureuser@$VM_IP     # its own terminal, whole session
```

```bash
VLLM_CUDA_BASE_URL=http://127.0.0.1:8001/v1
VLLM_CUDA_MODEL_ID=forge-bf16
VLLM_CUDA_GPU_NAME=Tesla T4                 # exactly what nvidia-smi printed
VLLM_CUDA_PROVIDER=azure
VLLM_CUDA_INSTANCE_TYPE=Standard_NC4as_T4_v3
VLLM_CUDA_REGION=centralindia
VLLM_CUDA_GPU_MEMORY_GB=16
VLLM_CUDA_GPU_MEMORY_BANDWIDTH_GB_S=320     # A100 80GB PCIe: 1935
VLLM_CUDA_HOURLY_USD=0.579                  # az_price, step 1 of provisioning
VLLM_CUDA_SERVER_ARGS={"dtype": "float16", "max_num_seqs": 256, "gpu_memory_utilization": 0.9, "chat_template": "chat_template.jinja"}
```

The vLLM version is detected from the server (`/version`) when the sweep
starts; `SERVER_ARGS` is the part it can't detect. Keep it in step with the
`vllm serve` line you actually ran.

**8. Run, score, price.**

```bash
curl -s -o /dev/null -w 'rtt %{time_total}s\n' http://127.0.0.1:8001/v1/models   # before
uv run python -m forge.phase2.sweep --arms vllm_cuda --skip-thermal \
  --max-retries 0 --request-timeout-s 300
curl -s -o /dev/null -w 'rtt %{time_total}s\n' http://127.0.0.1:8001/v1/models   # after
uv run python -m forge.phase2.score_accuracy
uv run python -m forge.phase3.build_report
```

The full concurrency ladder (1–64) is right here — unlike 8 vCPUs, a GPU with
continuous batching is expected to absorb 64. About 1,524 requests; the
preflight refuses to start if the tunnel is down or `forge-bf16` isn't served.
`build_report` adds the run to the "Cloud VMs" table and chart, priced from
the measured throughput and the recorded rate.

**9. Keep the server's own record, then tear down.** vLLM's startup log states
the resolved dtype, KV-cache size and maximum concurrency — the configuration
that actually ran. Copy it next to the local servers' logs before the VM goes:

```bash
mkdir -p logs/servers
scp azureuser@$VM_IP:~/vllm-serve.log "logs/servers/vllm_cuda_bf16_$(date -u +%Y%m%dT%H%M%SZ).log"
az group delete -n $RG --yes --no-wait
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
| `docs/cost-model.md` "Cloud VMs" section | says "Not measured yet" — regenerated automatically by `build_report`, but re-read the prose around it |
| `build_report.py` `ASSUMPTIONS` | the estimated factor uses 2,039 GB/s (A100 **SXM**) while pricing an A100 80GB **PCIe** (1,935 GB/s). Moot once measured — but if the estimate stays in the doc, correct it or say which part it means |

Set the scaling factor to `1.0` and pass the measured throughput directly —
`cloud_gpu_cost_per_query()` needs no rewrite, only honest inputs.
