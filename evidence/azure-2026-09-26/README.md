# Evidence: FORGE on Azure, 26 September 2026

The raw record behind every cloud number, and the cache-busted local re-run,
in [docs/write-up.md](../../docs/write-up.md), [docs/cost-model.md](../../docs/cost-model.md)
and failure #5 of [docs/failure-gallery.md](../../docs/failure-gallery.md). The committed
summaries (`results/sweep_cloud_summary.json`, `results/sweep_cold_summary.json`) are
derived from `raw-rows.tar.gz`; everything else here is what they can't show.

The raw rows live in `results/sweep*/`, which is gitignored — this folder is the
committed copy. Verify before trusting it:

```bash
cd evidence/azure-2026-09-26 && shasum -a 256 -c SHA256SUMS
```

## Contents

| File | What it is | What it proves |
|---|---|---|
| `raw-rows.tar.gz` | Every per-request row: `sweep/` (33 cloud cells, cached), `sweep_cold/` (141 cells, cache-busted, all arms), `sweep_cold_discarded/` (36 `vllm_metal` cells replaced by a re-run) | Each row carries its own `server_hardware` (processor, instance type, region, $/h), `server_software` (stack, version read from the live server, daemon settings) and, for cache-busted rows, a unique `prompt_nonce`. A cloud row cannot be mistaken for a local one. |
| `sweep-logs.tar.gz` | The seven sweep runs' structured logs, numbered in the order they ran | `run_id`s, `remote_preflight_ok` before each remote sweep, the detected server versions, every `cell_finish` with `n_failed`, every request error |
| `server-logs.tar.gz` | vLLM's startup+access log from the T4; local `mlx_lm` / `vllm_metal` server logs from the cache-busted run (home-directory paths replaced with `~`) | The T4's resolved configuration (fp16, Triton attention, 358,016-token KV cache) and that every request it received returned 200 — the basis for "lost in transit, not failed by the server" |
| `SHA256SUMS` | Checksums of the three archives | That this folder is what was committed |

Rows from `sweep_cold/` were run with a perturbed system prompt, so they carry
no accuracy; accuracy for every arm comes from the cached rows, in
`results/accuracy_by_group.json`.

## The machines

| | CPU arm (`ollama_cloud`) | GPU arm (`vllm_cuda`) |
|---|---|---|
| Size | `Standard_D4ps_v6`, Central India | `Standard_NC4as_T4_v3`, Central India |
| Hardware | 4× Neoverse-N2 (Cobalt 100), 16 GB, `aarch64` | Tesla T4 16 GB (compute 7.5), 4 vCPUs, 27 GB |
| Image | Ubuntu 24.04 Arm64 | Ubuntu 22.04 gen2, Secure Boot off, driver 595.91.07 |
| Server | Ollama 0.32.14; `NUM_PARALLEL=1`, flash attention off, `KEEP_ALIVE=5m` — read from the Mac daemon's own startup log and matched | vLLM 0.30.0; `--dtype float16` (no bf16 on Turing), `--max-num-seqs 256`, `--gpu-memory-utilization 0.9`, explicit chat template |
| Weights | GGUF sha256 `31c8a27c…e1bb3` — matched `models/MANIFEST.json` on the VM | bf16 directory, manifest hash `dc38d548…2467`; copied by `scp`, all 7 files matched Mac↔VM by sha256 |
| Rate (Azure Retail Prices API) | $0.0924/h | $0.579/h |
| SSH-tunnel RTT, before / after | 72 / 70 ms | 77 / 74 ms |

## The runs

| # | Sweep | `run_id` | Cells | Failed |
|---|---|---|---|---|
| 1 | `ollama_cloud`, cached, c 1–8 | `9796f9e6-7d92-4131-a78b-afcfef62d42a` | 12 | 0 |
| 2 | `ollama_cloud`, cache-busted | `072b2aca-a227-4db1-a7c2-6a222af292e1` | 12 | 0 |
| 3 | 3 local arms × 3 variants, cache-busted, c 1–8 | `e89f20f5-6689-44bf-a564-a8777893e976` | 108 | 23 (`mlx_lm`, c8 — listen backlog) |
| 4 | `vllm_metal` cache-busted re-run (replaces run 3's `vllm_metal` cells) | `e7cdbc3e-364a-44ae-b541-9cdd63daa9b6` | 36 | 0 |
| 5 | `vllm_cuda`, cached, c 1–64 | `76b38e44-da90-4781-a229-aa0587e7e816` | 21 | 19 at c ≥ 32 |
| 6 | `vllm_cuda`, cache-busted, c 1–64 | `22720666-6989-48ce-b67a-c3454e3fee17` | 21 | 3 at c64 |
| 7 | `vllm_cuda`, cached, c 32/64 re-run (replaces run 5's c32/c64 cells) | `6cb1025e-3f17-4500-9ea9-cf3b48e7f163` | 6 | 6 |

Run 5's c32/c64 cells were replaced because a 3 GB upload from the client
saturated its uplink during two of them (14:07:31–14:11:40 UTC, from file
timestamps on the VM). Run 4 replaced run 3's `vllm_metal` cells after an
unreproduced 17–58 s startup spike; the originals are kept in
`sweep_cold_discarded/` rather than deleted. Every run used
`--max-retries 0`: a failed request is recorded, never retried.

## Where it's committed

`cloud-arm-readiness` branch: `dc1f3f6` (CPU arm), `2e83493` (cache-busted
re-measurement), `eb7469e` (GPU arm). All Azure resources (`forge-bench`) were
deleted the same day; total spend was about $1–1.50 of sign-up credit.
