# FORGE — Local SLM Serving & Cost Benchmark

![CI](https://github.com/tarun1125/forge-slm-serving-benchmark/actions/workflows/ci.yml/badge.svg)

A LoRA fine-tuned Qwen2.5-Coder-1.5B, served five ways — Apple Silicon MLX, Ollama, vLLM on
Metal, vLLM on CUDA, and a hosted API — measured for latency, throughput, quality, and cost
per 1,000 queries.

**Headline finding this benchmark is built to surface:** prefill (TTFT) is compute-bound;
decode (inter-token latency) is memory-bandwidth-bound. They do not scale the same way, and
this repo reports them as two separate metrics everywhere, never blended into one "latency"
number. See `docs/` for the full write-up once Phase 5 lands.

## Status

Phase 0 (environment), Phase 1 (model prep), Phase 2 (benchmark harness + sweep), and Phase 3
(cost model) are done. Phase 4 (deployment: HF Hub model, Gradio demo, vllm-metal upstream
contribution) is done except for a live hosted demo, which needs a Hugging Face PRO
subscription this project isn't paying for — see `space/README.md`. Phase 5 (final write-up)
is next.

## Setup

```bash
uv sync --group dev
cp .env.example .env   # fill in GROQ_API_KEY / NIM_API_KEY for Phase 2; adjust paths if needed
```

Requires native arm64 Python 3.12 (`python -c "import platform; print(platform.machine())"`
must print `arm64`) and `vllm-metal` installed on the host (built from source against Apple's
Metal Performance Shaders — see https://github.com/vllm-project/vllm-metal).

## Phase 1 — model preparation

```bash
python -m forge.phase1.fuse                          # fuse LoRA into base -> models/fused-bf16
python -m forge.phase1.quantize                       # -> models/fused-8bit, models/fused-4bit
python -m forge.phase1.gguf_export                     # -> models/fused-gguf/*.gguf (needs llama.cpp; see script docstring)
python -m forge.phase1.parity_check                     # gate: fused model == adapter-applied model
python -m forge.phase1.manifest --adapter-path <path>    # -> models/MANIFEST.json
```

Read `docs/parity-check-design.md` before touching the parity-check thresholds — it documents
why they're set where they are, including a real gap the plan's own GGUF instructions had for
this specific model architecture.

## Repository layout

```
src/forge/            # library code (Claude Code's — the runner, adapters, CI, scaffolding)
  phase1/             # model fuse / quantize / GGUF export / parity check / manifest
  phase2/             # benchmark harness (arms, sweeps, MLflow) — not yet built
docs/                 # cost model, parity-check design, write-up
tests/                # pytest — mirrors src/forge structure
models/               # MANIFEST.json is committed; weight directories are gitignored
results/              # benchmark output, gitignored except summaries
notebooks/            # the one reproducible experiment (Phase 5 deliverable)
```

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
uv run pip-audit --strict --ignore-vuln CVE-2026-71211 --ignore-vuln PYSEC-2026-3552
```

## Division of labour

The cost model, judge/parity thresholds, and anything where the interesting interview
question is "why did you design it that way" are written and owned by the project author.
Provider adapters, sweep orchestration, chart generation, and CI are Claude Code's. See
`docs/parity-check-design.md` for one such decision made under explicit delegation, with its
reasoning kept on record.
