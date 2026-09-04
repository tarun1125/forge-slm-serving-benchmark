# vllm-metal supported_models.md contribution

**Submitted.** Opened as [vllm-project/vllm-metal#689](https://github.com/vllm-project/vllm-metal/pull/689),
from a fork at [tarun1125/vllm-metal](https://github.com/tarun1125/vllm-metal). This file is
kept as the record of the reasoning and the reviewed draft the actual PR was built from — see
the bottom of this file for exactly what was run to produce it.

## Why a prose note, not a new table row

`docs/supported_models.md` already lists Qwen2.5 as ✅ supported
(`mlx-community/Qwen2.5-7B-Instruct-4bit`), and the file's own header is
explicit: *"Other sizes and quantizations of the same family generally work
too... If a model or checkpoint does not work, please open an issue rather
than adding more rows or example checkpoints."* Qwen2.5-Coder-1.5B is the
same architecture family at a smaller size — adding a second row for it
would go against the maintainers' own stated preference.

What the file's own convention actually uses for reporting a specific
verified checkpoint+quantization combination is a **prose paragraph with a
PR/issue link**, matching the existing AWQ paragraph ("Verified for Qwen2.5,
Llama 3, and Mistral ([#340]...)") and GGUF paragraph ("Verified end-to-end
on Qwen3-0.6B Q4_1 and on Qwen3-0.6B, Llama-3.2-1B-Instruct, and
Mistral-7B-Instruct-v0.3 Q8_0 ([#415]...)"). This draft follows that pattern.

## What we can honestly claim

- Real checkpoint tested: `mlx-community/Qwen2.5-Coder-1.5B-Instruct-bf16`
  (public, unmodified) plus 8-bit and 4-bit MLX quantizations converted from
  it via `mlx_lm.convert` — not a synthetic or cherry-picked test.
- 135 real inference cells across concurrency 1/2/4/8/16, three prompt
  lengths (~128/~1.4k/~4-6k tokens), three quantization levels — **zero
  failures**. Well past the tracking issue's stated minimum ("loads without
  errors, generates coherent output on a standard prompt").
- Real measured throughput at concurrency=8 (medium/~1.4k-token prompts):
  bf16 148.9 tok/s, 8-bit 330.6 tok/s, 4-bit 459.8 tok/s — a real
  batching-scaling curve, not a single anecdote.
- Hardware: Apple M5 Pro, 15 CPU cores / 16 GPU cores, 24GB unified memory,
  macOS 26.5 (Tahoe).
- `vllm-metal` 0.28.0.dev20260903050220 (vLLM 0.28.0).
- **What we did NOT test**: `vllm-metal`'s own native GGUF serving path.
  The GGUF file we produced uses Q4_K_M (a K-quant), and this doc's own
  GGUF section explicitly states K-quants are rejected by that path —
  our GGUF was only ever served through Ollama, a completely different
  runtime, not vllm-metal's GGUF support. Not claiming that as tested.
- Also confirmed (secondary, not the citable public checkpoint): a LoRA
  fine-tune of this same model, fused into the base weights, loads and
  serves identically — expected, since it's the same architecture/config,
  just different learned weights, but worth a line for completeness.

## Proposed diff to `docs/supported_models.md`

Insert after the existing GGUF paragraph (after line 85, before the
`| Model | Support | ...` table header):

```markdown
Qwen2.5-Coder-1.5B-Instruct (`mlx-community/Qwen2.5-Coder-1.5B-Instruct-bf16`,
plus 8-bit and 4-bit MLX quantizations converted from it) verified on an
M5 Pro (15-core CPU / 16-core GPU, 24GB unified memory): 135 real inference
requests across concurrency 1-16 and three prompt lengths (~128 to ~6k
tokens), zero failures. Throughput scales with both quantization and
concurrency as expected — e.g. at concurrency 8 on ~1.4k-token prompts:
148.9 tok/s (bf16), 330.6 tok/s (8-bit), 459.8 tok/s (4-bit). Not tested:
vllm-metal's native GGUF path (our GGUF artifact uses Q4_K_M, a K-quant,
which this doc's own GGUF section notes is out of scope for that path — it
was served through Ollama instead, not vllm-metal). See #289.
```

## Proposed PR description

```
Reports on issue #289 (vLLM Metal Model Support).

Verified mlx-community/Qwen2.5-Coder-1.5B-Instruct-bf16, plus 8-bit and
4-bit MLX quantizations converted from it, on an Apple M5 Pro (15-core
CPU / 16-core GPU, 24GB unified memory) running vllm-metal
0.28.0.dev20260903050220 / vLLM 0.28.0.

135 real inference requests across concurrency 1/2/4/8/16 and three prompt
lengths (~128 / ~1.4k / ~4-6k tokens), across all three quantization
levels — zero failures, coherent output throughout. Real measured
throughput at concurrency 8 (medium-length prompts): 148.9 tok/s (bf16),
330.6 tok/s (8-bit), 459.8 tok/s (4-bit) — scaling with both concurrency
and quantization as expected.

Not tested: vllm-metal's native GGUF serving path. A GGUF export of this
model exists in my own benchmark project but uses Q4_K_M, a K-quant, which
this doc's own GGUF section states is out of scope for vllm-metal's GGUF
support — I served that variant through Ollama instead, a different
runtime, and I'm not claiming it as a vllm-metal GGUF test.

Added a note rather than a new table row, since Qwen2.5 is already ✅
listed and the doc asks not to add redundant rows/checkpoints for an
already-covered family — this documents a specific smaller-size,
code-specialized checkpoint + its quantized variants, matching the
existing prose-note pattern used for the AWQ and GGUF verification call-outs.
```

## What actually happened

1. Forked to [tarun1125/vllm-metal](https://github.com/tarun1125/vllm-metal)
   (`gh repo fork vllm-project/vllm-metal --clone=false`).
2. Re-fetched the live `docs/supported_models.md` from upstream first, to confirm
   nothing had drifted since this draft was written — it hadn't.
3. Applied the exact diff above, committed, pushed to the fork.
4. Opened [vllm-project/vllm-metal#689](https://github.com/vllm-project/vllm-metal/pull/689)
   with the description above via `gh pr create`.
5. GitHub's DCO check flagged the commit as missing a `Signed-off-by` trailer;
   fixed with `git commit --amend --signoff` and a force-push to the fork branch.
