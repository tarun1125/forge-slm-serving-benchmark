# Parity check — design and threshold

Phase 1 of `01-FORGE-serving-benchmark.md` assigns the parity-check pass/fail
threshold to the project owner, not to Claude Code, because every assumption
in it is arguable and it's the kind of thing an interviewer asks "why did you
pick that number?" about. For this repo, that call was explicitly delegated
back to Claude Code — this document is the record of that decision, so it can
still be defended on its merits rather than treated as a black box.

## What's being checked and why

`mlx_lm.fuse` folds a LoRA adapter into its base model's linear layers:
`W' = W + scale * B @ A`. That is a linear-algebra identity, not an
approximation — a correct fuse should be numerically indistinguishable from
running the base model with the adapter attached, up to bf16 rounding. If the
fused model's outputs diverge from the adapter-applied model's outputs by more
than rounding noise, the fuse is wrong, and every Phase 2/3 number produced
against the fused weights is invalid. This check exists to catch that before
any benchmark number gets produced, per the plan's own gate.

## The subset

50 cases, stratified by complexity (easy/medium/hard), sampled with a fixed
seed (42) from the capstone's existing **304-case held-out test set**
(`rag/data/rag_test.json`) — not re-derived from the full 1,517-case dataset.
Reasoning:

- The 304-case set is already the held-out split with zero train/test overlap,
  already has verified gold results (`data/gold_results.json`, computed by
  actually executing each gold query against Atlas), and already has the exact
  per-database training prompts recorded (`fine_tuning/data_23db/split_manifest.json`).
  Re-deriving a 50-case subset from the raw 1,517 would mean re-solving a
  problem (stratified split, gold generation, prompt reconstruction) the
  capstone repo already solved and validated.
- Reusing it is also how this check avoids ever calling `mlx_lm.generate`
  with a prompt that wasn't part of the model's own training-prompt
  distribution — the split_manifest's system prompts are read verbatim, the
  same way `fine_tuning/generate_predictions_23db.py` does, not reconstructed.

## The two-part threshold

Both conditions must hold. Constants live in
`src/forge/phase1/parity_check.py` (`EXACT_MATCH_THRESHOLD`, `N_CASES`).

**1. Exact-text-match rate ≥ 96% (48/50), greedy-decoded.**
Not 100%: fusing occasionally moves a token's logit across a tie-break
boundary purely from bf16 rounding order, which can flip one greedy decode
without indicating anything wrong. Not lower than 96%: any more than one
flipped case stops being explainable by rounding and starts looking like an
actual behavioral change from the fuse.

**2. Zero execution-accuracy regressions — a hard gate, independent of the
aggregate rate.**
Every case the adapter-applied model executed correctly (verified against
`gold_results.json` via the capstone's own `evaluation/execute_queries.py`
execution-and-compare logic — never string match) must still execute
correctly under the fused model. This is checked per-case, not as an
aggregate percentage, because an aggregate could hide a regression: a fused
model that loses one case but gains a different one nets to the same
aggregate accuracy while still being a broken fuse on the case it lost.

A run that fails either condition should not be treated as "close enough" —
re-run the fuse, and if it fails twice, treat that as a genuine finding for
the failure gallery (Phase 5), not something to explain away.

## Why execution accuracy, not string match

`evaluation/execute_queries.py` in the capstone repo already established
(with its own documented findings — key-name-tolerant matching, BSON→JSON
conversion, an AST allowlist against `eval()`) that string-comparing
generated PyMongo against gold is the wrong tool: two syntactically different
queries can be semantically identical, and two syntactically similar queries
can return different rows. That logic is imported (`forge.capstone_bridge`),
not reimplemented, for the same DRY reasons the capstone repo's own comments
give for not duplicating it a third time internally.

## Why there's no `MONGODB_URI` in FORGE's own `.env`

The capstone's `atlas_env.connect()` auto-discovers
`atlas-credentials.env` by walking up from its own file location. Importing
that function directly (rather than re-implementing a Mongo connection in
FORGE) means the Atlas secret has exactly one copy on disk — duplicating it
into a second `.env` would violate AGENTS.md's "no secrets in the repo, ever"
principle in spirit even while gitignored, by creating a second place a leak
could happen from. `forge.capstone_bridge.CapstoneHarness.check_atlas_credentials()`
fails loudly with the exact file path if it's missing, rather than silently
falling back to anything.

## A known gap in the plan this surfaced: GGUF export for Qwen2

`01-FORGE-serving-benchmark.md` Phase 1 calls for a GGUF variant "so you're
comparing the same weights, not Ollama's convenience download of a different
checkpoint," and suggests `mlx_lm.fuse --export-gguf` for this. That flag only
supports `model_type in {llama, mixtral, mistral}` (`mlx_lm/fuse.py`,
`mlx_lm/gguf.py`) — Qwen2.5-Coder's `model_type` is `"qwen2"`, so calling it
raises `ValueError`. `src/forge/phase1/gguf_export.py` routes around this via
llama.cpp's `convert_hf_to_gguf.py` (which does support Qwen2), run against
the *fused* BF16 directory — so the "same weights" requirement is still met,
just via a different converter than the plan assumed. This is exactly the
kind of thing the plan calls "the most common way this kind of benchmark is
silently invalid" — worth a line in the failure gallery (Phase 5) rather than
silently swapped without comment.
