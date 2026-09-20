"""Phase 4: upload models/fused-bf16 to a PRIVATE Hugging Face Hub repo, so a
rented cloud VM can pull the weights the `vllm_cuda` arm serves.

The sibling module upload_model.py deliberately uploads only the Q4_K_M GGUF —
its docstring says the MLX fused variants are "a separate, optional step if
ever wanted". The cloud arm is what makes that step wanted: fused-bf16 is the
only artifact in models/ that vLLM on CUDA can load (fused-4bit and fused-8bit
are MLX affine-quantised; the GGUFs are a third format), and it is gitignored,
so a VM has no way to reach it. 2.9 GB up a home connection once, then every
future VM pulls it from HF at datacentre speed for free.

Private by default, unlike upload_model.py. That module publishes the demo's
model deliberately; this one exists to move weights to a machine you rented,
and defaulting a 3 GB fine-tune to public because it was convenient is not a
decision worth making by accident. Pass --public to override.

Two guards run before anything is uploaded, both of which refuse rather than
warn:

  1. The directory must still hash to what models/MANIFEST.json recorded, and
     that manifest's parity check must have passed. Uploading a checkpoint
     that has drifted from the verified one would put unverified weights on a
     GPU and produce numbers that look real. This is the same promise
     sweep.py's load_verified_variants() makes locally, applied at the point
     the artifact leaves this machine.
  2. Every file vLLM needs must be present — chat_template.jinja above all.
     tokenizer_config.json in this directory has NO chat_template key (checked:
     the template lives only in the sibling .jinja file), so a repo missing it
     serves a model that formats prompts differently from every other arm in
     this benchmark. That failure is silent: the server starts, answers, and
     scores near zero for reasons that look like a model problem.

This is an outward-facing action that creates a repo under your account. Like
upload_model.py it is NOT run by anything automatically. Run it yourself:

    python -m forge.phase4.upload_fused_bf16 --dry-run   # verify, upload nothing
    python -m forge.phase4.upload_fused_bf16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

from forge.artifact_hash import hash_directory, hash_file
from forge.config import get_settings
from forge.logging_config import configure_logging, get_logger, start_run

log = get_logger(__name__)

DEFAULT_MODEL_DIR = Path("models/fused-bf16")
DEFAULT_MANIFEST_PATH = Path("models/MANIFEST.json")
DEFAULT_REPO_SUFFIX = "forge-qwen2.5-coder-1.5b-mongodb-bf16"
DEFAULT_GITHUB_REPO_URL = "https://github.com/tarun1125/forge-slm-serving-benchmark"

# Exactly what vLLM needs to serve this model, and nothing else. The
# directory's own README.md is mlx_lm's generated stub and is replaced by the
# card this module builds, which is why it isn't in this list.
REQUIRED_FILES = [
    "model.safetensors",
    "model.safetensors.index.json",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    # Not optional. See this module's docstring — without it the served
    # prompts stop matching every other arm's, silently.
    "chat_template.jinja",
]

MODEL_CARD_TEMPLATE = """\
---
license: apache-2.0
base_model: mlx-community/Qwen2.5-Coder-1.5B-Instruct-bf16
library_name: transformers
tags:
  - text-generation
  - mongodb
  - pymongo
  - lora
  - vllm
---

# {repo_name}

A LoRA fine-tune of Qwen2.5-Coder-1.5B-Instruct for natural-language ->
PyMongo query generation, fused into the base weights and kept at bf16 in
standard Hugging Face safetensors format.

This is the artifact the FORGE benchmark's cloud-GPU arm serves. It exists so
a rented NVIDIA GPU can `vllm serve` the same weights the Apple Silicon arms
run, rather than a separately quantised approximation of them. The MLX 4-bit
and 8-bit variants in that project use MLX's affine quantisation and cannot be
loaded by vLLM on CUDA at all; the Q4_K_M GGUF is a different format again and
is published separately.

Full benchmark methodology, cost model and results: {github_repo_url}

## Serving

```bash
vllm serve <local-dir> \\
  --served-model-name forge-bf16 \\
  --dtype bfloat16 \\
  --max-num-seqs 256 \\
  --gpu-memory-utilization 0.9 \\
  --chat-template <local-dir>/chat_template.jinja
```

`--dtype float16` instead on Turing GPUs (T4), which have no bf16 units.

`--chat-template` is passed explicitly on purpose: `tokenizer_config.json`
here carries no `chat_template` key, so the template is loaded from the
sibling `chat_template.jinja`. A server that misses it formats prompts
differently from the rest of the benchmark and scores near zero without
erroring.

## Provenance

Fused from `{base_model}`, adapter `{adapter_hash}`.

Phase 1's parity check (fused model vs. adapter-applied model, {n_cases}
cases): **{exact_match_rate:.0%} exact match, passed**. The source directory
hashes to `{dir_hash}`, matching `models/MANIFEST.json` at upload time.

Per-file SHA-256 of what was uploaded:

| File | Bytes | SHA-256 |
|---|---:|---|
{file_table}
"""


def verify_against_manifest(
    model_dir: Path, manifest_path: Path, variant_name: str = "bf16"
) -> dict:
    """Refuses unless the directory still IS the artifact Phase 1 verified.

    Returns the manifest's variant entry. Raises on: a missing manifest, a
    manifest whose parity check didn't pass, a variant it doesn't describe, or
    a directory whose content hash has drifted from the recorded one.
    """
    if not model_dir.is_dir():
        raise RuntimeError(f"{model_dir} does not exist — run forge.phase1.fuse first.")
    if not manifest_path.exists():
        raise RuntimeError(
            f"{manifest_path} not found — run forge.phase1.manifest first. This module "
            "refuses to publish weights whose parity check it cannot confirm."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parity = manifest.get("parity_check", {})
    if not parity.get("passed"):
        raise RuntimeError(
            f"{manifest_path} shows parity_check.passed={parity.get('passed')!r}. Publishing "
            "a checkpoint that failed (or never had) its parity check would put unverified "
            "weights on a GPU and produce numbers that look real. Re-run "
            "forge.phase1.parity_check first."
        )

    entry = next((v for v in manifest.get("variants", []) if v["name"] == variant_name), None)
    if entry is None:
        raise RuntimeError(f"{manifest_path} has no variant named {variant_name!r}.")

    actual = "sha256:" + hash_directory(model_dir)
    if actual != entry["hash"]:
        raise RuntimeError(
            f"{model_dir} no longer matches {manifest_path}.\n"
            f"  manifest: {entry['hash']}\n"
            f"  actual:   {actual}\n"
            "The directory has changed since the parity check verified it. Re-run "
            "forge.phase1.parity_check and forge.phase1.manifest, or restore the directory."
        )
    return entry


def check_required_files(model_dir: Path) -> list[Path]:
    """Returns the files to upload, in REQUIRED_FILES order. Raises naming
    every missing one — a partial upload is worse than none, because the
    resulting repo starts and serves."""
    paths = [model_dir / name for name in REQUIRED_FILES]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise RuntimeError(
            f"{model_dir} is missing {', '.join(missing)}. vLLM needs every file in "
            f"{REQUIRED_FILES} to serve this model the way the other arms do — see this "
            "module's docstring on chat_template.jinja in particular."
        )
    return paths


def build_model_card(
    repo_name: str, manifest_entry: dict, manifest: dict, files: list[Path], github_repo_url: str
) -> str:
    parity = manifest["parity_check"]
    file_table = "\n".join(
        f"| `{p.name}` | {p.stat().st_size:,} | `{hash_file(p)}` |" for p in files
    )
    return MODEL_CARD_TEMPLATE.format(
        repo_name=repo_name,
        github_repo_url=github_repo_url,
        base_model=manifest["base_model"],
        adapter_hash=manifest["adapter_path_hash"],
        n_cases=parity["n_cases"],
        exact_match_rate=parity["exact_match_rate"],
        dir_hash=manifest_entry["hash"],
        file_table=file_table,
    )


def upload(
    model_dir: Path,
    manifest_path: Path,
    hf_token: str,
    hf_username: str,
    repo_suffix: str,
    github_repo_url: str,
    private: bool = True,
    dry_run: bool = False,
) -> str:
    entry = verify_against_manifest(model_dir, manifest_path)
    files = check_required_files(model_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    repo_id = f"{hf_username}/{repo_suffix}"
    total_mb = round(sum(p.stat().st_size for p in files) / 1e6, 1)
    log.info(
        "upload_fused_bf16.verified",
        repo_id=repo_id,
        dir_hash=entry["hash"],
        n_files=len(files),
        total_mb=total_mb,
        private=private,
        dry_run=dry_run,
    )

    card = build_model_card(repo_suffix, entry, manifest, files, github_repo_url)
    if dry_run:
        log.info("upload_fused_bf16.dry_run_complete", repo_id=repo_id)
        print(f"\n--- dry run: nothing uploaded ---\nWould create: {repo_id} (private={private})")
        for p in files:
            print(f"  {p.name:32} {p.stat().st_size / 1e6:>9.1f} MB")
        print(f"\n--- generated model card ---\n{card}")
        return repo_id

    api = HfApi(token=hf_token)
    log.info("upload_fused_bf16.create_repo", repo_id=repo_id, private=private)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    for path in files:
        log.info(
            "upload_fused_bf16.upload_file",
            repo_id=repo_id,
            file=path.name,
            size_mb=round(path.stat().st_size / 1e6, 1),
        )
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=repo_id,
            repo_type="model",
        )

    card_path = model_dir / "README.md.generated"
    card_path.write_text(card, encoding="utf-8")
    try:
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
        )
    finally:
        # In a temp name inside model_dir, and removed even on failure — a
        # stray file here would change the directory hash and make the next
        # run's manifest check fail for a reason that has nothing to do with
        # the weights.
        card_path.unlink(missing_ok=True)

    log.info("upload_fused_bf16.finish", repo_id=repo_id)
    return repo_id


def main() -> None:
    configure_logging()
    settings = get_settings()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--repo-suffix", default=DEFAULT_REPO_SUFFIX)
    parser.add_argument("--github-repo-url", default=DEFAULT_GITHUB_REPO_URL)
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create a public repo instead of a private one. These are 3 GB of fine-tuned "
        "weights being moved to a machine you rented — publishing them should be a choice, "
        "not a default.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run both guards and print the model card, then stop. Contacts no network and "
        "needs no HF_TOKEN.",
    )
    args = parser.parse_args()

    if not args.dry_run:
        if not settings.hf_token:
            raise RuntimeError("HF_TOKEN is not set in .env — see .env.example.")
        if not settings.hf_username:
            raise RuntimeError("HF_USERNAME is not set in .env — see .env.example.")

    run_id = start_run(log, phase="phase4.upload_fused_bf16")
    repo_id = upload(
        model_dir=args.model_dir,
        manifest_path=args.manifest_path,
        hf_token=settings.hf_token or "",
        hf_username=settings.hf_username or "DRY-RUN-USER",
        repo_suffix=args.repo_suffix,
        github_repo_url=args.github_repo_url,
        private=not args.public,
        dry_run=args.dry_run,
    )
    log.info("run.finish", run_id=run_id, phase="phase4.upload_fused_bf16", repo_id=repo_id)

    if not args.dry_run:
        print(f"\nUploaded to: https://huggingface.co/{repo_id}")
        print("\nOn the VM:")
        print(f"  hf download {repo_id} --local-dir ~/models/fused-bf16")
        print("\nA private repo needs a read token there too: `hf auth login`.")


if __name__ == "__main__":
    main()
