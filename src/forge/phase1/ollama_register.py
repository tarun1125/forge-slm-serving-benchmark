"""Register the fine-tuned GGUF variants with Ollama under names distinct
from any stock pull, so the Ollama arm in Phase 2 provably serves the same
fine-tuned weights as every other arm — not a convenience download of a
different, un-fine-tuned checkpoint (see docs/parity-check-design.md).

Ollama resolves a Modelfile's `FROM <path>` relative to the *Modelfile's own
directory*, not the caller's CWD or the repo root — confirmed empirically
(a `./models/...`-relative Modelfile written for CWD resolution fails with
a generic "invalid model name" error that has nothing to do with the model
name). Modelfiles here therefore live in ollama/ and reference
../models/fused-gguf/<file> accordingly; don't "simplify" that path without
re-testing against a running Ollama daemon.

Requires `ollama serve` running locally.

Usage:
    python -m forge.phase1.ollama_register
    python -m forge.phase1.ollama_register --gguf-dir models/fused-gguf \\
        --model-prefix forge-qwen-coder-ft
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from forge.logging_config import configure_logging, get_logger, start_run

log = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
OLLAMA_DIR = REPO_ROOT / "ollama"

# GGUF filename fragment -> short Ollama tag.
TAG_BY_FRAGMENT = {
    "f16": "f16",
    "Q8_0": "q8",
    "Q4_K_M": "q4",
}


def tag_for(gguf_file: Path) -> str | None:
    for fragment, tag in TAG_BY_FRAGMENT.items():
        if fragment in gguf_file.stem:
            return tag
    return None


def write_modelfile(gguf_file: Path, tag: str) -> Path:
    OLLAMA_DIR.mkdir(parents=True, exist_ok=True)
    modelfile_path = OLLAMA_DIR / f"Modelfile.{tag}"
    relative_from = Path("..") / gguf_file.relative_to(REPO_ROOT)
    modelfile_path.write_text(f"FROM {relative_from}\n", encoding="utf-8")
    return modelfile_path


def register(gguf_file: Path, tag: str, model_prefix: str) -> str:
    modelfile_path = write_modelfile(gguf_file, tag)
    model_name = f"{model_prefix}:{tag}"

    log.info("ollama_register.start", model_name=model_name, gguf_file=str(gguf_file))
    subprocess.run(
        ["ollama", "create", model_name, "-f", str(modelfile_path)],
        check=True,
        cwd=REPO_ROOT,
    )
    log.info("ollama_register.finish", model_name=model_name)
    return model_name


def main() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf-dir", type=Path, default=Path("models/fused-gguf"))
    parser.add_argument("--model-prefix", default="forge-qwen-coder-ft")
    args = parser.parse_args()

    gguf_dir = (REPO_ROOT / args.gguf_dir) if not args.gguf_dir.is_absolute() else args.gguf_dir
    if not gguf_dir.exists():
        raise RuntimeError(f"{gguf_dir} does not exist — run forge.phase1.gguf_export first.")

    gguf_files = sorted(gguf_dir.glob("*.gguf"))
    if not gguf_files:
        raise RuntimeError(f"No .gguf files found in {gguf_dir}.")

    run_id = start_run(log, phase="phase1.ollama_register", n_files=len(gguf_files))
    registered = []
    for gguf_file in gguf_files:
        tag = tag_for(gguf_file)
        if tag is None:
            log.warning("ollama_register.unrecognized_variant", gguf_file=str(gguf_file))
            continue
        registered.append(register(gguf_file, tag, args.model_prefix))

    log.info("run.finish", run_id=run_id, phase="phase1.ollama_register", registered=registered)
    print(f"Registered: {', '.join(registered)}")
    print("Verify with: ollama list")


if __name__ == "__main__":
    main()
