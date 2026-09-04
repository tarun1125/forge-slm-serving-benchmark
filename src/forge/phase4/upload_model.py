"""Phase 4: upload the fused, quantized GGUF checkpoint to a new Hugging Face
Hub model repo, so the HF Space demo (space/app.py) can download it via
hf_hub_download() instead of needing FORGE_LOCAL_MODEL_PATH.

Uploads models/fused-gguf/model-Q4_K_M.gguf specifically — the same file
space/app.py requests (MODEL_FILENAME) and the same one tested end-to-end
locally via FORGE_LOCAL_MODEL_PATH. Does NOT upload the MLX fused variants
(fused-bf16/8bit/4bit): those only run via mlx_lm on Apple Silicon, aren't
what the Space serves, and uploading them is a separate, optional step if
ever wanted for citability — not needed to unblock the demo.

This is an outward-facing action (creates a public-by-default HF repo under
your account) — it is NOT run automatically by anything in this project.
Run it yourself, explicitly, once HF_TOKEN and HF_USERNAME are set in .env:

    python -m forge.phase4.upload_model

Then copy the printed repo id into space/app.py's MODEL_REPO_ID.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

from forge.config import get_settings
from forge.logging_config import configure_logging, get_logger, start_run

log = get_logger(__name__)

DEFAULT_GGUF_PATH = Path("models/fused-gguf/model-Q4_K_M.gguf")
DEFAULT_REPO_SUFFIX = "forge-qwen2.5-coder-1.5b-mongodb-gguf"

MODEL_CARD_TEMPLATE = """\
---
license: apache-2.0
base_model: mlx-community/Qwen2.5-Coder-1.5B-Instruct-bf16
tags:
  - gguf
  - text-generation
  - mongodb
  - pymongo
  - lora
---

# {repo_name}

A LoRA fine-tune of Qwen2.5-Coder-1.5B-Instruct, trained for natural-language
-> PyMongo query generation, fused into the base weights and quantized here
as {quant_type} GGUF (llama.cpp K-quant) for CPU serving via
`llama-cpp-python` or Ollama.

Served in the [FORGE benchmark's Space demo](https://huggingface.co/spaces/{hf_username}/forge-nl-to-mongodb).
Full write-up, benchmark methodology, and results across serving stacks
(mlx_lm, Ollama, vLLM-Metal) and quantization levels:
https://github.com/{hf_username}/forge-slm-serving-benchmark

Not the same quantization implementation as the MLX 4-bit variant used
elsewhere in that benchmark (llama.cpp K-quant vs. MLX's own INT4 scheme) —
this GGUF is specifically the one portable to non-Apple-Silicon infrastructure.
"""


def build_model_card(hf_username: str, repo_name: str, quant_type: str) -> str:
    return MODEL_CARD_TEMPLATE.format(
        repo_name=repo_name, quant_type=quant_type, hf_username=hf_username
    )


def upload(
    gguf_path: Path,
    hf_token: str,
    hf_username: str,
    repo_suffix: str,
    quant_type: str,
) -> str:
    if not gguf_path.exists():
        raise RuntimeError(f"{gguf_path} does not exist — run forge.phase1.gguf_export first.")

    repo_id = f"{hf_username}/{repo_suffix}"
    api = HfApi(token=hf_token)

    log.info("upload_model.create_repo", repo_id=repo_id)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    card_path = gguf_path.parent / "README.md.generated"
    card_path.write_text(build_model_card(hf_username, repo_suffix, quant_type), encoding="utf-8")

    log.info(
        "upload_model.upload_file",
        repo_id=repo_id,
        gguf_path=str(gguf_path),
        size_mb=round(gguf_path.stat().st_size / 1e6, 1),
    )
    api.upload_file(
        path_or_fileobj=str(gguf_path),
        path_in_repo=gguf_path.name,
        repo_id=repo_id,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    card_path.unlink()

    log.info("upload_model.finish", repo_id=repo_id)
    return repo_id


def main() -> None:
    configure_logging()
    settings = get_settings()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf-path", type=Path, default=DEFAULT_GGUF_PATH)
    parser.add_argument("--repo-suffix", default=DEFAULT_REPO_SUFFIX)
    parser.add_argument(
        "--quant-type",
        default="Q4_K_M",
        help="Label only, for the generated model card — does not re-quantize.",
    )
    args = parser.parse_args()

    if not settings.hf_token:
        raise RuntimeError("HF_TOKEN is not set in .env — see .env.example.")
    if not settings.hf_username:
        raise RuntimeError("HF_USERNAME is not set in .env — see .env.example.")

    run_id = start_run(log, phase="phase4.upload_model")
    repo_id = upload(
        gguf_path=args.gguf_path,
        hf_token=settings.hf_token,
        hf_username=settings.hf_username,
        repo_suffix=args.repo_suffix,
        quant_type=args.quant_type,
    )
    log.info("run.finish", run_id=run_id, phase="phase4.upload_model", repo_id=repo_id)
    print(f"\nUploaded to: https://huggingface.co/{repo_id}")
    print(f"Set space/app.py's MODEL_REPO_ID to: {repo_id!r}")


if __name__ == "__main__":
    main()
