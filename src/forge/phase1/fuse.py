"""Phase 1, step 1: fuse the LoRA adapter into the base model, producing the
BF16 reference weights every other variant (8-bit, 4-bit, GGUF) derives from.

Usage:
    python -m forge.phase1.fuse
    python -m forge.phase1.fuse --adapter-path /path/to/adapter --save-path models/fused-bf16

Note: mlx_lm.fuse's own --export-gguf flag only supports model_type in
{llama, mixtral, mistral} (see mlx_lm/fuse.py) — Qwen2.5-Coder's model_type
is "qwen2", so it is NOT eligible. GGUF export is handled separately by
gguf_export.py via llama.cpp, using this fused BF16 directory as its input,
so the GGUF and MLX arms are provably built from the same fused weights.
See docs/parity-check-design.md.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from mlx_lm.fuse import main as mlx_fuse_main

from forge.config import get_settings
from forge.logging_config import configure_logging, get_logger, hash_for_log, start_run

log = get_logger(__name__)


def fuse(base_model: str, adapter_path: Path, save_path: Path) -> Path:
    save_path.mkdir(parents=True, exist_ok=True)
    log.info(
        "fuse.start",
        base_model=base_model,
        adapter_path_hash=hash_for_log(str(adapter_path)),
        save_path=str(save_path),
    )
    start = time.monotonic()

    # mlx_lm.fuse.main() reads sys.argv directly (it's a CLI entrypoint, not a
    # library function) — patch argv for the duration of the call rather than
    # shelling out, so we stay in-process and keep our own logging/timing.
    import sys

    argv_backup = sys.argv
    sys.argv = [
        "mlx_lm.fuse",
        "--model",
        base_model,
        "--adapter-path",
        str(adapter_path),
        "--save-path",
        str(save_path),
    ]
    try:
        mlx_fuse_main()
    finally:
        sys.argv = argv_backup

    elapsed_ms = (time.monotonic() - start) * 1000
    log.info("fuse.finish", save_path=str(save_path), latency_ms=round(elapsed_ms, 1))
    return save_path


def main() -> None:
    configure_logging()
    settings = get_settings()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=settings.base_model)
    parser.add_argument("--adapter-path", type=Path, default=settings.adapter_path)
    parser.add_argument("--save-path", type=Path, default=Path("models/fused-bf16"))
    args = parser.parse_args()

    if not args.adapter_path.exists():
        raise RuntimeError(f"Adapter path does not exist: {args.adapter_path}")

    run_id = start_run(log, phase="phase1.fuse")
    fuse(args.base_model, args.adapter_path, args.save_path)
    log.info("run.finish", run_id=run_id, phase="phase1.fuse")


if __name__ == "__main__":
    main()
