"""Phase 1, step 2: produce 8-bit and 4-bit MLX variants from the fused BF16
model, so every quantization level in the benchmark traces back to the same
fused weights (not three independently-fused checkpoints that could silently
drift from each other).

Usage:
    python -m forge.phase1.quantize --fused-path models/fused-bf16
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from mlx_lm.convert import convert

from forge.logging_config import configure_logging, get_logger, start_run

log = get_logger(__name__)

DEFAULT_GROUP_SIZE = 64


def quantize_variant(fused_path: Path, out_path: Path, bits: int, group_size: int) -> Path:
    if out_path.exists():
        log.warning("quantize.skip_existing", out_path=str(out_path), bits=bits)
        shutil.rmtree(out_path)  # mlx_lm.convert refuses to write into an existing dir

    log.info("quantize.start", bits=bits, group_size=group_size, out_path=str(out_path))
    start = time.monotonic()
    convert(
        hf_path=str(fused_path),
        mlx_path=str(out_path),
        quantize=True,
        q_bits=bits,
        q_group_size=group_size,
    )
    elapsed_ms = (time.monotonic() - start) * 1000
    log.info("quantize.finish", bits=bits, out_path=str(out_path), latency_ms=round(elapsed_ms, 1))
    return out_path


def main() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fused-path", type=Path, default=Path("models/fused-bf16"))
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    args = parser.parse_args()

    if not args.fused_path.exists():
        raise RuntimeError(f"{args.fused_path} does not exist — run forge.phase1.fuse first.")

    run_id = start_run(log, phase="phase1.quantize")
    for bits in (8, 4):
        out_path = args.out_dir / f"fused-{bits}bit"
        quantize_variant(args.fused_path, out_path, bits=bits, group_size=args.group_size)
    log.info("run.finish", run_id=run_id, phase="phase1.quantize")


if __name__ == "__main__":
    main()
