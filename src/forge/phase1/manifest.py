"""Phase 1 gate deliverable: models/MANIFEST.json — every variant with its
hash, size on disk, and the parity-check execution accuracy it was verified
against. This is the file Phase 2 refuses to run without (see its own gate
check), so a variant that was never verified can't silently enter the
benchmark.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from forge.artifact_hash import dir_size_bytes, hash_directory, hash_file
from forge.hardware import get_hardware_dict
from forge.logging_config import configure_logging, get_logger, start_run

log = get_logger(__name__)

DEFAULT_VARIANTS = {
    "bf16": Path("models/fused-bf16"),
    "8bit": Path("models/fused-8bit"),
    "4bit": Path("models/fused-4bit"),
}
DEFAULT_GGUF_DIR = Path("models/fused-gguf")


def describe_directory_variant(name: str, path: Path) -> dict | None:
    if not path.exists():
        log.warning("manifest.variant_missing", variant=name, path=str(path))
        return None
    return {
        "name": name,
        "path": str(path),
        "format": "mlx",
        "hash": f"sha256:{hash_directory(path)}",
        "size_bytes": dir_size_bytes(path),
    }


def describe_gguf_variants(gguf_dir: Path) -> list[dict]:
    if not gguf_dir.exists():
        log.warning("manifest.gguf_dir_missing", path=str(gguf_dir))
        return []
    variants = []
    for gguf_file in sorted(gguf_dir.glob("*.gguf")):
        variants.append(
            {
                "name": gguf_file.stem,
                "path": str(gguf_file),
                "format": "gguf",
                "hash": f"sha256:{hash_file(gguf_file)}",
                "size_bytes": gguf_file.stat().st_size,
            }
        )
    return variants


def load_parity_report(report_path: Path) -> dict | None:
    if not report_path.exists():
        log.warning("manifest.parity_report_missing", path=str(report_path))
        return None
    return json.loads(report_path.read_text(encoding="utf-8"))


def build_manifest(
    base_model: str,
    adapter_path: Path,
    variants: dict[str, Path],
    gguf_dir: Path,
    parity_report_path: Path,
) -> dict:
    parity_report = load_parity_report(parity_report_path)

    entries = [describe_directory_variant(name, path) for name, path in variants.items()]
    entries = [e for e in entries if e is not None]
    entries.extend(describe_gguf_variants(gguf_dir))

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "base_model": base_model,
        "adapter_path_hash": f"sha256:{hash_directory(adapter_path)}"
        if adapter_path.exists()
        else None,
        "hardware": get_hardware_dict(),
        "parity_check": {
            "passed": parity_report.get("passed") if parity_report else None,
            "exact_match_rate": parity_report.get("exact_match_rate") if parity_report else None,
            "n_cases": parity_report.get("n_cases") if parity_report else None,
            "report_path": str(parity_report_path) if parity_report else None,
        },
        "variants": entries,
    }


def main() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="mlx-community/Qwen2.5-Coder-1.5B-Instruct-bf16")
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--gguf-dir", type=Path, default=DEFAULT_GGUF_DIR)
    parser.add_argument(
        "--parity-report", type=Path, default=Path("results/parity_check_report.json")
    )
    parser.add_argument("--out", type=Path, default=Path("models/MANIFEST.json"))
    args = parser.parse_args()

    run_id = start_run(log, phase="phase1.manifest")
    manifest = build_manifest(
        args.base_model, args.adapter_path, DEFAULT_VARIANTS, args.gguf_dir, args.parity_report
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info(
        "run.finish",
        run_id=run_id,
        phase="phase1.manifest",
        out=str(args.out),
        n_variants=len(manifest["variants"]),
    )


if __name__ == "__main__":
    main()
