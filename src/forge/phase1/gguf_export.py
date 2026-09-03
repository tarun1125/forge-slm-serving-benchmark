"""Phase 1, step 3: export a GGUF variant of the FUSED weights for the Ollama
arm, so Ollama serves the same fine-tuned model as every other arm instead of
its own convenience download of a different (base, un-fine-tuned) checkpoint.
The plan calls this out as "the most common way this kind of benchmark is
silently invalid" — this script exists specifically to close that hole.

Why this shells out to llama.cpp instead of using mlx_lm.fuse --export-gguf:
mlx_lm's built-in GGUF exporter only supports model_type in
{llama, mixtral, mistral} (see mlx_lm/fuse.py, mlx_lm/gguf.py). Qwen2.5-Coder's
model_type is "qwen2", which is not in that list — calling --export-gguf on
this model raises ValueError. llama.cpp's convert_hf_to_gguf.py supports Qwen2
directly, so we use it against the *fused* BF16 directory (Hugging-Face-format
safetensors + config.json, produced by fuse.py) as input. This still satisfies
the "same weights" requirement: the GGUF is derived from the identical fused
checkpoint the MLX variants are derived from, just via a different converter.

One-time setup (not automated here deliberately — it's a build step with its
own toolchain requirements, not something to run unattended):
    git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp
    python3 -m venv ~/llama.cpp/.venv          # DEDICATED venv — see note below
    ~/llama.cpp/.venv/bin/pip install -r ~/llama.cpp/requirements.txt
    cmake -B ~/llama.cpp/build -DCMAKE_BUILD_TYPE=Release
    cmake --build ~/llama.cpp/build --config Release -j
    # llama-quantize will be at ~/llama.cpp/build/bin/llama-quantize

Why a DEDICATED venv rather than just `pip install -r requirements.txt` into
whatever's active: convert_hf_to_gguf.py needs torch (~2GB, CPU wheel) plus
sentencepiece/gguf — none of which FORGE itself needs, so they don't belong
in this project's own pyproject.toml. It also has to be a real venv, not a
plain `pip install --user`: running `python -m forge.phase1.gguf_export`
from FORGE's own activated .venv shadows the `python3` on PATH with FORGE's
venv interpreter, which never sees `--user` site-packages installed against
a different (e.g. pyenv global) interpreter — that mismatch is exactly what
produced a `ModuleNotFoundError: No module named 'torch'` the first time
this was set up, despite torch being verifiably installed elsewhere. This
script resolves llama.cpp's own venv explicitly (see
find_conversion_python()) rather than trusting ambient PATH state, so it
works correctly regardless of which venv FORGE itself is running under.

A second, unrelated upstream gap this hits: the mlx-community checkpoint's
tokenizer_config.json carries "extra_special_tokens" as a JSON *list*, but
transformers>=4.5x's tokenizer loader (used internally by
convert_hf_to_gguf.py's AutoTokenizer.from_pretrained(), not by mlx_lm's own
tokenizer loading) requires it to be a dict and crashes with
`AttributeError: 'list' object has no attribute 'keys'` otherwise. Every
token that field would have named is already present in tokenizer.json's own
added_tokens (verified: 22 entries, including all the ones listed) — the
field is a redundant, misformatted duplicate, not load-bearing. sanitize_
tokenizer_config() corrects it to {} in place before conversion. This DOES
change models/fused-bf16's content hash — re-run forge.phase1.manifest after
a GGUF export to keep MANIFEST.json accurate.

Usage:
    python -m forge.phase1.gguf_export --fused-path models/fused-bf16
    python -m forge.phase1.gguf_export --llama-cpp-path ~/llama.cpp
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from forge.logging_config import configure_logging, get_logger, start_run

log = get_logger(__name__)

SETUP_INSTRUCTIONS = """\
llama.cpp not found at {path}.

FORGE needs llama.cpp's convert_hf_to_gguf.py to export a GGUF of the fused
Qwen2.5-Coder weights (mlx_lm's own GGUF exporter doesn't support Qwen2 — see
this module's docstring). One-time setup:

    git clone https://github.com/ggerganov/llama.cpp {path}
    python3 -m venv {path}/.venv
    {path}/.venv/bin/pip install -r {path}/requirements.txt
    cmake -B {path}/build -DCMAKE_BUILD_TYPE=Release
    cmake --build {path}/build --config Release -j

Then re-run this script, or pass --llama-cpp-path if you cloned it elsewhere.
"""


def find_llama_cpp(explicit_path: Path | None) -> Path:
    candidate = explicit_path or Path(os.environ.get("LLAMA_CPP_PATH", Path.home() / "llama.cpp"))
    convert_script = candidate / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        raise RuntimeError(SETUP_INSTRUCTIONS.format(path=candidate))
    return candidate


def find_conversion_python(llama_cpp_path: Path) -> str:
    """Resolve the interpreter that has llama.cpp's own requirements
    installed (torch, sentencepiece, gguf), rather than trusting whatever
    `python3` PATH resolution happens to pick — which, run from inside
    FORGE's own activated .venv, resolves to FORGE's interpreter and never
    sees these packages. See this module's docstring for the concrete
    failure this caused. Falls back to plain "python3" with a warning if
    the dedicated venv wasn't set up (e.g. an older manual install)."""
    dedicated = llama_cpp_path / ".venv" / "bin" / "python3"
    if dedicated.exists():
        return str(dedicated)
    log.warning(
        "gguf_export.no_dedicated_venv",
        expected_path=str(dedicated),
        hint=f"python3 -m venv {llama_cpp_path}/.venv && "
        f"{llama_cpp_path}/.venv/bin/pip install -r {llama_cpp_path}/requirements.txt",
    )
    return "python3"


def sanitize_tokenizer_config(fused_path: Path) -> None:
    """Fix the extra_special_tokens list-vs-dict mismatch described in this
    module's docstring, in place. No-op if already a dict (e.g. a
    differently-sourced checkpoint that never had the bug)."""
    config_path = fused_path / "tokenizer_config.json"
    if not config_path.exists():
        return

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if isinstance(config.get("extra_special_tokens"), list):
        log.info(
            "gguf_export.sanitize_tokenizer_config",
            fused_path=str(fused_path),
            note="extra_special_tokens list -> {} (already present in tokenizer.json added_tokens)",
        )
        config["extra_special_tokens"] = {}
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def export_f16_gguf(llama_cpp_path: Path, fused_path: Path, out_file: Path) -> Path:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    convert_script = llama_cpp_path / "convert_hf_to_gguf.py"
    python = find_conversion_python(llama_cpp_path)
    sanitize_tokenizer_config(fused_path)

    log.info("gguf_export.start", fused_path=str(fused_path), out_file=str(out_file))
    start = time.monotonic()
    subprocess.run(
        [
            python,
            str(convert_script),
            str(fused_path),
            "--outfile",
            str(out_file),
            "--outtype",
            "f16",
        ],
        check=True,
    )
    elapsed_ms = (time.monotonic() - start) * 1000
    log.info("gguf_export.finish", out_file=str(out_file), latency_ms=round(elapsed_ms, 1))
    return out_file


def quantize_gguf(llama_cpp_path: Path, f16_gguf: Path, out_file: Path, quant_type: str) -> Path:
    """Optional: further quantize the F16 GGUF (e.g. Q8_0, Q4_K_M) for the
    Ollama arm's own quantization sweep, via the compiled llama-quantize
    binary. Skipped with a clear warning if the binary hasn't been built."""
    binary = llama_cpp_path / "build" / "bin" / "llama-quantize"
    if not binary.exists():
        log.warning(
            "gguf_quantize.skip_missing_binary",
            expected_path=str(binary),
            hint="cmake --build build --config Release -j",
        )
        return f16_gguf

    log.info("gguf_quantize.start", quant_type=quant_type, out_file=str(out_file))
    start = time.monotonic()
    subprocess.run([str(binary), str(f16_gguf), str(out_file), quant_type], check=True)
    elapsed_ms = (time.monotonic() - start) * 1000
    log.info("gguf_quantize.finish", out_file=str(out_file), latency_ms=round(elapsed_ms, 1))
    return out_file


def main() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fused-path", type=Path, default=Path("models/fused-bf16"))
    parser.add_argument("--out-dir", type=Path, default=Path("models/fused-gguf"))
    parser.add_argument("--llama-cpp-path", type=Path, default=None)
    parser.add_argument(
        "--quant-types",
        nargs="*",
        default=["Q8_0", "Q4_K_M"],
        help="Additional GGUF quant levels to produce from the F16 export, via llama-quantize.",
    )
    args = parser.parse_args()

    if not args.fused_path.exists():
        raise RuntimeError(f"{args.fused_path} does not exist — run forge.phase1.fuse first.")

    llama_cpp_path = find_llama_cpp(args.llama_cpp_path)
    run_id = start_run(log, phase="phase1.gguf_export")

    f16_path = args.out_dir / "model-f16.gguf"
    export_f16_gguf(llama_cpp_path, args.fused_path, f16_path)

    for quant_type in args.quant_types:
        out_file = args.out_dir / f"model-{quant_type}.gguf"
        quantize_gguf(llama_cpp_path, f16_path, out_file, quant_type)

    log.info("run.finish", run_id=run_id, phase="phase1.gguf_export")


if __name__ == "__main__":
    main()
