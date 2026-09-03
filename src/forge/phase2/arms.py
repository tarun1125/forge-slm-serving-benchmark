"""Arm registry. Every arm in the plan's Phase 2 table speaks the same
OpenAI-compatible chat-completions protocol — confirmed directly against
this machine's actual installs, not assumed:
  - mlx_lm:      `mlx_lm.server` ships an OpenAI-compatible HTTP server.
  - ollama:      native OpenAI-compatible endpoint at /v1/chat/completions.
  - vllm_metal:  `vllm serve <model>` — standard vLLM OpenAI-compatible server.
  - hosted_api:  Groq and NVIDIA NIM are OpenAI-compatible by design.
One client (client.py) therefore serves every arm; ArmConfig is the only
thing that differs per arm.

A critical, easy-to-miss default: mlx_lm.server defaults to
--decode-concurrency 32 --prompt-concurrency 8, i.e. it DOES batch multiple
concurrent requests by default. The plan's "mlx_lm direct" arm is
specifically meant to isolate the no-batching baseline ("Raw single-stream
Apple Silicon baseline. No scheduler, no batching.") — running it with
mlx_lm.server's defaults would silently turn it into a second, cruder
continuous-batching arm and erase the exact comparison Phase 2 exists to
make (vLLM's batching advantage only shows up against a genuine
no-batching baseline). MLX_LM_ARGS below forces --decode-concurrency 1
--prompt-concurrency 1 for this reason — do not remove it to "simplify".
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

MODELS_DIR = Path("models")

# quantization-level name -> path/model-id, per arm. Kept separate per arm
# because each serving stack wants a different artifact format for "the
# same" quantization level (MLX safetensors dir vs. GGUF file vs. Ollama's
# own registered tag) — see docs/parity-check-design.md and
# ollama_register.py for why these are provably the same fine-tuned weights
# despite the different file formats.
MLX_MODEL_VARIANTS = {
    "bf16": MODELS_DIR / "fused-bf16",
    "8bit": MODELS_DIR / "fused-8bit",
    "4bit": MODELS_DIR / "fused-4bit",
}
OLLAMA_MODEL_VARIANTS = {
    "f16": "forge-qwen-coder-ft:f16",
    "q8": "forge-qwen-coder-ft:q8",
    "q4": "forge-qwen-coder-ft:q4",
}
# vllm-metal loads the same HF-format directories mlx_lm produces (fuse.py's
# save() writes standard HF safetensors + config.json — verified: this is
# the same directory llama.cpp's convert_hf_to_gguf.py successfully read).
VLLM_METAL_MODEL_VARIANTS = MLX_MODEL_VARIANTS


@dataclass(frozen=True)
class ArmConfig:
    name: str  # "mlx_lm" | "ollama" | "vllm_metal" | "hosted_api"
    model_variant: str  # e.g. "bf16", "q4", "groq-llama-70b" — arm-specific label
    base_url: str
    model_id: str  # the string sent as `model` in the chat-completions request
    api_key: str | None = None
    launch_command: list[str] | None = None  # None => already running (Ollama, hosted API)
    # Relative to base_url, which already ends in /v1 (see mlx_lm_arm/ollama_arm/
    # vllm_metal_arm below) — "/models", not "/v1/models", or server_lifecycle.py's
    # health check hits .../v1/v1/models and 404s. Caught live against a real
    # mlx_lm.server before this default shipped anywhere it could bite silently.
    health_check_path: str = "/models"
    extra_request_params: dict = field(default_factory=dict)


def mlx_lm_arm(model_variant: str, port: int = 8080) -> ArmConfig:
    model_path = MLX_MODEL_VARIANTS[model_variant]
    return ArmConfig(
        name="mlx_lm",
        model_variant=model_variant,
        base_url=f"http://127.0.0.1:{port}/v1",
        model_id=str(model_path),
        launch_command=[
            "mlx_lm.server",
            "--model",
            str(model_path),
            "--port",
            str(port),
            "--decode-concurrency",
            "1",  # forces the true no-batching baseline — see module docstring
            "--prompt-concurrency",
            "1",
        ],
    )


def ollama_arm(model_variant: str, port: int = 11434) -> ArmConfig:
    model_id = OLLAMA_MODEL_VARIANTS[model_variant]
    return ArmConfig(
        name="ollama",
        model_variant=model_variant,
        base_url=f"http://127.0.0.1:{port}/v1",
        model_id=model_id,
        launch_command=None,  # `ollama serve` is a persistent daemon, started once, not per-sweep
    )


def vllm_metal_arm(
    model_variant: str,
    port: int = 8000,
    max_num_seqs: int = 256,
    gpu_memory_utilization: float = 0.9,
    venv_path: Path | None = None,
) -> ArmConfig:
    model_path = VLLM_METAL_MODEL_VARIANTS[model_variant]
    # Default resolved here, not in the signature — Path.home() as a default
    # argument value is evaluated once at import time, not per call.
    vllm_binary = (venv_path or Path.home() / ".venv-vllm-metal") / "bin" / "vllm"
    return ArmConfig(
        name="vllm_metal",
        model_variant=model_variant,
        base_url=f"http://127.0.0.1:{port}/v1",
        model_id=str(model_path),
        launch_command=[
            str(vllm_binary),
            "serve",
            str(model_path),
            "--port",
            str(port),
            "--max-num-seqs",
            str(max_num_seqs),
            "--gpu-memory-utilization",
            str(gpu_memory_utilization),
        ],
        extra_request_params={"max_num_seqs": max_num_seqs},
    )


def hosted_api_arm(provider: str) -> ArmConfig:
    """provider: "groq" | "nim". Requires GROQ_API_KEY / NIM_API_KEY in the
    environment — raises loudly if missing rather than silently skipping
    the arm, per AGENTS.md's "no raw dicts / validate at the boundary" spirit
    applied to config as much as request bodies."""
    if provider == "groq":
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set — see .env.example")
        return ArmConfig(
            name="hosted_api",
            model_variant="groq",
            base_url="https://api.groq.com/openai/v1",
            model_id="llama-3.3-70b-versatile",
            api_key=api_key,
            launch_command=None,
        )
    if provider == "nim":
        api_key = os.environ.get("NIM_API_KEY")
        if not api_key:
            raise RuntimeError("NIM_API_KEY not set — see .env.example")
        return ArmConfig(
            name="hosted_api",
            model_variant="nim",
            base_url="https://integrate.api.nvidia.com/v1",
            model_id="meta/llama-3.1-70b-instruct",
            api_key=api_key,
            launch_command=None,
        )
    raise ValueError(f"Unknown hosted API provider: {provider!r} (expected 'groq' or 'nim')")
