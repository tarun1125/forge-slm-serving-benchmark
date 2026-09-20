"""Arm registry. Every arm in the plan's Phase 2 table speaks the same
OpenAI-compatible chat-completions protocol — confirmed directly against
this machine's actual installs, not assumed:
  - mlx_lm:      `mlx_lm.server` ships an OpenAI-compatible HTTP server.
  - ollama:      native OpenAI-compatible endpoint at /v1/chat/completions.
  - vllm_metal:  `vllm serve <model>` — standard vLLM OpenAI-compatible server.
  - vllm_cuda:   the same `vllm serve`, on a rented NVIDIA GPU — this
                 process never launches it, it only points at it.
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

from dataclasses import dataclass, field
from pathlib import Path

from forge.config import Settings
from forge.hardware import ServerHardware

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
    name: str  # "mlx_lm" | "ollama" | "vllm_metal" | "vllm_cuda" | "hosted_api"
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
    # None means "the client machine IS the serving machine" — true for every
    # local arm, and the reason this isn't just a string with a "local"
    # sentinel. Set only for arms that serve from somewhere else; client.py
    # copies it onto every RequestResult. See hardware.ServerHardware.
    server_hardware: ServerHardware | None = None


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


def vllm_cuda_arm(model_variant: str, settings: Settings | None = None) -> ArmConfig:
    """The cloud-GPU arm — the fifth arm the plan called for and Phase 3 has
    so far only estimated (see docs/cost-model.md's scaling-factor row).

    Three things make this different from vllm_metal_arm even though both
    run the same `vllm serve`:

    launch_command is None. The server is already running on a rented VM,
    so ManagedServer treats this exactly like Ollama or a hosted API: a
    no-op start/stop. That also means nothing health-checks the endpoint
    before the sweep starts — curl base_url + "/models" yourself first, or
    a dead tunnel produces a full run of failed rows rather than an error.

    bf16 only, and the caller can't ask for anything else. models/fused-4bit
    and fused-8bit are MLX affine-quantised (see their config.json's
    `"mode": "affine"`); vLLM on CUDA cannot load them, and the GGUF exports
    are a third format again. A CUDA-native quantisation would be new
    weights, which by this repo's rules means a new parity check and a new
    manifest entry — a separate phase, not a silently-accepted argument here.

    server_hardware is mandatory. The sweep runs on the Mac, so
    client.py's own get_hardware_info() will (correctly) stamp "Apple M5 Pro"
    onto every row this arm produces; without the serving side recorded
    alongside it, a cloud row and a local row are indistinguishable in the
    saved data. Missing VLLM_CUDA_GPU_NAME raises rather than defaulting to
    None, for the same reason require_capstone_repo() raises rather than
    guessing a path.
    """
    if model_variant != "bf16":
        raise ValueError(
            f"vllm_cuda only serves 'bf16', got {model_variant!r}. The 4bit/8bit "
            "variants are MLX affine-quantised and cannot be loaded by vLLM on CUDA — "
            "see this function's docstring."
        )
    if settings is None:
        from forge.config import get_settings

        settings = get_settings()

    if not settings.vllm_cuda_base_url:
        raise RuntimeError(
            "VLLM_CUDA_BASE_URL is not set. Point it at the remote vLLM server as "
            "reachable from this machine — an SSH-forwarded local port in the "
            "documented setup, e.g. http://127.0.0.1:8001/v1. See .env.example and "
            "docs/cloud-arm.md."
        )
    if not settings.vllm_cuda_gpu_name:
        raise RuntimeError(
            "VLLM_CUDA_GPU_NAME is not set. This arm serves from another machine, so "
            "the result rows cannot say what hardware produced them unless you say so "
            "here — run `nvidia-smi --query-gpu=name --format=csv,noheader` on the VM. "
            "See hardware.ServerHardware and docs/cloud-arm.md."
        )

    return ArmConfig(
        name="vllm_cuda",
        model_variant=model_variant,
        base_url=settings.vllm_cuda_base_url,
        model_id=settings.vllm_cuda_model_id or "forge-bf16",
        api_key=settings.vllm_cuda_api_key,
        launch_command=None,
        server_hardware=ServerHardware(
            accelerator=settings.vllm_cuda_gpu_name,
            provider=settings.vllm_cuda_provider,
            instance_type=settings.vllm_cuda_instance_type,
            region=settings.vllm_cuda_region,
            accelerator_memory_gb=settings.vllm_cuda_gpu_memory_gb,
            memory_bandwidth_gb_s=settings.vllm_cuda_gpu_memory_bandwidth_gb_s,
            hourly_usd=settings.vllm_cuda_hourly_usd,
        ),
    )


def hosted_api_arm(provider: str, settings: Settings | None = None) -> ArmConfig:
    """provider: "groq" | "nim". Requires GROQ_API_KEY / NIM_API_KEY in
    Settings (from .env) — raises loudly if missing rather than silently
    skipping the arm, applying this project's "no raw dicts / validate at
    the boundary" convention to config as much as request bodies. Reads
    through forge.config.Settings, not os.environ directly — an earlier
    version read os.environ.get(...) here, which this module's own
    docstring already said not to do, and never actually saw the key at
    runtime (pydantic-settings loads .env into Settings' own fields, not
    into the process environment — os.environ.get("GROQ_API_KEY") was
    silently None the whole time this went unnoticed).

    model_id was originally llama-3.3-70b-versatile per the plan's own
    resume-entry text — confirmed dead against the real key: a live call
    returns 404 model_not_found, and client.models.list() shows no Llama
    models on this account at all (Groq appears to have moved that model,
    and Llama models generally, to enterprise-only access at some point
    before this was checked). openai/gpt-oss-120b is the largest
    general-purpose model this key can actually reach, confirmed via
    models.list(), with confirmed real pricing ($0.15/$0.60 per 1M input/
    output tokens) — the right fit for "the big hosted model you'd reach
    for instead of self-hosting," not a downgrade in spirit even though
    the specific model changed. Re-check client.models.list() before
    trusting this if it's been a while — this account's available models
    already changed once mid-project.
    """
    if settings is None:
        from forge.config import get_settings

        settings = get_settings()

    if provider == "groq":
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY not set — see .env.example")
        return ArmConfig(
            name="hosted_api",
            model_variant="groq",
            base_url="https://api.groq.com/openai/v1",
            model_id="openai/gpt-oss-120b",
            api_key=settings.groq_api_key,
            launch_command=None,
        )
    if provider == "nim":
        if not settings.nim_api_key:
            raise RuntimeError("NIM_API_KEY not set — see .env.example")
        return ArmConfig(
            name="hosted_api",
            model_variant="nim",
            base_url="https://integrate.api.nvidia.com/v1",
            model_id="meta/llama-3.1-70b-instruct",
            api_key=settings.nim_api_key,
            launch_command=None,
        )
    raise ValueError(f"Unknown hosted API provider: {provider!r} (expected 'groq' or 'nim')")
