"""Arm registry. Every arm in the plan's Phase 2 table speaks the same
OpenAI-compatible chat-completions protocol — confirmed directly against
this machine's actual installs, not assumed:
  - mlx_lm:      `mlx_lm.server` ships an OpenAI-compatible HTTP server.
  - ollama:      native OpenAI-compatible endpoint at /v1/chat/completions.
  - vllm_metal:  `vllm serve <model>` — standard vLLM OpenAI-compatible server.
  - vllm_cuda:   the same `vllm serve`, on a rented NVIDIA GPU — this
                 process never launches it, it only points at it.
  - ollama_cloud: the same Ollama daemon as the local arm, on a rented CPU.
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
from importlib import metadata
from pathlib import Path

from forge.config import Settings
from forge.hardware import ServerHardware, ServerSoftware
from forge.logging_config import get_logger

log = get_logger(__name__)

# Relative to the server ROOT (base_url minus "/v1"), not to base_url — these
# are each server's own endpoints, outside the OpenAI-compatible surface.
OLLAMA_VERSION_PATH = "/api/version"
VLLM_VERSION_PATH = "/version"

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
    name: str  # "mlx_lm" | "ollama" | "vllm_metal" | "vllm_cuda" | "ollama_cloud" | "hosted_api"
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
    # Stack, version and settings of whatever served the request — see
    # hardware.ServerSoftware. version is usually None here and filled in by
    # server_lifecycle from version_path once the server is reachable.
    server_software: ServerSoftware | None = None
    version_path: str | None = None


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _stringify(settings: dict[str, str | int | float | bool] | None) -> dict[str, str]:
    return {k: str(v) for k, v in (settings or {}).items()}


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
        # mlx_lm.server has no version endpoint, but it runs from this venv,
        # so the installed package version is the served version.
        server_software=ServerSoftware(
            stack="mlx_lm.server",
            version=_package_version("mlx-lm"),
            settings={"decode_concurrency": "1", "prompt_concurrency": "1"},
        ),
    )


def ollama_arm(
    model_variant: str, port: int = 11434, settings: Settings | None = None
) -> ArmConfig:
    if settings is None:
        from forge.config import get_settings

        settings = get_settings()
    model_id = OLLAMA_MODEL_VARIANTS[model_variant]
    return ArmConfig(
        name="ollama",
        model_variant=model_variant,
        base_url=f"http://127.0.0.1:{port}/v1",
        model_id=model_id,
        launch_command=None,  # `ollama serve` is a persistent daemon, started once, not per-sweep
        server_software=ServerSoftware(
            stack="ollama", settings=_stringify(settings.ollama_server_env)
        ),
        version_path=OLLAMA_VERSION_PATH,
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
        # vllm-metal lives in its own venv, so its version comes from the
        # running server (VLLM_VERSION_PATH), not this process's packages.
        server_software=ServerSoftware(
            stack="vllm-metal",
            settings={
                "max_num_seqs": str(max_num_seqs),
                "gpu_memory_utilization": str(gpu_memory_utilization),
            },
        ),
        version_path=VLLM_VERSION_PATH,
    )


def _remote_arm(
    *,
    name: str,
    model_variant: str,
    model_id: str,
    base_url: str | None,
    processor: str | None,
    base_url_env: str,
    processor_env: str,
    processor_probe: str,
    server_software: ServerSoftware,
    version_path: str,
    api_key: str | None = None,
    provider: str | None = None,
    instance_type: str | None = None,
    region: str | None = None,
    processor_memory_gb: float | None = None,
    memory_bandwidth_gb_s: float | None = None,
    hourly_usd: float | None = None,
) -> ArmConfig:
    """The shape every arm shares when it serves from a machine this process
    can only reach over a socket. Factored out because both remote arms need
    the identical pair of refusals, and a near-copy of a guard is how the two
    copies drift.

    launch_command is always None here: the server is already running
    somewhere else, so ManagedServer treats these exactly as it treats Ollama
    and the hosted API — a no-op start and stop, plus a one-shot preflight
    (ManagedServer._preflight_remote) that refuses to start unless base_url +
    "/models" answers and lists model_id. A tunnel that drops mid-sweep still
    produces failed rows rather than an abort.

    Both refusals exist for the same reason. A missing base URL fails fast
    instead of at the first request; a missing processor name would produce
    rows that cannot say what hardware made them, on a machine that is about
    to be deleted. See hardware.ServerHardware.
    """
    if not base_url:
        raise RuntimeError(
            f"{base_url_env} is not set. Point it at the remote server as reachable from "
            "THIS machine — an SSH-forwarded local port in the documented setup, e.g. "
            "http://127.0.0.1:8001/v1. See .env.example and docs/cloud-arm.md."
        )
    if not processor:
        raise RuntimeError(
            f"{processor_env} is not set. This arm serves from another machine, so the "
            "result rows cannot say what hardware produced them unless you say so here — "
            f"run `{processor_probe}` on the VM. See hardware.ServerHardware and "
            "docs/cloud-arm.md."
        )
    if not server_software.settings:
        # A warning, not a refusal: the version is still detected, and a run
        # without settings is usable — just not provably like-for-like.
        log.warning(
            "arms.server_settings_unrecorded",
            arm=name,
            hint="set the *_SERVER_ENV / *_SERVER_ARGS JSON in .env — see .env.example",
        )
    return ArmConfig(
        name=name,
        model_variant=model_variant,
        base_url=base_url,
        model_id=model_id,
        api_key=api_key,
        launch_command=None,
        server_hardware=ServerHardware(
            processor=processor,
            provider=provider,
            instance_type=instance_type,
            region=region,
            processor_memory_gb=processor_memory_gb,
            memory_bandwidth_gb_s=memory_bandwidth_gb_s,
            hourly_usd=hourly_usd,
        ),
        server_software=server_software,
        version_path=version_path,
    )


def vllm_cuda_arm(model_variant: str, settings: Settings | None = None) -> ArmConfig:
    """The cloud-GPU arm — the fifth arm the plan called for and Phase 3 has
    so far only estimated (see docs/cost-model.md's scaling-factor row).

    bf16 only, and the caller can't ask for anything else. models/fused-4bit
    and fused-8bit are MLX affine-quantised (see their config.json's
    `"mode": "affine"`); vLLM on CUDA cannot load them, and the GGUF exports
    are a third format again. A CUDA-native quantisation would be new
    weights, which by this repo's rules means a new parity check and a new
    manifest entry — a separate phase, not a silently-accepted argument here.

    Everything else about serving from a rented machine — the refusals, the
    absent launch_command, the mandatory processor label — is in _remote_arm.
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

    return _remote_arm(
        name="vllm_cuda",
        model_variant=model_variant,
        model_id=settings.vllm_cuda_model_id or "forge-bf16",
        base_url=settings.vllm_cuda_base_url,
        processor=settings.vllm_cuda_gpu_name,
        base_url_env="VLLM_CUDA_BASE_URL",
        processor_env="VLLM_CUDA_GPU_NAME",
        processor_probe="nvidia-smi --query-gpu=name --format=csv,noheader",
        server_software=ServerSoftware(
            stack="vllm", settings=_stringify(settings.vllm_cuda_server_args)
        ),
        version_path=VLLM_VERSION_PATH,
        api_key=settings.vllm_cuda_api_key,
        provider=settings.vllm_cuda_provider,
        instance_type=settings.vllm_cuda_instance_type,
        region=settings.vllm_cuda_region,
        processor_memory_gb=settings.vllm_cuda_gpu_memory_gb,
        memory_bandwidth_gb_s=settings.vllm_cuda_gpu_memory_bandwidth_gb_s,
        hourly_usd=settings.vllm_cuda_hourly_usd,
    )


def ollama_cloud_arm(model_variant: str, settings: Settings | None = None) -> ArmConfig:
    """The cloud-CPU arm: the same Ollama daemon, the same Modelfile, the same
    registered tag as ollama_arm above — running on a rented commodity CPU
    instead of this laptop.

    That sameness is the entire design. Holding the serving software constant
    means the only variable between an `ollama` row and an `ollama_cloud` row
    is the hardware, which is a far cleaner comparison than swapping stack and
    machine at once. It is also why this needs its own arm NAME rather than
    being ollama_arm with a different base_url: sweep.py writes results to
    f"{arm}_{variant}_c{n}_{bucket}.jsonl", so reusing "ollama" would have the
    cloud run silently OVERWRITE the local results in results/sweep/, and
    score_accuracy.py groups by (arm, variant, bucket) and would merge the two
    populations into one number.

    Needs no GPU quota and no Pay-As-You-Go upgrade, which is why this is the
    arm to run first — see docs/cloud-arm.md on sequencing.

    model_id defaults to the local arm's tag for the same reason: register the
    GGUF on the VM with the repo's own ollama/Modelfile.q4 and the two arms
    are asking the identical daemon for the identical model.
    """
    if settings is None:
        from forge.config import get_settings

        settings = get_settings()

    if model_variant not in OLLAMA_MODEL_VARIANTS:
        raise ValueError(
            f"Unknown ollama_cloud variant {model_variant!r} "
            f"(expected one of {sorted(OLLAMA_MODEL_VARIANTS)})."
        )

    return _remote_arm(
        name="ollama_cloud",
        model_variant=model_variant,
        model_id=settings.ollama_cloud_model_id or OLLAMA_MODEL_VARIANTS[model_variant],
        base_url=settings.ollama_cloud_base_url,
        processor=settings.ollama_cloud_cpu_name,
        base_url_env="OLLAMA_CLOUD_BASE_URL",
        processor_env="OLLAMA_CLOUD_CPU_NAME",
        processor_probe="lscpu | grep 'Model name'",
        server_software=ServerSoftware(
            stack="ollama", settings=_stringify(settings.ollama_cloud_server_env)
        ),
        version_path=OLLAMA_VERSION_PATH,
        provider=settings.ollama_cloud_provider,
        instance_type=settings.ollama_cloud_instance_type,
        region=settings.ollama_cloud_region,
        processor_memory_gb=settings.ollama_cloud_memory_gb,
        memory_bandwidth_gb_s=settings.ollama_cloud_memory_bandwidth_gb_s,
        hourly_usd=settings.ollama_cloud_hourly_usd,
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
