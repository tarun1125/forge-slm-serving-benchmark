"""Hardware fingerprint, captured once per process and embedded in every
result file (models/MANIFEST.json, benchmark result rows, MLflow run tags).

FORGE's entire claim is hardware-conditional — a throughput or latency number
means nothing without the chip it was measured on, since Apple Silicon
generations differ enough in memory bandwidth and GPU core count to move the
result on their own. An unlabelled number is worthless — this module is the
single place that fingerprint gets produced so every artifact agrees.

Two fingerprints, not one. HardwareInfo describes THIS machine, which for
every local arm is also the machine that served the request. That stops being
true the moment an arm points at a remote server: the sweep still runs here,
so HardwareInfo still (correctly) says "Apple M5 Pro", but the inference
happened on something else entirely. ServerHardware is that other machine,
and it is None for exactly the arms where the client IS the server. Without
it a cloud row is indistinguishable from a local one in the saved data —
same arm code, same client fingerprint, different silicon — which is the
unlabelled-number failure this module exists to prevent.
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import asdict, dataclass, field
from functools import lru_cache


@dataclass(frozen=True)
class HardwareInfo:
    machine: str  # "arm64" or "x86_64" — the Rosetta tripwire from Phase 0
    macos_version: str
    chip: str
    cpu_core_count: int
    gpu_core_count: int | None
    unified_memory_gb: float | None


@dataclass(frozen=True)
class ServerHardware:
    """The machine that actually ran inference, when that isn't this one.

    `processor`, not `accelerator`: a cloud CPU arm serving GGUF through
    Ollama has no accelerator, and calling a Graviton4 one would be a small
    lie in a field whose entire job is to label honestly. It covers whatever
    actually ran the model — a GPU, or the CPU itself.

    It has no default, on purpose. Every other field here is genuinely
    optional metadata, but a remote row with no processor name is precisely
    the artifact this module's docstring calls worthless, so the type refuses
    to be constructed without one rather than quietly recording a null.
    memory_bandwidth_gb_s is called out separately from the rest because it
    is the independent variable in Phase 3's cloud scaling factor (see
    cost_model.cloud_gpu_cost_per_query) — capturing it per-row is what lets
    a later analysis test that factor instead of assuming it.
    """

    processor: str  # "NVIDIA A100 80GB PCIe", "Tesla T4", "AWS Graviton4"
    provider: str | None = None  # "azure" | "aws"
    instance_type: str | None = None  # "Standard_NC4as_T4_v3", "c8g.2xlarge"
    region: str | None = None
    # VRAM for a GPU, system RAM for a CPU VM — the memory the processor
    # above is actually working out of.
    processor_memory_gb: float | None = None
    memory_bandwidth_gb_s: float | None = None
    hourly_usd: float | None = None


@dataclass(frozen=True)
class ServerSoftware:
    """What served the request, as opposed to what it ran on.

    Lives beside ServerHardware because it answers the same question — could
    two rows differ for a reason the data doesn't show? — for the half of the
    answer hardware can't: the `ollama` and `ollama_cloud` arms exist to hold
    the serving stack constant, and that claim is only checkable if each row
    says which version ran and with which daemon settings (OLLAMA_NUM_PARALLEL
    alone decides whether concurrency 8 is batching or queueing).

    `version` is detected from the live server where it exposes one (Ollama's
    /api/version, vLLM's /version) — see server_lifecycle — rather than typed
    by hand. `settings` cannot be detected (neither server reports its own
    env), so for daemons this process doesn't launch it comes from .env; for
    servers it does launch, it is the launch flags themselves.
    """

    stack: str  # "ollama", "vllm", "vllm-metal", "mlx_lm.server"
    version: str | None = None
    settings: dict[str, str] = field(default_factory=dict)


def _sysctl(name: str) -> str | None:
    try:
        out = subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, timeout=5, check=True
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def _sw_vers() -> str:
    try:
        out = subprocess.run(
            ["sw_vers", "-productVersion"], capture_output=True, text=True, timeout=5, check=True
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def _chip_name() -> str:
    return _sysctl("machdep.cpu.brand_string") or "unknown"


def _gpu_core_count() -> int | None:
    """Apple Silicon exposes this via `system_profiler SPDisplaysDataType`,
    not sysctl. Best-effort — a missing value is reported as null, never
    guessed."""
    try:
        out = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    import json as _json

    try:
        data = _json.loads(out.stdout)
        for display in data.get("SPDisplaysDataType", []):
            cores = display.get("sppci_cores")
            if cores is not None:
                return int(cores)
    except (ValueError, KeyError, TypeError):
        pass
    return None


def _unified_memory_gb() -> float | None:
    raw = _sysctl("hw.memsize")
    if raw is None or not raw.isdigit():
        return None
    return round(int(raw) / (1024**3), 1)


@lru_cache(maxsize=1)
def get_hardware_info() -> HardwareInfo:
    total_cpu = _sysctl("hw.physicalcpu")
    return HardwareInfo(
        machine=platform.machine(),
        macos_version=_sw_vers(),
        chip=_chip_name(),
        cpu_core_count=int(total_cpu) if total_cpu and total_cpu.isdigit() else 0,
        gpu_core_count=_gpu_core_count(),
        unified_memory_gb=_unified_memory_gb(),
    )


def get_hardware_dict() -> dict:
    return asdict(get_hardware_info())


def assert_native_arm64() -> None:
    """Phase 0 gate: refuse to proceed under Rosetta."""
    machine = platform.machine()
    if machine != "arm64":
        raise RuntimeError(
            f"Detected machine={machine!r}, expected 'arm64'. You are running under Rosetta — "
            "vllm-metal will refuse this outright. Install a native arm64 Python 3.12 "
            "(e.g. via `arch -arm64 brew install python@3.12`, not the Rosetta-translated "
            "system Python) before continuing."
        )


if __name__ == "__main__":
    import json

    assert_native_arm64()
    print(json.dumps(get_hardware_dict(), indent=2))
