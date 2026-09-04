"""Hardware fingerprint, captured once per process and embedded in every
result file (models/MANIFEST.json, benchmark result rows, MLflow run tags).

FORGE's entire claim is hardware-conditional — a throughput or latency number
means nothing without the chip it was measured on, since Apple Silicon
generations differ enough in memory bandwidth and GPU core count to move the
result on their own. An unlabelled number is worthless — this module is the
single place that fingerprint gets produced so every artifact agrees.
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import asdict, dataclass
from functools import lru_cache


@dataclass(frozen=True)
class HardwareInfo:
    machine: str  # "arm64" or "x86_64" — the Rosetta tripwire from Phase 0
    macos_version: str
    chip: str
    cpu_core_count: int
    gpu_core_count: int | None
    unified_memory_gb: float | None


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
