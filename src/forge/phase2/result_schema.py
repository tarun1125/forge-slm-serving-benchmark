"""The one result schema every arm writes to. Per this project's
validate-at-the-boundary convention: this is the single validated shape a
raw benchmark request's outcome takes, whatever arm produced it —
everything downstream (MLflow logging, percentile aggregation, cost
modeling in Phase 3) reads this shape, never a raw dict.

Granularity is per-REQUEST, not pre-aggregated: TTFT/ITL percentiles need
the full distribution across many requests, so aggregation happens at
analysis time over a collection of these rows (see metrics.py), not here.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from forge.hardware import HardwareInfo


class RequestResult(BaseModel):
    """One request's outcome against one arm, at one concurrency level, one
    prompt-length bucket, one quantization level."""

    run_id: str
    arm: str  # "mlx_lm" | "ollama" | "vllm_metal" | "vllm_cuda" | "hosted_api"
    model_variant: str  # e.g. "bf16", "8bit", "4bit", "q4_k_m", "groq-llama-70b"
    concurrency: int
    prompt_bucket: str  # "short" | "medium" | "long"
    case_id: str

    # Timing — see metrics.py for TTFT/ITL/throughput derivation from these.
    request_start_monotonic: float
    first_token_monotonic: float | None = None  # None only on total failure
    last_token_monotonic: float | None = None
    token_monotonics: list[float] = Field(default_factory=list)  # every token's arrival time

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    generated_text: str | None = None

    error: str | None = None  # None means success

    # Thermal (Mac arms) — logged per-request so drift correlates with time,
    # not with arm (see thermal.py's module docstring for the randomization
    # rationale this feeds into). No raw Celsius here deliberately: this
    # macOS version's powermetrics has no sampler that reports one (the
    # historical "smc" sampler is gone) — confirmed against real captured
    # output, not assumed. thermal_pressure_level is macOS's own
    # Nominal/Fair/Serious/Critical throttling judgment, arguably a more
    # authoritative signal than a raw temperature would have been anyway.
    thermal_pressure_level: str | None = None
    cpu_power_mw: float | None = None
    gpu_power_mw: float | None = None
    peak_memory_mb: float | None = None

    hardware: HardwareInfo

    @property
    def succeeded(self) -> bool:
        return self.error is None and self.first_token_monotonic is not None

    @property
    def ttft_ms(self) -> float | None:
        if self.first_token_monotonic is None:
            return None
        return (self.first_token_monotonic - self.request_start_monotonic) * 1000

    @property
    def total_latency_ms(self) -> float | None:
        if self.last_token_monotonic is None:
            return None
        return (self.last_token_monotonic - self.request_start_monotonic) * 1000

    @property
    def inter_token_latencies_ms(self) -> list[float]:
        """Per-gap ITL, not an average — feeds percentile computation across
        the whole sweep, not just this one request."""
        timestamps = self.token_monotonics
        if len(timestamps) < 2:
            return []
        return [(b - a) * 1000 for a, b in zip(timestamps, timestamps[1:], strict=False)]
