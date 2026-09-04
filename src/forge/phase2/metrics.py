"""Aggregation over a sweep's raw RequestResults into Phase 2's metrics table.
TTFT and inter-token latency are ALWAYS reported separately — never blended
into one "latency" number. That separation is the headline finding this
whole project is built to surface, so it is not a formatting nicety here:
there is no function in this module that combines them into a single number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from forge.phase2.result_schema import RequestResult


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile — no interpolation, so a returned value is
    always one that was actually observed (defensible in an interview:
    "what does your p95 mean" has a literal, not interpolated, answer)."""
    if not values:
        raise ValueError("percentile() of an empty list is undefined")
    if not 0 <= p <= 100:
        raise ValueError(f"p must be in [0, 100], got {p}")
    ordered = sorted(values)
    rank = math.ceil(p / 100 * len(ordered))
    index = max(0, min(len(ordered) - 1, rank - 1))
    return ordered[index]


@dataclass(frozen=True)
class LatencyPercentiles:
    p50: float
    p95: float
    p99: float
    n: int


def latency_percentiles(values: list[float]) -> LatencyPercentiles | None:
    if not values:
        return None
    return LatencyPercentiles(
        p50=percentile(values, 50),
        p95=percentile(values, 95),
        p99=percentile(values, 99),
        n=len(values),
    )


@dataclass(frozen=True)
class ArmMetrics:
    """One aggregation cell: a fixed (arm, model_variant, concurrency,
    prompt_bucket) group, per Phase 2's sweep dimensions."""

    arm: str
    model_variant: str
    concurrency: int
    prompt_bucket: str

    ttft_ms: LatencyPercentiles | None
    inter_token_latency_ms: LatencyPercentiles | None
    throughput_tokens_per_sec: float | None
    n_requests: int
    n_succeeded: int
    n_failed: int
    execution_accuracy: float | None  # fraction correct, None if not scored


def aggregate(results: list[RequestResult]) -> ArmMetrics:
    """Aggregate a single (arm, model_variant, concurrency, prompt_bucket)
    cell's raw requests. Caller is responsible for grouping — this function
    does not group across cells, since silently mixing cells would produce
    a percentile that answers no real question (see module docstring)."""
    if not results:
        raise ValueError("aggregate() of an empty result list is undefined")

    keys = {(r.arm, r.model_variant, r.concurrency, r.prompt_bucket) for r in results}
    if len(keys) > 1:
        raise ValueError(
            f"aggregate() received results from {len(keys)} different cells: {keys}. "
            "Group by (arm, model_variant, concurrency, prompt_bucket) before aggregating."
        )
    arm, model_variant, concurrency, prompt_bucket = keys.pop()

    succeeded = [r for r in results if r.succeeded]
    failed = [r for r in results if not r.succeeded]

    ttft_values = [r.ttft_ms for r in succeeded if r.ttft_ms is not None]
    itl_values = [gap for r in succeeded for gap in r.inter_token_latencies_ms]

    total_completion_tokens = sum(r.completion_tokens or 0 for r in succeeded)
    wall_clock_span = _wall_clock_span_seconds(succeeded)
    throughput = (
        total_completion_tokens / wall_clock_span
        if wall_clock_span and wall_clock_span > 0
        else None
    )

    return ArmMetrics(
        arm=arm,
        model_variant=model_variant,
        concurrency=concurrency,
        prompt_bucket=prompt_bucket,
        ttft_ms=latency_percentiles(ttft_values),
        inter_token_latency_ms=latency_percentiles(itl_values),
        throughput_tokens_per_sec=throughput,
        n_requests=len(results),
        n_succeeded=len(succeeded),
        n_failed=len(failed),
        execution_accuracy=None,  # filled in by the accuracy-scoring pass, not here
    )


def _wall_clock_span_seconds(results: list[RequestResult]) -> float | None:
    """Total tokens / wall-clock span across concurrent requests, NOT the
    sum of individual request durations — the latter would double-count
    time that concurrent requests spent overlapping, understating
    throughput exactly where concurrency is supposed to help."""
    starts = [r.request_start_monotonic for r in results]
    ends = [r.last_token_monotonic for r in results if r.last_token_monotonic is not None]
    if not starts or not ends:
        return None
    return max(ends) - min(starts)
