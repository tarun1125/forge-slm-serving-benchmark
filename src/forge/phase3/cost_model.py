"""The cost model — Phase 3's core deliverable. Per the plan, this is
explicitly the piece where "every assumption is arguable," so every number
below either traces to something measured in Phase 2 or is cited from a
real, dated source — nothing here is invented. See docs/cost-model.md for
the full assumptions table with sources; this module is the arithmetic on
top of those assumptions, kept separate and unit-tested because the
arithmetic itself (amortization, break-even) is easy to get subtly wrong
even when every input is correct.

The core idea, stated once here rather than re-derived at each call site:
self-hosting's cost per query is NOT constant — it falls as query volume
rises, because the same fixed hardware cost gets spread over more queries.
A hosted API's cost per query IS constant (pure pay-per-token). That's what
makes a break-even CURVE the right artifact rather than a single number:
the crossover point is wherever the declining local curve meets the flat
hosted-API line, and that point moves with every assumption — utilization,
hardware price, amortization window — which is exactly why the plan calls
for presenting a curve, not a number.
"""

from __future__ import annotations

from dataclasses import dataclass

HOURS_PER_YEAR = 365 * 24
SECONDS_PER_HOUR = 3600


@dataclass(frozen=True)
class CostAssumptions:
    """Every field here is a claim that should be independently checkable —
    see docs/cost-model.md for the citation behind each one. Defaults are
    the values actually used in the shipped analysis, not placeholders."""

    # --- Local (Mac) ---
    mac_hardware_cost_inr: float  # 14" MacBook Pro M5 Pro, 24GB/1TB, official Apple India price
    mac_useful_life_years: float  # amortization window — a stated assumption, not measured
    mac_power_draw_watts: float  # sustained-inference power draw estimate
    electricity_tariff_inr_per_kwh: float  # Indian residential average

    # --- Cloud GPU (estimated — the vLLM-on-CUDA arm was deferred, never measured) ---
    cloud_gpu_hourly_usd: float
    cloud_gpu_throughput_scaling_factor: float  # applied to a measured vllm_metal throughput number

    # --- Hosted API (Groq, gpt-oss-120b) ---
    groq_input_usd_per_1m_tokens: float
    groq_output_usd_per_1m_tokens: float

    # --- FX ---
    usd_to_inr: float


@dataclass(frozen=True)
class LocalCostBreakdown:
    monthly_query_volume: int
    utilization: float  # fraction of the machine's useful life spent serving this workload
    amortized_hardware_cost_inr: float  # per query
    electricity_cost_inr: float  # per query
    total_cost_inr: float  # per query


def local_cost_per_query(
    assumptions: CostAssumptions,
    throughput_tokens_per_sec: float,
    avg_completion_tokens: float,
    monthly_query_volume: int,
) -> LocalCostBreakdown:
    """The declining-cost curve: as monthly_query_volume rises, the same
    fixed hardware_cost_inr gets spread over more total queries served over
    the machine's life, so amortized_hardware_cost_inr falls roughly as
    1/volume. electricity_cost_inr is the marginal cost of one query's
    worth of active compute time and stays roughly flat with volume (idle
    time is assumed to draw negligible power for this purpose — a stated
    simplification, not a measurement, see docs/cost-model.md)."""
    if throughput_tokens_per_sec <= 0 or avg_completion_tokens <= 0:
        raise ValueError("throughput and avg_completion_tokens must be positive")

    lifetime_hours = assumptions.mac_useful_life_years * HOURS_PER_YEAR
    lifetime_months = assumptions.mac_useful_life_years * 12
    max_lifetime_queries = (
        throughput_tokens_per_sec * lifetime_hours * SECONDS_PER_HOUR / avg_completion_tokens
    )
    total_queries_over_life = monthly_query_volume * lifetime_months

    utilization = (
        min(total_queries_over_life / max_lifetime_queries, 1.0)
        if max_lifetime_queries > 0
        else 0.0
    )
    # Capped at the machine's actual lifetime capacity: demand beyond that
    # isn't servable by one machine, not modeled as "free" extra queries.
    servable_queries = min(total_queries_over_life, max_lifetime_queries)

    amortized_hardware_cost = (
        assumptions.mac_hardware_cost_inr / servable_queries
        if servable_queries > 0
        else float("inf")
    )

    seconds_per_query = avg_completion_tokens / throughput_tokens_per_sec
    electricity_cost = (
        (assumptions.mac_power_draw_watts / 1000)
        * assumptions.electricity_tariff_inr_per_kwh
        * (seconds_per_query / SECONDS_PER_HOUR)
    )

    return LocalCostBreakdown(
        monthly_query_volume=monthly_query_volume,
        utilization=utilization,
        amortized_hardware_cost_inr=amortized_hardware_cost,
        electricity_cost_inr=electricity_cost,
        total_cost_inr=amortized_hardware_cost + electricity_cost,
    )


def hosted_api_cost_per_query(
    assumptions: CostAssumptions,
    avg_prompt_tokens: float,
    avg_completion_tokens: float,
) -> float:
    """Pure pay-per-token — no utilization dependence, no amortization,
    which is exactly why this is a flat line against local's declining
    curve rather than another curve of its own."""
    cost_usd = (avg_prompt_tokens / 1_000_000) * assumptions.groq_input_usd_per_1m_tokens + (
        avg_completion_tokens / 1_000_000
    ) * assumptions.groq_output_usd_per_1m_tokens
    return cost_usd * assumptions.usd_to_inr


def cloud_gpu_cost_per_query(
    assumptions: CostAssumptions,
    measured_local_throughput_tokens_per_sec: float,
    avg_completion_tokens: float,
    gpu_utilization: float,
) -> float:
    """Estimated, not measured — the vLLM-on-CUDA arm was deferred (see
    docs/cost-model.md for why). Throughput is extrapolated from a REAL
    measured vllm_metal number via cloud_gpu_throughput_scaling_factor
    (the A100/M5-Pro memory-bandwidth ratio — decode is memory-bandwidth-
    bound, per Phase 2's own headline finding, so this is a principled
    scaling, not an arbitrary multiplier, but it is still an estimate, and
    the least certain number in this whole model). gpu_utilization applies
    the plan's own "an endpoint at 10% utilization costs 10x per query"
    rule directly: the same $/hour is spread over fewer served queries."""
    if gpu_utilization <= 0:
        raise ValueError("gpu_utilization must be positive")
    estimated_throughput = (
        measured_local_throughput_tokens_per_sec * assumptions.cloud_gpu_throughput_scaling_factor
    )
    queries_per_hour_at_full_utilization = (
        estimated_throughput * SECONDS_PER_HOUR / avg_completion_tokens
    )
    effective_queries_per_hour = queries_per_hour_at_full_utilization * gpu_utilization
    cost_per_hour_usd = assumptions.cloud_gpu_hourly_usd
    cost_per_query_usd = cost_per_hour_usd / effective_queries_per_hour
    return cost_per_query_usd * assumptions.usd_to_inr


def find_break_even_volume(
    assumptions: CostAssumptions,
    throughput_tokens_per_sec: float,
    avg_completion_tokens: float,
    hosted_cost_per_query_inr: float,
    volume_grid: list[int],
) -> int | None:
    """Returns the smallest monthly_query_volume in volume_grid at which
    local_cost_per_query <= hosted_cost_per_query_inr — the break-even
    point. None if local never catches up within the given grid (raise the
    upper bound rather than assume it always exists).

    Takes a PRE-COMPUTED hosted_cost_per_query_inr rather than prompt/
    completion token counts to recompute it from — an earlier version took
    avg_prompt_tokens too and called hosted_api_cost_per_query() internally,
    which silently used the LOCAL model's own token counts to price the
    HOSTED side. Caught by hand-checking a break-even result against a
    direct calculation: the two disagreed (this function said the crossover
    was at 1,000,000/month for vllm_metal 4-bit; recomputing hosted cost
    correctly using Groq's own real measured tokens — 1469 prompt / 356
    completion, very different from the local model's 1408/75 — moved it to
    300,000). The caller now computes the hosted baseline once, correctly,
    and passes the single number in — impossible to accidentally cross the
    streams between two different workloads' token counts this way."""
    for volume in sorted(volume_grid):
        local = local_cost_per_query(
            assumptions, throughput_tokens_per_sec, avg_completion_tokens, volume
        )
        if local.total_cost_inr <= hosted_cost_per_query_inr:
            return volume
    return None


def cost_per_accuracy_point(cost_per_query_inr: float, execution_accuracy: float) -> float | None:
    """cost_per_query / accuracy — "what does one percentage point of
    correctness cost." None (not a divide-by-zero crash) when accuracy is
    literally 0, which the short prompt bucket's real data hits — an
    infinite cost-per-accuracy-point is a genuine, reportable result for
    that condition, not an error to hide."""
    if execution_accuracy <= 0:
        return None
    return cost_per_query_inr / execution_accuracy
