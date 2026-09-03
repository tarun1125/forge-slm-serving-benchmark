import pytest

from forge.phase3.cost_model import (
    CostAssumptions,
    cloud_gpu_cost_per_query,
    cost_per_accuracy_point,
    find_break_even_volume,
    hosted_api_cost_per_query,
    local_cost_per_query,
)


def _assumptions(**overrides) -> CostAssumptions:
    defaults = dict(
        mac_hardware_cost_inr=249_900.0,
        mac_useful_life_years=3.0,
        mac_power_draw_watts=75.0,
        electricity_tariff_inr_per_kwh=5.5,
        cloud_gpu_hourly_usd=1.39,
        cloud_gpu_throughput_scaling_factor=6.64,
        groq_input_usd_per_1m_tokens=0.15,
        groq_output_usd_per_1m_tokens=0.60,
        usd_to_inr=88.0,
    )
    defaults.update(overrides)
    return CostAssumptions(**defaults)


class TestLocalCostPerQuery:
    def test_rejects_nonpositive_throughput(self):
        with pytest.raises(ValueError):
            local_cost_per_query(_assumptions(), 0, 100, 1000)

    def test_rejects_nonpositive_completion_tokens(self):
        with pytest.raises(ValueError):
            local_cost_per_query(_assumptions(), 100, 0, 1000)

    def test_cost_per_query_declines_as_volume_rises(self):
        a = _assumptions()
        low = local_cost_per_query(
            a, throughput_tokens_per_sec=100, avg_completion_tokens=200, monthly_query_volume=100
        )
        high = local_cost_per_query(
            a,
            throughput_tokens_per_sec=100,
            avg_completion_tokens=200,
            monthly_query_volume=100_000,
        )
        assert high.total_cost_inr < low.total_cost_inr

    def test_utilization_rises_toward_one_as_volume_approaches_capacity(self):
        a = _assumptions(mac_useful_life_years=1.0)
        # throughput=100 tok/s, 200 tok/query -> 0.5 queries/sec -> capacity over 1 year:
        capacity_per_month = 100 / 200 * 3600 * 24 * 365 / 12
        result = local_cost_per_query(
            a,
            throughput_tokens_per_sec=100,
            avg_completion_tokens=200,
            monthly_query_volume=int(capacity_per_month),
        )
        assert result.utilization == pytest.approx(1.0, abs=0.01)

    def test_utilization_capped_at_one_beyond_capacity(self):
        a = _assumptions(mac_useful_life_years=1.0)
        capacity_per_month = 100 / 200 * 3600 * 24 * 365 / 12
        result = local_cost_per_query(
            a,
            throughput_tokens_per_sec=100,
            avg_completion_tokens=200,
            monthly_query_volume=int(capacity_per_month * 10),  # way beyond capacity
        )
        assert result.utilization == 1.0

    def test_electricity_cost_is_independent_of_volume(self):
        a = _assumptions()
        low = local_cost_per_query(a, 100, 200, 100)
        high = local_cost_per_query(a, 100, 200, 100_000)
        assert low.electricity_cost_inr == pytest.approx(high.electricity_cost_inr)

    def test_total_is_sum_of_parts(self):
        a = _assumptions()
        r = local_cost_per_query(a, 100, 200, 5000)
        assert r.total_cost_inr == pytest.approx(
            r.amortized_hardware_cost_inr + r.electricity_cost_inr
        )


class TestHostedApiCostPerQuery:
    def test_scales_linearly_with_tokens(self):
        a = _assumptions()
        base = hosted_api_cost_per_query(a, avg_prompt_tokens=1000, avg_completion_tokens=1000)
        doubled = hosted_api_cost_per_query(a, avg_prompt_tokens=2000, avg_completion_tokens=2000)
        assert doubled == pytest.approx(base * 2)

    def test_matches_hand_computed_value(self):
        a = _assumptions(
            groq_input_usd_per_1m_tokens=1.0, groq_output_usd_per_1m_tokens=2.0, usd_to_inr=100.0
        )
        # 1000 prompt @ $1/1M = $0.001; 500 completion @ $2/1M = $0.001; total $0.002 -> ₹0.20
        cost = hosted_api_cost_per_query(a, avg_prompt_tokens=1000, avg_completion_tokens=500)
        assert cost == pytest.approx(0.20)


class TestCloudGpuCostPerQuery:
    def test_rejects_nonpositive_utilization(self):
        with pytest.raises(ValueError):
            cloud_gpu_cost_per_query(_assumptions(), 100, 200, 0)

    def test_lower_utilization_costs_more_per_query(self):
        a = _assumptions()
        low_util = cloud_gpu_cost_per_query(a, 100, 200, gpu_utilization=0.1)
        high_util = cloud_gpu_cost_per_query(a, 100, 200, gpu_utilization=1.0)
        assert low_util > high_util
        # the plan's own "10% utilization costs 10x" rule, exactly:
        assert low_util == pytest.approx(high_util * 10, rel=1e-9)


class TestFindBreakEvenVolume:
    def test_finds_a_crossover_when_one_exists(self):
        a = _assumptions()
        grid = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
        result = find_break_even_volume(
            a,
            throughput_tokens_per_sec=500,
            avg_completion_tokens=150,
            avg_prompt_tokens=1200,
            volume_grid=grid,
        )
        assert result is not None
        assert result in grid

    def test_returns_none_when_local_never_catches_up_in_grid(self):
        # Absurdly expensive hardware relative to a tiny grid ensures no crossover.
        a = _assumptions(mac_hardware_cost_inr=10_000_000_000.0)
        result = find_break_even_volume(
            a,
            throughput_tokens_per_sec=500,
            avg_completion_tokens=150,
            avg_prompt_tokens=1200,
            volume_grid=[100, 1000],
        )
        assert result is None


class TestCostPerAccuracyPoint:
    def test_none_when_accuracy_is_zero(self):
        assert cost_per_accuracy_point(cost_per_query_inr=1.0, execution_accuracy=0.0) is None

    def test_divides_cost_by_accuracy(self):
        result = cost_per_accuracy_point(cost_per_query_inr=1.0, execution_accuracy=0.5)
        assert result == pytest.approx(2.0)
