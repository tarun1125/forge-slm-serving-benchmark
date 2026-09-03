import pytest

from forge.hardware import get_hardware_info
from forge.phase2.metrics import aggregate, latency_percentiles, percentile
from forge.phase2.result_schema import RequestResult


def _result(
    start: float,
    first_token: float | None,
    last_token: float | None,
    token_monotonics: list[float] | None = None,
    completion_tokens: int = 0,
    error: str | None = None,
    arm: str = "mlx_lm",
    model_variant: str = "bf16",
    concurrency: int = 1,
    prompt_bucket: str = "short",
) -> RequestResult:
    return RequestResult(
        run_id="test-run",
        arm=arm,
        model_variant=model_variant,
        concurrency=concurrency,
        prompt_bucket=prompt_bucket,
        case_id="c",
        request_start_monotonic=start,
        first_token_monotonic=first_token,
        last_token_monotonic=last_token,
        token_monotonics=token_monotonics or [],
        completion_tokens=completion_tokens,
        error=error,
        hardware=get_hardware_info(),
    )


class TestPercentile:
    def test_p50_of_odd_length_is_the_middle_value(self):
        assert percentile([1, 2, 3, 4, 5], 50) == 3

    def test_nearest_rank_never_interpolates(self):
        # nearest-rank p95 of 10 values is the 10th (last) value, not an
        # interpolated point between the 9th and 10th.
        values = list(range(1, 11))  # 1..10
        assert percentile(values, 95) in values

    def test_p100_is_the_max(self):
        assert percentile([5, 1, 3], 100) == 5

    def test_p0_is_the_min(self):
        assert percentile([5, 1, 3], 0) == 1

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            percentile([], 50)

    def test_out_of_range_p_raises(self):
        with pytest.raises(ValueError):
            percentile([1, 2, 3], 101)


class TestLatencyPercentiles:
    def test_empty_returns_none(self):
        assert latency_percentiles([]) is None

    def test_reports_n(self):
        result = latency_percentiles([1, 2, 3])
        assert result is not None
        assert result.n == 3


class TestAggregate:
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            aggregate([])

    def test_mixed_cells_raises(self):
        results = [
            _result(0, 0.1, 0.5, arm="mlx_lm"),
            _result(0, 0.1, 0.5, arm="ollama"),
        ]
        with pytest.raises(ValueError, match="different cells"):
            aggregate(results)

    def test_counts_success_and_failure_separately(self):
        results = [
            _result(0, 0.1, 0.5, completion_tokens=10),
            _result(0, None, None, error="timeout"),
        ]
        metrics = aggregate(results)
        assert metrics.n_requests == 2
        assert metrics.n_succeeded == 1
        assert metrics.n_failed == 1

    def test_ttft_percentiles_only_use_succeeded_requests(self):
        results = [
            _result(0, 0.1, 0.5, completion_tokens=5),
            _result(0, 0.2, 0.6, completion_tokens=5),
            _result(0, None, None, error="timeout"),
        ]
        metrics = aggregate(results)
        assert metrics.ttft_ms is not None
        assert metrics.ttft_ms.n == 2  # the failed request contributes no TTFT

    def test_throughput_uses_wall_clock_span_not_summed_durations(self):
        # Two fully-overlapping concurrent requests, each running 0..1s,
        # each producing 10 tokens. Summing individual durations would give
        # 20 tokens / 2s = 10 tok/s; the correct answer accounts for the
        # overlap: 20 tokens / 1s = 20 tok/s.
        results = [
            _result(0.0, 0.1, 1.0, completion_tokens=10),
            _result(0.0, 0.1, 1.0, completion_tokens=10),
        ]
        metrics = aggregate(results)
        assert metrics.throughput_tokens_per_sec == pytest.approx(20.0)

    def test_throughput_is_none_when_no_request_succeeded(self):
        results = [_result(0, None, None, error="timeout")]
        metrics = aggregate(results)
        assert metrics.throughput_tokens_per_sec is None
