import json

import pytest

from forge.hardware import HardwareInfo, ServerHardware, ServerSoftware
from forge.phase2.result_schema import RequestResult
from forge.phase3 import build_report

HW = HardwareInfo("arm64", "26.5", "Apple M5 Pro", 15, 16, 24.0)


def _row(
    i: int,
    *,
    hourly_usd: float | None = 0.1,
    processor: str = "Neoverse-N2",
    arm: str = "ollama_cloud",
    variant: str = "q4",
) -> str:
    start = float(i)
    return RequestResult(
        run_id="r",
        arm=arm,
        model_variant=variant,
        concurrency=8,
        prompt_bucket="medium",
        case_id=f"c{i}",
        request_start_monotonic=start,
        first_token_monotonic=start + 1,
        last_token_monotonic=start + 10,
        token_monotonics=[start + 1 + k for k in range(10)],
        prompt_tokens=1400,
        completion_tokens=100,
        generated_text="db.x.find({})",
        hardware=HW,
        server_hardware=ServerHardware(
            processor=processor,
            provider="azure",
            instance_type="Standard_D4ps_v6",
            hourly_usd=hourly_usd,
        ),
        server_software=ServerSoftware(stack="ollama", version="0.32.14"),
    ).model_dump_json()


@pytest.fixture
def sweep_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(build_report, "SWEEP_DIR", tmp_path)
    accuracy = tmp_path / "accuracy.json"
    accuracy.write_text(json.dumps({"ollama_cloud/q4/medium": {"execution_accuracy": 0.5}}))
    monkeypatch.setattr(build_report, "ACCURACY_PATH", accuracy)
    return tmp_path


def _write_cell(sweep_dir, rows):
    (sweep_dir / "ollama_cloud_q4_c8_medium.jsonl").write_text("\n".join(rows))


class TestCloudVmTable:
    def test_a_cloud_arm_that_never_ran_is_skipped_not_fatal(self, sweep_dir):
        assert build_report.build_cloud_vm_table(hosted_cost_per_query_inr=0.1) == []
        assert "Not measured yet" in "\n".join(build_report.render_cloud_vm_section([]))

    def test_prices_from_the_rows_own_rate_and_throughput(self, sweep_dir):
        _write_cell(sweep_dir, [_row(i) for i in range(3)])
        [row] = build_report.build_cloud_vm_table(hosted_cost_per_query_inr=0.1)

        # 3 requests x 100 tokens over a span of 0 -> 12 s
        assert row["throughput_tokens_per_sec"] == pytest.approx(300 / 12)
        assert row["hourly_usd"] == 0.1
        assert row["hourly_usd_source"] == "recorded on result rows"
        assert row["server_software"] == ["ollama 0.32.14"]
        assert row["execution_accuracy"] == 0.5
        # always-on: 730 h x $0.1 x 88 INR / 10k queries
        assert row["cost_per_query_inr_at_10k_monthly"] == pytest.approx(730 * 0.1 * 88 / 10_000)

        section = "\n".join(build_report.render_cloud_vm_section([row]))
        assert "Neoverse-N2 · Standard_D4ps_v6" in section

    def test_refuses_a_cell_mixing_machines(self, sweep_dir):
        _write_cell(sweep_dir, [_row(0), _row(1, processor="Ampere Altra")])
        with pytest.raises(RuntimeError, match="same server_hardware"):
            build_report.build_cloud_vm_table(hosted_cost_per_query_inr=0.1)

    def test_refuses_to_price_without_any_hourly_rate(self, sweep_dir, monkeypatch):
        monkeypatch.delenv("OLLAMA_CLOUD_HOURLY_USD", raising=False)
        monkeypatch.chdir(sweep_dir)  # no .env here
        _write_cell(sweep_dir, [_row(0, hourly_usd=None)])
        with pytest.raises(RuntimeError, match="OLLAMA_CLOUD_HOURLY_USD"):
            build_report.build_cloud_vm_table(hosted_cost_per_query_inr=0.1)

    def test_a_gpu_run_is_priced_too_with_its_own_rate_fallback(self, sweep_dir, monkeypatch):
        """vllm_cuda rows written without a rate fall back to VLLM_CUDA_HOURLY_USD —
        not the CPU arm's setting."""
        monkeypatch.chdir(sweep_dir)
        monkeypatch.setenv("VLLM_CUDA_HOURLY_USD", "0.579")
        monkeypatch.setenv("OLLAMA_CLOUD_HOURLY_USD", "99")
        rows = [
            _row(i, hourly_usd=None, processor="Tesla T4", arm="vllm_cuda", variant="bf16")
            for i in range(2)
        ]
        (sweep_dir / "vllm_cuda_bf16_c8_medium.jsonl").write_text("\n".join(rows))

        [row] = build_report.build_cloud_vm_table(hosted_cost_per_query_inr=0.1)
        assert (row["arm"], row["processor"]) == ("vllm_cuda", "Tesla T4")
        assert row["hourly_usd"] == 0.579
        assert row["hourly_usd_source"] == ".env at report time"
