import json

from forge.hardware import get_hardware_info
from forge.phase2.result_schema import RequestResult
from forge.phase5.build_summary import build_summary, group_by_cell, load_accuracy_lookup


def _result(
    arm: str,
    model_variant: str,
    concurrency: int,
    prompt_bucket: str,
    case_id: str,
    completion_tokens: int = 10,
) -> RequestResult:
    return RequestResult(
        run_id="test-run",
        arm=arm,
        model_variant=model_variant,
        concurrency=concurrency,
        prompt_bucket=prompt_bucket,
        case_id=case_id,
        request_start_monotonic=0.0,
        first_token_monotonic=0.1,
        last_token_monotonic=0.5,
        token_monotonics=[0.1, 0.2, 0.3, 0.4, 0.5],
        completion_tokens=completion_tokens,
        prompt_tokens=100,
        hardware=get_hardware_info(),
    )


class TestGroupByCell:
    def test_groups_by_arm_variant_concurrency_bucket(self):
        results = [
            _result("mlx_lm", "bf16", 1, "short", "c1"),
            _result("mlx_lm", "bf16", 1, "short", "c2"),
            _result("mlx_lm", "bf16", 2, "short", "c3"),
            _result("ollama", "bf16", 1, "short", "c4"),
        ]
        groups = group_by_cell(results)
        assert len(groups) == 3
        assert len(groups[("mlx_lm", "bf16", 1, "short")]) == 2
        assert len(groups[("mlx_lm", "bf16", 2, "short")]) == 1
        assert len(groups[("ollama", "bf16", 1, "short")]) == 1


class TestLoadAccuracyLookup:
    def test_keys_by_arm_variant_bucket_ignoring_concurrency(self, tmp_path):
        path = tmp_path / "accuracy_by_group.json"
        path.write_text(
            json.dumps(
                {
                    "mlx_lm/bf16/short": {
                        "arm": "mlx_lm",
                        "model_variant": "bf16",
                        "prompt_bucket": "short",
                        "execution_accuracy": 0.0,
                    }
                }
            ),
            encoding="utf-8",
        )
        lookup = load_accuracy_lookup(path)
        assert lookup == {("mlx_lm", "bf16", "short"): 0.0}

    def test_missing_file_returns_empty_dict(self, tmp_path):
        assert load_accuracy_lookup(tmp_path / "nonexistent.json") == {}


class TestBuildSummary:
    def test_produces_one_row_per_cell_with_accuracy_attached(self, tmp_path):
        sweep_dir = tmp_path / "sweep"
        sweep_dir.mkdir()
        rows = [
            _result("mlx_lm", "bf16", 1, "short", "c1"),
            _result("mlx_lm", "bf16", 2, "short", "c2"),
        ]
        (sweep_dir / "mlx_lm_bf16_c1_short.jsonl").write_text(
            rows[0].model_dump_json() + "\n", encoding="utf-8"
        )
        (sweep_dir / "mlx_lm_bf16_c2_short.jsonl").write_text(
            rows[1].model_dump_json() + "\n", encoding="utf-8"
        )
        accuracy_path = tmp_path / "accuracy_by_group.json"
        accuracy_path.write_text(
            json.dumps(
                {
                    "mlx_lm/bf16/short": {
                        "arm": "mlx_lm",
                        "model_variant": "bf16",
                        "prompt_bucket": "short",
                        "execution_accuracy": 0.5,
                    }
                }
            ),
            encoding="utf-8",
        )

        summary = build_summary(sweep_dir, accuracy_path)

        assert len(summary) == 2
        assert all(row["execution_accuracy"] == 0.5 for row in summary)
        concurrencies = {row["concurrency"] for row in summary}
        assert concurrencies == {1, 2}

    def test_raises_on_empty_sweep_dir(self, tmp_path):
        sweep_dir = tmp_path / "empty_sweep"
        sweep_dir.mkdir()
        try:
            build_summary(sweep_dir, tmp_path / "accuracy_by_group.json")
            raise AssertionError("expected RuntimeError")
        except RuntimeError as e:
            assert "No sweep results found" in str(e)
