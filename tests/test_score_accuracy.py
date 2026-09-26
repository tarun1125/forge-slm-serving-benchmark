import json

import pytest

from forge.phase2.score_accuracy import (
    ACCURACY_METHOD,
    ScoredCase,
    collect_unique_generations,
    merge_and_write,
    summarize_group,
)


def _case(case_id: str, text: str, correct: bool | None, n: int = 1) -> ScoredCase:
    return ScoredCase(case_id, "db", text, correct, n)


class TestSummarizeGroup:
    def test_a_nondeterministic_case_counts_once_not_once_per_generation(self):
        """The bug this replaced: two distinct outputs for case b made it
        count twice, so the group read 1/3 instead of the per-case 0.25."""
        scored = [
            _case("a", "q", False, n=4),
            _case("b", "right", True, n=1),
            _case("b", "wrong", False, n=3),
        ]
        summary = summarize_group(scored)
        assert summary["n_cases"] == 2
        assert summary["n_unique_generations"] == 3
        # a: 0/4, b: 1/4 -> mean 0.125
        assert summary["execution_accuracy"] == pytest.approx(0.125)
        assert summary["correct_case_credit"] == pytest.approx(0.25)
        assert summary["nondeterministic_case_ids"] == ["b"]
        assert summary["accuracy_method"] == ACCURACY_METHOD

    def test_stable_cases_score_exactly_as_the_old_method_did(self):
        scored = [_case("a", "q", True, n=3), _case("b", "q", False, n=3)]
        assert summarize_group(scored)["execution_accuracy"] == pytest.approx(0.5)

    def test_cases_without_gold_are_excluded_from_the_denominator(self):
        scored = [_case("a", "q", True), _case("b", "q", None)]
        summary = summarize_group(scored)
        assert summary["n_scoreable_cases"] == 1
        assert summary["execution_accuracy"] == pytest.approx(1.0)

    def test_nothing_scoreable_is_none_not_zero(self):
        assert summarize_group([_case("a", "q", None)])["execution_accuracy"] is None


class TestCollectUniqueGenerations:
    def test_counts_repeats_and_skips_failures(self, tmp_path):
        rows = [
            {"arm": "x", "model_variant": "v", "prompt_bucket": "b", "case_id": "a",
             "generated_text": "q", "error": None},
            {"arm": "x", "model_variant": "v", "prompt_bucket": "b", "case_id": "a",
             "generated_text": "q", "error": None},
            {"arm": "x", "model_variant": "v", "prompt_bucket": "b", "case_id": "a",
             "generated_text": None, "error": "Connection error."},
        ]  # fmt: skip
        (tmp_path / "cell.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        groups = collect_unique_generations(tmp_path)
        assert groups[("x", "v", "b")] == {("a", "q"): 2}


class TestMergeAndWrite:
    def test_keeps_groups_it_did_not_score(self, tmp_path):
        """Scoring only the cloud arm's directory must not delete the local groups."""
        out = tmp_path / "accuracy_by_group.json"
        out.write_text(json.dumps({"mlx_lm/4bit/long": {"execution_accuracy": 0.2}}))

        replaced, kept = merge_and_write(out, {"ollama_cloud/q4/long": {"execution_accuracy": 0.3}})

        data = json.loads(out.read_text())
        assert set(data) == {"mlx_lm/4bit/long", "ollama_cloud/q4/long"}
        assert (replaced, kept) == ([], ["mlx_lm/4bit/long"])
        assert not (tmp_path / "accuracy_by_group.json.tmp").exists()

    def test_rescored_groups_replace_their_own_entry(self, tmp_path):
        out = tmp_path / "accuracy_by_group.json"
        out.write_text(json.dumps({"g": {"execution_accuracy": 0.2}}))
        replaced, _ = merge_and_write(out, {"g": {"execution_accuracy": 0.9}})
        assert json.loads(out.read_text())["g"]["execution_accuracy"] == 0.9
        assert replaced == ["g"]

    def test_creates_the_file_when_absent(self, tmp_path):
        out = tmp_path / "sub" / "acc.json"
        merge_and_write(out, {"g": {}})
        assert json.loads(out.read_text()) == {"g": {}}
