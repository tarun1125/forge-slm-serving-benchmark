"""Scores execution accuracy for the Phase 2 sweep's generated outputs,
against the same gold-results/execution-and-compare logic Phase 1's parity
check already validated (see forge.capstone_bridge, evaluation/execute_queries.py
in the capstone repo). Not part of the sweep itself: accuracy shouldn't vary
with concurrency (same model, same weights, temperature=0), so scoring is
done once per (arm, model_variant, prompt_bucket) — not once per concurrency
level — to avoid redundant Atlas round-trips for what should be near-duplicate
generations.

Dedup by (case_id, generated_text): a first assumption that concurrency
never changes a given case's output at temp=0 is checked here rather than
trusted — server-side batching can perturb floating-point summation order
(the same phenomenon Phase 1's parity-check design doc documents for the
fuse itself), so if a case_id shows more than one distinct generated_text
across concurrency levels within one (arm, variant, bucket) group, every
distinct variant is scored and logged, not silently collapsed to one.

Usage:
    python -m forge.phase2.score_accuracy --sweep-dir results/sweep
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from forge.capstone_bridge import CapstoneHarness
from forge.config import get_settings
from forge.logging_config import configure_logging, get_logger, start_run
from forge.query_guard import assert_generated_query_safe

log = get_logger(__name__)


@dataclass(frozen=True)
class ScoredCase:
    case_id: str
    database: str
    generated_text: str
    execution_accuracy: bool | None  # None = no gold result available for this case


def load_case_id_to_database(harness: CapstoneHarness) -> dict[str, str]:
    cases = json.loads(harness.rag_test_path.read_text(encoding="utf-8"))
    return {str(c["id"]): c["database"] for c in cases}


def collect_unique_generations(
    sweep_dir: Path,
) -> dict[tuple[str, str, str], set[tuple[str, str]]]:
    """Groups sweep result rows by (arm, model_variant, prompt_bucket) and
    collects the set of unique (case_id, generated_text) pairs seen across
    ALL concurrency levels and repeats for that group — the dedup step the
    module docstring explains."""
    groups: dict[tuple[str, str, str], set[tuple[str, str]]] = defaultdict(set)
    for path in sorted(sweep_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["error"] is not None or not row["generated_text"]:
                continue
            key = (row["arm"], row["model_variant"], row["prompt_bucket"])
            groups[key].add((row["case_id"], row["generated_text"]))
    return groups


def score_generation(
    harness: CapstoneHarness,
    client: object,
    gold_by_id: dict[str, dict],
    case_id: str,
    database: str,
    generated_text: str,
) -> bool | None:
    gold = gold_by_id.get(case_id)
    if gold is None:
        return None

    execute_queries = harness.execute_queries
    normalize = harness.normalize
    db = client[database]  # type: ignore[index]

    try:
        query = normalize.normalize(generated_text)
        # Second gate, in front of the capstone harness's own allowlist — see
        # forge.query_guard for the two verified gaps in that allowlist this
        # closes. Runs post-normalize so it sees exactly the string that will
        # be eval'd, not the pre-normalized form.
        assert_generated_query_safe(query)
        result = execute_queries.safe_eval_query(query, db)
        if not isinstance(result, (int, float, str, bool)) and not isinstance(result, list):
            result = list(result)
        result = execute_queries.to_json_safe(result)
        return bool(execute_queries.results_match(result, gold["result"]))
    except Exception as exc:  # noqa: BLE001 — a rejected/malformed query is a real (False) result,
        # not a crash — matches parity_check.py's and the capstone's own run_model() behavior.
        log.warning("score_accuracy.execution_error", case_id=case_id, error=str(exc))
        return False


def main() -> None:
    configure_logging()
    settings = get_settings()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, default=Path("results/sweep"))
    parser.add_argument("--output", type=Path, default=Path("results/accuracy_by_group.json"))
    args = parser.parse_args()

    harness = CapstoneHarness(settings)
    harness.check_atlas_credentials()
    case_id_to_database = load_case_id_to_database(harness)
    gold_raw = json.loads(harness.gold_results_path.read_text(encoding="utf-8"))
    gold_by_id = {str(r["id"]): r for r in gold_raw if r.get("status") == "PASS"}

    run_id = start_run(log, phase="phase2.score_accuracy")
    groups = collect_unique_generations(args.sweep_dir)
    log.info("score_accuracy.groups", n_groups=len(groups))

    client = harness.execute_queries.connect()

    results_by_group: dict[str, dict] = {}
    for (arm, model_variant, prompt_bucket), pairs in sorted(groups.items()):
        group_key = f"{arm}/{model_variant}/{prompt_bucket}"
        scored: list[ScoredCase] = []
        for case_id, generated_text in pairs:
            database = case_id_to_database.get(case_id)
            if database is None:
                log.warning("score_accuracy.unknown_case", case_id=case_id)
                continue
            accuracy = score_generation(
                harness, client, gold_by_id, case_id, database, generated_text
            )
            scored.append(ScoredCase(case_id, database, generated_text, accuracy))

        scoreable = [s for s in scored if s.execution_accuracy is not None]
        correct = sum(1 for s in scoreable if s.execution_accuracy)
        accuracy_pct = correct / len(scoreable) if scoreable else None

        # A case_id with more than one distinct generated_text across
        # concurrency levels means generation wasn't perfectly deterministic
        # under load — worth knowing about, not silently averaged away.
        case_id_counts: dict[str, int] = defaultdict(int)
        for s in scored:
            case_id_counts[s.case_id] += 1
        nondeterministic_case_ids = [cid for cid, n in case_id_counts.items() if n > 1]

        results_by_group[group_key] = {
            "arm": arm,
            "model_variant": model_variant,
            "prompt_bucket": prompt_bucket,
            "n_unique_generations": len(scored),
            "n_scoreable": len(scoreable),
            "n_correct": correct,
            "execution_accuracy": accuracy_pct,
            "nondeterministic_case_ids": nondeterministic_case_ids,
        }
        log.info(
            "score_accuracy.group_finish",
            group=group_key,
            n_scoreable=len(scoreable),
            n_correct=correct,
            accuracy=round(accuracy_pct, 3) if accuracy_pct is not None else None,
            nondeterministic=len(nondeterministic_case_ids),
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results_by_group, indent=2), encoding="utf-8")
    log.info("run.finish", run_id=run_id, phase="phase2.score_accuracy", output=str(args.output))


if __name__ == "__main__":
    main()
