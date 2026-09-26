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

Weighting — one vote per CASE, not per distinct generation. An earlier
version averaged over the deduplicated (case_id, generated_text) pairs, so a
case that produced two different outputs counted twice and a stable case
once: results/accuracy_by_group.json's mlx_lm/4bit/long entry scored 13
generations drawn from 10 cases. Now each distinct generation is weighted by
how many requests produced it, averaged within its case, and the group's
accuracy is the mean over cases (summarize_group). A case whose output never
varied scores exactly as before; only the nondeterministic ones change.

The output file is MERGED into, never replaced wholesale: groups scored in
this run overwrite their own keys, every other group already in the file is
kept. Scoring just a cloud arm's directory (--sweep-dir results/cloud) used
to silently delete every local group from results/accuracy_by_group.json.
Written atomically (temp file + rename), so an interrupted run can't leave
it half-written.

Usage:
    python -m forge.phase2.score_accuracy --sweep-dir results/sweep
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from forge.capstone_bridge import CapstoneHarness
from forge.config import get_settings
from forge.logging_config import configure_logging, get_logger, start_run
from forge.query_guard import assert_generated_query_safe

log = get_logger(__name__)

# Stamped on every group this module writes, so an entry scored under the old
# per-generation weighting (which has no such key) is distinguishable in a
# merged file. See the module docstring.
ACCURACY_METHOD = "per_case_mean_request_weighted"


@dataclass(frozen=True)
class ScoredCase:
    case_id: str
    database: str
    generated_text: str
    execution_accuracy: bool | None  # None = no gold result available for this case
    n_requests: int = 1  # how many sweep rows produced exactly this generation


def load_case_id_to_database(harness: CapstoneHarness) -> dict[str, str]:
    cases = json.loads(harness.rag_test_path.read_text(encoding="utf-8"))
    return {str(c["id"]): c["database"] for c in cases}


def collect_unique_generations(
    sweep_dir: Path,
) -> dict[tuple[str, str, str], Counter[tuple[str, str]]]:
    """Groups sweep result rows by (arm, model_variant, prompt_bucket) and
    counts each unique (case_id, generated_text) pair seen across ALL
    concurrency levels and repeats for that group — the dedup step the module
    docstring explains. Counted, not just collected: the count is each
    generation's weight within its case (see summarize_group)."""
    groups: dict[tuple[str, str, str], Counter[tuple[str, str]]] = defaultdict(Counter)
    for path in sorted(sweep_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["error"] is not None or not row["generated_text"]:
                continue
            key = (row["arm"], row["model_variant"], row["prompt_bucket"])
            groups[key][(row["case_id"], row["generated_text"])] += 1
    return groups


def summarize_group(scored: list[ScoredCase]) -> dict:
    """Pure aggregation, no Atlas: per case, the request-weighted fraction of
    its generations that scored correct; for the group, the mean of those over
    cases. Generations with no gold result (execution_accuracy None) are left
    out of both numerator and denominator."""
    per_case_correct: dict[str, int] = defaultdict(int)
    per_case_total: dict[str, int] = defaultdict(int)
    generations_per_case: Counter[str] = Counter()
    for s in scored:
        generations_per_case[s.case_id] += 1
        if s.execution_accuracy is None:
            continue
        per_case_total[s.case_id] += s.n_requests
        per_case_correct[s.case_id] += s.n_requests if s.execution_accuracy else 0

    case_scores = {cid: per_case_correct[cid] / n for cid, n in per_case_total.items()}
    return {
        "n_cases": len(generations_per_case),
        "n_scoreable_cases": len(case_scores),
        "n_unique_generations": len(scored),
        # Sum of per-case credit: an integer when every case was stable, a
        # fraction when a nondeterministic case was right only some of the time.
        "correct_case_credit": sum(case_scores.values()),
        "execution_accuracy": (
            sum(case_scores.values()) / len(case_scores) if case_scores else None
        ),
        "nondeterministic_case_ids": sorted(
            cid for cid, n in generations_per_case.items() if n > 1
        ),
        "accuracy_method": ACCURACY_METHOD,
    }


def merge_and_write(output: Path, new_groups: dict[str, dict]) -> tuple[list[str], list[str]]:
    """Merges new_groups into whatever output already holds and writes the
    result atomically. Returns (replaced, kept) group keys for logging."""
    existing: dict[str, dict] = (
        json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
    )
    replaced = sorted(k for k in new_groups if k in existing)
    kept = sorted(k for k in existing if k not in new_groups)
    merged = {**existing, **new_groups}
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    tmp.write_text(json.dumps(dict(sorted(merged.items())), indent=2), encoding="utf-8")
    os.replace(tmp, output)
    return replaced, kept


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
        for (case_id, generated_text), n_requests in pairs.items():
            database = case_id_to_database.get(case_id)
            if database is None:
                log.warning("score_accuracy.unknown_case", case_id=case_id)
                continue
            accuracy = score_generation(
                harness, client, gold_by_id, case_id, database, generated_text
            )
            scored.append(ScoredCase(case_id, database, generated_text, accuracy, n_requests))

        # A case_id with more than one distinct generated_text across
        # concurrency levels means generation wasn't perfectly deterministic
        # under load — reported, and weighted by request count rather than
        # silently averaged away. See summarize_group.
        summary = summarize_group(scored)
        results_by_group[group_key] = {
            "arm": arm,
            "model_variant": model_variant,
            "prompt_bucket": prompt_bucket,
            **summary,
        }
        group_accuracy = summary["execution_accuracy"]
        log.info(
            "score_accuracy.group_finish",
            group=group_key,
            n_scoreable_cases=summary["n_scoreable_cases"],
            accuracy=round(group_accuracy, 3) if group_accuracy is not None else None,
            nondeterministic=len(summary["nondeterministic_case_ids"]),
        )

    replaced, kept = merge_and_write(args.output, results_by_group)
    log.info(
        "run.finish",
        run_id=run_id,
        phase="phase2.score_accuracy",
        output=str(args.output),
        n_scored=len(results_by_group),
        replaced=replaced,
        kept_from_previous_runs=kept,
    )


if __name__ == "__main__":
    main()
