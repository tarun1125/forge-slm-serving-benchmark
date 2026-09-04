"""Phase 5: aggregate the full raw sweep (results/sweep/*.jsonl — 2,511 real
requests, gitignored because the raw per-token timelines are large and
regenerable) into one small, COMMITTED summary file the Phase 5 notebook
reads from. Without this, the notebook's "reproducible run producing the
headline chart" claim would be false for anyone who hasn't re-run the full
multi-hour, three-server sweep on identical Apple Silicon hardware — cloning
the repo wouldn't actually reproduce anything. This script is the fix:
regenerating results/sweep_summary.json from the raw files is fast (no
servers needed, just the already-collected JSONL), and the notebook only
ever reads the small output, not the raw sweep.

Reuses forge.phase2.metrics.aggregate() rather than recomputing percentiles
here — that function is the one place TTFT/ITL/throughput derivation lives,
per its own module docstring.

Usage:
    python -m forge.phase5.build_summary
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from forge.logging_config import configure_logging, get_logger, start_run
from forge.phase2.metrics import aggregate
from forge.phase2.result_schema import RequestResult

log = get_logger(__name__)

SWEEP_DIR = Path("results/sweep")
ACCURACY_PATH = Path("results/accuracy_by_group.json")
OUTPUT_PATH = Path("results/sweep_summary.json")


def load_all_results(sweep_dir: Path) -> list[RequestResult]:
    results = []
    for path in sorted(sweep_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                results.append(RequestResult(**json.loads(line)))
    return results


def group_by_cell(
    results: list[RequestResult],
) -> dict[tuple[str, str, int, str], list[RequestResult]]:
    groups: dict[tuple[str, str, int, str], list[RequestResult]] = {}
    for r in results:
        key = (r.arm, r.model_variant, r.concurrency, r.prompt_bucket)
        groups.setdefault(key, []).append(r)
    return groups


def load_accuracy_lookup(path: Path) -> dict[tuple[str, str, str], float]:
    """Keyed by (arm, model_variant, prompt_bucket) — accuracy is scored
    independent of concurrency (see score_accuracy.py), so the same value
    applies to every concurrency cell for that arm/variant/bucket."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    lookup = {}
    for entry in data.values():
        key = (entry["arm"], entry["model_variant"], entry["prompt_bucket"])
        lookup[key] = entry["execution_accuracy"]
    return lookup


def build_summary(sweep_dir: Path, accuracy_path: Path) -> list[dict]:
    results = load_all_results(sweep_dir)
    if not results:
        raise RuntimeError(
            f"No sweep results found under {sweep_dir} — run forge.phase2.sweep first."
        )
    accuracy_lookup = load_accuracy_lookup(accuracy_path)

    rows = []
    for (arm, variant, _concurrency, bucket), cell_results in sorted(
        group_by_cell(results).items()
    ):
        metrics = aggregate(cell_results)
        row = asdict(metrics)
        row["execution_accuracy"] = accuracy_lookup.get((arm, variant, bucket))
        rows.append(row)
    return rows


def main() -> None:
    configure_logging()
    run_id = start_run(log, phase="phase5.build_summary")

    rows = build_summary(SWEEP_DIR, ACCURACY_PATH)
    OUTPUT_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    log.info(
        "run.finish",
        run_id=run_id,
        phase="phase5.build_summary",
        output_path=str(OUTPUT_PATH),
        n_cells=len(rows),
    )
    print(f"Wrote {len(rows)} cells to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
