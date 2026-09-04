"""Phase 1 gate: verify the fused model is equivalent to the base+adapter
model it was fused from, BEFORE any benchmark number gets produced against it.
If the fuse is wrong, every downstream Phase 2/3 number is garbage.

Design (full rationale in docs/parity-check-design.md — read that before
changing the constants below):

  - 50 cases sampled from the capstone's existing 304-case held-out test set
    (rag/data/rag_test.json), stratified by complexity, seed=42.
  - PASS requires:
      1. ZERO execution-accuracy regressions — the hard gate. Every case the
         adapter model got right, the fused model must also get right. A
         regression means the fuse corrupted something; the aggregate rate
         alone can hide this (a fused model that gets a *different* case
         right by luck nets to the same aggregate while still losing one).
      2. Exact-text-match rate >= EXACT_MATCH_SANITY_FLOOR — a loose belt-
         and-suspenders check, NOT the primary signal (see below).

  Originally EXACT_MATCH_SANITY_FLOOR was 0.96, on the theory that fusing is
  a linear-algebra identity so divergence should only be bf16-rounding
  noise. First real run measured 90% (45/50) with zero regressions —
  inspecting the 5 mismatches by hand showed why 96% was the wrong bar: at
  MAX_TOKENS of free-form autoregressive decoding, one legitimate bf16
  tie-break early in a sequence cascades into a fully different completion
  (every later token conditions on it), even though the fuse transform
  itself is exact. That's expected decoding sensitivity, not fuse
  corruption — of the 5 mismatches, one was the fused model *fixing* a
  degenerate repetition-loop bug in the adapter output, two were
  semantically-identical rephrasings (both executed correctly), and two
  were pre-existing model failures identical in kind on both sides (not
  introduced by fusing). Net effect was 20/50 -> 21/50 correct — fusing
  improved, not regressed. The floor is now 0.70: loose enough not to flag
  normal decoding-cascade divergence, tight enough to still catch a fuse
  that produces mostly-unrelated output.

Usage:
    python -m forge.phase1.parity_check --fused-path models/fused-bf16
    python -m forge.phase1.parity_check --n-cases 50 --skip-atlas   # text-match only, no DB needed
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from mlx_lm import generate, load

from forge.capstone_bridge import CapstoneHarness
from forge.config import get_settings
from forge.logging_config import configure_logging, get_logger, start_run
from forge.query_guard import assert_generated_query_safe

log = get_logger(__name__)

SEED = 42
N_CASES = 50
MAX_TOKENS = 300
EXACT_MATCH_SANITY_FLOOR = 0.70  # diagnostic floor, not the primary gate — see module docstring


def stratified_sample(cases: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_complexity: dict[str, list[dict]] = {}
    for case in cases:
        by_complexity.setdefault(case.get("complexity", "unknown"), []).append(case)

    total = len(cases)
    allocation = {
        complexity: round(len(group) / total * n) for complexity, group in by_complexity.items()
    }
    # Rounding can drift the total off n by a case or two — correct against
    # the largest stratum so the sample size is always exactly n.
    drift = n - sum(allocation.values())
    largest = max(allocation, key=lambda k: len(by_complexity[k]))
    allocation[largest] += drift

    sample = []
    for complexity, count in allocation.items():
        group = by_complexity[complexity]
        sample.extend(rng.sample(group, min(count, len(group))))
    rng.shuffle(sample)
    return sample


def build_prompt(tokenizer: Any, system_prompt: str, question: str) -> Any:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True)


def run_generation(
    model: Any, tokenizer: Any, cases: list[dict], db_to_prompt: dict[str, str], clean: Any
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for case in cases:
        prompt = build_prompt(tokenizer, db_to_prompt[case["database"]], case["question"])
        raw = generate(model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS, verbose=False)
        outputs[case["id"]] = clean(raw)
    return outputs


def execution_accuracy(
    harness: CapstoneHarness,
    client: Any,
    cases: list[dict],
    generated_by_id: dict[str, str],
    gold_by_id: dict[str, dict],
) -> dict[str, bool | None]:
    execute_queries = harness.execute_queries
    normalize = harness.normalize
    accuracy: dict[str, bool | None] = {}

    for case in cases:
        case_id = str(case["id"])
        gold = gold_by_id.get(case_id)
        if gold is None:
            accuracy[case_id] = None
            continue

        db = client[case["database"]]
        try:
            query = normalize.normalize(generated_by_id[case["id"]])
            # Second gate in front of the capstone harness's own allowlist —
            # see forge.query_guard for the verified gaps it closes.
            assert_generated_query_safe(query)
            result = execute_queries.safe_eval_query(query, db)
            if not isinstance(result, (int, float, str, bool)) and not isinstance(result, list):
                result = list(result)
            result = execute_queries.to_json_safe(result)
            accuracy[case_id] = execute_queries.results_match(result, gold["result"])
        except Exception as exc:  # noqa: BLE001 — matches capstone's own run_model behavior
            log.warning("execution_accuracy.error", case_id=case_id, error=str(exc))
            accuracy[case_id] = False

    return accuracy


def main() -> None:
    configure_logging()
    settings = get_settings()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fused-path", type=Path, default=Path("models/fused-bf16"))
    parser.add_argument("--n-cases", type=int, default=N_CASES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--skip-atlas",
        action="store_true",
        help="Skip the execution-accuracy check (text-match only). Useful without a live "
        "Atlas connection, but this is NOT a substitute for the full check before Phase 2.",
    )
    parser.add_argument(
        "--report-path", type=Path, default=Path("results/parity_check_report.json")
    )
    args = parser.parse_args()

    if not args.fused_path.exists():
        raise RuntimeError(f"{args.fused_path} does not exist — run forge.phase1.fuse first.")

    run_id = start_run(log, phase="phase1.parity_check", fused_path=str(args.fused_path))
    harness = CapstoneHarness(settings)

    all_cases = json.loads(harness.rag_test_path.read_text(encoding="utf-8"))
    cases = stratified_sample(all_cases, args.n_cases, args.seed)
    log.info(
        "parity_check.sample",
        n_cases=len(cases),
        by_complexity=dict(Counter(c.get("complexity") for c in cases)),
    )

    db_to_prompt = {
        db: batch["system_prompt"]
        for batch in json.loads(harness.split_manifest_path.read_text(encoding="utf-8"))["batches"]
        for db in batch["databases"]
    }
    clean = harness.spot_check.clean

    log.info("parity_check.load_adapter_model", base_model=settings.base_model)
    # load()'s return type is a Union keyed on return_config (unused here, always False),
    # so mypy can't narrow it to the 2-tuple this call always produces at runtime.
    adapter_model, adapter_tokenizer = load(  # type: ignore[misc]
        settings.base_model, adapter_path=str(settings.require_adapter_path())
    )
    adapter_outputs = run_generation(adapter_model, adapter_tokenizer, cases, db_to_prompt, clean)
    del adapter_model  # free unified memory before loading the second model

    log.info("parity_check.load_fused_model", fused_path=str(args.fused_path))
    fused_model, fused_tokenizer = load(str(args.fused_path))  # type: ignore[misc]
    fused_outputs = run_generation(fused_model, fused_tokenizer, cases, db_to_prompt, clean)
    del fused_model

    exact_matches = sum(adapter_outputs[c["id"]] == fused_outputs[c["id"]] for c in cases)
    exact_match_rate = exact_matches / len(cases)

    adapter_accuracy: dict[str, bool | None] = {}
    fused_accuracy: dict[str, bool | None] = {}
    regressions: list[str] = []

    if not args.skip_atlas:
        harness.check_atlas_credentials()
        execute_queries = harness.execute_queries
        gold_raw = json.loads(harness.gold_results_path.read_text(encoding="utf-8"))
        gold_by_id = {str(r["id"]): r for r in gold_raw if r.get("status") == "PASS"}

        client = execute_queries.connect()
        adapter_accuracy = execution_accuracy(harness, client, cases, adapter_outputs, gold_by_id)
        fused_accuracy = execution_accuracy(harness, client, cases, fused_outputs, gold_by_id)

        regressions = [
            str(c["id"])
            for c in cases
            if adapter_accuracy.get(str(c["id"])) is True
            and fused_accuracy.get(str(c["id"])) is False
        ]

    passed = not regressions and exact_match_rate >= EXACT_MATCH_SANITY_FLOOR

    report = {
        "run_id": run_id,
        "fused_path": str(args.fused_path),
        "n_cases": len(cases),
        "seed": args.seed,
        "exact_match_rate": exact_match_rate,
        "exact_match_sanity_floor": EXACT_MATCH_SANITY_FLOOR,
        "skipped_atlas": args.skip_atlas,
        "adapter_execution_correct": sum(v is True for v in adapter_accuracy.values()),
        "fused_execution_correct": sum(v is True for v in fused_accuracy.values()),
        "regressions": regressions,
        "passed": passed,
        "cases": [
            {
                "id": c["id"],
                "database": c["database"],
                "complexity": c.get("complexity"),
                "adapter_output": adapter_outputs[c["id"]],
                "fused_output": fused_outputs[c["id"]],
                "text_match": adapter_outputs[c["id"]] == fused_outputs[c["id"]],
                "adapter_execution_accuracy": adapter_accuracy.get(str(c["id"])),
                "fused_execution_accuracy": fused_accuracy.get(str(c["id"])),
            }
            for c in cases
        ],
    }

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    log.info(
        "run.finish",
        run_id=run_id,
        phase="phase1.parity_check",
        passed=passed,
        exact_match_rate=round(exact_match_rate, 3),
        regressions=len(regressions),
        report_path=str(args.report_path),
    )

    print(
        f"\n{'PASS' if passed else 'FAIL'}: regressions={len(regressions)} (hard gate), "
        f"exact_match_rate={exact_match_rate:.1%} (sanity floor {EXACT_MATCH_SANITY_FLOOR:.0%})"
    )
    print(f"Full report -> {args.report_path}")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
