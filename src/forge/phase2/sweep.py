"""The Phase 2 sweep orchestrator — wires arms.py, client.py,
server_lifecycle.py, thermal.py, prompts.py, metrics.py, and
mlflow_tracking.py together into the actual concurrency x prompt-length x
quantization sweep the plan calls for.

Grouping and randomization, and the trade-off between them: cells are
grouped by (arm, model_variant) so each distinct model is only loaded once
(loading a server is the expensive step — seconds to tens of seconds, times
dozens of cells, adds up fast on a laptop). Within that constraint, this
sweep randomizes as much as it practically can: the (arm, model_variant)
group order itself, per thermal.randomize_sweep_order, AND the
(concurrency, prompt_bucket) sub-order within each group. Full
cross-product randomization (every single cell an independent server
restart) would be the more statistically pure choice per the plan's own
"randomise configuration order so thermal drift doesn't correlate with arm"
rule, but it multiplies total runtime by however many extra server restarts
that implies. This is a deliberate, documented compromise, not an
oversight — note it in the write-up (Phase 5) if the thermal data ever
shows a correlation with (arm, model_variant) group position despite the
inner randomization.

Refuses to benchmark a model variant that Phase 1's parity check didn't
verify — see load_verified_variants() — matching the promise
models/manifest.py's own docstring makes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from mlx_lm.utils import load_tokenizer
from openai import AsyncOpenAI

from forge.capstone_bridge import CapstoneHarness
from forge.config import get_settings
from forge.hardware import assert_native_arm64
from forge.logging_config import configure_logging, get_logger, start_run
from forge.phase2 import mlflow_tracking
from forge.phase2.arms import ArmConfig, mlx_lm_arm, ollama_arm, vllm_metal_arm
from forge.phase2.client import make_client, run_request
from forge.phase2.metrics import aggregate
from forge.phase2.prompts import PromptCase, build_prompt_buckets
from forge.phase2.result_schema import RequestResult
from forge.phase2.server_lifecycle import ManagedServer
from forge.phase2.thermal import ThermalMonitor, randomize_sweep_order, wait_for_cooldown

log = get_logger(__name__)

DEFAULT_CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32, 64]
DEFAULT_N_REPEATS = 3  # per the plan: "3 runs minimum per configuration"
DEFAULT_N_WARMUP = 1  # per the plan: "Warm-up requests discarded before measurement"
DEFAULT_MAX_TOKENS = 300
DEFAULT_N_PER_BUCKET = 10

ARM_BUILDERS = {
    "mlx_lm": mlx_lm_arm,
    "ollama": ollama_arm,
    "vllm_metal": vllm_metal_arm,
}


@dataclass(frozen=True)
class SweepCell:
    arm: str
    model_variant: str
    concurrency: int
    prompt_bucket: str


def load_verified_variants(manifest_path: Path) -> set[str]:
    """Reads models/MANIFEST.json and returns the variant names Phase 1's
    parity check actually verified. Raises if the check didn't pass at
    all — a sweep run against an unverified fuse produces numbers that
    look real but aren't, exactly the failure mode Phase 1's gate exists
    to catch before it reaches here."""
    if not manifest_path.exists():
        raise RuntimeError(
            f"{manifest_path} not found — run forge.phase1.manifest first. "
            "The sweep refuses to benchmark an unverified model."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parity = manifest.get("parity_check", {})
    if not parity.get("passed"):
        raise RuntimeError(
            f"{manifest_path} shows parity_check.passed={parity.get('passed')!r} — "
            "the sweep refuses to run against a model that failed (or never had) its "
            "Phase 1 parity check. Re-run forge.phase1.parity_check first."
        )
    return {v["name"] for v in manifest.get("variants", [])}


def build_arm_config(arm: str, model_variant: str) -> ArmConfig:
    builder = ARM_BUILDERS.get(arm)
    if builder is None:
        raise ValueError(f"Unknown arm: {arm!r} (expected one of {sorted(ARM_BUILDERS)})")
    return builder(model_variant)


async def run_cell(
    client: AsyncOpenAI,
    arm_config: ArmConfig,
    prompt_cases: list[PromptCase],
    concurrency: int,
    run_id: str,
    n_repeats: int = DEFAULT_N_REPEATS,
    n_warmup: int = DEFAULT_N_WARMUP,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> list[RequestResult]:
    """Fires `concurrency` requests at once per round, cycling through
    prompt_cases; n_warmup rounds are discarded, n_repeats rounds are kept.
    """
    all_results: list[RequestResult] = []
    total_rounds = n_warmup + n_repeats

    for round_idx in range(total_rounds):
        round_cases = [
            prompt_cases[(round_idx * concurrency + i) % len(prompt_cases)]
            for i in range(concurrency)
        ]
        tasks = [
            run_request(
                client,
                arm_config,
                run_id=run_id,
                concurrency=concurrency,
                prompt_bucket=case.prompt_bucket,
                case_id=case.case_id,
                system_prompt=case.system_prompt,
                question=case.question,
                max_tokens=max_tokens,
            )
            for case in round_cases
        ]
        round_results = await asyncio.gather(*tasks)
        if round_idx >= n_warmup:
            all_results.extend(round_results)
        else:
            log.info("sweep.warmup_round_discarded", round=round_idx, arm=arm_config.name)

    return all_results


async def run_sweep(
    cells_by_group: dict[tuple[str, str], list[SweepCell]],
    prompt_buckets: dict[str, list[PromptCase]],
    run_id: str,
    output_dir: Path,
    thermal_monitor: ThermalMonitor | None,
    n_repeats: int = DEFAULT_N_REPEATS,
    n_warmup: int = DEFAULT_N_WARMUP,
) -> None:
    # No seed here, deliberately: unlike Phase 1's parity-check sampling
    # (which needs the exact same 50 cases reproduced across runs), sweep
    # ORDER only needs to avoid correlating with thermal drift, not be
    # reproducible bit-for-bit — a fresh shuffle each run is fine and one
    # less parameter to thread through.
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = randomize_sweep_order(list(cells_by_group.items()))

    for (arm, model_variant), cells in groups:
        arm_config = build_arm_config(arm, model_variant)
        cells = randomize_sweep_order(cells)

        log.info("sweep.group_start", arm=arm, model_variant=model_variant, n_cells=len(cells))
        try:
            with ManagedServer(arm_config) as _server:
                async with make_client(arm_config) as client:
                    for cell in cells:
                        cooled_down = True
                        if thermal_monitor is not None:
                            cooled_down = wait_for_cooldown(thermal_monitor)

                        prompt_cases = prompt_buckets[cell.prompt_bucket]
                        results = await run_cell(
                            client,
                            arm_config,
                            prompt_cases,
                            cell.concurrency,
                            run_id,
                            n_repeats=n_repeats,
                            n_warmup=n_warmup,
                        )
                        # Stamped once per cell, not per request: thermal state
                        # doesn't meaningfully change within the seconds-to-tens-
                        # of-seconds one cell takes, and this is exactly the point
                        # wait_for_cooldown() already checked state at.
                        if thermal_monitor is not None:
                            reading = thermal_monitor.current()
                            results = [
                                r.model_copy(
                                    update={
                                        "thermal_pressure_level": reading.pressure_level,
                                        "cpu_power_mw": reading.cpu_power_mw,
                                        "gpu_power_mw": reading.gpu_power_mw,
                                    }
                                )
                                for r in results
                            ]
                        metrics = aggregate(results)

                        with mlflow_tracking.sweep_cell_run(
                            run_id,
                            cell.arm,
                            cell.model_variant,
                            cell.concurrency,
                            cell.prompt_bucket,
                        ):
                            mlflow_tracking.log_arm_metrics(metrics)
                            mlflow_tracking.log_raw_results(results)
                            mlflow_tracking.log_thermal_flag(cooled_down)

                        cell_filename = (
                            f"{cell.arm}_{cell.model_variant}_c{cell.concurrency}_"
                            f"{cell.prompt_bucket}.jsonl"
                        )
                        cell_path = output_dir / cell_filename
                        with cell_path.open("w", encoding="utf-8") as f:
                            for result in results:
                                f.write(result.model_dump_json() + "\n")

                        log.info(
                            "sweep.cell_finish",
                            arm=cell.arm,
                            model_variant=cell.model_variant,
                            concurrency=cell.concurrency,
                            prompt_bucket=cell.prompt_bucket,
                            n_succeeded=metrics.n_succeeded,
                            n_failed=metrics.n_failed,
                            cooled_down=cooled_down,
                        )
        except Exception:
            log.exception("sweep.group_failed", arm=arm, model_variant=model_variant)
            raise
        finally:
            log.info("sweep.group_finish", arm=arm, model_variant=model_variant)


def main() -> None:
    configure_logging()
    assert_native_arm64()
    settings = get_settings()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=["mlx_lm", "ollama", "vllm_metal"])
    parser.add_argument(
        "--concurrency-levels", nargs="+", type=int, default=DEFAULT_CONCURRENCY_LEVELS
    )
    parser.add_argument("--n-repeats", type=int, default=DEFAULT_N_REPEATS)
    parser.add_argument("--n-warmup", type=int, default=DEFAULT_N_WARMUP)
    parser.add_argument("--n-per-bucket", type=int, default=DEFAULT_N_PER_BUCKET)
    parser.add_argument("--manifest-path", type=Path, default=Path("models/MANIFEST.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/sweep"))
    parser.add_argument(
        "--fused-model-path",
        type=Path,
        default=Path("models/fused-bf16"),
        help="Only used to load the tokenizer for prompt-bucket token counting — "
        "any variant's tokenizer is identical, bf16 is just the canonical one.",
    )
    parser.add_argument(
        "--skip-thermal",
        action="store_true",
        help="Skip thermal monitoring/cooldown entirely — needs sudo otherwise.",
    )
    args = parser.parse_args()

    verified_variants = load_verified_variants(args.manifest_path)
    log.info("sweep.verified_variants", variants=sorted(verified_variants))

    arm_variant_map = {
        "mlx_lm": ["bf16", "8bit", "4bit"],
        "vllm_metal": ["bf16", "8bit", "4bit"],
        "ollama": ["f16", "q8", "q4"],
    }
    cells_by_group: dict[tuple[str, str], list[SweepCell]] = {}
    for arm in args.arms:
        for model_variant in arm_variant_map.get(arm, []):
            # Ollama's own variant names (f16/q8/q4) don't literally appear
            # in the MLX-format MANIFEST.json — only mlx_lm/vllm_metal's
            # bf16/8bit/4bit do, since Ollama serves the separately-tracked
            # GGUF export. Gate those two arms on the manifest; trust
            # ollama_register.py's own success (already verified at
            # registration time) for the Ollama arm.
            if arm in ("mlx_lm", "vllm_metal") and model_variant not in verified_variants:
                log.warning(
                    "sweep.skipping_unverified_variant", arm=arm, model_variant=model_variant
                )
                continue
            cells_by_group[(arm, model_variant)] = [
                SweepCell(arm, model_variant, concurrency, bucket)
                for concurrency in args.concurrency_levels
                for bucket in ("short", "medium", "long")
            ]

    harness = CapstoneHarness(settings)
    tokenizer = load_tokenizer(str(args.fused_model_path))
    prompt_buckets = build_prompt_buckets(harness, tokenizer, n_per_bucket=args.n_per_bucket)

    run_id = start_run(log, phase="phase2.sweep")

    thermal_monitor = None if args.skip_thermal else ThermalMonitor()
    if thermal_monitor is not None:
        thermal_monitor.start()

    try:
        asyncio.run(
            run_sweep(
                cells_by_group,
                prompt_buckets,
                run_id,
                args.output_dir,
                thermal_monitor,
                n_repeats=args.n_repeats,
                n_warmup=args.n_warmup,
            )
        )
    finally:
        if thermal_monitor is not None:
            thermal_monitor.stop()

    log.info("run.finish", run_id=run_id, phase="phase2.sweep")


if __name__ == "__main__":
    main()
