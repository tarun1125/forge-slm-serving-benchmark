"""MLflow integration — this closes gap 5 (experiment tracking) per the
master plan, which explicitly calls out not bolting tracking on afterwards.
Every sweep cell (one arm, one quantization level, one concurrency level,
one prompt bucket) gets its own MLflow run: params identify the cell,
metrics are the aggregated ArmMetrics, and the raw per-request JSONL is
logged as an artifact so a later analysis can recompute percentiles
differently without re-running the sweep.

Uses the plain file-based tracking URI from settings (MLFLOW_TRACKING_URI,
default ./mlruns) — no tracking server needed for a local benchmark.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import mlflow

from forge.config import get_settings
from forge.hardware import get_hardware_dict
from forge.logging_config import get_logger
from forge.phase2.metrics import ArmMetrics
from forge.phase2.result_schema import RequestResult

log = get_logger(__name__)

EXPERIMENT_NAME = "forge-serving-benchmark"


def configure_tracking() -> None:
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)


@contextmanager
def sweep_cell_run(
    run_id: str, arm: str, model_variant: str, concurrency: int, prompt_bucket: str
) -> Iterator[None]:
    """One MLflow run per (arm, model_variant, concurrency, prompt_bucket)
    cell — matches metrics.aggregate()'s own grouping exactly, so a run's
    params always identify precisely the population its metrics summarize."""
    configure_tracking()
    run_name = f"{arm}-{model_variant}-c{concurrency}-{prompt_bucket}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(
            {
                "forge.run_id": run_id,
                "forge.arm": arm,
                "forge.chip": get_hardware_dict()["chip"],
            }
        )
        mlflow.log_params(
            {
                "arm": arm,
                "model_variant": model_variant,
                "concurrency": concurrency,
                "prompt_bucket": prompt_bucket,
                **{f"hardware.{k}": v for k, v in get_hardware_dict().items()},
            }
        )
        yield


def log_arm_metrics(metrics: ArmMetrics) -> None:
    """Logs the aggregated cell metrics. TTFT and inter-token latency are
    logged under clearly separate metric name prefixes — never combined —
    per the plan's core "don't blend TTFT and ITL into one latency number"
    rule, applied here too, not just in the raw result schema."""
    mlflow.log_metrics(
        {
            "n_requests": metrics.n_requests,
            "n_succeeded": metrics.n_succeeded,
            "n_failed": metrics.n_failed,
        }
    )
    if metrics.ttft_ms is not None:
        mlflow.log_metrics(
            {
                "ttft_ms_p50": metrics.ttft_ms.p50,
                "ttft_ms_p95": metrics.ttft_ms.p95,
                "ttft_ms_p99": metrics.ttft_ms.p99,
            }
        )
    if metrics.inter_token_latency_ms is not None:
        mlflow.log_metrics(
            {
                "inter_token_latency_ms_p50": metrics.inter_token_latency_ms.p50,
                "inter_token_latency_ms_p95": metrics.inter_token_latency_ms.p95,
                "inter_token_latency_ms_p99": metrics.inter_token_latency_ms.p99,
            }
        )
    if metrics.throughput_tokens_per_sec is not None:
        mlflow.log_metric("throughput_tokens_per_sec", metrics.throughput_tokens_per_sec)
    if metrics.execution_accuracy is not None:
        mlflow.log_metric("execution_accuracy", metrics.execution_accuracy)


def log_raw_results(results: list[RequestResult]) -> None:
    """Logs every request's full raw record as a JSONL artifact — the
    point of keeping per-request granularity (see result_schema.py's
    docstring) is that a later analysis can recompute percentiles under a
    different definition without re-running the sweep against real
    hardware. Written to a temp file first since mlflow.log_artifact wants
    a real path, not an in-memory buffer."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "requests.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for result in results:
                f.write(result.model_dump_json() + "\n")
        mlflow.log_artifact(str(path))


def log_thermal_flag(cooled_down: bool) -> None:
    """Logs whether the configuration's pre-run cooldown wait actually
    succeeded (see thermal.wait_for_cooldown) — a config that ran hot
    should be visibly flagged in the results, not silently treated the
    same as a properly-cooled run."""
    mlflow.log_param("cooled_down_before_run", cooled_down)


def log_manifest_reference(manifest_path: Path) -> None:
    """Logs models/MANIFEST.json's content as a param snapshot, so a run
    is traceable back to exactly which model artifact hash it benchmarked
    against, even if the models/ directory has since changed."""
    if not manifest_path.exists():
        log.warning("mlflow_tracking.manifest_missing", path=str(manifest_path))
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mlflow.log_dict(manifest, "model_manifest.json")
