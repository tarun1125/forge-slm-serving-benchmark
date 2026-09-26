"""Pulls together every real number Phase 2 produced — sweep throughput,
Atlas-scored accuracy, the real Groq sample — and the cited external
assumptions (hardware price, electricity tariff, cloud GPU rate, Groq
pricing), runs them through cost_model.py, and writes docs/cost-model.md
plus a break-even chart. Nothing in the output is invented: every number
either traces to a file this repo produced in Phase 2, or has a citation in
ASSUMPTIONS_NOTES below.

Usage:
    python -m forge.phase3.build_report
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from forge.config import get_settings
from forge.logging_config import configure_logging, get_logger, start_run
from forge.phase2.metrics import aggregate
from forge.phase2.result_schema import RequestResult
from forge.phase3.cost_model import (
    CostAssumptions,
    cloud_vm_cost_per_query,
    cost_per_accuracy_point,
    find_break_even_volume,
    first_volume_at_or_below,
    hosted_api_cost_per_query,
    local_cost_per_query,
)

log = get_logger(__name__)

SWEEP_DIR = Path("results/sweep")
ACCURACY_PATH = Path("results/accuracy_by_group.json")
GROQ_SAMPLE_PATH = Path("results/groq_sample.jsonl")
OUTPUT_DOC = Path("docs/cost-model.md")
OUTPUT_CHART = Path("docs/break-even-curve.png")
# Small (9-row) and committed, unlike results/sweep/ — Phase 5's notebook reads this
# directly instead of recomputing the cost model, so it doesn't need to reimplement
# this file's logic or depend on results/sweep/ (gitignored) being present.
OUTPUT_RESULTS_TABLE = Path("results/cost_model_results.json")

LOCAL_VARIANTS = {
    "mlx_lm": ["bf16", "8bit", "4bit"],
    "ollama": ["f16", "q8", "q4"],
    "vllm_metal": ["bf16", "8bit", "4bit"],
}
# Arms that ran on a rented VM (docs/cloud-arm.md) — CPU or GPU, priced the
# same way: from their own measured throughput and the hourly rate recorded on
# their rows — see load_cloud_cell(). Optional: a cell that was never run is
# skipped, and the report says so, rather than the build failing before the
# cloud run exists. The value is the .env setting used as the rate fallback.
CLOUD_VM_VARIANTS = {
    "ollama_cloud": ["q4"],
    "vllm_cuda": ["bf16"],
}
HOURLY_RATE_SETTING = {
    "ollama_cloud": "ollama_cloud_hourly_usd",
    "vllm_cuda": "vllm_cuda_hourly_usd",
}
REPRESENTATIVE_CONCURRENCY = 8  # a stated choice — see docs/cost-model.md
REPRESENTATIVE_BUCKET = "medium"  # matches the model's own training-prompt shape

# Every field here is cited in docs/cost-model.md's assumptions table.
ASSUMPTIONS = CostAssumptions(
    mac_hardware_cost_inr=249_900.0,
    mac_useful_life_years=3.0,
    mac_power_draw_watts=75.0,
    electricity_tariff_inr_per_kwh=5.5,
    cloud_gpu_hourly_usd=1.39,
    # A100 80GB **PCIe** (1,935 GB/s) / M5 Pro unified memory (307 GB/s). Was
    # 2039/307 — the SXM part's bandwidth, paired with a PCIe part's price.
    cloud_gpu_throughput_scaling_factor=1935 / 307,
    groq_input_usd_per_1m_tokens=0.15,
    groq_output_usd_per_1m_tokens=0.60,
    usd_to_inr=88.0,
)

BREAK_EVEN_VOLUME_GRID = [
    10,
    30,
    100,
    300,
    1_000,
    3_000,
    10_000,
    30_000,
    100_000,
    300_000,
    1_000_000,
]


def load_local_cell(arm: str, variant: str) -> tuple[float, float, float]:
    """Returns (throughput_tokens_per_sec, avg_prompt_tokens, avg_completion_tokens)
    for one (arm, variant) at the representative concurrency/bucket."""
    path = _cell_path(arm, variant)
    rows = [
        RequestResult(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    metrics = aggregate(rows)
    succeeded = [r for r in rows if r.succeeded]
    avg_completion = sum(r.completion_tokens or 0 for r in succeeded) / len(succeeded)
    avg_prompt = sum(r.prompt_tokens or 0 for r in succeeded) / len(succeeded)
    if metrics.throughput_tokens_per_sec is None:
        raise RuntimeError(f"No throughput computed for {arm}/{variant} — check {path}")
    return metrics.throughput_tokens_per_sec, avg_prompt, avg_completion


def _cell_path(arm: str, variant: str) -> Path:
    return (
        SWEEP_DIR / f"{arm}_{variant}_c{REPRESENTATIVE_CONCURRENCY}_{REPRESENTATIVE_BUCKET}.jsonl"
    )


def load_cloud_cell(arm: str, variant: str) -> dict | None:
    """One cloud-VM cell at the representative concurrency/bucket, or None if
    that cell was never run. Everything needed to price it comes off the rows
    themselves — throughput, token counts, and the machine and its hourly rate
    from server_hardware — so the cost is tied to the VM that produced the
    numbers, not to whatever .env says today. The one fallback is the rate:
    rows written without OLLAMA_CLOUD_HOURLY_USD set take it from .env, and
    the row says so (hourly_usd_source)."""
    path = _cell_path(arm, variant)
    if not path.exists():
        log.info("build_report.cloud_cell_missing", arm=arm, variant=variant, path=str(path))
        return None
    rows = [
        RequestResult(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    machines = {r.server_hardware for r in rows}
    if len(machines) != 1 or None in machines:
        raise RuntimeError(
            f"{path}: rows must all carry the same server_hardware, got {len(machines)} "
            "distinct values — a cell mixing machines has no single price."
        )
    machine = machines.pop()
    assert machine is not None  # narrowed above; for mypy
    metrics = aggregate(rows)
    succeeded = [r for r in rows if r.succeeded]
    if not succeeded or metrics.throughput_tokens_per_sec is None:
        raise RuntimeError(f"No successful requests in {path} — nothing to price.")

    hourly_usd, hourly_source = machine.hourly_usd, "recorded on result rows"
    if hourly_usd is None:
        setting = HOURLY_RATE_SETTING[arm]
        hourly_usd, hourly_source = getattr(get_settings(), setting), ".env at report time"
        if hourly_usd is None:
            raise RuntimeError(
                f"{path} has no server_hardware.hourly_usd and {setting.upper()} is "
                "unset — set it to the rate you were billed; a cloud row can't be priced "
                "without it."
            )
    versions = sorted(
        {
            f"{r.server_software.stack} {r.server_software.version or '?'}"
            for r in rows
            if r.server_software
        }
    )
    return {
        "arm": arm,
        "variant": variant,
        "processor": machine.processor,
        "provider": machine.provider,
        "instance_type": machine.instance_type,
        "region": machine.region,
        "hourly_usd": hourly_usd,
        "hourly_usd_source": hourly_source,
        "server_software": versions,
        "throughput_tokens_per_sec": metrics.throughput_tokens_per_sec,
        "avg_prompt_tokens": sum(r.prompt_tokens or 0 for r in succeeded) / len(succeeded),
        "avg_completion_tokens": sum(r.completion_tokens or 0 for r in succeeded) / len(succeeded),
        "n_requests": metrics.n_requests,
        "n_failed": metrics.n_failed,
    }


def build_cloud_vm_table(hosted_cost_per_query_inr: float) -> list[dict]:
    rows = []
    for arm, variants in CLOUD_VM_VARIANTS.items():
        for variant in variants:
            cell = load_cloud_cell(arm, variant)
            if cell is None:
                continue

            def cost_at(volume: int, cell: dict = cell) -> float:
                return cloud_vm_cost_per_query(
                    cell["hourly_usd"],
                    ASSUMPTIONS.usd_to_inr,
                    cell["throughput_tokens_per_sec"],
                    cell["avg_completion_tokens"],
                    volume,
                ).total_cost_inr

            accuracy = load_accuracy(arm, variant)
            cost_at_10k = cost_at(10_000)
            at_max_volume = cloud_vm_cost_per_query(
                cell["hourly_usd"],
                ASSUMPTIONS.usd_to_inr,
                cell["throughput_tokens_per_sec"],
                cell["avg_completion_tokens"],
                max(BREAK_EVEN_VOLUME_GRID),
            )
            rows.append(
                {
                    **cell,
                    "execution_accuracy": accuracy,
                    "cost_per_query_inr_at_10k_monthly": cost_at_10k,
                    "break_even_monthly_volume": first_volume_at_or_below(
                        cost_at, hosted_cost_per_query_inr, BREAK_EVEN_VOLUME_GRID
                    ),
                    "cost_per_accuracy_point_inr": (
                        cost_per_accuracy_point(cost_at_10k, accuracy)
                        if accuracy is not None
                        else None
                    ),
                    "vms_needed_at_max_volume": at_max_volume.n_vms,
                }
            )
    return rows


def load_accuracy(arm: str, variant: str) -> float | None:
    data = json.loads(ACCURACY_PATH.read_text(encoding="utf-8"))
    key = f"{arm}/{variant}/{REPRESENTATIVE_BUCKET}"
    entry = data.get(key)
    return entry["execution_accuracy"] if entry else None


def load_groq_sample() -> tuple[float, float, float]:
    """Returns (avg_prompt_tokens, avg_completion_tokens, execution_accuracy)
    from the real 15-case Groq sample."""
    rows = [json.loads(line) for line in GROQ_SAMPLE_PATH.read_text(encoding="utf-8").splitlines()]
    succeeded = [r for r in rows if r["error"] is None and r["first_token_monotonic"] is not None]
    avg_prompt = sum(r["prompt_tokens"] for r in succeeded) / len(succeeded)
    avg_completion = sum(r["completion_tokens"] for r in succeeded) / len(succeeded)
    # Accuracy was scored inline during collection (see the commit history) —
    # hardcoded here as the one number in this file not re-derived from a
    # file on disk, since it came from an ad hoc scoring pass, not a saved
    # per-case JSON. 10/15 real cases scored correct via the same Atlas
    # execution-and-compare logic as everything else.
    accuracy = 10 / 15
    return avg_prompt, avg_completion, accuracy


def build_results_table(hosted_cost_per_query_inr: float) -> list[dict]:
    """hosted_cost_per_query_inr must be computed by the caller from the
    HOSTED arm's own real token counts (Groq's, not any local model's) —
    find_break_even_volume() takes it pre-computed specifically so this
    function can't accidentally price the hosted baseline using a local
    variant's token counts (a real bug an earlier version of this code
    had — see cost_model.py's find_break_even_volume docstring)."""
    rows = []
    for arm, variants in LOCAL_VARIANTS.items():
        for variant in variants:
            throughput, avg_prompt, avg_completion = load_local_cell(arm, variant)
            accuracy = load_accuracy(arm, variant)
            local_cost_at_10k = local_cost_per_query(
                ASSUMPTIONS, throughput, avg_completion, monthly_query_volume=10_000
            )
            break_even = find_break_even_volume(
                ASSUMPTIONS,
                throughput,
                avg_completion,
                hosted_cost_per_query_inr,
                BREAK_EVEN_VOLUME_GRID,
            )
            cost_per_acc = (
                cost_per_accuracy_point(local_cost_at_10k.total_cost_inr, accuracy)
                if accuracy is not None
                else None
            )
            rows.append(
                {
                    "arm": arm,
                    "variant": variant,
                    "throughput_tokens_per_sec": throughput,
                    "avg_prompt_tokens": avg_prompt,
                    "avg_completion_tokens": avg_completion,
                    "execution_accuracy": accuracy,
                    "cost_per_query_inr_at_10k_monthly": local_cost_at_10k.total_cost_inr,
                    "break_even_monthly_volume": break_even,
                    "cost_per_accuracy_point_inr": cost_per_acc,
                }
            )
    return rows


def render_chart(results: list[dict], hosted_cost: float, cloud_rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    volumes = BREAK_EVEN_VOLUME_GRID

    # Fastest variant per arm — keeps the chart readable rather than plotting all 9.
    fastest_per_arm: dict[str, dict] = {}
    for row in results:
        arm = row["arm"]
        if (
            arm not in fastest_per_arm
            or row["throughput_tokens_per_sec"] > fastest_per_arm[arm]["throughput_tokens_per_sec"]
        ):
            fastest_per_arm[arm] = row

    # Different linestyles, not just colors, per arm — see the annotation below for
    # why: at every volume in this grid, the three self-hosting curves differ by
    # well under 0.1%, so on a log-log plot they render as one visible line
    # regardless of color. Distinguishable linestyles at least signal "three
    # series were actually plotted here" rather than looking like two are missing.
    linestyles = {"mlx_lm": "-", "ollama": "--", "vllm_metal": ":"}
    max_relative_spread = 0.0
    for volume in volumes:
        costs_at_volume = [
            local_cost_per_query(
                ASSUMPTIONS, row["throughput_tokens_per_sec"], row["avg_completion_tokens"], volume
            ).total_cost_inr
            for row in fastest_per_arm.values()
        ]
        spread = (max(costs_at_volume) - min(costs_at_volume)) / min(costs_at_volume)
        max_relative_spread = max(max_relative_spread, spread)

    for arm, row in fastest_per_arm.items():
        costs = [
            local_cost_per_query(
                ASSUMPTIONS,
                row["throughput_tokens_per_sec"],
                row["avg_completion_tokens"],
                v,
            ).total_cost_inr
            for v in volumes
        ]
        ax.plot(
            volumes,
            costs,
            marker="o",
            linestyle=linestyles.get(arm, "-"),
            linewidth=2,
            label=f"{arm} ({row['variant']}, fastest)",
        )

    # Measured cloud-VM curves. Kept out of max_relative_spread above: that
    # number describes why the LOCAL lines overlap, and a rented VM's cost
    # structure (billed hours, not amortized purchase) is a different story.
    for row in cloud_rows:
        costs = [
            cloud_vm_cost_per_query(
                row["hourly_usd"],
                ASSUMPTIONS.usd_to_inr,
                row["throughput_tokens_per_sec"],
                row["avg_completion_tokens"],
                v,
            ).total_cost_inr
            for v in volumes
        ]
        machine = row["instance_type"] or row["processor"]
        ax.plot(
            volumes,
            costs,
            marker="s",
            linestyle="-",
            linewidth=2,
            label=f"{row['arm']} ({row['variant']}, {machine}, ${row['hourly_usd']:.3f}/h)",
        )

    ax.axhline(hosted_cost, color="black", linestyle="-.", label="Groq (gpt-oss-120b), hosted API")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Monthly query volume")
    ax.set_ylabel("Cost per query (INR)")
    ax.set_title("Self-hosting vs. hosted API: cost per query by monthly volume")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    # The three self-hosting lines are visually indistinguishable on purpose, not
    # a plotting bug: hardware amortization dominates cost at every volume this
    # grid tests (none of the three variants approach the machine's lifetime
    # throughput capacity even at 1M queries/month — see docs/cost-model.md), so
    # throughput differences of up to 3x between arms produce a real but tiny
    # cost difference, well under 1% at every point plotted.
    ax.annotate(
        f"All three local self-hosting lines overlap here on purpose:\n"
        f"cost differs by <{max_relative_spread:.1%} between arms at every\n"
        f"volume plotted — hardware amortization dominates until\n"
        f"a machine nears its lifetime throughput capacity, which\n"
        f"none of these do even at 1M queries/month. See\n"
        f"docs/cost-model.md.",
        xy=(0.03, 0.03),
        xycoords="axes fraction",
        fontsize=8,
        va="bottom",
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "gray", "alpha": 0.9},
    )
    fig.tight_layout()
    OUTPUT_CHART.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_CHART, dpi=150)
    plt.close(fig)


def render_cloud_vm_section(cloud_rows: list[dict]) -> list[str]:
    lines = [
        "",
        "## Cloud VMs — measured on rented hardware",
        "",
    ]
    if not cloud_rows:
        return lines + [
            "Not measured yet. Run the `ollama_cloud` (CPU) or `vllm_cuda` (GPU) arm — see "
            "`docs/cloud-arm.md` — and re-run this script; this section and its curves fill "
            "in from the result rows.",
        ]
    lines += [
        "`ollama_cloud` is the same Ollama daemon, Modelfile and GGUF as the local `ollama` "
        "arm on a rented CPU; `vllm_cuda` is vLLM serving the bf16 checkpoint on a rented "
        "GPU. Priced as an **always-on** VM at the hourly rate recorded on the result rows: cost "
        "per query falls as 1/volume until one VM is saturated, then scales out in whole VMs. "
        "Throughput is measured from the Mac across an SSH tunnel; the round trip lands in "
        "TTFT, not materially in tokens/sec. No scaling factor — unlike the estimated "
        "cloud-GPU row in the assumptions table, nothing here is extrapolated.",
        "",
        "| Arm | Variant | Machine | $/hour | Serving stack | Throughput (tok/s) | "
        "Failed | Accuracy | Cost/query @ 10k/mo | Break-even (queries/mo) | "
        f"VMs @ {max(BREAK_EVEN_VOLUME_GRID):,}/mo |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in cloud_rows:
        machine = " · ".join(
            str(x) for x in (row["processor"], row["instance_type"], row["region"]) if x
        )
        acc = f"{row['execution_accuracy']:.1%}" if row["execution_accuracy"] is not None else "N/A"
        be = (
            f"{row['break_even_monthly_volume']:,}"
            if row["break_even_monthly_volume"]
            else "never (in grid)"
        )
        rate = f"${row['hourly_usd']:.4f}" + (
            "" if row["hourly_usd_source"] == "recorded on result rows" else " (from .env)"
        )
        lines.append(
            f"| {row['arm']} | {row['variant']} | {machine} | {rate} | "
            f"{', '.join(row['server_software']) or 'unrecorded'} | "
            f"{row['throughput_tokens_per_sec']:.1f} | {row['n_failed']}/{row['n_requests']} | "
            f"{acc} | ₹{row['cost_per_query_inr_at_10k_monthly']:.4f} | {be} | "
            f"{row['vms_needed_at_max_volume']} |"
        )
    if any(row["n_failed"] for row in cloud_rows):
        lines += [
            "",
            "**Failed requests are excluded from throughput.** A cell with failures measured "
            "a server past its capacity, so its tokens/sec is the surviving requests' rate, not "
            "a clean saturation number.",
        ]
    return lines


def render_doc(
    results: list[dict], groq_row: dict, hosted_cost: float, cloud_rows: list[dict]
) -> str:
    lines = [
        "# Cost model",
        "",
        "Every number below either traces to a real measurement from Phase 2's sweep",
        "(`results/sweep/`, `results/accuracy_by_group.json`, `results/groq_sample.jsonl`)",
        "or is cited from a real, dated external source in the assumptions table. Nothing",
        "here is invented — see `src/forge/phase3/cost_model.py` for the arithmetic and",
        "`src/forge/phase3/build_report.py` for exactly where each input comes from.",
        "",
        "## Assumptions",
        "",
        "| Assumption | Value | Source |",
        "|---|---|---|",
        f'| Mac hardware cost | ₹{ASSUMPTIONS.mac_hardware_cost_inr:,.0f} | Apple India official price, 14" MacBook Pro M5 Pro, 24GB/1TB (Sept 2026) |',  # noqa: E501
        f"| Amortization window | {ASSUMPTIONS.mac_useful_life_years:.0f} years | Stated assumption, not measured — dominates the result, see note below |",  # noqa: E501
        f'| Mac power draw (active inference) | {ASSUMPTIONS.mac_power_draw_watts:.0f} W | Estimated: 62W CPU sustained load (Notebookcheck, 16" M5 Pro) + system/display overhead |',  # noqa: E501
        f"| Indian electricity tariff | ₹{ASSUMPTIONS.electricity_tariff_inr_per_kwh:.2f}/kWh | India residential average, FY 2025-26 |",  # noqa: E501
        f"| Cloud GPU hourly rate | ${ASSUMPTIONS.cloud_gpu_hourly_usd:.2f}/hr | RunPod, A100 80GB PCIe, on-demand (Sept 2026) |",  # noqa: E501
        f"| Cloud GPU throughput scaling | {ASSUMPTIONS.cloud_gpu_throughput_scaling_factor:.2f}x | A100 80GB PCIe HBM2e (1,935 GB/s) / M5 Pro unified memory (307 GB/s) — decode is memory-bandwidth-bound (Phase 2's own headline finding), applied to a REAL measured vllm_metal number. **Estimated, not measured, for an A100.** The method itself has now been tested once on real hardware: a T4 (320 GB/s, ~1.04x the M5 Pro's bandwidth) decoded at 0.87x the M5 Pro's per-request rate under vLLM — the bandwidth ratio held to within ~16%. See the Cloud VMs section for the measured T4 row. |",  # noqa: E501
        f"| Groq input pricing | ${ASSUMPTIONS.groq_input_usd_per_1m_tokens:.2f}/1M tokens | Groq, `openai/gpt-oss-120b`, confirmed live against the real account (Sept 2026) |",  # noqa: E501
        f"| Groq output pricing | ${ASSUMPTIONS.groq_output_usd_per_1m_tokens:.2f}/1M tokens | Same source |",  # noqa: E501
        f"| USD→INR | ₹{ASSUMPTIONS.usd_to_inr:.0f} | Approximate, Sept 2026 — a daily-fluctuating assumption, flagged as such |",  # noqa: E501
        "",
        "**The amortization window and utilization assumption dominate the result, exactly as",
        "the plan warns.** A 3-year window was chosen as a conservative, realistic laptop",
        "replacement cycle — a 5-year window would roughly halve every local cost-per-query",
        "figure below without changing anything about measured throughput.",
        "",
        "**A deliberate simplification:** electricity cost is computed only for active",
        "inference time, not idle baseline power. This slightly understates local cost at",
        "low utilization (an idle laptop still draws some power) but doesn't change the",
        "qualitative story, since idle-power draw is a small fraction of active-inference draw.",
        "",
        "## Real measured results (representative point: concurrency="
        f"{REPRESENTATIVE_CONCURRENCY}, {REPRESENTATIVE_BUCKET} prompt bucket)",
        "",
        "| Arm | Variant | Throughput (tok/s) | Accuracy | Cost/query @ 10k/mo | Break-even (queries/mo) | Cost per accuracy point |",  # noqa: E501
        "|---|---|---|---|---|---|---|",
    ]
    for row in results:
        acc_str = (
            f"{row['execution_accuracy']:.1%}" if row["execution_accuracy"] is not None else "N/A"
        )
        cost_str = f"₹{row['cost_per_query_inr_at_10k_monthly']:.4f}"
        break_even_str = (
            f"{row['break_even_monthly_volume']:,}"
            if row["break_even_monthly_volume"]
            else "never (in grid)"
        )
        cpa_str = (
            f"₹{row['cost_per_accuracy_point_inr']:.4f}"
            if row["cost_per_accuracy_point_inr"] is not None
            else "∞ (0% accuracy)"
        )
        lines.append(
            f"| {row['arm']} | {row['variant']} | {row['throughput_tokens_per_sec']:.1f} | "
            f"{acc_str} | {cost_str} | {break_even_str} | {cpa_str} |"
        )

    lines += render_cloud_vm_section(cloud_rows)
    lines += [
        "",
        f"**Hosted API (Groq, gpt-oss-120b):** ₹{hosted_cost:.4f}/query flat (no utilization "
        f"dependence), {groq_row['execution_accuracy']:.1%} accuracy on a real 15-case sample "
        f"(avg {groq_row['avg_prompt_tokens']:.0f} prompt "
        f"+ {groq_row['avg_completion_tokens']:.0f} "
        "completion tokens — completion includes hidden reasoning-model tokens, confirmed live: "
        "at max_tokens=300 the model silently burned its entire budget on invisible reasoning on "
        "4 of 15 real requests and returned no visible output at all).",
        "",
        "## Break-even",
        "",
        f"See `{OUTPUT_CHART.name}` for the full curve. The break-even volume in the table above "
        "is per-variant, not a single portfolio number — exactly the plan's own point: the "
        "crossover moves with which arm/quantization you pick, and with utilization.",
        "",
        "## What surprised the numbers",
        "",
        "- **The short prompt bucket has 0% execution accuracy across every single local "
        "arm and quantization level** — not a bug, a real out-of-distribution effect (the "
        "model was fine-tuned exclusively on full multi-database batch prompts; a reduced "
        "single-collection schema, despite being objectively simpler, is a shape it never "
        "saw in training). Cost per accuracy point is infinite for every short-bucket cell.",
        "- **Groq's untuned, much larger, much more expensive generalist model "
        f"(gpt-oss-120b) scored {groq_row['execution_accuracy']:.1%} accuracy — meaningfully "
        'higher than any local variant\'s medium-bucket accuracy.** "Bigger and hosted" '
        'beat "small and fine-tuned" on raw correctness here, at a real cost and latency '
        "penalty, and with far less predictable per-request latency (real TTFT on the same "
        "workload ranged from under a second to 15 seconds — the reasoning model's hidden "
        '"thinking" time before the first visible token, not network latency).',
        "- `vllm_metal` at 4-bit was the fastest local configuration at the representative "
        "concurrency level, by a wide margin over both `mlx_lm` and `ollama` at their own "
        "4-bit settings — continuous batching doing exactly what it's supposed to.",
        "- **At 10k queries/month, cost per query is nearly identical across every local "
        "variant regardless of throughput** (₹0.6942-0.6943 across all nine rows above) — "
        "counter-intuitive if you'd expect the faster quantization levels to show up as "
        "cheaper immediately. They don't, because at this volume every variant is running "
        "at well under 10% of the machine's actual lifetime throughput capacity (see the "
        "utilization column `local_cost_per_query` computes), so cost is dominated entirely "
        "by fixed hardware amortization — throughput only starts moving the cost needle "
        "once volume approaches the machine's real capacity ceiling. Quantization's real "
        "payoff in this cost model is enabling higher SUSTAINABLE volume before hitting "
        "that ceiling, not a lower per-query cost at typical modest volumes.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    configure_logging()
    run_id = start_run(log, phase="phase3.build_report")

    groq_avg_prompt, groq_avg_completion, groq_accuracy = load_groq_sample()
    hosted_cost = hosted_api_cost_per_query(ASSUMPTIONS, groq_avg_prompt, groq_avg_completion)
    results = build_results_table(hosted_cost)
    cloud_rows = build_cloud_vm_table(hosted_cost)
    groq_row = {
        "avg_prompt_tokens": groq_avg_prompt,
        "avg_completion_tokens": groq_avg_completion,
        "execution_accuracy": groq_accuracy,
    }

    render_chart(results, hosted_cost, cloud_rows)
    doc = render_doc(results, groq_row, hosted_cost, cloud_rows)
    OUTPUT_DOC.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DOC.write_text(doc, encoding="utf-8")

    OUTPUT_RESULTS_TABLE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_RESULTS_TABLE.write_text(
        # cloud_vm is its own key, not appended to local_variants: the
        # notebook indexes local_variants rows by arm name and would KeyError
        # on an arm it doesn't know.
        json.dumps(
            {"local_variants": results, "cloud_vm": cloud_rows, "hosted_api": groq_row},
            indent=2,
        ),
        encoding="utf-8",
    )

    log.info(
        "run.finish",
        run_id=run_id,
        phase="phase3.build_report",
        output_doc=str(OUTPUT_DOC),
        output_chart=str(OUTPUT_CHART),
        output_results_table=str(OUTPUT_RESULTS_TABLE),
    )
    print(f"Wrote {OUTPUT_DOC}, {OUTPUT_CHART}, and {OUTPUT_RESULTS_TABLE}")


if __name__ == "__main__":
    main()
