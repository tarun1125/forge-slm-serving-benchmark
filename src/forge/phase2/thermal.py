"""Thermal control for the benchmark sweep. Per the plan: "This is a laptop.
Sustained load throttles. Log CPU/GPU temperature, insert a cooldown between
configurations, and randomise configuration order so thermal drift doesn't
correlate with arm."

IMPORTANT — could not be live-tested in the environment that wrote this:
`powermetrics --samplers smc` requires sudo, and sudo needs an interactive
password prompt that this session cannot supply (confirmed: `sudo -n
powermetrics ...` fails immediately with no cached credential, and there is
no way for an automated session to type a real password). The parsing logic
(parse_smc_line) is unit-tested against captured sample output text below,
but the live subprocess path — does `sudo powermetrics` actually prompt
correctly when stdout is piped, does the output format match across macOS
versions — has NOT been verified end-to-end. Run
`python -m forge.phase2.thermal` standalone once (it prints live readings for
10s) before trusting it inside a real sweep.

The cooldown THRESHOLD (what temperature counts as "cooled down") is
deliberately left as a parameter with a placeholder default, not hardcoded —
per the plan, "the thermal-control protocol" is explicitly called out as the
project owner's decision to make and defend, not Claude Code's. Watch a few
real sweep runs, note what temperature this machine idles at vs. what it
reaches under sustained load, and set COOLDOWN_THRESHOLD_C from that
measurement rather than this file's default.
"""

from __future__ import annotations

import random
import re
import subprocess
import threading
import time
from types import TracebackType
from typing import Protocol

from forge.logging_config import get_logger

log = get_logger(__name__)

# Placeholder — see module docstring. Set this from your own machine's
# observed idle-vs-loaded temperatures before trusting cooldown timing.
COOLDOWN_THRESHOLD_C = 70.0
COOLDOWN_TIMEOUT_S = 120.0  # give up waiting and proceed anyway past this, logging that it happened
COOLDOWN_POLL_INTERVAL_S = 2.0

_CPU_DIE_RE = re.compile(r"CPU die temperature:\s*([\d.]+)\s*C", re.IGNORECASE)
_GPU_DIE_RE = re.compile(r"GPU die temperature:\s*([\d.]+)\s*C", re.IGNORECASE)


def parse_smc_line(line: str) -> tuple[str | None, float | None]:
    """Pure parsing, no subprocess — testable without sudo. Returns
    (kind, celsius) where kind is "cpu" | "gpu" | None (line didn't match
    either pattern)."""
    cpu_match = _CPU_DIE_RE.search(line)
    if cpu_match:
        return "cpu", float(cpu_match.group(1))
    gpu_match = _GPU_DIE_RE.search(line)
    if gpu_match:
        return "gpu", float(gpu_match.group(1))
    return None, None


class ThermalMonitor:
    """Runs `sudo powermetrics --samplers smc` as one persistent background
    process for the life of the sweep, so `current()` is a cheap in-memory
    read rather than spawning (and re-prompting sudo for) a new
    powermetrics invocation per request."""

    def __init__(self, sample_interval_ms: int = 1000):
        self._sample_interval_ms = sample_interval_ms
        self._process: subprocess.Popen[str] | None = None
        self._reader_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest_cpu_c: float | None = None
        self._latest_gpu_c: float | None = None
        self._stop_requested = False

    def start(self) -> None:
        log.info("thermal.start", sample_interval_ms=self._sample_interval_ms)
        self._process = subprocess.Popen(
            [
                "sudo",
                "powermetrics",
                "--samplers",
                "smc",
                "-i",
                str(self._sample_interval_ms),
                "-n",
                "0",  # 0 = sample forever, until the process is killed
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
        )
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

    def _read_loop(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        for line in self._process.stdout:
            if self._stop_requested:
                break
            kind, celsius = parse_smc_line(line)
            if kind == "cpu":
                with self._lock:
                    self._latest_cpu_c = celsius
            elif kind == "gpu":
                with self._lock:
                    self._latest_gpu_c = celsius

    def current(self) -> tuple[float | None, float | None]:
        """Returns (cpu_temp_c, gpu_temp_c) — either may be None if no
        sample has arrived yet, or the parse pattern didn't match this
        macOS version's powermetrics output (see module docstring)."""
        with self._lock:
            return self._latest_cpu_c, self._latest_gpu_c

    def stop(self) -> None:
        self._stop_requested = True
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        log.info("thermal.stop")

    def __enter__(self) -> ThermalMonitor:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()


class TemperatureSource(Protocol):
    """Structural type for wait_for_cooldown()'s monitor parameter — lets
    tests pass a lightweight fake instead of a real ThermalMonitor (which
    would spawn a sudo subprocess) without a concrete-class coupling."""

    def current(self) -> tuple[float | None, float | None]: ...


def wait_for_cooldown(
    monitor: TemperatureSource,
    threshold_c: float = COOLDOWN_THRESHOLD_C,
    timeout_s: float = COOLDOWN_TIMEOUT_S,
    poll_interval_s: float = COOLDOWN_POLL_INTERVAL_S,
) -> bool:
    """Blocks until CPU temp drops below threshold_c or timeout_s elapses.
    Returns True if it cooled down in time, False if it gave up — the
    sweep should log this flag onto the next configuration's results so an
    analysis can flag "this config ran hot" rather than silently treating
    it the same as a properly-cooled run."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        cpu_c, _gpu_c = monitor.current()
        if cpu_c is not None and cpu_c < threshold_c:
            return True
        time.sleep(poll_interval_s)
    log.warning(
        "thermal.cooldown_timeout",
        threshold_c=threshold_c,
        timeout_s=timeout_s,
        last_cpu_c=monitor.current()[0],
    )
    return False


def randomize_sweep_order(configs: list) -> list:
    """Shuffle the full config grid before running the sweep, so thermal
    drift over the session's duration doesn't correlate with which arm or
    quantization level happens to run late. Returns a new list — does not
    mutate the input."""
    shuffled = list(configs)
    random.shuffle(shuffled)
    return shuffled


if __name__ == "__main__":
    import sys

    print("Starting powermetrics — this WILL prompt for your sudo password.")
    with ThermalMonitor() as monitor:
        for _ in range(10):
            time.sleep(1)
            cpu_c, gpu_c = monitor.current()
            print(f"CPU: {cpu_c} C  GPU: {gpu_c} C", file=sys.stderr)
