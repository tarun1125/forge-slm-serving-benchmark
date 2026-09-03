"""Thermal control for the benchmark sweep. Per the plan: "This is a laptop.
Sustained load throttles. Log CPU/GPU temperature, insert a cooldown between
configurations, and randomise configuration order so thermal drift doesn't
correlate with arm."

The plan says "temperature," but this macOS version's powermetrics has no
sampler that reports a raw Celsius value — the historical `--samplers smc`
this module originally targeted returns "unrecognized sampler: smc" here.
`powermetrics -h` lists what's actually available: tasks, battery, network,
disk, interrupts, cpu_power, thermal, sfi, gpu_power, ane_power. Confirmed
against real captured output (`sudo powermetrics --samplers
thermal,cpu_power,gpu_power -i 1000 -n1`), the relevant fields are:

    **** Thermal pressure ****
    Current pressure level: Nominal

    CPU Power: 372 mW
    GPU Power: 46 mW

`thermal_pressure_level` (Nominal/Fair/Serious/Critical, presumably — only
Nominal has been observed so far) is macOS's own judgment about whether it's
throttling, which is arguably more authoritative for this project's purpose
than a raw temperature would have been: it's literally the signal the OS
uses to decide whether to slow the chip down. Cooldown-waiting is therefore
built around "wait for pressure to return to Nominal," not a numeric
threshold — see wait_for_cooldown().

Design history — three live round-trips to get here, worth reading before
"simplifying" this back to a persistent streamed process:

  1. --samplers smc doesn't exist on this macOS version at all (renamed/
     removed upstream). Fixed by switching to thermal,cpu_power,gpu_power,
     found via `powermetrics -h`, not by guessing another name.
  2. Fixing the sampler alone still produced all-None readings from a
     persistent `-n 0` background process, even though the identical
     parsing logic passed unit tests AND the user's own one-shot
     `-n1 > file` run showed the right fields. Suspected cause: powermetrics
     fully buffers stdout when it isn't a tty, so under continuous sampling
     nothing reaches the pipe until an internal buffer fills, and
     `.terminate()` discards whatever was still held. Added `-b 1`
     (line-buffer powermetrics' own output) and fixed a real second bug
     where the reader loop checked a stop flag before processing each line,
     which would've discarded a final flushed burst even if it arrived.
  3. Still all-None after both fixes. Asked for a live side-by-side: piping
     `-n 5 -b 1` straight to the terminal streamed samples one at a time —
     but that doesn't actually prove `-b 1` fixed anything, because a real
     terminal is a tty, and stdio line-buffers a tty by default regardless
     of the buffer-size flag. It says nothing about the PIPE case this
     module actually uses (stdout=subprocess.PIPE is never a tty). Rather
     than guess a fourth flag blind, this version abandons continuous
     streaming entirely: `_sample_once()` runs a fresh one-shot `-n1`
     invocation per poll, which — unlike continuous sampling — flushes
     unconditionally on process exit and has now been confirmed working
     every single time it was tried, including by the user directly. A
     background thread just calls this on a timer instead of parsing a
     live stream. Slightly more subprocess-spawn overhead; categorically
     more reliable. Sudo's credential cache means only the first sample in
     a session prompts for a password — confirmed empirically across many
     one-shot invocations in this same debugging session, none of which
     re-prompted after the first.
"""

from __future__ import annotations

import random
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from types import TracebackType
from typing import Literal, Protocol

from forge.logging_config import get_logger

log = get_logger(__name__)

# The pressure level considered "cooled down" enough to start the next
# configuration. Nominal is the only level observed on this machine so far;
# if Fair/Serious/Critical ever show up in a real sweep, note what triggered
# them in the failure gallery (Phase 5) rather than just waiting them out.
COOLDOWN_TARGET_LEVEL = "Nominal"
COOLDOWN_TIMEOUT_S = 120.0  # give up waiting and proceed anyway past this, logging that it happened
COOLDOWN_POLL_INTERVAL_S = 2.0

_PRESSURE_RE = re.compile(r"Current pressure level:\s*(\w+)")
_CPU_POWER_RE = re.compile(r"^CPU Power:\s*([\d.]+)\s*mW")
_GPU_POWER_RE = re.compile(r"^GPU Power:\s*([\d.]+)\s*mW")

PowerMetricsField = Literal["pressure", "cpu_power_mw", "gpu_power_mw"]


def parse_powermetrics_line(line: str) -> tuple[PowerMetricsField | None, float | str | None]:
    """Pure parsing, no subprocess — testable without sudo. Returns
    (field, value) where field is None if the line matched nothing.
    "pressure" values are strings (Nominal/Fair/...); the power fields
    are floats in mW.

    ^-anchored on the power lines deliberately: "Combined Power (CPU + GPU +
    ANE): 418 mW" contains "GPU" as a substring but does not start with
    "GPU Power:" — anchoring avoids matching it by accident."""
    pressure_match = _PRESSURE_RE.search(line)
    if pressure_match:
        return "pressure", pressure_match.group(1)
    cpu_match = _CPU_POWER_RE.match(line)
    if cpu_match:
        return "cpu_power_mw", float(cpu_match.group(1))
    gpu_match = _GPU_POWER_RE.match(line)
    if gpu_match:
        return "gpu_power_mw", float(gpu_match.group(1))
    return None, None


@dataclass(frozen=True)
class ThermalReading:
    pressure_level: str | None
    cpu_power_mw: float | None
    gpu_power_mw: float | None


_EMPTY_READING = ThermalReading(pressure_level=None, cpu_power_mw=None, gpu_power_mw=None)


class ThermalMonitor:
    """Polls a fresh one-shot `sudo powermetrics ... -n1` sample on a
    background timer, rather than parsing a persistent continuous stream —
    see module docstring for why the streaming approach was abandoned after
    three separate live failures. `current()` is a cheap in-memory read of
    whatever the last completed poll found."""

    def __init__(self, sample_interval_ms: int = 1000, subprocess_timeout_s: float = 10.0):
        self._sample_interval_ms = sample_interval_ms
        self._subprocess_timeout_s = subprocess_timeout_s
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest = _EMPTY_READING

    def _sample_once(self) -> ThermalReading:
        try:
            result = subprocess.run(
                [
                    "sudo",
                    "powermetrics",
                    "--samplers",
                    "thermal,cpu_power,gpu_power",
                    "-i",
                    str(self._sample_interval_ms),
                    "-n",
                    "1",
                ],
                capture_output=True,
                text=True,
                timeout=self._sample_interval_ms / 1000 + self._subprocess_timeout_s,
            )
        except subprocess.TimeoutExpired:
            log.warning("thermal.sample_timeout")
            return _EMPTY_READING

        if result.returncode != 0:
            log.warning(
                "thermal.sample_error",
                returncode=result.returncode,
                stderr=result.stderr[:200] if result.stderr else None,
            )
            return _EMPTY_READING

        pressure: str | None = None
        cpu_power_mw: float | None = None
        gpu_power_mw: float | None = None
        for line in result.stdout.splitlines():
            field, value = parse_powermetrics_line(line)
            if field == "pressure":
                pressure = str(value)
            elif field == "cpu_power_mw":
                cpu_power_mw = float(value)  # type: ignore[arg-type]
            elif field == "gpu_power_mw":
                gpu_power_mw = float(value)  # type: ignore[arg-type]

        return ThermalReading(pressure, cpu_power_mw, gpu_power_mw)

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            reading = self._sample_once()
            with self._lock:
                self._latest = reading

    def start(self) -> None:
        log.info("thermal.start", sample_interval_ms=self._sample_interval_ms)
        # Take one sample synchronously first — fail loud (well, log loud)
        # if sudo/powermetrics access isn't actually working, rather than
        # silently polling None forever in the background thread.
        initial = self._sample_once()
        with self._lock:
            self._latest = initial
        if initial.pressure_level is None:
            log.warning(
                "thermal.start_initial_sample_empty",
                hint="check sudo access and that powermetrics accepts these samplers",
            )
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def current(self) -> ThermalReading:
        with self._lock:
            return self._latest

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._sample_interval_ms / 1000 + self._subprocess_timeout_s)
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

    def current(self) -> ThermalReading: ...


def wait_for_cooldown(
    monitor: TemperatureSource,
    target_level: str = COOLDOWN_TARGET_LEVEL,
    timeout_s: float = COOLDOWN_TIMEOUT_S,
    poll_interval_s: float = COOLDOWN_POLL_INTERVAL_S,
) -> bool:
    """Blocks until thermal pressure returns to target_level or timeout_s
    elapses. Returns True if it cooled down in time, False if it gave up —
    the sweep should log this flag onto the next configuration's results so
    an analysis can flag "this config ran hot" rather than silently
    treating it the same as a properly-cooled run."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        reading = monitor.current()
        if reading.pressure_level == target_level:
            return True
        time.sleep(poll_interval_s)
    log.warning(
        "thermal.cooldown_timeout",
        target_level=target_level,
        timeout_s=timeout_s,
        last_pressure_level=monitor.current().pressure_level,
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
            reading = monitor.current()
            print(
                f"pressure: {reading.pressure_level}  "
                f"CPU: {reading.cpu_power_mw} mW  GPU: {reading.gpu_power_mw} mW",
                file=sys.stderr,
            )
