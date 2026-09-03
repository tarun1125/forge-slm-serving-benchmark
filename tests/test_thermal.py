from forge.phase2.thermal import parse_smc_line, randomize_sweep_order, wait_for_cooldown

# Sample lines shaped like real `powermetrics --samplers smc` output on
# Apple Silicon. Parsing is unit-tested against this captured shape since
# the live subprocess path itself needs an interactive sudo password this
# environment can't supply — see thermal.py's module docstring.
SAMPLE_OUTPUT = """
*** Sampled system activity (Thu Sep  3 10:30:00 2026 +0000) (1000.00ms elapsed) ***


**** SMC ****

CPU die temperature: 45.23 C
GPU die temperature: 42.10 C
""".splitlines()


class TestParseSmcLine:
    def test_parses_cpu_temperature(self):
        kind, celsius = parse_smc_line("CPU die temperature: 45.23 C")
        assert kind == "cpu"
        assert celsius == 45.23

    def test_parses_gpu_temperature(self):
        kind, celsius = parse_smc_line("GPU die temperature: 42.10 C")
        assert kind == "gpu"
        assert celsius == 42.10

    def test_is_case_insensitive(self):
        kind, celsius = parse_smc_line("cpu die temperature: 50.0 c")
        assert kind == "cpu"
        assert celsius == 50.0

    def test_unrelated_line_returns_none(self):
        kind, celsius = parse_smc_line("*** Sampled system activity ***")
        assert kind is None
        assert celsius is None

    def test_blank_line_returns_none(self):
        kind, celsius = parse_smc_line("")
        assert kind is None
        assert celsius is None

    def test_full_sample_output_extracts_both_readings(self):
        readings = dict(parse_smc_line(line) for line in SAMPLE_OUTPUT)
        readings.pop(None, None)
        assert readings == {"cpu": 45.23, "gpu": 42.10}


class TestRandomizeSweepOrder:
    def test_returns_same_elements(self):
        configs = [1, 2, 3, 4, 5]
        shuffled = randomize_sweep_order(configs)
        assert sorted(shuffled) == sorted(configs)

    def test_does_not_mutate_input(self):
        configs = [1, 2, 3]
        original = list(configs)
        randomize_sweep_order(configs)
        assert configs == original


class _FakeMonitor:
    """Feeds a scripted sequence of (cpu, gpu) readings to
    wait_for_cooldown() without needing a real powermetrics subprocess."""

    def __init__(self, readings: list[tuple[float | None, float | None]]):
        self._readings = readings
        self._index = 0

    def current(self) -> tuple[float | None, float | None]:
        reading = self._readings[min(self._index, len(self._readings) - 1)]
        self._index += 1
        return reading


class TestWaitForCooldown:
    def test_returns_true_immediately_when_already_below_threshold(self):
        monitor = _FakeMonitor([(60.0, 55.0)])
        result = wait_for_cooldown(monitor, threshold_c=70.0, timeout_s=1.0, poll_interval_s=0.01)
        assert result is True

    def test_returns_false_after_timeout_when_never_cools(self):
        monitor = _FakeMonitor([(95.0, 90.0)])
        result = wait_for_cooldown(monitor, threshold_c=70.0, timeout_s=0.05, poll_interval_s=0.01)
        assert result is False

    def test_treats_unknown_reading_as_not_cooled(self):
        monitor = _FakeMonitor([(None, None)])
        result = wait_for_cooldown(monitor, threshold_c=70.0, timeout_s=0.05, poll_interval_s=0.01)
        assert result is False
