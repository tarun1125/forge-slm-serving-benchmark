from forge.phase2.thermal import (
    ThermalReading,
    parse_powermetrics_line,
    randomize_sweep_order,
    wait_for_cooldown,
)

# Real output captured on the dev machine via:
#   sudo powermetrics --samplers thermal,cpu_power,gpu_power -i 1000 -n1
# This macOS version has no "smc" sampler (the historical way to get a raw
# Celsius reading) — thermal.py's module docstring has the full story.
REAL_SAMPLE_LINES = """
CPU Power: 372 mW
GPU Power: 46 mW
ANE Power: 0 mW
Combined Power (CPU + GPU + ANE): 418 mW


**** Thermal pressure ****

Current pressure level: Nominal

**** GPU usage ****

GPU HW active frequency: 338 MHz
GPU idle residency:  86.09%
GPU Power: 46 mW
""".splitlines()


class TestParsePowermetricsLine:
    def test_parses_pressure_level(self):
        field, value = parse_powermetrics_line("Current pressure level: Nominal")
        assert field == "pressure"
        assert value == "Nominal"

    def test_parses_cpu_power(self):
        field, value = parse_powermetrics_line("CPU Power: 372 mW")
        assert field == "cpu_power_mw"
        assert value == 372.0

    def test_parses_gpu_power(self):
        field, value = parse_powermetrics_line("GPU Power: 46 mW")
        assert field == "gpu_power_mw"
        assert value == 46.0

    def test_combined_power_line_is_not_mistaken_for_gpu_power(self):
        # This line contains "GPU" as a substring but must not match the
        # anchored ^GPU Power: pattern — a naive `"GPU" in line` check
        # would have wrongly parsed 418 as the GPU's own power draw.
        field, value = parse_powermetrics_line("Combined Power (CPU + GPU + ANE): 418 mW")
        assert field is None
        assert value is None

    def test_unrelated_line_returns_none(self):
        field, value = parse_powermetrics_line("**** GPU usage ****")
        assert field is None
        assert value is None

    def test_blank_line_returns_none(self):
        field, value = parse_powermetrics_line("")
        assert field is None
        assert value is None

    def test_real_captured_sample_extracts_all_three_fields(self):
        results = {}
        for line in REAL_SAMPLE_LINES:
            field, value = parse_powermetrics_line(line)
            if field is not None:
                results[field] = value
        assert results == {
            "cpu_power_mw": 372.0,
            "gpu_power_mw": 46.0,  # last occurrence wins in this reduction; both were 46.0 anyway
            "pressure": "Nominal",
        }


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
    """Feeds a scripted sequence of ThermalReadings to wait_for_cooldown()
    without needing a real powermetrics subprocess."""

    def __init__(self, readings: list[ThermalReading]):
        self._readings = readings
        self._index = 0

    def current(self) -> ThermalReading:
        reading = self._readings[min(self._index, len(self._readings) - 1)]
        self._index += 1
        return reading


class TestWaitForCooldown:
    def test_returns_true_immediately_when_already_at_target_level(self):
        monitor = _FakeMonitor([ThermalReading("Nominal", 300.0, 40.0)])
        result = wait_for_cooldown(
            monitor, target_level="Nominal", timeout_s=1.0, poll_interval_s=0.01
        )
        assert result is True

    def test_returns_false_after_timeout_when_never_reaches_target(self):
        monitor = _FakeMonitor([ThermalReading("Serious", 5000.0, 2000.0)])
        result = wait_for_cooldown(
            monitor, target_level="Nominal", timeout_s=0.05, poll_interval_s=0.01
        )
        assert result is False

    def test_treats_unknown_reading_as_not_cooled(self):
        monitor = _FakeMonitor([ThermalReading(None, None, None)])
        result = wait_for_cooldown(
            monitor, target_level="Nominal", timeout_s=0.05, poll_interval_s=0.01
        )
        assert result is False
