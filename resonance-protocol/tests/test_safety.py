"""
RESONANCE PROTOCOL - Safety Tests

Tests for stimulation safety and thermal monitoring.

GATE 5: SAFETY_COMPLIANCE
    - Test: all simulated pulses < 30 uC/cm^2
    - Test: thermal limit check halts on 1C rise
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stim_controller import (
    calculate_charge_density,
    check_shannon_limit,
    generate_pulse,
    monitor_thermal,
    test_shannon_limit,
    emit_stimulation_receipt,
    StimulationController,
    StimulationSafetyViolation,
    STIM_CHARGE_DENSITY_LIMIT,
    THERMAL_LIMIT_DELTA_C,
    SHANNON_K,
)

from core import StopRule, enforce_thermal_limit


class TestChargeDensity:
    """Tests for charge density calculation."""

    def test_charge_density_calculation(self):
        """Charge density calculated correctly."""
        # 100 uA, 200 us, 0.001 cm^2
        # Q = 100 * 200 * 1e-6 = 0.02 uC
        # Q/A = 0.02 / 0.001 = 20 uC/cm^2
        density = calculate_charge_density(100, 200, 0.001)

        assert abs(density - 20.0) < 0.01

    def test_charge_density_zero_current(self):
        """Zero current gives zero density."""
        density = calculate_charge_density(0, 200, 0.001)

        assert density == 0.0

    def test_charge_density_rejects_zero_area(self):
        """Zero area raises error."""
        with pytest.raises(ValueError):
            calculate_charge_density(100, 200, 0)


class TestShannonLimit:
    """Tests for Shannon limit safety check."""

    def test_safe_pulse_passes(self):
        """Low charge density passes Shannon check."""
        # 10 uC/cm^2 is well within limit
        result = check_shannon_limit(10.0, 0.01)

        assert result == True

    def test_unsafe_pulse_fails(self):
        """High charge density fails Shannon check."""
        # 50 uC/cm^2 exceeds limit
        result = check_shannon_limit(50.0, 0.05)

        assert result == False

    def test_borderline_fails(self):
        """Charge density at limit fails."""
        result = check_shannon_limit(STIM_CHARGE_DENSITY_LIMIT, 0.03)

        assert result == False


class TestPulseGeneration:
    """Tests for pulse generation with safety check."""

    def test_safe_pulse_generated(self, temp_receipts_dir):
        """Safe pulse parameters generate valid pulse spec."""
        pulse = generate_pulse(
            target_region="hippocampus",
            amplitude_ua=50,
            width_us=100,
            electrode_area_cm2=0.001
        )

        assert pulse is not None
        assert pulse["safety_check_passed"] == True
        assert pulse["target_region"] == "hippocampus"

    def test_unsafe_pulse_rejected(self, temp_receipts_dir):
        """Unsafe pulse parameters return None."""
        pulse = generate_pulse(
            target_region="hippocampus",
            amplitude_ua=1000,  # Very high
            width_us=500,  # Long duration
            electrode_area_cm2=0.0001  # Small electrode
        )

        assert pulse is None

    def test_pulse_includes_charge_metrics(self, temp_receipts_dir):
        """Pulse spec includes charge density and charge per phase."""
        pulse = generate_pulse(
            target_region="cortex",
            amplitude_ua=50,
            width_us=100,
            electrode_area_cm2=0.001
        )

        assert "charge_density_uc_cm2" in pulse
        assert "charge_per_phase_uc" in pulse


class TestThermalMonitoring:
    """Tests for thermal safety monitoring."""

    def test_safe_temperature_passes(self, temp_receipts_dir):
        """Temperature within limits passes check."""
        temps = [37.0, 37.2, 37.5, 37.3]  # All within 1C of baseline
        result = monitor_thermal(temps, baseline_temp=37.0)

        assert result == True

    def test_unsafe_temperature_fails(self, temp_receipts_dir):
        """Temperature exceeding limit fails check."""
        temps = [37.0, 37.5, 38.5]  # 1.5C rise
        result = monitor_thermal(temps, baseline_temp=37.0)

        assert result == False

    def test_thermal_limit_raises_stoprule(self, temp_receipts_dir):
        """GATE 5: Thermal limit violation raises StopRule."""
        with pytest.raises(StopRule):
            enforce_thermal_limit(38.5, 37.0)


class TestShannonLimitTest:
    """Tests for Shannon limit validation."""

    def test_shannon_limit_passes(self, temp_receipts_dir):
        """GATE 5: All conservative pulses pass safety check."""
        result = test_shannon_limit(n_tests=100)

        assert result == True


class TestStimulationController:
    """Tests for stimulation controller class."""

    def test_controller_tracks_pulses(self, temp_receipts_dir):
        """Controller tracks pulse count and total charge."""
        controller = StimulationController(electrode_area_cm2=0.001)

        for _ in range(5):
            controller.request_pulse("hippocampus", 50, 100)

        stats = controller.get_stats()
        assert stats["pulse_count"] == 5
        assert stats["total_charge_uc"] > 0

    def test_controller_rejects_unsafe_pulse(self, temp_receipts_dir):
        """Controller rejects unsafe pulse request."""
        controller = StimulationController(electrode_area_cm2=0.0001)

        pulse = controller.request_pulse("cortex", 1000, 500)

        assert pulse is None

    def test_controller_checks_thermal(self, temp_receipts_dir):
        """Controller rejects pulse when temperature too high."""
        controller = StimulationController()

        pulse = controller.request_pulse(
            "hippocampus",
            50,
            100,
            current_temp=38.5  # Too hot
        )

        assert pulse is None


class TestReceipt:
    """Tests for stimulation receipt emission."""

    def test_stimulation_receipt(self, temp_receipts_dir):
        """Stimulation receipt contains required fields."""
        receipt = emit_stimulation_receipt(
            target_region="hippocampus",
            charge_density=15.0,
            pulse_width_us=100,
            safety_check_passed=True
        )

        assert receipt["type"] == "stimulation"
        assert "target_region" in receipt
        assert "charge_density" in receipt
        assert "safety_check_passed" in receipt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
