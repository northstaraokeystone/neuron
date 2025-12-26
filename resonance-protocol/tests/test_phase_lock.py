"""
RESONANCE PROTOCOL - Phase Locking Tests

Tests for phase prediction and stimulation timing.

GATE 4: PHASE_LOCKING
    - Test: phase prediction error < pi/4 radians
    - Test: stimulation trigger within <2ms window
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phase_predictor import (
    fit_oscillation,
    predict_next_peak,
    calculate_phase_error,
    trigger_stimulation,
    test_phase_lock,
    emit_phase_lock_receipt,
    PhaseLockController,
    PHASE_LOCK_LATENCY_MS,
    STIMULATION_WINDOW_MS,
)


class TestOscillationFitting:
    """Tests for oscillation model fitting."""

    def test_fit_empty_events(self):
        """Fit returns zero model for empty events."""
        model = fit_oscillation([], "swr")

        assert model["frequency_hz"] == 0.0
        assert model["n_events"] == 0

    def test_fit_single_event(self):
        """Fit handles single event gracefully."""
        events = [{"timestamp_ns": 1000000, "amplitude": 1.0}]
        model = fit_oscillation(events, "swr")

        assert model["n_events"] == 1

    def test_fit_regular_events(self):
        """Fit correctly estimates frequency from regular events."""
        # Events at 5ms intervals = 200 Hz
        events = [
            {"timestamp_ns": i * 5_000_000, "amplitude": 1.0}
            for i in range(10)
        ]
        model = fit_oscillation(events, "swr")

        assert model["n_events"] == 10
        assert abs(model["frequency_hz"] - 200.0) < 10  # Within 10 Hz


class TestPeakPrediction:
    """Tests for peak timing prediction."""

    def test_predict_future_peak(self):
        """Prediction returns timestamp in the future."""
        model = {"frequency_hz": 200.0, "phase_offset": 0.0, "amplitude": 1.0}
        current_ns = 1_000_000_000  # 1 second

        predicted_ns = predict_next_peak(model, current_ns)

        assert predicted_ns > current_ns

    def test_predict_period_correct(self):
        """Predicted peak is approximately one period ahead."""
        model = {"frequency_hz": 200.0, "phase_offset": 0.0, "amplitude": 1.0}
        current_ns = 1_000_000_000

        predicted_ns = predict_next_peak(model, current_ns)

        # 200 Hz = 5ms period = 5,000,000 ns
        expected_period_ns = 5_000_000
        time_to_peak = predicted_ns - current_ns

        assert abs(time_to_peak - expected_period_ns) < expected_period_ns


class TestPhaseError:
    """Tests for phase error calculation."""

    def test_zero_error_exact_prediction(self):
        """Zero error when prediction is exact."""
        error = calculate_phase_error(1000, 1000, frequency_hz=200.0)

        assert abs(error) < 1e-6

    def test_half_period_error(self):
        """Half period timing error gives pi phase error."""
        # 200 Hz = 5ms period, half period = 2.5ms = 2,500,000 ns
        predicted = 1_000_000_000
        actual = predicted + 2_500_000

        error = calculate_phase_error(predicted, actual, frequency_hz=200.0)

        assert abs(abs(error) - math.pi) < 0.1


class TestStimulationTrigger:
    """Tests for stimulation triggering logic."""

    def test_trigger_at_target_phase(self):
        """Trigger returns True when at target phase."""
        result = trigger_stimulation(
            predicted_phase=0.0,
            target_phase=0.0,
            tolerance=math.pi / 4
        )

        assert result == True

    def test_no_trigger_outside_tolerance(self):
        """Trigger returns False when outside tolerance."""
        result = trigger_stimulation(
            predicted_phase=0.0,
            target_phase=math.pi,  # Half cycle away
            tolerance=math.pi / 4
        )

        assert result == False

    def test_trigger_within_tolerance(self):
        """Trigger returns True when within tolerance."""
        result = trigger_stimulation(
            predicted_phase=0.1,
            target_phase=0.0,
            tolerance=math.pi / 4
        )

        assert result == True


class TestPhaseLockController:
    """Tests for phase lock controller."""

    def test_controller_initialization(self):
        """Controller initializes correctly."""
        controller = PhaseLockController(target_oscillation="swr")

        assert controller.target_oscillation == "swr"
        assert len(controller.event_history) == 0

    def test_controller_adds_events(self):
        """Controller tracks event history."""
        controller = PhaseLockController()

        for i in range(5):
            controller.add_event({
                "timestamp_ns": i * 5_000_000,
                "amplitude": 1.0,
                "frequency_hz": 200.0,
            })

        assert len(controller.event_history) == 5

    def test_controller_predicts_window(self):
        """Controller predicts stimulation window."""
        controller = PhaseLockController()

        # Add regular events
        for i in range(10):
            controller.add_event({
                "timestamp_ns": i * 5_000_000,
                "amplitude": 1.0,
                "frequency_hz": 200.0,
            })

        window = controller.predict_next_window(50_000_000)

        assert "start_ns" in window
        assert "end_ns" in window
        assert window["end_ns"] > window["start_ns"]


class TestPhaseLockTest:
    """Tests for phase lock validation."""

    def test_phase_lock_passes(self, temp_receipts_dir):
        """GATE 4: Phase prediction error < pi/4 radians for 95% of events."""
        result = test_phase_lock(n_trials=100)

        threshold = math.pi / 4
        assert result["p95_error_rad"] < threshold
        assert result["passed"] == True


class TestReceipt:
    """Tests for receipt emission."""

    def test_phase_lock_receipt(self, temp_receipts_dir):
        """Phase lock receipt contains required fields."""
        receipt = emit_phase_lock_receipt(
            target_oscillation="swr",
            predicted_phase=0.1,
            actual_phase=0.15,
            lock_error_ms=0.5
        )

        assert receipt["type"] == "phase_lock"
        assert "target_oscillation" in receipt
        assert "predicted_phase" in receipt
        assert "actual_phase" in receipt
        assert "phase_error_rad" in receipt
        assert "within_tolerance" in receipt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
