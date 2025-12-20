"""
NEURON v4.5 Gate 2: Frequency Tuning Tests
Test load_frequencies, tune_frequency, and frequency sources.
"""

import os
import tempfile

# Set up isolated test environment BEFORE importing
_test_dir = tempfile.mkdtemp()
os.environ["NEURON_LEDGER"] = os.path.join(_test_dir, "test_receipts.jsonl")
os.environ["NEURON_ARCHIVE"] = os.path.join(_test_dir, "test_archive.jsonl")
os.environ["NEURON_RECEIPTS"] = os.path.join(_test_dir, "test_receipts.jsonl")
os.environ["NEURON_BASE"] = _test_dir

import pytest

from frequency import (
    load_frequencies,
    tune_frequency,
    adaptive_frequency,
    FREQUENCY_SOURCES,
)


@pytest.fixture(autouse=True)
def clean_files():
    """Clean test files before and after each test."""
    from pathlib import Path

    for f in Path(_test_dir).glob("*.jsonl"):
        f.unlink(missing_ok=True)

    yield

    for f in Path(_test_dir).glob("*.jsonl"):
        f.unlink(missing_ok=True)


class TestLoadFrequencies:
    """Test load_frequencies returns all sources."""

    def test_load_frequencies(self):
        """Returns dict with all 5 sources."""
        config = load_frequencies()

        assert "frequencies" in config
        frequencies = config["frequencies"]

        # Should have all 5 sources
        for source in FREQUENCY_SOURCES:
            assert source in frequencies, f"Missing source: {source}"


class TestTuneHumanCircadian:
    """Test HUMAN_CIRCADIAN frequency."""

    def test_tune_human_circadian(self):
        """period = 86400."""
        receipt = tune_frequency("HUMAN_CIRCADIAN")

        assert receipt["source"] == "HUMAN_CIRCADIAN"
        assert receipt["period_seconds"] == 86400
        assert receipt["frequency_hz"] == pytest.approx(1.0 / 86400, rel=1e-6)


class TestTuneHumanFocus:
    """Test HUMAN_FOCUS frequency."""

    def test_tune_human_focus(self):
        """period = 5400."""
        receipt = tune_frequency("HUMAN_FOCUS")

        assert receipt["source"] == "HUMAN_FOCUS"
        assert receipt["period_seconds"] == 5400
        assert receipt["frequency_hz"] == pytest.approx(1.0 / 5400, rel=1e-6)


class TestTuneMarsDelay:
    """Test MARS_LIGHT_DELAY frequency."""

    def test_tune_mars_delay(self):
        """period in (180, 1320)."""
        receipt = tune_frequency("MARS_LIGHT_DELAY")

        assert receipt["source"] == "MARS_LIGHT_DELAY"
        # Should be average of range (180, 1320) = 750
        assert 180 <= receipt["period_seconds"] <= 1320


class TestTuneInvalidSource:
    """Test invalid frequency source raises ValueError."""

    def test_tune_invalid_source(self):
        """Raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            tune_frequency("INVALID_SOURCE")

        assert "Invalid frequency source" in str(exc_info.value)


class TestAdaptiveFrequency:
    """Test adaptive frequency tuning."""

    def test_adaptive_frequency(self):
        """Returns float > 0."""
        ledger = [
            {"event_type": "task", "alpha": 0.5},
            {"event_type": "phase_transition", "alpha": 0.8},
        ]

        freq_hz = adaptive_frequency(ledger, target_transitions=1)

        assert isinstance(freq_hz, float)
        assert freq_hz > 0
