"""
NEURON v4.5 Gate 4: Human Phase Direction Tests
Test human_direct_phase and manual override functionality.
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

from neuron import (
    human_direct_phase,
    get_current_phase,
    reset_oscillation_state,
)


@pytest.fixture(autouse=True)
def clean_state():
    """Reset oscillation state before and after each test."""
    from neuron import _get_ledger_path, _get_receipts_path

    reset_oscillation_state()

    for path_fn in [_get_ledger_path, _get_receipts_path]:
        path = path_fn()
        if path.exists():
            path.unlink()

    yield

    reset_oscillation_state()

    for path_fn in [_get_ledger_path, _get_receipts_path]:
        path = path_fn()
        if path.exists():
            path.unlink()


class TestHumanInjectOverride:
    """Test human inject direction."""

    def test_human_inject_override(self):
        """direction='inject' sets phase."""
        result = human_direct_phase("inject")

        assert result["direction"] == "inject"
        assert get_current_phase() == "inject"


class TestHumanSurgeOverride:
    """Test human surge direction."""

    def test_human_surge_override(self):
        """direction='surge' sets phase."""
        result = human_direct_phase("surge")

        assert result["direction"] == "surge"
        assert get_current_phase() == "surge"


class TestHumanPhaseReceipt:
    """Test human_phase_receipt is returned."""

    def test_human_phase_receipt(self):
        """Returns human_phase_receipt."""
        result = human_direct_phase("inject")

        assert result["receipt_type"] == "human_phase"
        assert "previous_phase" in result
        assert "human_id" in result
        assert "ts" in result
        assert "hash" in result


class TestInvalidDirection:
    """Test invalid direction raises ValueError."""

    def test_invalid_direction(self):
        """Raises ValueError for unknown direction."""
        with pytest.raises(ValueError) as exc_info:
            human_direct_phase("invalid")

        assert "inject" in str(exc_info.value)
        assert "surge" in str(exc_info.value)


class TestHumanOverridesAuto:
    """Test human timing takes precedence."""

    def test_human_overrides_auto(self):
        """Human timing takes precedence."""
        # Start in inject phase
        reset_oscillation_state()
        assert get_current_phase() == "inject"

        # Human overrides to surge
        human_direct_phase("surge")
        assert get_current_phase() == "surge"

        # Human overrides back to inject
        human_direct_phase("inject")
        assert get_current_phase() == "inject"


class TestHumanLogged:
    """Test human_id is recorded."""

    def test_human_logged(self):
        """human_id recorded."""
        result = human_direct_phase("surge", human_id="test_user_123")

        assert result["human_id"] == "test_user_123"
