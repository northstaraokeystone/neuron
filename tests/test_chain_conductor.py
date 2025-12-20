"""
NEURON v4.6 Chain Conductor Tests
Test gap-derived rhythm, persistence alpha, human meta-steering, self-conduct verification.
"""

import os
import tempfile

# Set up isolated test environment BEFORE importing
_test_dir = tempfile.mkdtemp()
os.environ["NEURON_LEDGER"] = os.path.join(_test_dir, "test_receipts.jsonl")
os.environ["NEURON_ARCHIVE"] = os.path.join(_test_dir, "test_archive.jsonl")
os.environ["NEURON_RECEIPTS"] = os.path.join(_test_dir, "test_receipts.jsonl")
os.environ["NEURON_BASE"] = _test_dir

# Create data directory and spec file for tests
_data_dir = os.path.join(_test_dir, "data")
os.makedirs(_data_dir, exist_ok=True)
_spec_path = os.path.join(_data_dir, "self_conduct_spec.json")
with open(_spec_path, "w") as f:
    f.write("""{
  "rhythm_source": "gaps_live",
  "alpha_mode": "persistence_survival",
  "human_role": "meta_steer_optional",
  "self_conduct_enabled": true,
  "persistence_window_entries": 1000,
  "min_gap_ms_for_rhythm": 100
}""")

import pytest
from pathlib import Path

from chain_conductor import (
    load_self_conduct_spec,
    derive_rhythm_from_gaps,
    calculate_persistence_alpha,
    human_meta_append,
    verify_self_conduct,
    detect_induced_oscillation,
    ConductorStopRule,
    RHYTHM_SOURCE,
    ALPHA_MODE,
    HUMAN_ROLE,
)


@pytest.fixture(autouse=True)
def clean_files():
    """Clean test files before and after each test."""
    for f in Path(_test_dir).glob("*.jsonl"):
        f.unlink(missing_ok=True)

    yield

    for f in Path(_test_dir).glob("*.jsonl"):
        f.unlink(missing_ok=True)


# ============================================
# GATE 1: SELF-CONDUCT SPEC TESTS
# ============================================


class TestLoadSelfConductSpec:
    """Test load_self_conduct_spec validates correctly."""

    def test_load_valid_spec(self):
        """Valid spec loads and returns dict."""
        spec = load_self_conduct_spec(_spec_path)

        assert spec["rhythm_source"] == "gaps_live"
        assert spec["alpha_mode"] == "persistence_survival"
        assert spec["human_role"] == "meta_steer_optional"
        assert spec["self_conduct_enabled"] is True
        assert "_spec_hash" in spec
        assert "_ingest_receipt" in spec

    def test_spec_hash_computed(self):
        """Spec has dual-hash computed."""
        spec = load_self_conduct_spec(_spec_path)

        # Dual hash format: SHA256:BLAKE3
        assert ":" in spec["_spec_hash"]

    def test_stoprule_on_invalid_rhythm_source(self):
        """StopRule if rhythm_source != gaps_live."""
        bad_spec_path = os.path.join(_test_dir, "bad_spec.json")
        with open(bad_spec_path, "w") as f:
            f.write(
                '{"rhythm_source": "external", "alpha_mode": "persistence_survival", "human_role": "meta_steer_optional", "self_conduct_enabled": true}'
            )

        with pytest.raises(ConductorStopRule) as exc_info:
            load_self_conduct_spec(bad_spec_path)

        assert "invalid_rhythm_source" in str(exc_info.value)


# ============================================
# GATE 2: GAP RHYTHM DERIVATION TESTS
# ============================================


class TestDeriveRhythmFromGapsOnly:
    """Test rhythm derived from gaps only, no external input."""

    def test_rhythm_derived_from_gaps_only(self):
        """rhythm_pattern comes solely from gap_history, no external input."""
        gap_history = [
            {"ts": "2025-01-01T00:00:00Z", "duration_ms": 500},
            {"ts": "2025-01-01T00:01:00Z", "duration_ms": 500},
            {"ts": "2025-01-01T00:02:00Z", "duration_ms": 500},
        ]

        result = derive_rhythm_from_gaps(gap_history)

        assert result["derivation_method"] == "gaps_live"
        assert result["rhythm_pattern"] in ["regular", "syncopated", "chaotic"]
        assert result["gap_count"] == 3

    def test_empty_gaps_returns_silent(self):
        """Empty gap history returns 'silent' pattern."""
        result = derive_rhythm_from_gaps([])

        assert result["rhythm_pattern"] == "silent"
        assert result["gap_count"] == 0

    def test_regular_pattern_on_consistent_gaps(self):
        """Consistent gap durations produce 'regular' pattern."""
        gap_history = [
            {"ts": "2025-01-01T00:00:00Z", "duration_ms": 500},
            {"ts": "2025-01-01T00:01:00Z", "duration_ms": 500},
            {"ts": "2025-01-01T00:02:00Z", "duration_ms": 500},
            {"ts": "2025-01-01T00:03:00Z", "duration_ms": 500},
        ]

        result = derive_rhythm_from_gaps(gap_history)

        assert result["rhythm_pattern"] == "regular"


class TestNoInducedOscillation:
    """Test detect_induced_oscillation returns False for gap-derived rhythm."""

    def test_no_induced_oscillation(self):
        """detect_induced_oscillation() returns False for clean entries."""
        clean_entries = [
            {"id": "a", "ts": "2025-01-01T00:00:00Z", "event_type": "task"},
            {"id": "b", "ts": "2025-01-01T00:01:00Z", "event_type": "task"},
        ]

        assert detect_induced_oscillation(clean_entries) is False

    def test_detects_resonance_injection(self):
        """Detects resonance_injection event type."""
        bad_entries = [
            {
                "id": "a",
                "ts": "2025-01-01T00:00:00Z",
                "event_type": "resonance_injection",
            },
        ]

        assert detect_induced_oscillation(bad_entries) is True

    def test_detects_oscillation_phase(self):
        """Detects oscillation_phase in source_context."""
        bad_entries = [
            {
                "id": "a",
                "ts": "2025-01-01T00:00:00Z",
                "source_context": {"oscillation_phase": "inject"},
            },
        ]

        assert detect_induced_oscillation(bad_entries) is True

    def test_detects_surge_count(self):
        """Detects surge_count > 0."""
        bad_entries = [
            {"id": "a", "ts": "2025-01-01T00:00:00Z", "surge_count": 1},
        ]

        assert detect_induced_oscillation(bad_entries) is True


# ============================================
# GATE 3: PERSISTENCE ALPHA TESTS
# ============================================


class TestPersistenceAlphaWeightsSurvivors:
    """Test entries surviving longest gaps have highest alpha."""

    def test_persistence_alpha_weights_survivors(self):
        """Entries surviving longest gaps have highest alpha."""
        entries = [
            {"id": "old", "ts": "2025-01-01T00:00:00Z"},
            {"id": "new", "ts": "2025-01-01T01:00:00Z"},
        ]
        gap_history = [
            {"ts": "2025-01-01T00:30:00Z", "duration_ms": 60000},  # 1 minute gap
        ]

        result = calculate_persistence_alpha(entries, gap_history)

        # Old entry should have higher weight (survived the gap)
        assert result["weights"]["old"] >= result["weights"]["new"]

    def test_alpha_zero_for_gap_failures(self):
        """Entries created after gaps don't get survival credit."""
        entries = [
            {"id": "after_gap", "ts": "2025-01-01T01:00:00Z"},
        ]
        gap_history = [
            {"ts": "2025-01-01T00:30:00Z", "duration_ms": 60000},
        ]

        result = calculate_persistence_alpha(entries, gap_history)

        # Entry created after gap has zero survival credit
        assert result["weights"]["after_gap"] == 0.0


# ============================================
# GATE 4: HUMAN META STEERING TESTS
# ============================================


class TestHumanMetaOptional:
    """Test human meta-steering is optional."""

    def test_human_meta_optional(self):
        """System functions identically with meta_entry=None."""
        result = human_meta_append(None)

        assert result is None

    def test_human_meta_steers_not_directs(self):
        """Meta adds to composition but doesn't change tempo/pattern."""
        meta_entry = {
            "steering_type": "observation",
            "note": "Human observes the rhythm",
        }

        result = human_meta_append(meta_entry)

        assert result is not None
        assert "_meta_hash" in result
        assert "_receipt" in result

    def test_stoprule_on_human_direction(self):
        """StopRule if human tries to direct (not steer)."""
        bad_meta = {
            "steering_type": "direct",  # Not allowed!
            "command": "inject now",
        }

        with pytest.raises(ConductorStopRule) as exc_info:
            human_meta_append(bad_meta)

        assert "human_direction_attempted" in str(exc_info.value)


# ============================================
# GATE 5: SELF-CONDUCT VERIFICATION TESTS
# ============================================


class TestSelfConductVerified:
    """Test verify_self_conduct confirms emergent rhythm."""

    def test_self_conduct_verified(self):
        """verify_self_conduct() returns self_conducting=True for gap-derived."""
        rhythm = {
            "rhythm_pattern": "regular",
            "tempo_ms": 500,
            "gap_count": 3,
            "derivation_method": "gaps_live",
        }
        alpha_weights = {
            "weights": {"a": 0.8, "b": 0.6},
            "max_gap_survived_ms": 60000,
        }
        entry_history = [
            {"id": "a", "ts": "2025-01-01T00:00:00Z"},
            {"id": "b", "ts": "2025-01-01T00:01:00Z"},
        ]

        result = verify_self_conduct(rhythm, alpha_weights, entry_history)

        assert result["self_conducting"] is True
        assert result["conductor_type"] == "self"

    def test_stoprule_on_induced_oscillation(self):
        """StopRule raised if induced oscillation detected."""
        rhythm = {
            "rhythm_pattern": "regular",
            "derivation_method": "gaps_live",
        }
        alpha_weights = {"weights": {}}
        bad_entries = [
            {"id": "a", "event_type": "resonance_injection"},  # Induced!
        ]

        with pytest.raises(ConductorStopRule) as exc_info:
            verify_self_conduct(rhythm, alpha_weights, bad_entries)

        assert "external_conductor_detected" in str(exc_info.value)


# ============================================
# RECEIPTS TESTS
# ============================================


class TestReceiptsEmitted:
    """Test all functions emit correct receipt types with dual-hash."""

    def test_gap_rhythm_receipt_emitted(self):
        """derive_rhythm_from_gaps emits gap_rhythm_receipt."""
        gaps = [{"ts": "2025-01-01T00:00:00Z", "duration_ms": 500}]
        result = derive_rhythm_from_gaps(gaps)

        assert "_receipt" in result
        assert result["_receipt"]["type"] == "gap_rhythm"
        assert ":" in result["_receipt"]["hash"]  # Dual hash

    def test_persistence_alpha_receipt_emitted(self):
        """calculate_persistence_alpha emits persistence_alpha_receipt."""
        entries = [{"id": "a", "ts": "2025-01-01T00:00:00Z"}]
        gaps = [{"ts": "2025-01-01T00:30:00Z", "duration_ms": 60000}]

        result = calculate_persistence_alpha(entries, gaps)

        assert "_receipt" in result
        assert result["_receipt"]["type"] == "persistence_alpha"

    def test_meta_steer_receipt_emitted(self):
        """human_meta_append emits meta_steer_receipt when meta provided."""
        meta = {"steering_type": "observation", "note": "test"}

        result = human_meta_append(meta)

        assert "_receipt" in result
        assert result["_receipt"]["type"] == "meta_steer"

    def test_self_conduct_receipt_emitted(self):
        """verify_self_conduct emits self_conduct_receipt."""
        rhythm = {"rhythm_pattern": "regular", "derivation_method": "gaps_live"}
        alpha_weights = {"weights": {}}
        entries = []

        result = verify_self_conduct(rhythm, alpha_weights, entries)

        assert "_receipt" in result
        assert result["_receipt"]["type"] == "self_conduct"


# ============================================
# CONSTANTS VERIFICATION TESTS
# ============================================


class TestConstants:
    """Test v4.6 constants are correct."""

    def test_rhythm_source(self):
        """RHYTHM_SOURCE is gaps_live."""
        assert RHYTHM_SOURCE == "gaps_live"

    def test_alpha_mode(self):
        """ALPHA_MODE is persistence_survival."""
        assert ALPHA_MODE == "persistence_survival"

    def test_human_role(self):
        """HUMAN_ROLE is meta_steer_optional."""
        assert HUMAN_ROLE == "meta_steer_optional"
