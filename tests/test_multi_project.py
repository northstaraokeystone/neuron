"""
NEURON v4.3: Multi-Project Append Tests (Gate 1)
Tests for triad direct writes to shared ledger.
"""

import os
import tempfile
from pathlib import Path

import pytest

# Set up isolated test environment BEFORE imports
_test_dir = tempfile.mkdtemp()
os.environ["NEURON_LEDGER"] = str(Path(_test_dir) / "test_receipts.jsonl")
os.environ["NEURON_ARCHIVE"] = str(Path(_test_dir) / "test_archive.jsonl")
os.environ["NEURON_RECEIPTS"] = str(Path(_test_dir) / "test_stress_receipts.jsonl")

from neuron import (
    append,
    append_from_agentproof,
    append_from_axiom,
    append_from_grok,
    load_ledger,
    _read_ledger,
    _write_ledger,
    ALLOWED_PROJECTS,
    AI_EVENT_TYPES,
    StopRule,
)


@pytest.fixture(autouse=True)
def clean_ledger():
    """Clean ledger before each test."""
    ledger_path = Path(os.environ["NEURON_LEDGER"])
    if ledger_path.exists():
        ledger_path.unlink()
    yield
    if ledger_path.exists():
        ledger_path.unlink()


class TestAppendHumanDefault:
    """Test that append() defaults to project='human' is not needed - actually defaults to first positional."""

    def test_append_with_human_project(self):
        """Test appending with explicit human project."""
        entry = append(project="human", task="Human task", next_action="next")
        assert entry["project"] == "human"
        assert entry["event_type"] == "task"

    def test_append_human_event_types(self):
        """Test human-specific event types."""
        entry = append(project="human", task="Interrupt", next_action="pause",
                      event_type="human_interrupt")
        assert entry["event_type"] == "human_interrupt"


class TestAppendAgentproof:
    """Test append_from_agentproof wrapper."""

    def test_append_from_agentproof_rollback(self):
        """Test AgentProof rollback event."""
        entry = append_from_agentproof("rollback", {"tx_hash": "0xabc123"})
        assert entry["project"] == "agentproof"
        assert entry["event_type"] == "agentproof_rollback"
        assert entry["source_context"]["tx_hash"] == "0xabc123"
        assert entry["source_context"]["trigger"] == "auto"

    def test_append_from_agentproof_anchor(self):
        """Test AgentProof anchor event."""
        entry = append_from_agentproof("anchor", {"tx_hash": "0xdef456", "block": 12345})
        assert entry["project"] == "agentproof"
        assert entry["event_type"] == "agentproof_anchor"


class TestAppendAxiom:
    """Test append_from_axiom wrapper."""

    def test_append_from_axiom_discovery(self):
        """Test AXIOM law discovery event."""
        entry = append_from_axiom("law_discovery", {"law": "V = sqrt(GM/r)", "compression": 0.94})
        assert entry["project"] == "axiom"
        assert entry["event_type"] == "axiom_law_discovery"
        assert entry["source_context"]["law"] == "V = sqrt(GM/r)"
        assert entry["source_context"]["trigger"] == "auto"

    def test_append_from_axiom_entropy(self):
        """Test AXIOM entropy spike event."""
        entry = append_from_axiom("entropy_spike", {"entropy": 0.95, "threshold": 0.8})
        assert entry["project"] == "axiom"
        assert entry["event_type"] == "axiom_entropy_spike"


class TestAppendGrok:
    """Test append_from_grok wrapper."""

    def test_append_from_grok_eviction(self):
        """Test Grok eviction event."""
        entry = append_from_grok("eviction", {"tokens_evicted": 5000})
        assert entry["project"] == "grok"
        assert entry["event_type"] == "grok_eviction"
        assert entry["source_context"]["tokens_evicted"] == 5000
        assert entry["source_context"]["trigger"] == "auto"

    def test_append_from_grok_reset(self):
        """Test Grok reset event."""
        entry = append_from_grok("reset", {"reason": "cold_start"})
        assert entry["project"] == "grok"
        assert entry["event_type"] == "grok_reset"


class TestMixedProjectsInLedger:
    """Test ledger contains entries from multiple projects."""

    def test_mixed_projects(self):
        """Test that ledger contains entries from multiple projects."""
        append(project="human", task="Human task", next_action="next")
        append_from_agentproof("rollback", {"tx_hash": "0xabc"})
        append_from_axiom("law_discovery", {"law": "F=ma"})
        append_from_grok("eviction", {"tokens_evicted": 1000})

        ledger = load_ledger()
        projects = set(e["project"] for e in ledger)

        assert len(projects) >= 3
        assert "human" in projects
        assert "agentproof" in projects
        assert "axiom" in projects
        assert "grok" in projects


class TestInvalidProjectRejected:
    """Test that invalid project raises StopRule."""

    def test_invalid_project_raises(self):
        """Test that invalid project raises StopRule."""
        with pytest.raises(StopRule) as exc_info:
            append(project="invalid_project", task="test", next_action="test")

        assert "invalid_project" in str(exc_info.value)


class TestSourceContextPreserved:
    """Test that source_context is stored in entry."""

    def test_source_context_preserved(self):
        """Test source_context is preserved in entry."""
        context = {"key1": "value1", "key2": 123, "nested": {"a": 1}}
        entry = append(project="neuron", task="test", next_action="test",
                      source_context=context)

        assert entry["source_context"]["key1"] == "value1"
        assert entry["source_context"]["key2"] == 123
        assert entry["source_context"]["nested"]["a"] == 1


class TestSharedReceiptEmitted:
    """Test that shared_ledger_append_receipt is emitted."""

    def test_receipt_emitted(self):
        """Test that receipt is emitted on append."""
        receipts_path = Path(os.environ["NEURON_RECEIPTS"])

        # Append an entry
        append(project="neuron", task="test", next_action="test")

        # Check receipts file exists and contains entry
        assert receipts_path.exists()
        with open(receipts_path) as f:
            content = f.read()
            assert "shared_ledger_append" in content


class TestEventTypeValidation:
    """Test event type validation."""

    def test_unknown_event_type_defaults(self):
        """Test that unknown event type defaults to 'unknown'."""
        entry = append(project="neuron", task="test", next_action="test",
                      event_type="totally_invalid")
        assert entry["event_type"] == "unknown"

    def test_valid_event_types_accepted(self):
        """Test that valid event types are accepted."""
        for event_type in ["task", "eviction", "rollback", "discovery"]:
            entry = append(project="neuron", task="test", next_action="test",
                          event_type=event_type)
            assert entry["event_type"] == event_type
