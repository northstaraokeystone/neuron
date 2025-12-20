"""
NEURON v4.3: Nerve Module Tests (Gate 3)
Tests for AI event auto-append triggers.
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

from nerve import (
    trigger_grok_eviction,
    trigger_agentproof_rollback,
    trigger_axiom_discovery,
    trigger_human_interrupt,
    simulate_triad_activity,
    register_grok_hook,
    register_agentproof_hook,
    register_axiom_hook,
    _hooks,
)
from neuron import load_ledger


@pytest.fixture(autouse=True)
def clean_ledger():
    """Clean ledger and hooks before each test."""
    ledger_path = Path(os.environ["NEURON_LEDGER"])
    if ledger_path.exists():
        ledger_path.unlink()

    # Clear hooks
    for key in _hooks:
        _hooks[key] = []

    yield

    if ledger_path.exists():
        ledger_path.unlink()


class TestTriggerGrokEviction:
    """Test Grok eviction auto-append."""

    def test_trigger_grok_eviction(self):
        """Test entry created with project='grok', event_type='grok_eviction'."""
        entry = trigger_grok_eviction(5000, "context_window_full")

        assert entry["project"] == "grok"
        assert entry["event_type"] == "grok_eviction"
        assert entry["source_context"]["tokens_evicted"] == 5000
        assert entry["source_context"]["trigger"] == "auto"

    def test_grok_eviction_in_ledger(self):
        """Test eviction entry appears in ledger."""
        trigger_grok_eviction(3000, "test_reason")
        ledger = load_ledger()

        assert len(ledger) == 1
        assert ledger[0]["project"] == "grok"


class TestTriggerAgentproofRollback:
    """Test AgentProof rollback auto-append."""

    def test_trigger_agentproof_rollback(self):
        """Test entry created with project='agentproof', event_type='agentproof_rollback'."""
        entry = trigger_agentproof_rollback("0xabc123", "proof_verification_failed")

        assert entry["project"] == "agentproof"
        assert entry["event_type"] == "agentproof_rollback"
        assert entry["source_context"]["tx_hash"] == "0xabc123"
        assert entry["source_context"]["anomaly_type"] == "proof_verification_failed"
        assert entry["source_context"]["trigger"] == "auto"


class TestTriggerAxiomDiscovery:
    """Test AXIOM discovery auto-append."""

    def test_trigger_axiom_discovery(self):
        """Test entry created with project='axiom', event_type='axiom_law_discovery'."""
        entry = trigger_axiom_discovery("V = sqrt(GM/r)", 0.94)

        assert entry["project"] == "axiom"
        assert entry["event_type"] == "axiom_law_discovery"
        assert entry["source_context"]["law"] == "V = sqrt(GM/r)"
        assert entry["source_context"]["compression"] == 0.94
        assert entry["source_context"]["trigger"] == "auto"


class TestTriggerHumanInterrupt:
    """Test human interrupt auto-append."""

    def test_trigger_human_interrupt(self):
        """Test entry created with project='human', event_type='human_interrupt'."""
        entry = trigger_human_interrupt("lunch_break")

        assert entry["project"] == "human"
        assert entry["event_type"] == "human_interrupt"
        assert entry["source_context"]["reason"] == "lunch_break"
        assert entry["source_context"]["trigger"] == "auto"


class TestAutoReceiptEmitted:
    """Test ai_event_auto_receipt is emitted."""

    def test_receipt_emitted(self):
        """Test that ai_event_auto_receipt is emitted."""
        receipts_path = Path(os.environ["NEURON_RECEIPTS"])

        trigger_grok_eviction(1000, "test")

        assert receipts_path.exists()
        with open(receipts_path) as f:
            content = f.read()
            assert "ai_event_auto" in content


class TestSimulateTriad:
    """Test simulate_triad_activity returns mixed entries."""

    def test_simulate_triad_returns_entries(self):
        """Test simulate_triad_activity returns list of entries."""
        entries = simulate_triad_activity(n_events=20, seed=42)

        assert len(entries) == 20
        assert all(isinstance(e, dict) for e in entries)

    def test_simulate_triad_mixed_projects(self):
        """Test simulation generates mixed project entries."""
        entries = simulate_triad_activity(n_events=50, seed=42)

        projects = set(e["project"] for e in entries)
        # Should have at least 3 different projects
        assert len(projects) >= 3


class TestHooksRegistered:
    """Test callbacks fire on trigger."""

    def test_hook_fires_on_grok_trigger(self):
        """Test registered Grok hook fires."""
        hook_called = []

        def my_hook(event, context):
            hook_called.append((event, context))

        register_grok_hook(my_hook)
        trigger_grok_eviction(1000, "test")

        assert len(hook_called) == 1
        assert hook_called[0][0] == "eviction"

    def test_hook_fires_on_agentproof_trigger(self):
        """Test registered AgentProof hook fires."""
        hook_called = []

        def my_hook(event, context):
            hook_called.append((event, context))

        register_agentproof_hook(my_hook)
        trigger_agentproof_rollback("0xtest", "test_anomaly")

        assert len(hook_called) == 1
        assert hook_called[0][0] == "rollback"

    def test_hook_fires_on_axiom_trigger(self):
        """Test registered AXIOM hook fires."""
        hook_called = []

        def my_hook(event, context):
            hook_called.append((event, context))

        register_axiom_hook(my_hook)
        trigger_axiom_discovery("E=mc2", 0.99)

        assert len(hook_called) == 1
        assert hook_called[0][0] == "law_discovery"
