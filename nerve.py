"""
NEURON v4.3: The Nerve Module
AI event auto-append triggers for the triad.
Coordinates bio-silicon continuity across AgentProof, AXIOM, and Grok.
~100 lines. Trigger-native. Receipt-generating.
"""

import random
import time
from datetime import datetime, timezone
from typing import Callable

from neuron import (
    append,
    append_from_agentproof,
    append_from_axiom,
    append_from_grok,
    emit_receipt,
    load_ledger,
    ALLOWED_PROJECTS,
    AI_EVENT_TYPES,
    MAX_TASK_LEN,
)


# Registered hooks for each project
_hooks = {
    "grok": [],
    "agentproof": [],
    "axiom": [],
    "human": [],
}


def register_grok_hook(callback: Callable) -> None:
    """Register callback for Grok events. Called when eviction/reset detected."""
    _hooks["grok"].append(callback)


def register_agentproof_hook(callback: Callable) -> None:
    """Register callback for AgentProof events."""
    _hooks["agentproof"].append(callback)


def register_axiom_hook(callback: Callable) -> None:
    """Register callback for AXIOM events."""
    _hooks["axiom"].append(callback)


def register_human_hook(callback: Callable) -> None:
    """Register callback for human events."""
    _hooks["human"].append(callback)


def _fire_hooks(project: str, event: str, context: dict) -> None:
    """Fire all registered hooks for a project."""
    for callback in _hooks.get(project, []):
        try:
            callback(event, context)
        except Exception:
            pass  # Hooks should not break the system


def trigger_grok_eviction(tokens_evicted: int, reason: str) -> dict:
    """Auto-append Grok eviction entry.

    Called when Grok context window truncates oldest entries.

    Args:
        tokens_evicted: Number of tokens evicted
        reason: Reason for eviction

    Returns:
        Entry dict
    """
    context = {
        "tokens_evicted": tokens_evicted,
        "reason": reason,
        "trigger": "auto"
    }

    entry = append_from_grok("eviction", context)
    _fire_hooks("grok", "eviction", context)

    # Emit ai_event_auto_receipt
    emit_receipt("ai_event_auto", {
        "tenant_id": "neuron",
        "project": "grok",
        "event_type": "eviction",
        "trigger": "auto",
        "source_context": context
    })

    return entry


def trigger_grok_reset(reason: str = "cold_start") -> dict:
    """Auto-append Grok reset entry.

    Called when Grok conversation is reset.

    Args:
        reason: Reason for reset

    Returns:
        Entry dict
    """
    context = {
        "tokens_evicted": 0,
        "reason": reason,
        "trigger": "auto"
    }

    entry = append_from_grok("reset", context)
    _fire_hooks("grok", "reset", context)

    emit_receipt("ai_event_auto", {
        "tenant_id": "neuron",
        "project": "grok",
        "event_type": "reset",
        "trigger": "auto",
        "source_context": context
    })

    return entry


def trigger_agentproof_rollback(tx_hash: str, anomaly_type: str) -> dict:
    """Auto-append AgentProof rollback entry.

    Called when AgentProof anomaly containment is triggered.

    Args:
        tx_hash: Transaction hash that triggered rollback
        anomaly_type: Type of anomaly detected

    Returns:
        Entry dict
    """
    context = {
        "tx_hash": tx_hash,
        "anomaly_type": anomaly_type,
        "trigger": "auto"
    }

    entry = append_from_agentproof("rollback", context)
    _fire_hooks("agentproof", "rollback", context)

    emit_receipt("ai_event_auto", {
        "tenant_id": "neuron",
        "project": "agentproof",
        "event_type": "rollback",
        "trigger": "auto",
        "source_context": context
    })

    return entry


def trigger_agentproof_anchor(tx_hash: str, block_height: int) -> dict:
    """Auto-append AgentProof anchor entry.

    Called when AgentProof commits a blockchain anchor.

    Args:
        tx_hash: Transaction hash of anchor
        block_height: Block height of anchor

    Returns:
        Entry dict
    """
    context = {
        "tx_hash": tx_hash,
        "block_height": block_height,
        "trigger": "auto"
    }

    entry = append_from_agentproof("anchor", context)
    _fire_hooks("agentproof", "anchor", context)

    emit_receipt("ai_event_auto", {
        "tenant_id": "neuron",
        "project": "agentproof",
        "event_type": "anchor",
        "trigger": "auto",
        "source_context": context
    })

    return entry


def trigger_axiom_discovery(law: str, compression: float) -> dict:
    """Auto-append AXIOM law discovery entry.

    Called when AXIOM KAN witnesses a new coordination law.

    Args:
        law: The discovered law (e.g., "V = sqrt(GM/r)")
        compression: Compression ratio achieved

    Returns:
        Entry dict
    """
    context = {
        "law": law,
        "compression": compression,
        "trigger": "auto"
    }

    entry = append_from_axiom("law_discovery", context)
    _fire_hooks("axiom", "law_discovery", context)

    emit_receipt("ai_event_auto", {
        "tenant_id": "neuron",
        "project": "axiom",
        "event_type": "discovery",
        "trigger": "auto",
        "source_context": context
    })

    return entry


def trigger_axiom_entropy_spike(entropy_value: float, threshold: float) -> dict:
    """Auto-append AXIOM entropy spike entry.

    Called when AXIOM swarm entropy exceeds threshold.

    Args:
        entropy_value: Current entropy value
        threshold: Threshold that was exceeded

    Returns:
        Entry dict
    """
    context = {
        "entropy": entropy_value,
        "threshold": threshold,
        "trigger": "auto"
    }

    entry = append_from_axiom("entropy_spike", context)
    _fire_hooks("axiom", "entropy_spike", context)

    emit_receipt("ai_event_auto", {
        "tenant_id": "neuron",
        "project": "axiom",
        "event_type": "entropy_spike",
        "trigger": "auto",
        "source_context": context
    })

    return entry


def trigger_human_interrupt(reason: str) -> dict:
    """Auto-append human interruption entry.

    Called when human task switch or interruption occurs.

    Args:
        reason: Reason for interruption

    Returns:
        Entry dict
    """
    context = {
        "reason": reason,
        "trigger": "auto"
    }

    entry = append(
        project="human",
        task=f"Human interrupt: {reason}"[:MAX_TASK_LEN],
        next_action="resume_when_ready",
        event_type="human_interrupt",
        salience=0.9,
        source_context=context
    )
    _fire_hooks("human", "interrupt", context)

    emit_receipt("ai_event_auto", {
        "tenant_id": "neuron",
        "project": "human",
        "event_type": "interrupt",
        "trigger": "auto",
        "source_context": context
    })

    return entry


def trigger_human_return(gap_minutes: float = 0.0) -> dict:
    """Auto-append human return entry.

    Called when human returns from a gap.

    Args:
        gap_minutes: Duration of the gap

    Returns:
        Entry dict
    """
    context = {
        "gap_minutes": gap_minutes,
        "trigger": "auto"
    }

    entry = append(
        project="human",
        task=f"Human return after {gap_minutes:.1f}m"[:MAX_TASK_LEN],
        next_action="continue_work",
        event_type="human_return",
        salience=0.8,
        source_context=context
    )
    _fire_hooks("human", "return", context)

    emit_receipt("ai_event_auto", {
        "tenant_id": "neuron",
        "project": "human",
        "event_type": "return",
        "trigger": "auto",
        "source_context": context
    })

    return entry


def simulate_triad_activity(n_events: int = 100, seed: int | None = None) -> list:
    """Generate realistic triad activity for testing.

    Creates a mix of events across all projects with realistic timing.

    Args:
        n_events: Number of events to generate
        seed: Random seed for reproducibility

    Returns:
        List of generated entries
    """
    if seed is not None:
        random.seed(seed)

    entries = []

    # Event generators with weights
    generators = [
        (0.25, lambda: trigger_human_interrupt(random.choice(["lunch", "meeting", "call", "break"]))),
        (0.15, lambda: trigger_human_return(random.uniform(5, 120))),
        (0.15, lambda: trigger_grok_eviction(random.randint(1000, 10000), "context_full")),
        (0.10, lambda: trigger_grok_reset("cold_start")),
        (0.10, lambda: trigger_agentproof_rollback(f"0x{random.randint(0, 0xFFFFFF):06x}", "proof_failed")),
        (0.10, lambda: trigger_agentproof_anchor(f"0x{random.randint(0, 0xFFFFFF):06x}", random.randint(1000, 100000))),
        (0.10, lambda: trigger_axiom_discovery(random.choice(["V=kr", "F=ma", "E=mc2"]), random.uniform(0.8, 0.99))),
        (0.05, lambda: trigger_axiom_entropy_spike(random.uniform(0.8, 1.5), 0.75)),
    ]

    # Build cumulative weights
    cumulative = []
    total = 0
    for weight, gen in generators:
        total += weight
        cumulative.append((total, gen))

    for _ in range(n_events):
        r = random.random()
        for threshold, gen in cumulative:
            if r <= threshold:
                entry = gen()
                entries.append(entry)
                break

        # Small delay between events for realistic timestamps
        time.sleep(0.001)

    return entries


if __name__ == "__main__":
    print("NEURON v4.3 - The Nerve Module")
    print("=" * 40)
    print(f"Projects: {list(_hooks.keys())}")
    print(f"Event types: {len(AI_EVENT_TYPES)}")
    print("\nQuick simulation (5 events):")
    entries = simulate_triad_activity(5, seed=42)
    for e in entries:
        print(f"  [{e.get('project')}] {e.get('task')}")
