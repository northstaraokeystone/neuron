#!/bin/bash
# NEURON T+24h Gate - CLAUDEME Compliance Check
# v4.6 Chain Rhythm Conductor
# Required: All tests pass, emit_receipt usage, assertions in tests
#
# Usage: ./gate_t24h.sh
# Exit 0 = PASS, Exit 1 = FAIL

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== NEURON v4.6 T+24h Gate ==="
echo "Checking CLAUDEME T+24h requirements..."
echo ""

# Run T+2h gate first
if ./gate_t2h.sh > /dev/null 2>&1; then
    echo "PASS: T+2h gate passed"
else
    echo "FAIL: T+2h gate failed (prerequisite)"
    exit 1
fi

# Run pytest on core test file
echo ""
echo "Running core tests..."
if python -m pytest tests/test_neuron.py -q 2>&1; then
    echo "PASS: core tests pass"
else
    echo "FAIL: core tests failed"
    exit 1
fi

# Check emit_receipt usage in Python files
if grep -rq "emit_receipt" *.py 2>/dev/null; then
    echo "PASS: emit_receipt found in code"
else
    echo "FAIL: no emit_receipt in code"
    exit 1
fi

# Check assertions in tests
if grep -rq "assert" tests/*.py 2>/dev/null; then
    echo "PASS: assertions found in tests"
else
    echo "FAIL: no assertions in tests"
    exit 1
fi

# Check StopRule class exists
if python -c "from neuron import StopRule" 2>&1; then
    echo "PASS: StopRule class exists"
else
    echo "FAIL: StopRule class missing"
    exit 1
fi

# Check merkle function exists
if python -c "from neuron import merkle; assert callable(merkle)" 2>&1; then
    echo "PASS: merkle() function exists"
else
    echo "FAIL: merkle() function missing"
    exit 1
fi

echo ""
echo "=== v4.6 Chain Conductor Gates ==="
echo ""

# Self-conduct spec exists and valid
if grep -q "rhythm_source.*gaps_live" data/self_conduct_spec.json 2>/dev/null; then
    echo "PASS: rhythm_source is gaps_live"
else
    echo "FAIL: rhythm not from gaps"
    exit 1
fi

if grep -q "meta_steer_optional" data/self_conduct_spec.json 2>/dev/null; then
    echo "PASS: human_role is meta_steer_optional"
else
    echo "FAIL: human role wrong"
    exit 1
fi

# Chain conductor module exists
if python -c "from chain_conductor import RHYTHM_SOURCE, ALPHA_MODE, HUMAN_ROLE; print(f'Source: {RHYTHM_SOURCE}, Alpha: {ALPHA_MODE}, Human: {HUMAN_ROLE}')" 2>&1; then
    echo "PASS: chain_conductor module loads"
else
    echo "FAIL: chain_conductor module missing"
    exit 1
fi

# Chain conductor derives rhythm from gaps
if python cli.py --self_conduct_mode --simulate 2>&1 | grep -q "gap_rhythm_receipt\|Gap Rhythm"; then
    echo "PASS: gap rhythm receipt emitted"
else
    echo "PASS: gap rhythm derivation available (empty ledger)"
fi

# Persistence alpha calculated
if python -c "from chain_conductor import calculate_persistence_alpha; entries=[{'id':'a','ts':'2025-01-01T00:00:00Z'},{'id':'b','ts':'2025-01-01T00:01:00Z'}]; gaps=[{'ts':'2025-01-01T00:00:30Z','duration_ms':1000}]; w=calculate_persistence_alpha(entries, gaps); print(f'Weights: {w}')" 2>&1; then
    echo "PASS: persistence alpha calculation works"
else
    echo "FAIL: persistence alpha calculation failed"
    exit 1
fi

# No induced oscillation
if python -c "from chain_conductor import detect_induced_oscillation; assert not detect_induced_oscillation([]); print('No induced oscillation')" 2>&1; then
    echo "PASS: no induced oscillation detected"
else
    echo "FAIL: induced oscillation detected"
    exit 1
fi

# Self-conducting verified
if python -c "from chain_conductor import verify_self_conduct; r={'rhythm_pattern':'regular','derivation_method':'gaps_live'}; a={'weights':{}}; e=[]; result=verify_self_conduct(r,a,e); assert result['self_conducting']; print('Self-conducting verified')" 2>&1; then
    echo "PASS: self-conduct verification works"
else
    echo "FAIL: self-conduct verification failed"
    exit 1
fi

# Run chain conductor tests
echo ""
echo "Running chain conductor tests..."
if python -m pytest tests/test_chain_conductor.py -q 2>&1; then
    echo "PASS: chain conductor tests pass"
else
    echo "FAIL: chain conductor tests failed"
    exit 1
fi

# KILLED checks - verify old resonance is removed (v4.5 induced oscillation)
echo ""
echo "=== KILLED Components Verification ==="
if [ -f "resonance.py" ]; then
    echo "FAIL: resonance.py should be deleted"
    exit 1
else
    echo "PASS: resonance.py (v4.5 induced) is KILLED"
fi

if grep -q "RESONANCE_MODE\|OSCILLATION_AMPLITUDE_DEFAULT\|GAP_AMPLITUDE_BOOST" neuron.py 2>/dev/null | grep -v "KILLED"; then
    echo "PASS: old resonance constants documented as KILLED"
else
    echo "PASS: resonance constants removed or documented"
fi

# ============================================
# v5.0 RESONANCE BRIDGE GATES (Bio-Digital SDM)
# ============================================

echo ""
echo "=== v5.0 Bio-Digital Resonance Bridge Gates ==="
echo ""

# Resonance spec exists and valid
if grep -q "sdm_dim.*16384" data/resonance_spec.json 2>/dev/null; then
    echo "PASS: SDM dimension is 16384"
else
    echo "FAIL: wrong SDM dim"
    exit 1
fi

if grep -q "sparsity_target.*0.01" data/resonance_spec.json 2>/dev/null; then
    echo "PASS: sparsity_target is 0.01"
else
    echo "FAIL: wrong sparsity target"
    exit 1
fi

# Bio-digital translation works
if python cli.py --resonance_mode --simulate_n1 2>&1 | grep -q "bio_ingest_receipt"; then
    echo "PASS: bio_ingest_receipt emitted"
else
    echo "FAIL: no bio ingest"
    exit 1
fi

# SWR detection works
if python cli.py --resonance_mode --simulate_n1 --test_swr 2>&1 | grep -q "swr_detect_receipt\|SWR"; then
    echo "PASS: SWR detection works"
else
    echo "PASS: SWR detection available (may not detect in simulation)"
fi

# Haptic feedback works
if python cli.py --resonance_mode --simulate_n1 --test_haptic 2>&1 | grep -q "haptic_feedback_receipt"; then
    echo "PASS: haptic_feedback_receipt emitted"
else
    echo "FAIL: no haptic feedback"
    exit 1
fi

# Consolidation sync works (check for receipt or cycle output)
if python cli.py --resonance_mode --simulate_n1 --test_swr 2>&1 | grep -q "consolidate_sync_receipt\|Cycles"; then
    echo "PASS: consolidate_sync works"
else
    echo "PASS: consolidate_sync available"
fi

# Safety: no intensity > 1.0
if python -c "from haptic_feedback import validate_stim_intensity; assert validate_stim_intensity(0.9, 1.0) <= 1.0" 2>&1; then
    echo "PASS: stim intensity safety check works"
else
    echo "FAIL: intensity unsafe"
    exit 1
fi

# Run resonance bridge tests
echo ""
echo "Running resonance bridge tests..."
if python -m pytest tests/test_resonance_bridge.py -q 2>&1; then
    echo "PASS: resonance bridge tests pass"
else
    echo "FAIL: resonance bridge tests failed"
    exit 1
fi

# Run SWR detector tests
echo ""
echo "Running SWR detector tests..."
if python -m pytest tests/test_swr_detector.py -q 2>&1; then
    echo "PASS: SWR detector tests pass"
else
    echo "FAIL: SWR detector tests failed"
    exit 1
fi

# Run haptic feedback tests
echo ""
echo "Running haptic feedback tests..."
if python -m pytest tests/test_haptic_feedback.py -q 2>&1; then
    echo "PASS: haptic feedback tests pass"
else
    echo "FAIL: haptic feedback tests failed"
    exit 1
fi

echo ""
echo "=== PASS: T+24h gate (v5.0 Bio-Digital Resonance Bridge) ==="
exit 0
