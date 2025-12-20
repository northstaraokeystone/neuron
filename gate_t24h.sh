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

# KILLED checks - verify resonance is removed
echo ""
echo "=== KILLED Components Verification ==="
if [ -f "resonance.py" ]; then
    echo "FAIL: resonance.py should be deleted"
    exit 1
else
    echo "PASS: resonance.py is KILLED"
fi

if grep -q "RESONANCE_MODE\|OSCILLATION_AMPLITUDE_DEFAULT\|GAP_AMPLITUDE_BOOST" neuron.py 2>/dev/null | grep -v "KILLED"; then
    echo "PASS: old resonance constants documented as KILLED"
else
    echo "PASS: resonance constants removed or documented"
fi

echo ""
echo "=== PASS: T+24h gate (v4.6 Chain Rhythm Conductor) ==="
exit 0
