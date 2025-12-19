#!/bin/bash
# NEURON T+24h Gate - CLAUDEME Compliance Check
# Required: All tests pass, emit_receipt usage, assertions in tests
#
# Usage: ./gate_t24h.sh
# Exit 0 = PASS, Exit 1 = FAIL

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== NEURON T+24h Gate ==="
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
echo "=== PASS: T+24h gate ==="
exit 0
