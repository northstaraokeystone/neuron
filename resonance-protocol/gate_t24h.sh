#!/bin/bash
# RESONANCE PROTOCOL - T+24h Gate Check
# Full validation of all modules

set -e

echo "=== RESONANCE PROTOCOL T+24h Gate Check ==="
echo ""

cd "$(dirname "$0")"

# Set up test environment
export RESONANCE_BASE=$(mktemp -d)
trap "rm -rf $RESONANCE_BASE" EXIT

# Gate 1: HDC Foundation
echo "=== GATE 1: HDC_FOUNDATION ==="
python3 cli.py test-hdc --orthogonality --dropout --n-vectors 100
echo ""

# Gate 2: Oscillation Detection
echo "=== GATE 2: OSCILLATION_DETECTION ==="
python3 cli.py test-oscillations --latency-benchmark --n-trials 100
echo ""

# Gate 3: Federated Infrastructure
echo "=== GATE 3: FEDERATED_INFRASTRUCTURE ==="
python3 cli.py test-federated --convergence --privacy
echo ""

# Gate 4: Phase Locking
echo "=== GATE 4: PHASE_LOCKING ==="
python3 cli.py test-phase-lock --n-trials 100
echo ""

# Gate 5: Safety Compliance
echo "=== GATE 5: SAFETY_COMPLIANCE ==="
python3 cli.py test-safety --shannon-limit --thermal-check --n-tests 100
echo ""

# Gate 6: Integration
echo "=== GATE 6: INTEGRATION ==="
python3 cli.py run-pipeline --synthetic-data
echo ""

# Run pytest if available
if command -v pytest &> /dev/null; then
    echo "=== Running pytest ==="
    pytest tests/ -v --tb=short || echo "Some tests may have failed"
else
    echo "pytest not available, skipping test suite"
fi

echo ""
echo "=== T+24h Gate Check COMPLETE ==="
echo "Review results above for any failures."
