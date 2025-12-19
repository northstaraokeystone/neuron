#!/bin/bash
# NEURON T+2h Gate - CLAUDEME Compliance Check
# Required artifacts: spec.md, ledger_schema.json, cli.py with receipt emission
#
# Usage: ./gate_t2h.sh
# Exit 0 = PASS, Exit 1 = FAIL

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== NEURON T+2h Gate ==="
echo "Checking CLAUDEME T+2h requirements..."
echo ""

# Check spec.md exists
if [ -f spec.md ]; then
    echo "PASS: spec.md exists"
else
    echo "FAIL: no spec.md"
    exit 1
fi

# Check ledger_schema.json exists
if [ -f ledger_schema.json ]; then
    echo "PASS: ledger_schema.json exists"
else
    echo "FAIL: no ledger_schema.json"
    exit 1
fi

# Check cli.py exists
if [ -f cli.py ]; then
    echo "PASS: cli.py exists"
else
    echo "FAIL: no cli.py"
    exit 1
fi

# Check cli.py emits receipt (look for "type": which indicates receipt format)
if python cli.py --test 2>&1 | grep -q '"type"'; then
    echo "PASS: cli.py emits receipts"
else
    echo "FAIL: cli.py does not emit receipts"
    exit 1
fi

# Check dual_hash format (SHA256:BLAKE3)
if python -c "from neuron import dual_hash; h=dual_hash('test'); assert ':' in h and len(h.split(':')[0])==64" 2>&1; then
    echo "PASS: dual_hash() format correct"
else
    echo "FAIL: dual_hash() format incorrect"
    exit 1
fi

echo ""
echo "=== PASS: T+2h gate ==="
exit 0
