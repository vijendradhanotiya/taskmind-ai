#!/usr/bin/env bash
set -euo pipefail

CYAN='\033[96m'
GREEN='\033[92m'
RED='\033[91m'
YELLOW='\033[93m'
BOLD='\033[1m'
DIM='\033[2m'
RESET='\033[0m'

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT_DIR="$ROOT_DIR/tests/reports"

clear

echo -e "${BOLD}${CYAN}"
echo "  ╔══════════════════════════════════════════════════════╗"
echo "  ║       TaskMind API — Full Test Suite Runner          ║"
echo "  ║       $(date '+%Y-%m-%d %H:%M:%S')  •  All Endpoints            ║"
echo "  ╚══════════════════════════════════════════════════════╝"
echo -e "${RESET}"

echo -e "${DIM}  Working dir : $ROOT_DIR${RESET}"
echo -e "${DIM}  Reports dir : $REPORT_DIR${RESET}"
echo ""

# Check server is up
echo -e "${YELLOW}  Checking server health ...${RESET}"
if curl -sf http://localhost:8001/health > /dev/null 2>&1; then
    echo -e "${GREEN}  ✓ Server is live at http://localhost:8001${RESET}\n"
else
    echo -e "${RED}  ✗ Server not reachable at http://localhost:8001${RESET}"
    echo -e "${YELLOW}  Start it with:${RESET}"
    echo -e "    python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8001\n"
    exit 1
fi

# Show model info
MODEL_INFO=$(curl -sf http://localhost:8001/v1/models 2>/dev/null || echo '{}')
HEALTH_INFO=$(curl -sf http://localhost:8001/health 2>/dev/null || echo '{}')
echo -e "${BOLD}  Model Details${RESET}"
echo -e "${DIM}  $(echo "$HEALTH_INFO" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'model={d.get(\"model_version\",\"?\")}  device={d.get(\"device\",\"?\")}  adapter={d.get(\"adapter_dir\",\"?\")}')  " 2>/dev/null || echo 'unavailable')${RESET}"
echo ""

echo -e "${BOLD}  Running tests across all endpoints:${RESET}"
echo -e "${DIM}  • /health  /metrics  /v1/models${RESET}"
echo -e "${DIM}  • /v1/classify       (30 prompts)${RESET}"
echo -e "${DIM}  • /v1/batch          (3 × 10 messages)${RESET}"
echo -e "${DIM}  • /v1/chat/completions (30 messages)${RESET}"
echo -e "${DIM}  • /v1/completions    (30 raw prompts)${RESET}"
echo ""
echo -e "${YELLOW}  Note: Each LLM call takes ~2–6 seconds on MPS.${RESET}"
echo -e "${YELLOW}  Total expected time: 6–10 minutes. Grab a coffee ☕${RESET}"
echo ""

START_EPOCH=$(date +%s)

echo -e "${BOLD}${CYAN}────────────────────────────────────────────────────────${RESET}"
echo -e "${BOLD}  Starting test run ...${RESET}"
echo -e "${BOLD}${CYAN}────────────────────────────────────────────────────────${RESET}\n"

cd "$ROOT_DIR"
python3 tests/test_api.py
EXIT_CODE=$?

END_EPOCH=$(date +%s)
ELAPSED=$((END_EPOCH - START_EPOCH))
MINS=$((ELAPSED / 60))
SECS=$((ELAPSED % 60))

echo ""
echo -e "${BOLD}${CYAN}────────────────────────────────────────────────────────${RESET}"
echo -e "${BOLD}  Test run complete  —  ${MINS}m ${SECS}s total${RESET}"
echo -e "${BOLD}${CYAN}────────────────────────────────────────────────────────${RESET}"

# Show latest report
LATEST=$(ls -t "$REPORT_DIR"/report_*.csv 2>/dev/null | head -1 || true)
if [[ -n "$LATEST" ]]; then
    LINE_COUNT=$(( $(wc -l < "$LATEST") - 1 ))
    PASS_COUNT=$(tail -n +2 "$LATEST" | cut -d',' -f6 | grep -c "True" || true)
    FAIL_COUNT=$(( LINE_COUNT - PASS_COUNT ))

    echo ""
    echo -e "${BOLD}  CSV Report${RESET}"
    echo -e "${DIM}  File    : $LATEST${RESET}"
    echo -e "${DIM}  Rows    : $LINE_COUNT${RESET}"
    echo -e "${GREEN}  Passed  : $PASS_COUNT${RESET}"
    if [[ $FAIL_COUNT -gt 0 ]]; then
        echo -e "${RED}  Failed  : $FAIL_COUNT${RESET}"
    fi

    echo ""
    echo -e "${BOLD}  First 5 rows preview:${RESET}"
    echo -e "${DIM}"
    python3 -c "
import csv, sys
with open('$LATEST') as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i >= 5: break
        ep = row['endpoint'][:22]
        inp = row['input'][:35]
        ok = row['ok']
        lat = row['latency_ms']
        intent = row['intent'][:18] if row['intent'] else '-'
        print(f'  {ep:<24} | {inp:<37} | ok={ok:<5} | {lat:>7}ms | {intent}')
"
    echo -e "${RESET}"
fi

echo -e "${BOLD}  To view full report:${RESET}"
echo -e "    ${CYAN}open $LATEST${RESET}   (opens in Numbers/Excel)"
echo -e "    ${CYAN}cat  $LATEST | column -t -s,${RESET}  (terminal view)"
echo ""

if [[ $EXIT_CODE -eq 0 ]]; then
    echo -e "${GREEN}${BOLD}  All done successfully.${RESET}\n"
else
    echo -e "${RED}${BOLD}  Some tests may have failed. Check the CSV report.${RESET}\n"
fi

exit $EXIT_CODE
