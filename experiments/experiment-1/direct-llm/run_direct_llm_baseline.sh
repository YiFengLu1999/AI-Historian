#!/usr/bin/env bash
set -euo pipefail

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$PKG_DIR/../.." && pwd)"

ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$ENV_FILE"
  set +a
fi

CASE_IDS="${CASE_IDS:-H-C1,H-C2,H-C3,H-C4,H-C5,H-C6}"
RUNS="${RUNS:-1}"
LABEL="${LABEL:-direct_llm_baseline}"
OUT="${OUT:-$PKG_DIR/outputs/current/generated_results_direct_llm_$(date +%Y%m%d_%H%M%S)}"

python3 "$PKG_DIR/direct-llm/run_direct_llm_baseline.py" \
  --case-ids "$CASE_IDS" \
  --runs "$RUNS" \
  --output-dir "$OUT" \
  --label "$LABEL"

node "$PKG_DIR/direct-llm/evaluation-scripts/score_direct_llm_variant.js" \
  "$OUT/tables/all_cases_direct_llm_prefill.csv" \
  "$LABEL"

echo "OUT=$OUT"
