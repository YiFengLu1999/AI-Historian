#!/bin/zsh
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$PACKAGE_DIR/../.." && pwd)"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

python3 "$PACKAGE_DIR/code/run_direct_llm.py" \
  --runs "${RUNS:-3}" \
  --forms "${FORMS:-T1,T2,T3}" \
  "$@"
