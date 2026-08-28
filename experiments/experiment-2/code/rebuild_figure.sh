#!/bin/zsh
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
node "$PACKAGE_DIR/code/build_strict_total_html.js"

