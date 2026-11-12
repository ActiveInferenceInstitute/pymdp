#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
./scripts/sync_docs_notebooks.sh
export NO_MKDOCS_2_WARNING=1
# --no-sync: serve from the prepared environment; see scripts/docs_build.sh.
uv run --no-sync mkdocs serve "$@"
