#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
./scripts/sync_docs_notebooks.sh
export NO_MKDOCS_2_WARNING=1
if ! uv run --no-sync python -c 'import mkdocs' >/dev/null 2>&1; then
  echo "mkdocs is not installed in the project environment." >&2
  echo "Install the docs extra first: uv sync --extra docs" >&2
  exit 1
fi
# --no-sync: build in the environment the caller prepared (CI syncs the docs
# extra explicitly), never silently re-resolving and pruning other groups.
uv run --no-sync mkdocs build --strict
