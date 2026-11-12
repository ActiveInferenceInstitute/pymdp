#!/usr/bin/env bash
# Single gate entry point for pymdp.
#
# Every arm is a complete, independently runnable check, and CI calls these same
# arms: there is exactly one definition of each command in the repository.
#
#   ./gate.sh lint               ruff, bounded to the dev group (same ruff as CI)
#   ./gate.sh test               unit gate: nightly-marked tests deselected
#   ./gate.sh test-full          every pytest test, including nightly-marked ones
#   ./gate.sh test-nightly       only the nightly-marked tests
#   ./gate.sh gnn                GNN pin reader + bridge tests, from the pinned parser tree
#   ./gate.sh notebooks          CI-tier notebook manifest
#   ./gate.sh notebooks-nightly  nightly-tier notebook manifest
#   ./gate.sh docs               curated-notebook sync + strict mkdocs build
#   ./gate.sh package            sdist/wheel build + install/import smoke test
#   ./gate.sh bench              re-measure every arm and rewrite the metrics doc
#   ./gate.sh all                lint + test + gnn + docs + notebooks
#
# Arguments after the arm are forwarded to the underlying tool, e.g.:
#   ./gate.sh test -- --cov=pymdp --cov-report=term
#
# Set GATE_PYTHON to choose the interpreter for the `package` arm's throwaway
# environment (default: 3.14, the version CI builds docs and packages with).
#
# The `gnn` arm runs the shipped reader and the bridge tests against the pinned
# GNN revision recorded in `.github/gnn-pin.json`. Set PYMDP_GNN_ROOT to a tree
# you exported, or PYMDP_GNN_CHECKOUT to a GNN checkout (default: a sibling
# `Generalized_Notation_Notation`) and the arm exports that revision itself.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

GATE_PYTHON="${GATE_PYTHON:-3.14}"
CI_NOTEBOOKS="test/notebooks/ci_notebooks.txt"
NIGHTLY_NOTEBOOKS="test/notebooks/nightly_notebooks.txt"
GNN_EXPORT_DIR=".cache/gnn-pin"
GNN_DEFAULT_CHECKOUT="../Generalized_Notation_Notation"

usage() {
  # Everything between the shebang and `set` is the arm list: keep one copy.
  sed -n '2,/^set /p' "${BASH_SOURCE[0]}" | sed -e '$d' -e 's/^# \{0,1\}//'
}

run() {
  local label="$1"
  shift
  printf '== %s: %s\n' "$label" "$*"
  local start=$SECONDS
  local status=0
  "$@" || status=$?
  printf '== %s: %ss (exit %s)\n' "$label" "$((SECONDS - start))" "$status"
  return "$status"
}

arm_lint() {
  run lint uv run --group dev ruff check --force-exclude "$@"
}

arm_test() {
  run test uv run --group test pytest test -n 2 -m "not nightly" "$@"
}

arm_test_full() {
  run test-full uv run --group test pytest test -n 2 "$@"
}

arm_test_nightly() {
  run test-nightly uv run --group test pytest test -n 2 -m nightly "$@"
}

gnn_bridge_pin() {
  # The code-facing pin `pymdp.gnn` ships, one value per line.
  uv run --group test --extra gnn python -c '
from pymdp.gnn import GNN_PINNED_RELEASE, GNN_PINNED_REVISION

print(GNN_PINNED_REVISION)
print(GNN_PINNED_RELEASE)
'
}

gnn_export() {
  # Export the bridge's pinned revision from a GNN checkout; print the directory
  # that holds the `gnn` package.
  uv run --group test --extra gnn python - "$1" "$GNN_EXPORT_DIR" <<'PY'
import sys

from pymdp import gnn

print(gnn.export_pinned_source(sys.argv[1], sys.argv[2]).path)
PY
}

arm_gnn() {
  # Two checks, both against the pinned parser:
  #   * the shipped reader validates `.github/gnn-pin.json` and `.github/GNN.md`
  #     and requires them to agree with the bridge's constants;
  #   * the bridge tests round-trip a model through that same parser.
  # The reader needs a materialized pinned tree, so the arm exports it when
  # PYMDP_GNN_ROOT does not already name one.
  local pin revision release root checkout
  pin="$(gnn_bridge_pin)" || return $?
  revision="$(printf '%s\n' "$pin" | sed -n 1p)"
  release="$(printf '%s\n' "$pin" | sed -n 2p)"
  checkout="${PYMDP_GNN_CHECKOUT:-}"
  root="${PYMDP_GNN_ROOT:-}"
  if [[ -z "$root" ]]; then
    checkout="${checkout:-$GNN_DEFAULT_CHECKOUT}"
    printf '== gnn-export: %s from %s into %s\n' "$revision" "$checkout" "$GNN_EXPORT_DIR"
    root="$(gnn_export "$checkout")" || return $?
  fi
  if [[ -z "$root" ]]; then
    printf 'gate.sh: no pinned GNN tree; set PYMDP_GNN_ROOT or PYMDP_GNN_CHECKOUT\n' >&2
    return 1
  fi
  run gnn-conformance env "PYTHONPATH=$root" uv run --group test --extra gnn \
    python -m gnn.importers check --repo . --require-pointer \
    --expected-commit "$revision" --expected-tag "$release" || return $?
  run gnn env "PYTHONPATH=$root" "PYMDP_GNN_ROOT=$root" \
    "PYMDP_GNN_CHECKOUT=$checkout" \
    uv run --group test --extra gnn pytest test/test_gnn_bridge.py -v "$@"
}

arm_notebooks() {
  run notebooks uv run --group test --extra nb python \
    scripts/run_notebook_manifest.py "$CI_NOTEBOOKS" -- "$@"
}

arm_notebooks_nightly() {
  run notebooks-nightly uv run --group test --extra nb python \
    scripts/run_notebook_manifest.py "$NIGHTLY_NOTEBOOKS" -- --nbval-cell-timeout=3600 "$@"
}

arm_docs() {
  run docs ./scripts/docs_build.sh "$@"
}

arm_package() {
  rm -rf dist
  run build uv build --out-dir dist || return $?
  local wheel
  wheel="$(ls dist/*.whl)"
  rm -rf .gate/package-env
  run package-venv uv venv .gate/package-env --python "$GATE_PYTHON" || return $?
  run package-install uv pip install --python .gate/package-env/bin/python "$wheel" \
    || return $?
  run package-import bash -c "cd /tmp && '$ROOT_DIR/.gate/package-env/bin/python' -c '
import pymdp
from pymdp.agent import Agent
assert \"site-packages\" in pymdp.__file__, pymdp.__file__
print(\"imported\", pymdp.__file__)
print(\"Agent:\", Agent.__name__)
'"
}

arm_bench() {
  run bench uv run --group dev --group test --extra nb --extra gnn \
    python scripts/gate_metrics.py "$@"
}

arm_all() {
  arm_lint && arm_test && arm_gnn && arm_docs && arm_notebooks
}

main() {
  local arm="${1:-}"
  case "$arm" in
    "") printf 'gate.sh: missing arm\n\n'; usage; exit 2 ;;
    -h|--help|help) usage; exit 0 ;;
    lint|test|test-full|test-nightly|gnn|notebooks|notebooks-nightly|docs|package|bench|all) ;;
    *) printf 'gate.sh: unknown arm %s\n\n' "$arm"; usage; exit 2 ;;
  esac
  shift
  "arm_${arm//-/_}" "$@"
}

main "$@"
