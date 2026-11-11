# Testing

`pymdp` is validated by a pytest suite under `test/` and a strict mkdocs build.
The documented unit gate excludes the slow, network- and accelerator-dependent
`nightly` tests.

## Install the test environment

```bash
uv venv .venv
uv sync --group test
```

Optional groups: `--group dev` (contributor hooks, ruff), `--extra docs`
(mkdocs tooling), `--extra nb` (notebook extras / Graphviz), `--extra modelfit`
(model-fitting extras). If `pygraphviz` fails to build, install Graphviz first —
see [`../getting-started/installation.md`](../getting-started/installation.md).

## Unit gate

```bash
uv run pytest test -n 2 -m "not nightly"
```

- `-n 2` runs two workers; `-m "not nightly"` deselects the nightly markers.
- Run long suites serially and give nested pytest runs their own temporary
  directory (`TMPDIR=$(mktemp -d)`) so parallel JAX compilation does not collide.

The suite covers the JAX-first core (`pymdp/agent.py`, `pymdp/inference.py`,
`pymdp/control.py`, `pymdp/learning.py`, `pymdp/math`, `pymdp/planning/`) and the
archived NumPy implementation under `pymdp/legacy/` used for cross-validation.

## Nightly tests

Tests marked `nightly` and the notebook suites are scheduled separately, so they
are out of scope for local runs. The notebook lists live in
`test/notebooks/ci_notebooks.txt` and `test/notebooks/nightly_notebooks.txt`; see
[`../test/notebooks/README.md`](https://github.com/ActiveInferenceInstitute/pymdp/blob/master/test/notebooks/README.md).

## GNN bridge tests (opt-in)

`test/test_gnn_bridge.py` publishes a flat POMDP as a Generalized Notation
Notation specification and validates it against the pinned GNN parser. It skips
cleanly unless `PYMDP_GNN_ROOT` is set:

```bash
PYMDP_GNN_ROOT=<exported-gnn>/src uv run pytest test/test_gnn_bridge.py -v
```

Never convert a skip into a pass. See
[`../guides/gnn-interoperability.md`](../guides/gnn-interoperability.md).

## Lint

```bash
uv run --group dev ruff check
```

`ruff` is capped at `0.15.x` in the dev group; see the repository `AGENTS.md`.

## Documentation build

```bash
./scripts/docs_build.sh
```

This runs the strict mkdocs build (`strict: true`), which fails on broken
internal links and missing nav entries. Every page added under `docs-mkdocs/`
must be listed in `mkdocs.yml`.
