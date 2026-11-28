# Testing and gates

`./gate.sh` is the single entry point for every check in this repository. Each
arm is a complete, independently runnable command, and the GitHub workflows call
the same arms, so a command is defined once and never re-typed in CI.

```bash
./gate.sh test      # the unit gate (nightly-marked tests deselected)
./gate.sh all       # lint + test + gnn + docs + notebooks
./gate.sh help      # every arm
```

One optional leading `--` after the arm is consumed; remaining arguments are
forwarded, e.g. `./gate.sh test -- --cov=pymdp --cov-report=term`.
The notebook runner's own internal separator is retained.

## Install the test environment

```bash
uv venv .venv
uv sync --group test
```

Optional groups: `--group dev` (contributor hooks, ruff), `--extra docs`
(mkdocs tooling), `--extra nb` (notebook extras / Graphviz), `--extra gnn`
(`defusedxml`, `psutil` for the pinned GNN parser), `--extra modelfit`
(model-fitting extras). If `pygraphviz` fails to build, install Graphviz first —
see [`../getting-started/installation.md`](../getting-started/installation.md).

The gate uses the Python that `uv` selects for the project; `GATE_PYTHON`
chooses the interpreter for the `package` arm's throwaway environment (default
`3.14`, the version CI builds docs and packages with).

## Arms

| Arm | What it does | CI job |
| --- | --- | --- |
| `lint` | Ruff over the tree, bounded by the `dev` group's `ruff>=0.15,<0.16`. | `Lint` |
| `test` | The unit gate: `nightly`-marked tests deselected. | `Test` (matrix) |
| `test-full` | Every pytest test, nothing deselected. | manual |
| `test-nightly` | Only the `nightly`-marked tests. | `Nightly Tests` |
| `gnn` | Shared pinned reader plus bridge/parser/executor tests; exports the pinned source if no explicit root is selected. | local / opt-in |
| `notebooks` | The CI-tier notebook manifest. | `Test` |
| `notebooks-nightly` | The nightly-tier notebook manifest. | `Nightly Tests` |
| `docs` | Curated-notebook sync, then `mkdocs build --strict`. | `Docs` |
| `package` | sdist/wheel build, wheel install into `.gate/package-env`, import from outside the source tree. | `Python package` |
| `bench` | Measures every arm and rewrites [`gate-metrics.md`](gate-metrics.md). | — |
| `all` | `lint`, `test`, `gnn`, `docs`, `notebooks`, in that order. | local pre-PR |

`./gate.sh help` prints this list, and the arm bodies in `gate.sh` are the only
place a command string lives; workflow files and the pre-commit hook invoke the
arms instead of repeating them.

There is no `types` arm: the repository configures no type checker (`mypy`,
`pyright` or similar) — `ruff` is the only static analysis, and `lint` is
therefore the whole static arm.

## CI shape

The workflows are a thin mapping onto arms, so a push runs each command once:

| Workflow | Jobs | What runs |
| --- | --- | --- |
| `Lint` | 1 | `./gate.sh lint` |
| `Test` | 4 unit interpreters (3.11–3.14) + 1 notebooks job | `./gate.sh test`; coverage is collected and uploaded once, on 3.14. `./gate.sh notebooks` runs once, on 3.14, instead of in every matrix cell. |
| `Docs` | 1 | `./gate.sh docs` after `uv sync --no-default-groups --extra docs`. |
| `Python package` | 1 | `./gate.sh package` on 3.14. |
| `Nightly Tests` | 1 | `./gate.sh test-nightly` and `./gate.sh notebooks-nightly` (manual dispatch on this fork; the schedule is disabled because the fork's delta is metadata only). |

Before the gate, the `Test` matrix executed the 16-notebook CI tier four times
and uploaded four coverage artifacts, and `Python package` built and installed
the distributions twice. Per-arm wall time did not change — the gate is a thin
dispatcher — so the saving is the removed duplicate runs, not a faster command.

## Fast versus full

The default arm is fast because it uses the repository's existing marker:
`-m "not nightly"` deselects the long, network- and accelerator-dependent
`nightly` tests. `test-full` is the same suite with nothing deselected, and
`test-nightly` is the nightly subset on its own. The exact selected/passed
counts are recorded in [`gate-metrics.md`](gate-metrics.md); that page is
generated, so do not copy its numbers elsewhere.

Run long suites serially, and give nested pytest runs their own temporary
directory (`TMPDIR=$(mktemp -d)`) so parallel JAX compilation does not collide.

The suite covers the JAX-first core (`pymdp/agent.py`, `pymdp/inference.py`,
`pymdp/control.py`, `pymdp/learning.py`, `pymdp/maths.py`, `pymdp/planning/`)
and the archived NumPy implementation under `pymdp/legacy/` used for
cross-validation.

## Notebook tiers

Notebook execution is tiered through manifests, and the gate wraps the same
runner CI uses:

- `test/notebooks/ci_notebooks.txt` — the PR tier, run by `./gate.sh notebooks`.
- `test/notebooks/nightly_notebooks.txt` — the scheduled tier, run by
  `./gate.sh notebooks-nightly`; see
  [`test/notebooks/README.md`](https://github.com/ActiveInferenceInstitute/pymdp/blob/master/test/notebooks/README.md).

Both arms default to `-n0` inside the runner to avoid xdist/nbval conflicts.

## GNN bridge and importer conformance (opt-in)

`test/test_gnn_bridge.py` publishes a flat POMDP as a Generalized Notation
Notation specification and validates it against the pinned GNN parser.

The pin is a file, [`.github/gnn-pin.json`](https://github.com/ActiveInferenceInstitute/pymdp/blob/master/.github/gnn-pin.json),
which records GNN `v5.2.0` at `4a7cccce6d4d7cb483f00323ec5cdd79300f3464`. It is
read by GNN's shipped reader — `python -m gnn.importers check --repo .` — not by
a local parser, and [`.github/GNN.md`](https://github.com/ActiveInferenceInstitute/pymdp/blob/master/.github/GNN.md)
is the pointer that reader checks. `pymdp.gnn.GNN_PINNED_RELEASE` /
`GNN_PINNED_REVISION` mirror the pin for library consumers; the arm requires the
mirror and the pin to agree.

```bash
PYMDP_GNN_CHECKOUT=/path/to/Generalized_Notation_Notation ./gate.sh gnn
# or point the arm at a tree you exported yourself:
GNN_EXPORT_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/pymdp-gnn-pin.XXXXXXXX")"
git -C /path/to/Generalized_Notation_Notation \
  archive 4a7cccce6d4d7cb483f00323ec5cdd79300f3464 src |
  tar -x -C "$GNN_EXPORT_ROOT"
PYMDP_GNN_ROOT="$GNN_EXPORT_ROOT/src" ./gate.sh gnn
```

The arm exports the pinned revision from `PYMDP_GNN_CHECKOUT` (default: a
sibling `Generalized_Notation_Notation` checkout) when `PYMDP_GNN_ROOT` is
unset, so `./gate.sh gnn` always runs the reader and the pinned-parser tests
against the revision the pin names instead of skipping. The emission, refusal,
and shape tests also run under `./gate.sh test` without any GNN tree. See
[`../guides/gnn-interoperability.md`](../guides/gnn-interoperability.md).

Each automatic export gets a fresh `.cache/gnn-pin.*` directory. Existing
exports, including `.cache/gnn-pin`, remain untouched.

## Lint

`./gate.sh lint` runs the exact command CI runs, using the `ruff` pinned in the
`dev` dependency group (currently `>=0.15,<0.16`; the ceiling is deliberate —
0.16.0 broadened the default rule set). The pre-commit hook installs the same
entry point, so a hook run and a CI run cannot diverge.

## Documentation build

```bash
./gate.sh docs
```

This syncs curated notebooks and runs the strict, no-sync MkDocs build in the
prepared environment. Options are forwarded, including
`./gate.sh docs -- --site-dir "/path/site with spaces"`.
Broken internal links and missing nav targets fail; add new pages to
`mkdocs.yml` `nav` as well. This command does not authorize environment changes.

## Packaging

```bash
./gate.sh package
```

Builds the sdist and wheel, installs the wheel into a throwaway environment
under `.gate/package-env`, and imports `pymdp` from outside the source tree so
the source checkout cannot mask a packaging error.

## Measured numbers

[`gate-metrics.md`](gate-metrics.md) records per-arm wall time (before and after
the gate existed), the selected/passed test counts, and the host and commit the
measurement came from. It is written by `./gate.sh bench` and verified by
`./gate.sh bench --check`; edit the producer, never the page.
