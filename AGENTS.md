# pymdp — agent guidance

`pymdp` is the JAX-first Active Inference library published as
`inferactively-pymdp`; `pymdp/legacy` is the archived NumPy implementation kept
for migration. Model shapes, factor/modality conventions and the agent loop are
documented in `docs-mkdocs/guides/generative-model-structure.md`.

This repository is a tracked fork of
[`infer-actively/pymdp`](https://github.com/infer-actively/pymdp). `origin` is
`ActiveInferenceInstitute/pymdp` and `master` is the fork's default branch.

## Layout

| Path | What it is |
| --- | --- |
| [`pymdp/`](pymdp/README.md) | The importable library (JAX-first; `legacy/` is the archived NumPy implementation). |
| [`test/`](test/) | pytest suite; see [`docs-mkdocs/development/testing.md`](docs-mkdocs/development/testing.md). |
| [`docs-mkdocs/`](docs-mkdocs/README.md) | Documentation source, built by [`mkdocs.yml`](mkdocs.yml) (`docs_dir: docs-mkdocs`). |
| [`examples/`](examples/README.md) | Curated notebooks and runnable examples. |
| [`scripts/`](scripts/README.md) | Docs build/serve and notebook tooling. |
| [`paper/`](paper/) | JOSS paper artifacts. |
| [`.aii/`](.aii/README.md) | InstituteOS sidecar metadata. |
| [`TODO.md`](TODO.md) · [`TASKS.md`](TASKS.md) | Owner backlog and planning signpost. |

Fork-local changes stay additive so that merges from `upstream/main` remain
clean: the fork delta is the `.aii` InstituteOS sidecar, `TASKS.md`, the two
fork-local nightly workflows, the `master` entries in the three CI branch
lists, and the GNN bridge. Never rewrite upstream code paths to make a
fork-local feature fit.

## Upstream sync

`upstream` points at `infer-actively/pymdp` with `pushurl` set to `DISABLED`,
so it is fetch-only and is never pushed to. Sync by merging `upstream/main`
into a branch off `master`, and land it as a merge commit (never squash or
rebase, because custody records pin commit SHAs). The divergence in
`manual-branch-nightly.yaml` and `nightly-tests.yaml` is deliberate: the fork
keeps `workflow_dispatch`-only scheduling and a `master` checkout ref and takes
upstream's step-level changes on top. The current fork delta is whatever
`git diff --stat upstream/main master` lists.

## Verification gate

`./gate.sh` is the single entry point; each arm is separately runnable and CI
calls the same arms, so a command is never defined twice.

```bash
uv venv .venv && uv sync --group test   # once (add --group dev --extra docs as needed)
./gate.sh lint                          # ruff, pinned by the dev group
./gate.sh test                          # unit gate: nightly marker deselected
./gate.sh test-full                     # everything, including nightly tests
./gate.sh notebooks                     # CI-tier notebook manifest
./gate.sh gnn                           # GNN bridge tests (skips without PYMDP_GNN_ROOT)
./gate.sh docs                          # notebook sync + strict mkdocs build
./gate.sh package                       # sdist/wheel build + install/import smoke test
./gate.sh all                           # lint, test, gnn, docs, notebooks
```

Measured arm wall time and test counts live in the generated
[`docs-mkdocs/development/gate-metrics.md`](docs-mkdocs/development/gate-metrics.md)
(`./gate.sh bench` refreshes it; `--check` verifies it). There is no type-checker
arm because the repository configures none.

Run long suites serially, and give nested pytest runs their own temporary
directory (`TMPDIR=$(mktemp -d)`) so parallel JAX compilation does not collide.

## GNN interoperability

`pymdp/gnn.py` publishes a flat discrete POMDP (one observation modality, one
hidden-state factor) as a Generalized Notation Notation specification
(`.gnn.md`) and validates it against the real GNN parser. Keep the bridge
additive and keep one canonical home per concept: joint composition for
factorized models belongs to GNN, so multi-modality/multi-factor models are
refused here rather than flattened by a second convention.

- Pin: `.github/gnn-pin.json` records GNN v5.1.0
  (`75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf`). `pymdp.gnn.GNN_PINNED_RELEASE` /
  `GNN_PINNED_REVISION` mirror it for library consumers, and `./gate.sh gnn`
  requires the two to agree (`--expected-tag` / `--expected-commit`), so a bump
  changes the pin first and the constants with it.
- The pin is read by the shipped upstream reader, never by a copy kept here:
  `python -m gnn.importers check --repo . --require-pointer` runs from the
  pinned parser tree. `.github/GNN.md` is the pointer it checks.
- Validation reads an exported pinned tree, never the live GNN working tree:
  `gnn.export_pinned_source(...)` or `git -C <gnn> archive <sha> src | tar -x`.
  `./gate.sh gnn` exports the pinned revision from `PYMDP_GNN_CHECKOUT` (default:
  a sibling `Generalized_Notation_Notation` checkout) unless `PYMDP_GNN_ROOT`
  already names an export.
- Tests: `./gate.sh gnn` runs the reader plus `test/test_gnn_bridge.py` against
  that tree (bridge round trips included; the executor test needs
  `PYMDP_GNN_CHECKOUT`). Without a materializable tree the arm fails rather than
  skipping, and `./gate.sh test` still runs the emission and refusal tests.
- Guide: `docs-mkdocs/guides/gnn-interoperability.md`.
