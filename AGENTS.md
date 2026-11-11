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

```bash
uv venv .venv && uv sync --group test          # once
uv run pytest test -n 2 -m "not nightly"       # documented unit gate
uv run --group dev ruff check                  # lint (ruff is capped at 0.15.x)
./scripts/docs_build.sh                        # strict mkdocs build
```

Run long suites serially, and give nested pytest runs their own temporary
directory (`TMPDIR=$(mktemp -d)`) so parallel JAX compilation does not collide.

## GNN interoperability

`pymdp/gnn.py` publishes a flat discrete POMDP (one observation modality, one
hidden-state factor) as a Generalized Notation Notation specification
(`.gnn.md`) and validates it against the real GNN parser. Keep the bridge
additive and keep one canonical home per concept: joint composition for
factorized models belongs to GNN, so multi-modality/multi-factor models are
refused here rather than flattened by a second convention.

- Pin: `pymdp.gnn.GNN_PINNED_REVISION` = GNN v5.0.3
  (`8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`). Bump it deliberately, in one
  place, with the validation test run.
- Validation reads an exported pinned tree, never the live GNN working tree:
  `gnn.export_pinned_source(...)` or `git -C <gnn> archive <sha> src | tar -x`.
- Opt-in tests: `PYMDP_GNN_ROOT=<exported>/src uv run pytest test/test_gnn_bridge.py -v`
  (add `PYMDP_GNN_CHECKOUT=<gnn checkout>` for the pinned-export test). They skip
  cleanly when the variable is unset; never convert a skip into a pass.
- Guide: `docs-mkdocs/guides/gnn-interoperability.md`.
