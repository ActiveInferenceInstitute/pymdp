# pymdp — agent guidance

`pymdp` is the JAX-first Active Inference library published as
`inferactively-pymdp`; `pymdp/legacy` is the archived NumPy implementation kept
for migration. Model shapes, factor/modality conventions and the agent loop are
documented in `docs-mkdocs/guides/generative-model-structure.md`.

This repository is a tracked fork of
[`infer-actively/pymdp`](https://github.com/infer-actively/pymdp). `origin` is
`ActiveInferenceInstitute/pymdp` and `master` is the fork's default branch.
Fork-local changes stay additive so that merges from `upstream/main` remain
clean: the fork delta today is the `.aii` InstituteOS sidecar, `TASKS.md`, and
two fork-local CI comments (`manual-branch-nightly.yaml`,
`nightly-tests.yaml`). Never rewrite upstream code paths to make a fork-local
feature fit.

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
