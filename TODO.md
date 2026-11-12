# TODO — pymdp

Owner-facing backlog for the `ActiveInferenceInstitute/pymdp` fork. Grouped by
kind; every item below is verified against the current tree. Planning signposts
also live in [`TASKS.md`](TASKS.md).

## Documentation gaps

- [ ] **No top-level `docs/`.** Documentation intentionally lives in
  [`docs-mkdocs/`](docs-mkdocs/README.md) and is built by
  [`mkdocs.yml`](mkdocs.yml) (`docs_dir: docs-mkdocs`). This is deliberate, not a
  missing tree; recorded here so it is not "fixed" by duplicating the tree.
- [x] ✓ **Single gate entry point.** `./gate.sh` now owns every check (lint,
  test, test-full, test-nightly, gnn, notebooks, notebooks-nightly, docs,
  package, bench, all) and the workflows and the pre-commit hook call those
  same arms. Measured arm times: the generated
  [`docs-mkdocs/development/gate-metrics.md`](docs-mkdocs/development/gate-metrics.md).
- [x] ✓ **Flat documentation index.** Added
  [`docs-mkdocs/documentation-index.md`](docs-mkdocs/documentation-index.md) and
  registered it in `mkdocs.yml` nav.
- [x] ✓ **Testing page.** Added
  [`docs-mkdocs/development/testing.md`](docs-mkdocs/development/testing.md)
  (pass 3), with the unit gate, nightly split, GNN opt-in tests, lint, and docs
  build; registered in `mkdocs.yml` nav.

## Environment prerequisites

- [ ] **Graphviz for `pygraphviz`.** `uv sync --group test`, `--extra nb`, and
  `--extra docs` can fail while building `pygraphviz`; install Graphviz first
  (macOS: Homebrew `graphviz`; Debian/Ubuntu: `graphviz libgraphviz-dev
  pkg-config build-essential python3-dev`). Notes in
  [`docs-mkdocs/getting-started/installation.md`](docs-mkdocs/getting-started/installation.md).
- [x] ✓ **GNN importer conformance adopted.** The pin is
  [`.github/gnn-pin.json`](.github/gnn-pin.json) (GNN `v5.1.0` at
  `75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf`, was `v5.0.3` at
  `8f293d9b` as code constants only), read by GNN's shipped reader
  (`python -m gnn.importers check --repo . --require-pointer`) and pointed at by
  [`.github/GNN.md`](.github/GNN.md). `./gate.sh gnn` exports the pinned tree
  itself (`PYMDP_GNN_CHECKOUT`, default a sibling checkout; or set
  `PYMDP_GNN_ROOT` to an existing export) and fails rather than skipping. There
  is still no hosted GNN conformance job: the arm is local and opt-in, since it
  needs a materialized pinned tree. See [`AGENTS.md`](AGENTS.md) and
  [`docs-mkdocs/guides/gnn-interoperability.md`](docs-mkdocs/guides/gnn-interoperability.md).

## Known defects

- [ ] None verified in this pass. Lint debt and any test failures are reported by
  the declared gate (`./gate.sh test` and `./gate.sh lint`), not tracked here as
  a standing list.

## Owner decisions

- [ ] **Fork custody.** The fork stays additive so `upstream/main` merges stay
  clean; the fork delta is `.aii`, `TASKS.md`, two nightly workflows, three CI
  branch lists, and the GNN bridge. Land upstream syncs as merge commits (never
  squash/rebase) because custody records pin commit SHAs.
- [ ] **Upstream `repo_url`.** `mkdocs.yml` still points `repo_url` at
  `infer-actively/pymdp` (upstream). Confirm whether the published site should
  advertise the fork instead.
