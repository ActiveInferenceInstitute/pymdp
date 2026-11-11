# TODO — pymdp

Owner-facing backlog for the `ActiveInferenceInstitute/pymdp` fork. Grouped by
kind; every item below is verified against the current tree. Planning signposts
also live in [`TASKS.md`](TASKS.md).

## Documentation gaps

- [ ] **No top-level `docs/`.** Documentation intentionally lives in
  [`docs-mkdocs/`](docs-mkdocs/README.md) and is built by
  [`mkdocs.yml`](mkdocs.yml) (`docs_dir: docs-mkdocs`). This is deliberate, not a
  missing tree; recorded here so it is not "fixed" by duplicating the tree.
- [ ] **README test command.** The README's "Run tests" block shows
  `pytest test`, which is weaker than the documented gate
  `uv run pytest test -n 2 -m "not nightly"` in [`AGENTS.md`](AGENTS.md). The
  gate is the source of truth; keep them consistent if the README is tightened.
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
- [ ] **GNN opt-in tests need a pinned export.** `test/test_gnn_bridge.py` skips
  unless `PYMDP_GNN_ROOT` points at an exported GNN tree (add
  `PYMDP_GNN_CHECKOUT` for the pinned-export test). A skip is never a pass. See
  [`AGENTS.md`](AGENTS.md) and
  [`docs-mkdocs/guides/gnn-interoperability.md`](docs-mkdocs/guides/gnn-interoperability.md).

## Known defects

- [ ] None verified in this pass. Lint debt and any test failures are reported by
  the declared gate (`uv run pytest test -n 2 -m "not nightly"` and
  `uv run --group dev ruff check`), not tracked here as a standing list.

## Owner decisions

- [ ] **Fork custody.** The fork stays additive so `upstream/main` merges stay
  clean; the fork delta is `.aii`, `TASKS.md`, two nightly workflows, three CI
  branch lists, and the GNN bridge. Land upstream syncs as merge commits (never
  squash/rebase) because custody records pin commit SHAs.
- [ ] **Upstream `repo_url`.** `mkdocs.yml` still points `repo_url` at
  `infer-actively/pymdp` (upstream). Confirm whether the published site should
  advertise the fork instead.
