# TODO — pymdp

Owner-facing backlog for the `ActiveInferenceInstitute/pymdp` fork. Grouped by
kind; every item below is verified against the current tree. Planning signposts
also live in [`TASKS.md`](TASKS.md).

Measured 2026-11-12 on `master` at `978f967` (the 1.0.5 sync) with
`./gate.sh lint`, `test`, `gnn` and `docs`. Hosted workflows cannot run — the
organisation's GitHub Actions billing is locked, so e.g. run `43281561063` on
that commit failed in 2 s with no executed steps — and every number below is a
local `./gate.sh` measurement.

## Upstream sync

- [x] ✓ **pymdp 1.0.5 merged.** PR #9 (`sync/upstream-1.0.5`), merge commit
  `978f967`, merging upstream tip `85fde61` (`chore(main): release 1.0.5
  (#450)`) as a merge commit, per the custody rule. `pyproject.toml` now
  declares `version = "1.0.5"`, and `CHANGELOG.md` carries upstream's 1.0.5
  entry (eight fixes plus the jax/jaxlib upper-bound drop, #461). This
  repository *is* the `inferactively-pymdp` distribution, so there is no
  dependency pin on itself to move; the cohort's `inferactively-pymdp` 1.0.5 is
  the version this fork now carries. Measured after the merge:
  `git rev-list --left-right --count 85fde61...978f967` → `0 28` (nothing left
  behind upstream's synced tip, 28 fork-only commits), and the fork delta
  against that tip is 47 files (`git diff --name-only 85fde61 978f967`).
- [ ] **The fork guide's divergence figures are stale, and its sync command
  cannot run in a fresh clone.** [`docs-mkdocs/guides/fork-and-upstream.md`](docs-mkdocs/guides/fork-and-upstream.md)
  still says "the fork is 16 commits ahead of and 1 commit behind
  `upstream/main`; the upstream-only commit is `c8be101`" — measured at
  `978f967` it is 28 ahead and 0 behind. The same page's "Checking the current
  delta" block starts with `git fetch upstream main`, but a clone of this fork
  has only `origin` (`git remote -v`), so the block needs a one-time
  `git remote add upstream https://github.com/infer-actively/pymdp.git` to work
  as written. Owner: repository maintainer.

## Documentation gaps

- [ ] **No top-level `docs/`.** Documentation intentionally lives in
  [`docs-mkdocs/`](docs-mkdocs/README.md) and is built by
  [`mkdocs.yml`](mkdocs.yml) (`docs_dir: docs-mkdocs`). This is deliberate, not a
  missing tree; recorded here so it is not "fixed" by duplicating the tree.
- [x] ✓ **Single gate entry point.** `./gate.sh` now owns every check (lint,
  test, test-full, test-nightly, gnn, notebooks, notebooks-nightly, docs,
  package, bench, all) and the workflows and the pre-commit hook call those
  same arms. Measured arm times: the generated
  [`docs-mkdocs/development/gate-metrics.md`](docs-mkdocs/development/gate-metrics.md)
  (currently stale — see Known defects).
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
  Re-run 2026-11-12 on `978f967`: `./gate.sh gnn` exported the pinned tree and
  reported `19 passed`, including
  `TestPinRecord::test_pin_agrees_with_the_bridge_constants`, so the pin, the
  bridge constants in `pymdp/gnn.py` and `.github/GNN.md` still agree.

## Known defects

- [ ] **The generated gate-metrics page is stale and nothing checks it.**
  [`docs-mkdocs/development/gate-metrics.md`](docs-mkdocs/development/gate-metrics.md)
  records inputs digest `6365024357e0174e`, `test` = 351 selected / 346 passed
  / 5 skipped and `gnn` = 13 passed / 5 skipped. On `978f967` today,
  `uv run --no-sync python scripts/gate_metrics.py --check` exits 1 —
  "inputs digest 6365024357e0174e does not match the tree (8671c0199c5326c5);
  re-run ./gate.sh bench" — while `./gate.sh test` selects 364 (358 passed, 6
  skipped) and `./gate.sh gnn` reports 19 passed. The page drifted because the
  1.0.5 merge landed after the last `./gate.sh bench`, and `--check` is unused
  in practice: no arm or workflow calls it (`git grep -n gate_metrics --
  gate.sh .github` finds only the `bench` arm forwarding `"$@"`). Its recorded
  `Repository commit 0bea4154…` is not itself a defect — PR #7 squashed that
  commit away and the generator documents the SHA as provenance only. Fix: run
  `./gate.sh bench` on `master`, and decide whether `docs` (or `all`) should run
  `--check` so the page cannot drift again. Owner: repository maintainer. Lint
  debt and test failures stay reported by the gate (`./gate.sh lint`,
  `./gate.sh test`) and are not tracked here as a standing list.

## Owner decisions

- [x] ✓ **Fork custody.** The fork stays additive so `upstream/main` merges stay
  clean, and PR #9 followed it: the 1.0.5 sync landed as a merge commit, never a
  squash or rebase, because custody records pin commit SHAs. The delta is wider
  than the earlier note here claimed: against upstream tip `85fde61` it is 47
  files (`git diff --name-only 85fde61 978f967`) — `.aii/`, `.github/GNN.md`,
  `.github/gnn-pin.json`, `.github/dependabot.yml`, eight modified workflows,
  `gate.sh`, `scripts/gate_metrics.py` and `scripts/docs_build.sh`/`docs_serve.sh`,
  `TASKS.md`, `TODO.md`, `AGENTS.md`, `README.md`, the `docs-mkdocs/` hub,
  index, development and guide pages, `pymdp/gnn.py`,
  `examples/gnn/publish_gnn_spec.py`, `test/test_gnn_bridge.py` and
  `test/test_fpi.py`. [`docs-mkdocs/guides/fork-and-upstream.md`](docs-mkdocs/guides/fork-and-upstream.md)
  remains the maintained description of what the fork adds.
- [ ] **Upstream repository URLs.** Three published surfaces still advertise
  upstream: `mkdocs.yml` sets `repo_url`/`repo_name` to `infer-actively/pymdp`,
  `pyproject.toml` sets `[project.urls].Repository` to
  `https://github.com/infer-actively/pymdp`, and `README.md:82` links the PyPI
  version badge at the pinned `https://pypi.org/project/inferactively-pymdp/1.0.0/`
  page. Confirm whether the published site and package metadata should
  advertise the fork, and whether that badge should link the project page
  rather than a release pinned at 1.0.0.
- [ ] **Python floor.** `pyproject.toml` declares `requires-python = ">=3.10"`,
  but the `Test` workflow matrix is `["3.11", "3.12", "3.13", "3.14"]`
  (`.github/workflows/test.yaml`), so 3.10 is declared and never run — and the
  `modelfit` group's `numpyro>=0.22; python_version >= '3.11'` marker exists
  precisely to keep 3.10 installable. Decide: drop 3.10 (raise the floor to the
  lowest tested interpreter) or add 3.10 to the matrix when hosted CI can run
  again. Owner: repository maintainer, together with upstream, since
  `requires-python` is an upstream file.
