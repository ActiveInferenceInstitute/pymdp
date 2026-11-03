# Fork and upstream

`ActiveInferenceInstitute/pymdp` is a tracked fork of
[`infer-actively/pymdp`](https://github.com/infer-actively/pymdp). The fork's
`origin` is `ActiveInferenceInstitute/pymdp`, its default branch is `master`, and
`upstream` is the fetch-only remote (`pushurl` is `DISABLED`).

## What the fork adds

Fork-local changes stay additive so that merges from `upstream/main` remain
clean. The delta is:

- the `.aii/` InstituteOS sidecar metadata;
- `TASKS.md` operator task discovery;
- the GNN bridge: `pymdp/gnn.py`, `examples/gnn/publish_gnn_spec.py`,
  `test/test_gnn_bridge.py`, and the guides
  [Generative Model Structure](generative-model-structure.md) and
  [Publishing Models as GNN Specifications](gnn-interoperability.md);
- fork-local nightly workflows and the `master` entries in the CI branch lists;
- the upstream sync convention in `AGENTS.md`.

The fork does not rewrite upstream code paths to make a fork-local feature fit.

## How the fork stays in sync

`upstream` points at `infer-actively/pymdp` and is never pushed to. Sync by
merging `upstream/main` into a branch off `master`, then land it as a merge
commit (never squash or rebase, because custody records pin commit SHAs). The
divergence in `manual-branch-nightly.yaml` and `nightly-tests.yaml` is
deliberate: the fork keeps `workflow_dispatch`-only scheduling and a `master`
checkout ref and takes upstream's step-level changes on top.

## Checking the current delta

```bash
git fetch upstream main
git log --oneline master ^upstream/main
git diff --stat upstream/main master
```

At the time of writing the fork is 16 commits ahead of and 1 commit behind
`upstream/main`; the upstream-only commit is `c8be101`
(`fix(graph_worlds): reject partial or mismatched batch locations`).

## Documentation boundaries

The upstream-inherited documentation trees (`docs-mkdocs/`, `examples/`,
`pymdp/`, `scripts/`, `test/`, `paper/`) are owned upstream. The fork's own
surfaces are the sidecar, the GNN bridge guides, `TASKS.md`, and the fork notes
in `AGENTS.md` and `README.md`. When a change touches an upstream-inherited
path, prefer sending it upstream; keep fork-local fixes additive.
