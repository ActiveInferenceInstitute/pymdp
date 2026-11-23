# Fork and upstream

`ActiveInferenceInstitute/pymdp` is a tracked fork of
[`infer-actively/pymdp`](https://github.com/infer-actively/pymdp). The fork's
`origin` is `ActiveInferenceInstitute/pymdp`, its default branch is `master`, and
`upstream` is the fetch-only remote (`pushurl` is `DISABLED`).

## What the fork adds

The default policy is additive fork-local work, keeping upstream merges small.
The explicit 2026-11-13 group19 correctness-repair exception covers the
preserved source repairs and associated documentation, with source-bound
qualification returned to the parent integration owner. Upstream forwarding
remains an owner decision, not a new general permission to diverge. The
original additive delta is:

- the `.aii/` InstituteOS sidecar metadata;
- `TASKS.md` operator task discovery;
- the GNN bridge: `pymdp/gnn.py`, `examples/gnn/publish_gnn_spec.py`,
  `test/test_gnn_bridge.py`, and the guides
  [Generative Model Structure](generative-model-structure.md) and
  [Publishing Models as GNN Specifications](gnn-interoperability.md);
- fork-local nightly workflows and the `master` entries in the CI branch lists;
- the upstream sync convention in `AGENTS.md`.

Outside that explicit correctness exception, fork-local features must not
rewrite upstream paths merely to make the feature fit.

## How the fork stays in sync

`upstream` points at `infer-actively/pymdp` and is never pushed to. Sync by
merging `upstream/main` into a branch off `master`, then land it as a merge
commit (never squash or rebase, because custody records pin commit SHAs). The
divergence in `manual-branch-nightly.yaml` and `nightly-tests.yaml` is
deliberate: the fork keeps `workflow_dispatch`-only scheduling and a `master`
checkout ref and takes upstream's step-level changes on top.

## Checking the current delta

The following setup is for a fresh clone only. Preserve existing remotes;
actual remote mutation/fetch requires separate repository-owner authorization.

```bash
git remote add upstream https://github.com/infer-actively/pymdp.git
git remote set-url --push upstream DISABLED
git fetch upstream main
git log --oneline master ^upstream/main
git diff --stat upstream/main master
```

Historical synchronization at fork `978f967` versus upstream `85fde61`
recorded 28 fork-only and zero upstream-only commits. This is not a claim
about current divergence.

## The measured documentation gaps are upstream's

The fleet scorecard measured 34 authored directories without a README or
`AGENTS.md` at fork tip `8a1cd26`. Every one is in an upstream-inherited tree:

- `docs-mkdocs/` and 13 subdirectories: `algos`, `api`, `assets`,
  `development`, `getting-started`, `guides`, `javascripts`, `migration`,
  `notebooks`, `overrides/modules`, `styles`, `tutorials`,
  `tutorials/notebooks`;
- `examples/` and 11 subdirectories: `advanced`,
  `advanced/infer_states_optimization`, `api`, `envs`,
  `experimental/sophisticated_inference`, `gnn`, `inductive_inference`,
  `inference_and_learning`, `learning`, `model_fitting`, `sparse`;
- `pymdp/`, `pymdp/envs`, `pymdp/envs/assets`, `pymdp/planning`;
- `scripts`, `test`, `test/matlab_crossval/generation`, `paper`.

Those gaps belong to `infer-actively/pymdp`. The fork does not add signposts to
upstream code paths, because that turns a reversible sync into a fork-only
divergence. The fork's own documentation surfaces are listed under
[What the fork adds](#what-the-fork-adds).

## Documentation boundaries

The upstream-inherited documentation trees (`docs-mkdocs/`, `examples/`,
`pymdp/`, `scripts/`, `test/`, `paper/`) are owned upstream. The fork's own
surfaces are the sidecar, the GNN bridge guides, `TASKS.md`, and the fork notes
in `AGENTS.md` and `README.md`. Prefer forwarding upstream-inherited repairs
upstream; the 2026-11-13 correctness exception does not decide publication or
upstream acceptance.
