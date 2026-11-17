# `pymdp/` — the library

The importable package published as `inferactively-pymdp`. The default backend
is JAX (`jax.numpy`); the NumPy implementation is preserved under
[`legacy/`](legacy/) for migration and cross-validation.

## Layout

| Module | Purpose |
| --- | --- |
| [`agent.py`](agent.py) | `Agent` — the active-inference agent (states, policies, learning). |
| [`inference.py`](inference.py) | Variational state inference (FPI, MMP, …). |
| [`control.py`](control.py) | Expected-free-energy policy inference and action selection. |
| [`learning.py`](learning.py) | Dirichlet parameter learning. |
| [`algos.py`](algos.py), [`maths.py`](maths.py), [`utils.py`](utils.py), [`likelihoods.py`](likelihoods.py), [`distribution.py`](distribution.py) | Math kernels, array helpers, distributions. |
| [`gnn.py`](gnn.py) | Generalized Notation Notation bridge (`to_gnn_spec`, `write_gnn_spec`, `validate_gnn_spec`). |
| `envs/` | Built-in environments: T-maze, generalized T-maze, grid worlds, graph worlds, cue chaining, and the `rollout` loop. |
| `planning/` | Sophisticated inference (SI), MCTS, and planning visualization. |
| `legacy/` | Archived NumPy implementation (agents, algos, envs, maths) kept for migration. |

## Conventions

- Model shapes, factor/modality ordering, and the agent loop are defined in
  [`../docs-mkdocs/guides/generative-model-structure.md`](../docs-mkdocs/guides/generative-model-structure.md).
  Read it before changing array conventions.
- Keep the default additive fork policy; the explicit 2026-11-13 correctness
  exception and upstream-forwarding boundary are recorded in
  [`../docs-mkdocs/guides/fork-and-upstream.md`](../docs-mkdocs/guides/fork-and-upstream.md).
- API reference pages live under [`../docs-mkdocs/api/`](../docs-mkdocs/api/);
  docstrings use the NumPy style consumed by mkdocstrings.

## Observed-action likelihoods

The NumPyro likelihood path requires observation and sampled-action tuples.
Blocks reset independently; there is no implicit inter-block carry or
missing-input generative fallback. Probability columns follow public joint
action-support indices. The joint `actions` sample site retains the original
`action_vectors` alongside those indices.
