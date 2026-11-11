# `examples/` — notebooks and runnable examples

Curated Jupyter notebooks and small scripts that demonstrate `pymdp` active
inference. Most are published in the site's
[Notebook Gallery](https://pymdp-rtd.readthedocs.io/en/latest/tutorials/notebooks/).

## Layout

| Path | Contents |
| --- | --- |
| `advanced/` | Larger models (action dependencies, neural encoder, inference optimisation). |
| `api/` | API-oriented tutorial notebooks (generative-model construction). |
| `envs/` | Environment demos: T-maze, generalized T-maze, graph worlds, cue chaining, knapsack. `chained_cue_navigation.py` generates the README GIF. |
| `experimental/` | Sophisticated-inference and MCTS prototypes. |
| `gnn/` | `publish_gnn_spec.py` — exports a model as a GNN `.gnn.md` specification. |
| `inductive_inference/` | Inductive-inference examples. |
| `inference_and_learning/` | Inference-method comparisons. |
| `learning/` | Learning/dirichlet demos. |
| `legacy/` | NumPy-era notebooks kept for migration reference. |
| `model_fitting/` | `pybefit`-based fitting notebooks and scripts. |
| `sparse/` | Sparse-backend benchmarks. |

## Running

Install the notebook extras and execute a notebook with Jupyter, or run the
plain scripts directly:

```bash
uv sync --group test --extra nb
jupyter notebook examples/          # browse and execute
uv run python examples/gnn/publish_gnn_spec.py
```

Notebook pages on the site are generated from these files by
`./scripts/sync_docs_notebooks.sh`; see [`../docs-mkdocs/README.md`](../docs-mkdocs/README.md).
