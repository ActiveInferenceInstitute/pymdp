# `scripts/` — docs and notebook tooling

Build and maintenance helpers for the documentation site and notebook gates.
Run them from the repository root with the docs/notebook extras installed.

| Script | Purpose |
| --- | --- |
| [`docs_build.sh`](docs_build.sh) | Strict one-shot mkdocs build (`./scripts/docs_build.sh`). |
| [`docs_serve.sh`](docs_serve.sh) | Serve the docs locally for preview. |
| [`docs_sync_and_serve.sh`](docs_sync_and_serve.sh) | Sync notebook pages, then build and serve. |
| [`sync_docs_notebooks.sh`](sync_docs_notebooks.sh) | Copy/convert `examples/` notebooks into the docs tree. |
| [`run_notebook_manifest.py`](run_notebook_manifest.py) | Execute the notebooks listed in a manifest (used by CI/nightly). |
| [`notebook_precommit.py`](notebook_precommit.py) | Pre-commit hook that keeps notebooks clean. |

## Usage

```bash
./scripts/docs_build.sh                 # build the site
./scripts/docs_sync_and_serve.sh        # sync notebooks + serve
uv run python scripts/run_notebook_manifest.py test/notebooks/ci_notebooks.txt
```

The build reads [`../mkdocs.yml`](../mkdocs.yml); every docs page must be listed
in its `nav`. See [`../docs-mkdocs/README.md`](../docs-mkdocs/README.md).
