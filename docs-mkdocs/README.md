# `docs-mkdocs/` — documentation source

This is the source tree for the published documentation site
(<https://pymdp-rtd.readthedocs.io/en/latest/>), built by [MkDocs](https://www.mkdocs.org/).
It is **not** a separate `docs/` tree — edit documents here.

## What is inside

| Path | Contents |
| --- | --- |
| [`index.md`](index.md) | Site home page. |
| `getting-started/` | Installation and the JAX quickstart. |
| `guides/` | Generative-model structure, `rollout()`, `PymdpEnv`, GNN interoperability, fork/upstream sync. |
| `migration/` | NumPy/legacy → JAX migration guide. |
| `tutorials/` | Tutorial index and the notebook gallery. |
| `api/` | Generated API reference (mkdocstrings) for each module. |
| `legacy/` | Archived NumPy-era documentation. |
| `development/` | Viewing docs locally, testing, release notes. |
| `assets/`, `styles/`, `javascripts/`, `overrides/` | Theme and build assets (not prose). |

## Build and serve

The build configuration is [`../mkdocs.yml`](../mkdocs.yml). `docs_dir` is
`docs-mkdocs`, and `strict: true` makes the build fail on broken links or a page
missing from `nav`.

```bash
./scripts/docs_build.sh          # strict one-shot build
./scripts/docs_serve.sh          # local preview
./scripts/sync_docs_notebooks.sh # refresh notebook pages from examples/
```

When you add a page, add it to the `nav:` section of `mkdocs.yml` as well — the
strict build rejects orphaned pages.
