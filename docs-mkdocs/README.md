# `docs-mkdocs/` — documentation source

This is the source tree for the published documentation site
(<https://pymdp-rtd.readthedocs.io/en/latest/>), built by [MkDocs](https://www.mkdocs.org/).
It is **not** a separate `docs/` tree — edit documents here.

The flat list of documents is [`documentation-index.md`](documentation-index.md).

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
| `development/` | Viewing docs locally, `./gate.sh` arms and measured times, release notes. |
| `assets/`, `styles/`, `javascripts/`, `overrides/` | Theme and build assets (not prose). |

## Build and serve

The build configuration is [`../mkdocs.yml`](../mkdocs.yml). `docs_dir` is
`docs-mkdocs`, and `strict: true` makes the build fail on broken links or a page
missing from `nav`.

```bash
./gate.sh docs                   # the docs arm: sync notebooks + strict build
./scripts/docs_serve.sh          # local preview
./scripts/sync_docs_notebooks.sh # refresh notebook pages from examples/
```

`./gate.sh docs` is the same command the `Docs` workflow runs; see
[`development/testing.md`](development/testing.md) for the other arms.

Add a new page to the `nav:` section of `mkdocs.yml`; the strict build fails on
a nav entry that points at a missing file or on a broken relative link
(`validation.omitted_files: ignore` only relaxes the orphan-page warning).
