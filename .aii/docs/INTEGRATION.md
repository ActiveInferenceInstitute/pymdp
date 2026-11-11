# InstituteOS integration — pymdp

## Identity

| Field | Value |
| --- | --- |
| Repository | `ActiveInferenceInstitute/pymdp` |
| Affiliation | `institute` |
| Category | `research-software` |
| Status | `active` |
| Maturity | `released` |
| Steward | Active Inference Institute |
| Default branch | `master` |
| Current release | — |
| License | `MIT` |
| Citation | Active Inference Institute — pymdp. |

## Federation

| Setting | Value |
| --- | --- |
| Dashboard mode | `repos` |
| Analytics | `true` |
| Dashboard | `true` |
| Sync policy | `pull` |
| Registries | — |
| Tags | pymdp, python, active-inference, markov-blankets, variational-message-passing |

## Ecosystem relations

| Repository | Relation | Note |
| --- | --- | --- |
| `ActiveInferenceInstitute/InstituteOS` | `referenced-by` | Federated via the InstituteOS .aii manifest. |
| `ActiveInferenceInstitute/Active_Inference_Ontology` | `references` | Uses Active Inference Ontology terms. |
| `ActiveInferenceInstitute/pymcp` | `feeds` | Exposes PyMDP over the Model Context Protocol and REST. |
| `ActiveInferenceInstitute/active_inference_language_model` | `feeds` | Conformance target for the AILM discrete language stack. |
| `ActiveInferenceInstitute/active_inference_power` | `feeds` | Agent engine for the statistical-power simulations. |
| `ActiveInferenceInstitute/policy_entanglement` | `feeds` | POMDP engine for the policy-entanglement experiments. |
| `ActiveInferenceInstitute/on_policy_distillation` | `feeds` | POMDP classroom engine for the distillation experiments. |
| `ActiveInferenceInstitute/active_fractal_rabbit` | `feeds` | Agent engine for the FractalRabbit scaffold. |
| `ActiveInferenceInstitute/Biofirm` | `feeds` | Agent engine for ecological control experiments. |
| `ActiveInferenceInstitute/Generalized_Notation_Notation` | `references` | Uses Generalized Notation Notation (GNN) for model specifications. |

## Key artifacts

| Path | Kind | Description |
| --- | --- | --- |
| `README.md` | `doc` | Repository overview. |
| `pyproject.toml` | `code` | Python package config. |
| `LICENSE` | `doc` | License. |
| `AGENTS.md` | `doc` | Agent guidance. |
| `CHANGELOG.md` | `doc` | Changelog. |
| `test` | `test` | Test suite. |
| `docs-mkdocs` | `doc` | Published documentation tree (mkdocs source). |
| `paper` | `publication` | Published paper artifacts. |

## Portable tasks

| Task | Command | Working directory | Description |
| --- | --- | --- | --- |
| `test` | `uv run pytest test -n 2 -m "not nightly"` | `.` | Run the unit test suite (nightly-marked tests excluded). |

## Validation

```bash
python -m aii_platform.aii_sidecar.validate <repo> --completeness
python -m aii_platform.aii_sidecar docs <repo> --overwrite
python -m aii_platform.aii_sidecar.run <repo> --list
```

The sidecar is valid when `validate` prints `OK` and reports no missing artifact,
manifest, link, or declared-doc path. Completeness is a curation target, not a
runtime requirement.

## Data boundary

This sidecar carries metadata only. It does not grant execution permission, does not
replace the repository's own documentation, and must not be used to move private
content into a public repository. Task execution is limited to trusted checkouts.
