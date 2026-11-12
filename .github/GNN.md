# GNN pointer

`pymdp` publishes generative models as Generalized Notation Notation
specifications and validates them with the upstream GNN parser.

- **Pin file:** [`.github/gnn-pin.json`](gnn-pin.json) — schema
  `gnn-importer-pin/v1`, repository
  `ActiveInferenceInstitute/Generalized_Notation_Notation`, tag `v5.1.0`,
  commit `75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf`.
- **Arm:** `./gate.sh gnn` runs the shipped upstream reader,
  `python -m gnn.importers check --repo . --require-pointer`, from a
  materialized pinned parser tree, then runs `test/test_gnn_bridge.py` there.
- **No hosted conformance job:** the arm stays local and opt-in because it needs
  a materialized pinned tree. `./gate.sh gnn` exports the revision the pin
  records when `PYMDP_GNN_ROOT` does not already name an export
  (`PYMDP_GNN_CHECKOUT`, default a sibling `Generalized_Notation_Notation`
  checkout) and fails when it cannot, rather than skipping.

The bridge constants `GNN_PINNED_RELEASE` / `GNN_PINNED_REVISION` in
[`../pymdp/gnn.py`](../pymdp/gnn.py) mirror this pin for library consumers. The
arm requires them to agree with the pin (`--expected-tag` / `--expected-commit`)
and the pin test reads this file with `gnn.importers.read_pin`.
