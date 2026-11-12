# Publishing `pymdp` models as GNN specifications

[Generalized Notation Notation (GNN)](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation)
is the plain-text specification language for Active Inference generative models.
`pymdp.gnn` is the pymdp-side bridge: it renders a pymdp model as a GNN
Markdown specification and validates that specification with the real GNN
parser, so a model defined in `pymdp` can be published, reviewed, and executed
by GNN tooling without re-entering its matrices by hand.

The bridge targets GNN **v5.1.0**, revision
[`75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf`](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/tree/75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf).

That revision is recorded in [`.github/gnn-pin.json`](https://github.com/ActiveInferenceInstitute/pymdp/blob/master/.github/gnn-pin.json),
read by GNN's own importer reader (`python -m gnn.importers check --repo .`),
and pointed at by [`.github/GNN.md`](https://github.com/ActiveInferenceInstitute/pymdp/blob/master/.github/GNN.md).
`pymdp.gnn.GNN_PINNED_RELEASE` / `GNN_PINNED_REVISION` mirror the pin for
library consumers; `./gate.sh gnn` runs the reader from the pinned parser tree
and requires the mirror to agree with the pin, so the checks cannot drift apart.

`pymdp` declares no hosted conformance job: the GNN arm is opt-in and local
because it needs the pinned GNN tree, which the arm exports itself when
`PYMDP_GNN_ROOT` does not already name one.

## Install

The bridge itself needs only `numpy` (already a pymdp dependency). Validating a
specification imports the GNN parser, which needs the parser's runtime
dependencies:

```bash
uv sync --group test --extra gnn        # defusedxml + psutil
# or, into an existing environment:
uv pip install defusedxml psutil
```

The GNN source tree stays external and pinned; you point the validator at an
exported copy of it (below).

## Emit a specification

```python
from jax import random as jr

from pymdp import gnn, utils
from pymdp.agent import Agent

key = jr.PRNGKey(0)
k1, k2 = jr.split(key)
A = utils.random_A_array(k1, [3], [3])
B = utils.create_controllable_B([3], [3])
C = utils.list_array_uniform([[3]])
D = utils.list_array_uniform([[3]])
agent = Agent(A=A, B=B, C=C, D=D, batch_size=1)

spec = gnn.to_gnn_spec(agent, name="three_state_agent", num_timesteps=10)
path = gnn.write_gnn_spec("three_state_agent.gnn.md", agent, name="three_state_agent", num_timesteps=10)
```

`to_gnn_spec` also accepts a `FlatPOMDP` (`gnn.FlatPOMDP.from_model(...)`,
`gnn.FlatPOMDP.from_agent(...)`) or a mapping with `A`/`B`/`C`/`D` (and optional
`E`) keys. A runnable version of this example lives in
`examples/gnn/publish_gnn_spec.py`.

The emitted file follows GNN's canonical discrete-POMDP section order:
`GNNSection`, `GNNVersionAndFlags`, `ModelName`, `ModelAnnotation`,
`StateSpaceBlock`, `Connections`, `InitialParameterization`, `Equations`,
`Time`, `ActInfOntologyAnnotation`, `ModelParameters`, `Footer`, `Signature`.
`B` is written in pymdp's canonical orientation
`(next_state, previous_state, action)` and the file declares
`b_tensor_order: next_state_previous_state_action` in `ModelParameters`, so the
orientation is self-describing rather than implied.

## Validate against the pinned GNN parser

Export the pinned revision into a directory of your choice (this is exactly the
committed tree, independent of any uncommitted work in the checkout):

```bash
git -C /path/to/Generalized_Notation_Notation archive 75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf src | tar -x -C /tmp/gnn-pin
export PYMDP_GNN_ROOT=/tmp/gnn-pin/src
```

The same export is available from Python:

```python
from pymdp import gnn

exported = gnn.export_pinned_source("/path/to/Generalized_Notation_Notation", "/tmp/gnn-pin")
print(exported.revision)  # 75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf
```

Then validate:

```python
report = gnn.validate_gnn_spec(agent, path, revision=gnn.GNN_PINNED_REVISION)
print(report.describe())
assert report.ok
```

`validate_gnn_spec` runs the parser in an isolated subprocess (interpreter from
`PYMDP_GNN_PYTHON`, default: the running interpreter), then compares:

- every parsed matrix (`A`, `B`, `C`, `D`, and `E` when present) against the
  source arrays, element for element at the source dtype;
- the declared dimensions in `StateSpaceBlock` against the array shapes;
- `num_hidden_states`, `num_obs`, `num_actions`, `num_timesteps`, and
  `b_tensor_order` in `ModelParameters`;
- the presence of all required sections and the parser's own success flag.

Failures are explicit: a failed import, a parser error, or any value mismatch
lands in the report (`mismatches`, `parser_errors`) and sets `report.ok` to
`False`. The test suite covers the negative direction too — validating a
specification against a *different* model, or with a section removed, must fail.

## Round trip through GNN's pymdp route

GNN consumes specifications in the opposite direction: it renders them to
pymdp runner scripts (`gnn.render.pymdp.pymdp_renderer.render_gnn_to_pymdp`)
and executes them (`gnn.execute.pymdp.simulation.run_pymdp_simulation`). The
bridge's integration test closes the loop: it emits a specification, renders
and executes it with GNN, and asserts that the recorded beliefs match pymdp's
own inference for the same observation/action sequence to within `1e-6`.

```bash
PYMDP_GNN_ROOT=/tmp/gnn-pin/src \
PYMDP_GNN_CHECKOUT=/path/to/Generalized_Notation_Notation \
./gate.sh gnn
```

`./gate.sh gnn` runs the shipped importer reader over the pin and pointer, then
this test file against the same tree (exporting the pinned revision first when
`PYMDP_GNN_ROOT` is unset). The emission, refusal, and shape tests also run
under `./gate.sh test`, which needs no GNN tree.

## Scope and boundaries

A GNN discrete-POMDP specification declares one `A`/`B`/`C`/`D` family — one
observation modality and one hidden-state factor — and that is exactly what
GNN's pymdp execution route consumes. Models with more than one modality or
factor are therefore **refused with an explicit error** rather than silently
flattened, because a joint composition convention belongs to GNN (its
`FACTORED` model kind) and not to a second, competing implementation here.

If you want to publish a joint model, build it explicitly and pass its arrays:
for example, lift a reduced likelihood with `pymdp.utils.make_A_full`, form the
joint transition as the Kronecker product of the per-factor `B` tensors, and
hand the resulting `A`/`B`/`C`/`D` to `gnn.FlatPOMDP.from_model`. Policy priors
are likewise only expressible when the number of policies equals the number of
actions (a single planning step); a multi-step policy prior is refused.

## Related documentation and projects

GNN at the pinned revision:

- [Framework integration guide](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/blob/75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf/docs/gnn/integration/framework_integration_guide.md)
- [GNN syntax specification](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/blob/75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf/docs/gnn/gnn_syntax.md)
- [PyMDP implementation guide](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/blob/75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf/docs/gnn/implementations/pymdp.md)

Newer than the pinned revision, on GNN `main`:

- [Downstream adoption: using GNN in a sibling repository](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/blob/main/docs/gnn/integration/downstream_adoption.md) — the surfaces a sibling repository consumes, including the parser and the extractor this bridge uses.

Other repositories that consume or specify models in GNN:

- [`active_inference_battleship`](https://github.com/ActiveInferenceInstitute/active_inference_battleship) — playable model specified in GNN and compiled at build time.
- [`active_inference_language_model`](https://github.com/ActiveInferenceInstitute/active_inference_language_model) — language-model stack with GNN specifications and conformance checks.
- [`ActiveBlockference`](https://github.com/ActiveInferenceInstitute/ActiveBlockference) and [`ActInf_RxInfer`](https://github.com/ActiveInferenceInstitute/ActInf_RxInfer) — block-based and RxInfer implementations in the same ecosystem.
