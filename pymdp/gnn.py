"""Publish `pymdp` generative models as Generalized Notation Notation (GNN) specifications.

GNN is the plain-text specification language for Active Inference generative
models (https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation).
This module is the pymdp-side bridge in the *publishing* direction:

* :func:`to_gnn_spec` renders a flat discrete POMDP (one observation modality,
  one hidden-state factor) as a GNN Markdown specification whose matrices are
  pymdp's own arrays.
* :func:`write_gnn_spec` writes that text to a ``.gnn.md`` file.
* :func:`validate_gnn_spec` re-parses the emitted file with the real upstream
  GNN parser and compares the parsed matrices, declared dimensions and model
  parameters against the source arrays. The parser is imported in an isolated
  subprocess from an explicit source tree, so validation is bound to the
  revision you hand it (see :func:`export_pinned_source`).

Usage
-----
::

    from jax import random as jr
    from pymdp import utils
    from pymdp.agent import Agent
    from pymdp import gnn

    key = jr.PRNGKey(0)
    A = utils.random_A_array(key, [3], [3])
    B = utils.create_controllable_B([3], [3])
    C = utils.list_array_uniform([[3]])
    D = utils.list_array_uniform([[3]])
    agent = Agent(A=A, B=B, C=C, D=D, batch_size=1)

    spec = gnn.to_gnn_spec(agent, name="my_model")
    path = gnn.write_gnn_spec("my_model.gnn.md", agent, name="my_model")

    # Bind validation to an exported, immutable copy of the pinned GNN tree:
    #   git -C <gnn-checkout> archive 8f293d9ba src | tar -x -C <destination>
    report = gnn.validate_gnn_spec(agent, path, gnn_source_root="<destination>/src")
    assert report.ok, report.describe()

The emitted file follows GNN's canonical discrete-POMDP section order
(``GNNSection``, ``GNNVersionAndFlags``, ``ModelName``, ``ModelAnnotation``,
``StateSpaceBlock``, ``Connections``, ``InitialParameterization``,
``Equations``, ``Time``, ``ActInfOntologyAnnotation``, ``ModelParameters``,
``Footer``, ``Signature``). ``B`` is written in pymdp's canonical orientation
``(next_state, previous_state, action)`` and the file declares
``b_tensor_order: next_state_previous_state_action`` in ``ModelParameters`` so
the orientation is self-describing.

Scope
-----
A GNN discrete-POMDP specification declares a single ``A``, ``B``, ``C``, ``D``
(and optional ``E``) family, and GNN's pymdp execution route consumes exactly
that flat shape. Models with more than one observation modality or more than
one hidden-state factor are therefore refused with an explicit error instead of
being silently flattened: pymdp's factorized models need a joint composition
convention, which belongs to the GNN side (see GNN's ``FACTORED`` model kind)
rather than to a second, competing implementation here. Users who want to
publish a joint model can construct it explicitly (for example with
:func:`pymdp.utils.make_A_full`) and pass the resulting arrays to this module.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "GNN_PIN_FILE",
    "GNN_PINNED_RELEASE",
    "GNN_PINNED_REVISION",
    "GNN_SPEC_SUFFIX",
    "GNN_SOURCE_ENV",
    "GNN_PYTHON_ENV",
    "GNN_B_ORDER",
    "GNNBridgeError",
    "FlatPOMDP",
    "GNNValidationReport",
    "PinnedGNNSource",
    "load_gnn_pin",
    "to_gnn_spec",
    "write_gnn_spec",
    "validate_gnn_spec",
    "export_pinned_source",
]

#: Committed GNN pin file, relative to the repository root. It is the
#: authoritative source of the pinned revision; the ``_GNN_PINNED_*`` fallback
#: constants are used only when the file is absent (for example a wheel install).
GNN_PIN_FILE = ".github/gnn-pin.json"

#: Fallback GNN release, used only when :data:`GNN_PIN_FILE` cannot be read.
_GNN_PINNED_RELEASE_FALLBACK = "v5.0.3"

#: Fallback commit of the pinned GNN release, used only when
#: :data:`GNN_PIN_FILE` cannot be read.
_GNN_PINNED_REVISION_FALLBACK = "8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1"


def load_gnn_pin(
    pin_file: str | os.PathLike[str] | None = None,
) -> dict[str, Any] | None:
    """Read the committed GNN pin and return it as a mapping, or ``None``.

    The pin lives at :data:`GNN_PIN_FILE` in the repository root. ``None`` means
    the file is absent or unreadable (for example a wheel install), in which
    case :data:`GNN_PINNED_REVISION` and :data:`GNN_PINNED_RELEASE` fall back to
    their in-code constants.
    """
    path = (
        Path(pin_file)
        if pin_file is not None
        else Path(__file__).resolve().parents[1] / GNN_PIN_FILE
    )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


_gnn_pin = load_gnn_pin() or {}

#: GNN release this bridge targets. Authoritative source: the ``tag`` field of
#: :data:`GNN_PIN_FILE`; the fallback constant applies only when it is absent.
GNN_PINNED_RELEASE = _gnn_pin.get("tag") or _GNN_PINNED_RELEASE_FALLBACK

#: Commit of the pinned GNN release. Authoritative source: the ``revision``
#: field of :data:`GNN_PIN_FILE`; the fallback constant applies only when it is
#: absent.
GNN_PINNED_REVISION = _gnn_pin.get("revision") or _GNN_PINNED_REVISION_FALLBACK

#: Suffix for emitted specifications (self-identifying; GNN parses it as Markdown).
GNN_SPEC_SUFFIX = ".gnn.md"

#: Environment variable naming the GNN source tree used for validation.
GNN_SOURCE_ENV = "PYMDP_GNN_ROOT"

#: Environment variable naming the interpreter used for the validation subprocess.
GNN_PYTHON_ENV = "PYMDP_GNN_PYTHON"

#: Orientation of ``B`` in emitted specifications (pymdp's canonical order).
GNN_B_ORDER = "next_state_previous_state_action"

_SECTIONS = (
    "GNNSection",
    "GNNVersionAndFlags",
    "ModelName",
    "ModelAnnotation",
    "StateSpaceBlock",
    "Connections",
    "InitialParameterization",
    "Equations",
    "Time",
    "ActInfOntologyAnnotation",
    "ModelParameters",
    "Footer",
    "Signature",
)

_ONTOLOGY = (
    ("A", "LikelihoodMatrix"),
    ("B", "TransitionMatrix"),
    ("C", "LogPreferenceVector"),
    ("D", "PriorOverHiddenStates"),
    ("E", "Habit"),
    ("F", "VariationalFreeEnergy"),
    ("G", "ExpectedFreeEnergy"),
    ("s", "HiddenState"),
    ("s_prime", "NextHiddenState"),
    ("o", "Observation"),
    ("π", "PolicyVector"),
    ("u", "Action"),
    ("t", "Time"),
)

_RESULT_MARKER = "PYMDP_GNN_RESULT "

_PARSER_RUNNER = """
import json
import sys

sys.path[:0] = [sys.argv[1]]
try:
    from gnn.parsers.markdown_parser import MarkdownGNNParser
except Exception as exc:  # noqa: BLE001 - reported, never swallowed
    print(
        "PYMDP_GNN_RESULT "
        + json.dumps(
            {
                "ok": False,
                "import_error": f"{type(exc).__name__}: {exc}",
                "source": sys.argv[1],
            }
        )
    )
    raise SystemExit(0)

try:
    result = MarkdownGNNParser().parse_file(sys.argv[2])
    model = result.model
    time_spec = model.time_specification
    payload = {
        "ok": True,
        "success": bool(result.success),
        "errors": [str(error) for error in result.errors],
        "warnings": [str(warning) for warning in result.warnings],
        "model_name": model.model_name,
        "version": model.version,
        "sections": list(model.raw_sections),
        "variables": [
            {"name": var.name, "dimensions": list(var.dimensions or [])}
            for var in model.variables
        ],
        "parameters": [
            {"name": param.name, "value": param.value} for param in model.parameters
        ],
        "ontology": [
            {"variable": mapping.variable_name, "term": mapping.ontology_term}
            for mapping in model.ontology_mappings
        ],
        "time": None
        if time_spec is None
        else {
            "time_type": time_spec.time_type,
            "discretization": time_spec.discretization,
            "horizon": time_spec.horizon,
        },
    }
except Exception as exc:  # noqa: BLE001 - reported, never swallowed
    payload = {"ok": False, "runner_error": f"{type(exc).__name__}: {exc}"}

print("PYMDP_GNN_RESULT " + json.dumps(payload, allow_nan=False, default=str))
"""


class GNNBridgeError(ValueError):
    """Raised when a pymdp model cannot be expressed as a GNN specification."""


def _pymdp_version() -> str:
    try:
        return metadata.version("inferactively-pymdp")
    except Exception:  # noqa: BLE001 - metadata is provenance, not behaviour
        return "unknown"


def _format_number(value: np.ndarray) -> str:
    """Shortest decimal literal that round-trips at the array's own dtype."""
    array = np.asarray(value)
    if array.dtype.kind in "iub":
        return str(int(array))
    return np.format_float_positional(array, unique=True, trim="0")


def _format_vector(vector: np.ndarray) -> str:
    values = ", ".join(_format_number(element) for element in np.asarray(vector).ravel())
    return "{" + f"({values})" + "}"


def _format_matrix(matrix: np.ndarray) -> str:
    rows = [
        "  (" + ", ".join(_format_number(element) for element in row) + ")"
        for row in np.asarray(matrix)
    ]
    return "{\n" + ",\n".join(rows) + "\n}"


def _format_tensor(tensor: np.ndarray) -> str:
    """Render a 3-D tensor as nested tuples over its leading axis."""
    blocks = []
    for block in np.asarray(tensor):
        rows = [
            "(" + ", ".join(_format_number(element) for element in row) + ")"
            for row in block
        ]
        blocks.append("  ( " + ", ".join(rows) + " )")
    return "{\n" + ",\n".join(blocks) + "\n}"


def _drop_batch(
    array: np.ndarray, expected_ranks: Sequence[int], name: str
) -> np.ndarray:
    """Return ``array`` with an optional leading batch axis of size 1 removed."""
    array = np.asarray(array)
    if array.ndim in expected_ranks:
        return array
    if array.ndim - 1 in expected_ranks and array.shape[0] == 1:
        return array[0]
    ranks = " or ".join(str(rank) for rank in expected_ranks)
    raise GNNBridgeError(
        f"{name} has shape {array.shape}; expected rank {ranks} "
        f"(optionally with a leading batch axis of size 1)"
    )


@dataclass(frozen=True)
class FlatPOMDP:
    """A flat discrete POMDP: one observation modality, one hidden-state factor.

    Attributes
    ----------
    A:
        Likelihood array, shape ``(num_obs, num_states)``.
    B:
        Transition tensor, shape ``(num_states, num_states, num_actions)`` in
        pymdp's canonical ``(next_state, previous_state, action)`` orientation.
        A 2-D ``B`` is promoted to a single-action tensor.
    C:
        Log-preference vector, shape ``(num_obs,)``.
    D:
        Prior over initial hidden states, shape ``(num_states,)``.
    E:
        Optional habit prior over actions, shape ``(num_actions,)``. Only
        expressible when the number of policies equals the number of actions
        (a single planning step).
    name:
        Model name written to ``## ModelName``.
    annotation:
        Free-text ``## ModelAnnotation``.
    num_timesteps:
        Simulation horizon written to ``## Time`` and ``## ModelParameters``.
    """

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    E: np.ndarray | None = None
    name: str = "pymdp_model"
    annotation: str = ""
    num_timesteps: int = 20

    def __post_init__(self) -> None:
        object.__setattr__(self, "A", np.asarray(self.A, dtype=np.float64))
        B = np.asarray(self.B, dtype=np.float64)
        if B.ndim == 2:
            B = B[:, :, np.newaxis]
        object.__setattr__(self, "B", B)
        object.__setattr__(self, "C", np.asarray(self.C, dtype=np.float64).reshape(-1))
        object.__setattr__(self, "D", np.asarray(self.D, dtype=np.float64).reshape(-1))
        if self.E is not None:
            object.__setattr__(
                self, "E", np.asarray(self.E, dtype=np.float64).reshape(-1)
            )
        if self.A.ndim != 2:
            raise GNNBridgeError(f"A must be 2-D (num_obs, num_states); got {self.A.shape}")
        if self.B.ndim != 3:
            raise GNNBridgeError(
                f"B must be 2-D or 3-D (next_state, previous_state, action); got {self.B.shape}"
            )
        if self.B.shape[0] != self.B.shape[1]:
            raise GNNBridgeError(
                f"B must be square over states; got {self.B.shape}"
            )
        if self.B.shape[0] != self.A.shape[1]:
            raise GNNBridgeError(
                f"A expects {self.A.shape[1]} states but B has {self.B.shape[0]}"
            )
        if self.C.shape != (self.A.shape[0],):
            raise GNNBridgeError(
                f"C must have shape ({self.A.shape[0]},); got {self.C.shape}"
            )
        if self.D.shape != (self.A.shape[1],):
            raise GNNBridgeError(
                f"D must have shape ({self.A.shape[1]},); got {self.D.shape}"
            )
        if self.E is not None and self.E.shape != (self.num_actions,):
            raise GNNBridgeError(
                f"E must have shape ({self.num_actions},) to be expressible as a "
                f"flat GNN habit prior; got {self.E.shape}. A policy prior over "
                "multi-step policies has no flat GNN equivalent."
            )
        if self.num_timesteps < 1:
            raise GNNBridgeError(f"num_timesteps must be positive; got {self.num_timesteps}")

    @property
    def num_obs(self) -> int:
        """Number of observation outcomes."""
        return int(self.A.shape[0])

    @property
    def num_states(self) -> int:
        """Number of hidden states."""
        return int(self.A.shape[1])

    @property
    def num_actions(self) -> int:
        """Number of actions (final axis of ``B``)."""
        return int(self.B.shape[2])

    @classmethod
    def from_model(
        cls,
        A: Any,
        B: Any,
        C: Any,
        D: Any,
        E: Any | None = None,
        *,
        name: str = "pymdp_model",
        annotation: str = "",
        num_timesteps: int = 20,
    ) -> "FlatPOMDP":
        """Build a flat model from arrays or single-element pymdp lists.

        Accepts pymdp's list-of-arrays convention (``[A]``, ``[B]``, ...) with an
        optional leading batch axis of size 1, and plain arrays.
        """
        return cls(
            A=_single(A, "A", expected_ranks=(2,)),
            B=_single(B, "B", expected_ranks=(2, 3)),
            C=_single(C, "C", expected_ranks=(1,)),
            D=_single(D, "D", expected_ranks=(1,)),
            E=None if E is None else _single(E, "E", expected_ranks=(1,)),
            name=name,
            annotation=annotation,
            num_timesteps=num_timesteps,
        )

    @classmethod
    def from_agent(cls, agent: Any, **kwargs: Any) -> "FlatPOMDP":
        """Build a flat model from a pymdp :class:`pymdp.agent.Agent`.

        Refuses agents with more than one observation modality, more than one
        hidden-state factor, or a policy prior that is not a single-step habit
        over actions.
        """
        num_modalities = int(getattr(agent, "num_modalities", len(agent.A)))
        num_factors = int(getattr(agent, "num_factors", len(agent.B)))
        if num_modalities != 1 or num_factors != 1:
            raise GNNBridgeError(
                "GNN's discrete-POMDP family (and GNN's pymdp execution route) "
                "declares one observation modality and one hidden-state factor; "
                f"this agent has num_modalities={num_modalities}, "
                f"num_factors={num_factors}. Publish an explicit joint model "
                "(for example by lifting reduced likelihoods with "
                "pymdp.utils.make_A_full) and pass its arrays instead."
            )
        num_controls = list(getattr(agent, "num_controls", [1]))
        if len(num_controls) != 1:
            raise GNNBridgeError(
                f"Expected a single control factor; got {num_controls}"
            )
        A = _drop_batch(np.asarray(agent.A[0]), (2,), "A[0]")
        B = _drop_batch(np.asarray(agent.B[0]), (2, 3), "B[0]")
        C = _drop_batch(np.asarray(agent.C[0]), (1,), "C[0]")
        D = _drop_batch(np.asarray(agent.D[0]), (1,), "D[0]")
        E = None
        if getattr(agent, "E", None) is not None:
            E = _drop_batch(np.asarray(agent.E), (1,), "E")
        return cls(
            A=A,
            B=B,
            C=C,
            D=D,
            E=E,
            **kwargs,
        )

    def to_gnn_spec(self) -> str:
        """Render the model as a GNN Markdown specification."""
        return _render_spec(self)

    def write(self, path: str | os.PathLike[str]) -> Path:
        """Write the GNN specification to ``path`` and return the written path."""
        return write_gnn_spec(path, self)

    def validate(
        self,
        spec: str | os.PathLike[str] | None = None,
        *,
        gnn_source_root: str | os.PathLike[str] | None = None,
        python: str | os.PathLike[str] | None = None,
        timeout: float = 300.0,
    ) -> "GNNValidationReport":
        """Validate this model's specification against the GNN parser."""
        return validate_gnn_spec(
            self,
            self.to_gnn_spec() if spec is None else spec,
            gnn_source_root=gnn_source_root,
            python=python,
            timeout=timeout,
        )


def _single(value: Any, name: str, *, expected_ranks: Sequence[int]) -> np.ndarray:
    """Accept a plain array or a one-element pymdp list, with optional batch axis."""
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise GNNBridgeError(
                f"{name} has {len(value)} entries; a flat GNN specification "
                "declares exactly one observation modality and one hidden-state "
                "factor."
            )
        value = value[0]
    return _drop_batch(np.asarray(value), expected_ranks, name)


def _render_spec(model: FlatPOMDP) -> str:
    no, ns, na = model.num_obs, model.num_states, model.num_actions
    annotation = model.annotation or (
        f"Flat discrete POMDP exported from pymdp: {no} observations, "
        f"{ns} hidden states, {na} actions."
    )
    state_space = [
        f"A[{no},{ns},type=float]   # Likelihood mapping hidden states to observations",
        f"B[{ns},{ns},{na},type=float]   # Transitions (next_state, previous_state, action)",
        f"C[{no},type=float]   # Log-preferences over observations",
        f"D[{ns},type=float]   # Prior over initial hidden states",
    ]
    if model.E is not None:
        state_space.append(f"E[{na},type=float]   # Habit prior over actions")
    state_space.extend(
        [
            f"s[{ns},1,type=float]   # Current hidden state distribution",
            f"s_prime[{ns},1,type=float]   # Next hidden state distribution",
            "F[π,type=float]   # Variational Free Energy",
            f"o[{no},1,type=int]   # Current observation (integer index)",
            f"π[{na},type=float]   # Policy (distribution over actions)",
            "u[1,type=int]   # Action taken",
            "G[π,type=float]   # Expected Free Energy (per policy)",
            "t[1,type=int]   # Discrete time step",
        ]
    )

    connections = ["D>s", "s-A", "s>s_prime", "A-o", "s-B", "C>G"]
    if model.E is not None:
        connections.append("E>π")
    connections.extend(["G>π", "π>u", "B>u", "u>s_prime"])

    parameters = [
        ("A", _format_matrix(model.A)),
        ("B", _format_tensor(model.B)),
        ("C", _format_vector(model.C)),
        ("D", _format_vector(model.D)),
    ]
    if model.E is not None:
        parameters.append(("E", _format_vector(model.E)))

    parameter_block = "\n\n".join(f"{key}={value}" for key, value in parameters)

    footer = (
        f"Emitted by pymdp {_pymdp_version()} (pymdp.gnn) from the model "
        f'"{model.name}". B is written in pymdp\'s canonical orientation '
        f"({GNN_B_ORDER.replace('_', ', ')}) and declared as b_tensor_order in "
        f"ModelParameters. Emitted for validation against GNN {GNN_PINNED_RELEASE} "
        f"({GNN_PINNED_REVISION[:8]})."
    )

    lines = [
        "# GNN specification emitted by pymdp (pymdp.gnn)",
        f"# Model: {model.name}",
        "",
        "## GNNSection",
        "ActInfPOMDP",
        "",
        "## GNNVersionAndFlags",
        "GNN v1.1",
        "",
        "## ModelName",
        model.name,
        "",
        "## ModelAnnotation",
        annotation,
        "",
        "## StateSpaceBlock",
        *state_space,
        "",
        "## Connections",
        *connections,
        "",
        "## InitialParameterization",
        parameter_block,
        "",
        "## Equations",
        "# Standard Active Inference update equations for discrete POMDPs:",
        "# state inference by variational free energy (infer_states),",
        "# policy inference by expected free energy (infer_policies),",
        "# action selection from the policy posterior (sample_action).",
        "",
        "## Time",
        "Time=t",
        "Dynamic",
        "Discrete",
        f"ModelTimeHorizon={model.num_timesteps}",
        "",
        "## ActInfOntologyAnnotation",
        *[f"{variable}={term}" for variable, term in _ONTOLOGY if variable != "E" or model.E is not None],
        "",
        "## ModelParameters",
        f"num_hidden_states: {ns}",
        f"num_obs: {no}",
        f"num_actions: {na}",
        f"num_timesteps: {model.num_timesteps}",
        f"b_tensor_order: {GNN_B_ORDER}",
        "",
        "## Footer",
        footer,
        "",
        "## Signature",
        "Unsigned specification emitted by pymdp; provenance is recorded in the Footer.",
        "",
    ]
    return "\n".join(lines)


def to_gnn_spec(model: FlatPOMDP | Any, **kwargs: Any) -> str:
    """Render ``model`` (a :class:`FlatPOMDP` or pymdp ``Agent``) as GNN text."""
    return _as_flat_pomdp(model, **kwargs).to_gnn_spec()


def write_gnn_spec(
    path: str | os.PathLike[str], model: FlatPOMDP | Any, **kwargs: Any
) -> Path:
    """Write ``model`` as a GNN specification at ``path`` and return the path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(to_gnn_spec(model, **kwargs), encoding="utf-8")
    return target


def _as_flat_pomdp(model: FlatPOMDP | Any, **kwargs: Any) -> FlatPOMDP:
    if isinstance(model, FlatPOMDP):
        if kwargs:
            model = FlatPOMDP(**{**model.__dict__, **kwargs})
        return model
    if hasattr(model, "A") and hasattr(model, "B"):
        return FlatPOMDP.from_agent(model, **kwargs)
    if isinstance(model, Mapping):
        return FlatPOMDP.from_model(**model, **kwargs)
    raise GNNBridgeError(
        "Expected a FlatPOMDP, a pymdp Agent, or a mapping with A/B/C/D keys; "
        f"got {type(model).__name__}"
    )


@dataclass(frozen=True)
class PinnedGNNSource:
    """An exported, immutable copy of a GNN source tree."""

    path: Path
    checkout: Path
    revision: str
    revision_requested: str


def export_pinned_source(
    gnn_checkout: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    revision: str = GNN_PINNED_REVISION,
    subdirectory: str = "src",
) -> PinnedGNNSource:
    """Export ``revision`` of ``gnn_checkout`` into ``destination``.

    Uses ``git archive`` so the exported tree is exactly the committed revision,
    independent of any uncommitted work in the checkout. Returns the exported
    directory (the one containing the ``gnn`` package) plus the resolved commit.
    """
    checkout = Path(gnn_checkout).resolve()
    target = Path(destination).resolve()
    resolved = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "--verify", f"{revision}^{{commit}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if resolved.returncode != 0:
        raise GNNBridgeError(
            f"Cannot resolve {revision!r} in {checkout}: {resolved.stderr.strip()}"
        )
    sha = resolved.stdout.strip()
    archive = subprocess.run(
        ["git", "-C", str(checkout), "archive", sha, subdirectory],
        capture_output=True,
        check=False,
    )
    if archive.returncode != 0:
        raise GNNBridgeError(
            f"git archive failed for {sha[:8]}: {archive.stderr.decode().strip()}"
        )
    target.mkdir(parents=True, exist_ok=True)
    _extract_tar(archive.stdout, target)
    source = target / subdirectory
    if not (source / "gnn" / "parsers" / "markdown_parser.py").is_file():
        raise GNNBridgeError(
            f"Exported tree does not contain gnn/parsers/markdown_parser.py under {source}"
        )
    return PinnedGNNSource(
        path=source, checkout=checkout, revision=sha, revision_requested=revision
    )


def _extract_tar(payload: bytes, destination: Path) -> None:
    """Extract a tar payload, refusing members that escape ``destination``."""
    root = destination.resolve()
    with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
        for member in archive.getmembers():
            member_path = (root / member.name).resolve()
            if root != member_path and root not in member_path.parents:
                raise GNNBridgeError(f"Refusing tar member outside destination: {member.name}")
        archive.extractall(root)


@dataclass(frozen=True)
class GNNValidationReport:
    """Outcome of parsing an emitted specification with the real GNN parser."""

    ok: bool
    parser_source: str
    parser_revision: str | None
    model_name: str | None
    version: str | None
    sections: tuple[str, ...]
    parsed_shapes: Mapping[str, tuple[int, ...]]
    mismatches: tuple[str, ...]
    parser_errors: tuple[str, ...]
    parser_warnings: tuple[str, ...]
    stdout: str = ""
    stderr: str = ""

    def describe(self) -> str:
        """Human-readable summary suitable for logs or assertions."""
        lines = [
            f"GNN validation: {'ok' if self.ok else 'FAILED'}",
            f"parser source: {self.parser_source}",
            f"parser revision: {self.parser_revision or 'unrecorded'}",
            f"model name: {self.model_name}",
            f"sections: {', '.join(self.sections)}",
            "parsed shapes: "
            + ", ".join(f"{key}{tuple(shape)}" for key, shape in self.parsed_shapes.items()),
        ]
        for label, entries in (
            ("mismatches", self.mismatches),
            ("parser errors", self.parser_errors),
            ("parser warnings", self.parser_warnings),
        ):
            lines.extend(f"{label}: {entry}" for entry in entries)
        if self.stderr.strip():
            lines.append(f"stderr: {self.stderr.strip()}")
        return "\n".join(lines)


def validate_gnn_spec(
    model: FlatPOMDP | Any,
    spec: str | os.PathLike[str],
    *,
    gnn_source_root: str | os.PathLike[str] | None = None,
    python: str | os.PathLike[str] | None = None,
    revision: str | None = None,
    timeout: float = 300.0,
    **model_kwargs: Any,
) -> GNNValidationReport:
    """Parse ``spec`` with the real GNN parser and compare it to ``model``.

    Parameters
    ----------
    model:
        The source :class:`FlatPOMDP` (or pymdp ``Agent``) the specification
        was emitted from.
    spec:
        Specification text or a path to a ``.gnn.md`` file.
    gnn_source_root:
        Directory containing the ``gnn`` package (for example ``<export>/src``).
        Defaults to the ``PYMDP_GNN_ROOT`` environment variable.
    python:
        Interpreter used for the isolated parser subprocess. Defaults to the
        ``PYMDP_GNN_PYTHON`` environment variable, then to the running
        interpreter. That interpreter needs the GNN parser's runtime
        dependencies (``numpy``, ``defusedxml``; the ``gnn`` extra of pymdp).
    revision:
        Optional revision label recorded in the report.
    timeout:
        Seconds to wait for the parser subprocess.

    Returns
    -------
    GNNValidationReport
        Structured outcome. A failed import, parser error or value mismatch is
        reported explicitly; nothing is silently skipped.
    """
    flat = _as_flat_pomdp(model, **model_kwargs)

    if gnn_source_root is None:
        gnn_source_root = os.environ.get(GNN_SOURCE_ENV)
    if gnn_source_root is None:
        raise GNNBridgeError(
            "No GNN source tree given. Pass gnn_source_root=... or set "
            f"{GNN_SOURCE_ENV} to a directory containing the gnn package "
            "(see pymdp.gnn.export_pinned_source)."
        )
    source_root = Path(gnn_source_root).resolve()
    if not (source_root / "gnn" / "parsers" / "markdown_parser.py").is_file():
        raise GNNBridgeError(
            f"{source_root} does not contain gnn/parsers/markdown_parser.py"
        )

    if python is None:
        python = os.environ.get(GNN_PYTHON_ENV, sys.executable)
    interpreter = str(python)

    with tempfile.TemporaryDirectory(prefix="pymdp-gnn-validate-") as workspace:
        spec_path = Path(workspace) / f"spec{GNN_SPEC_SUFFIX}"
        if isinstance(spec, Path) or (
            isinstance(spec, str) and "\n" not in spec and Path(spec).is_file()
        ):
            spec_path.write_text(Path(spec).read_text(encoding="utf-8"), encoding="utf-8")
        else:
            spec_path.write_text(str(spec), encoding="utf-8")

        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        completed = subprocess.run(
            [interpreter, "-c", _PARSER_RUNNER, str(source_root), str(spec_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=environment,
            check=False,
        )

    payload: dict[str, Any] | None = None
    for line in completed.stdout.splitlines():
        if line.startswith(_RESULT_MARKER):
            payload = json.loads(line[len(_RESULT_MARKER) :])
    if payload is None:
        raise GNNBridgeError(
            "The GNN parser subprocess produced no result. "
            f"exit={completed.returncode}; stderr={completed.stderr.strip()!r}"
        )

    parser_errors: list[str] = []
    if not payload.get("ok", False):
        for key in ("import_error", "runner_error"):
            if payload.get(key):
                parser_errors.append(f"{key}: {payload[key]}")
    parser_errors.extend(str(error) for error in payload.get("errors", ()))

    mismatches: list[str] = []
    parsed_shapes: dict[str, tuple[int, ...]] = {}

    if payload.get("ok", False):
        parameters = {
            str(entry["name"]).strip(): entry["value"]
            for entry in payload.get("parameters", ())
            if entry.get("name")
        }
        expected = {
            "A": flat.A,
            "B": flat.B,
            "C": flat.C,
            "D": flat.D,
        }
        if flat.E is not None:
            expected["E"] = flat.E
        for key, array in expected.items():
            if key not in parameters:
                mismatches.append(f"{key} is missing from the parsed parameters")
                continue
            parsed = np.asarray(parameters[key], dtype=np.float64).reshape(-1)
            source = np.asarray(array, dtype=np.float64).reshape(-1)
            parsed_shapes[key] = tuple(int(dim) for dim in np.shape(parameters[key]))
            if parsed.shape != source.shape:
                mismatches.append(
                    f"{key} parses to {parsed.shape} values; source has {source.shape}"
                )
            elif not np.array_equal(parsed, source):
                worst = float(np.max(np.abs(parsed - source)))
                mismatches.append(f"{key} values differ (max |delta| = {worst})")

        declared = {
            str(variable["name"]): tuple(int(dim) for dim in variable["dimensions"])
            for variable in payload.get("variables", ())
        }
        for key, array in expected.items():
            shape = tuple(int(dim) for dim in np.shape(array))
            if key not in declared:
                mismatches.append(f"{key} is not declared in StateSpaceBlock")
            elif declared[key] != shape:
                mismatches.append(
                    f"{key} is declared {declared[key]} but has shape {shape}"
                )

        model_parameters = {
            key: parameters.get(key)
            for key in (
                "num_hidden_states",
                "num_obs",
                "num_actions",
                "num_timesteps",
                "b_tensor_order",
            )
        }
        expected_parameters: dict[str, Any] = {
            "num_hidden_states": flat.num_states,
            "num_obs": flat.num_obs,
            "num_actions": flat.num_actions,
            "num_timesteps": flat.num_timesteps,
            "b_tensor_order": GNN_B_ORDER,
        }
        for key, value in expected_parameters.items():
            if model_parameters.get(key) != value:
                mismatches.append(
                    f"ModelParameters[{key}] = {model_parameters.get(key)!r}; "
                    f"expected {value!r}"
                )

        missing_sections = [
            section for section in _SECTIONS if section not in payload.get("sections", ())
        ]
        if missing_sections:
            mismatches.append(f"missing sections: {', '.join(missing_sections)}")

        if not payload.get("success", False):
            mismatches.append("the parser reported success=False")

    ok = bool(payload.get("ok", False)) and not mismatches and not parser_errors
    return GNNValidationReport(
        ok=ok,
        parser_source=str(source_root),
        parser_revision=revision,
        model_name=payload.get("model_name"),
        version=payload.get("version"),
        sections=tuple(payload.get("sections", ())),
        parsed_shapes=parsed_shapes,
        mismatches=tuple(mismatches),
        parser_errors=tuple(parser_errors),
        parser_warnings=tuple(str(warning) for warning in payload.get("warnings", ())),
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
