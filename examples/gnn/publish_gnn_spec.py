"""Publish a pymdp generative model as a Generalized Notation Notation (GNN) specification.

Run it with::

    uv run python examples/gnn/publish_gnn_spec.py
    uv run python examples/gnn/publish_gnn_spec.py --out-dir /tmp/gnn-publish

The script builds a small flat discrete POMDP, writes it as a ``.gnn.md``
specification, and — when ``PYMDP_GNN_ROOT`` points at an exported pinned GNN
tree — validates the written file with the real GNN parser. The export command
is printed when the variable is unset.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import jax.numpy as jnp

from pymdp import gnn
from pymdp.agent import Agent


def build_agent() -> Agent:
    """Build a three-state, three-action model with a habit prior."""
    # A[observation, hidden state]: each state emits its own observation most of the time.
    A = [jnp.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])]
    # B[next state, previous state, action]: each action moves to a fixed state.
    B = [
        jnp.array(
            [
                [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            ]
        )
    ]
    C = [jnp.zeros(3)]  # no preferences
    D = [jnp.ones(3) / 3]  # uniform prior over the initial state
    E = jnp.array([0.2, 0.3, 0.5])  # habit prior over the three actions
    return Agent(A=A, B=B, C=C, D=D, E=E, batch_size=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="directory for the emitted .gnn.md file (default: a fresh temporary directory)",
    )
    parser.add_argument("--name", default="example_three_state_pomdp")
    parser.add_argument("--timesteps", type=int, default=10)
    arguments = parser.parse_args()

    out_dir = arguments.out_dir or Path(tempfile.mkdtemp(prefix="pymdp-gnn-example-"))
    out_dir.mkdir(parents=True, exist_ok=True)

    agent = build_agent()
    model = gnn.FlatPOMDP.from_agent(
        agent, name=arguments.name, num_timesteps=arguments.timesteps
    )
    spec_path = gnn.write_gnn_spec(
        out_dir / f"{arguments.name}{gnn.GNN_SPEC_SUFFIX}", model
    )
    print(f"wrote {spec_path} ({spec_path.stat().st_size} bytes)")
    print("---- specification (first 24 lines) ----")
    print("\n".join(spec_path.read_text(encoding="utf-8").splitlines()[:24]))

    gnn_source_root = os.environ.get(gnn.GNN_SOURCE_ENV)
    if not gnn_source_root:
        print(
            f"\n{gnn.GNN_SOURCE_ENV} is not set; skipping parser validation.\n"
            "Export the pinned tree and retry, for example:\n"
            "  git -C <gnn-checkout> archive "
            f"{gnn.GNN_PINNED_REVISION} src | tar -x -C /tmp/gnn-pin\n"
            f"  {gnn.GNN_SOURCE_ENV}=/tmp/gnn-pin/src "
            "uv run python examples/gnn/publish_gnn_spec.py"
        )
        return 0

    report = gnn.validate_gnn_spec(
        model, spec_path, revision=gnn.GNN_PINNED_REVISION
    )
    print("\n---- validation ----")
    print(report.describe())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
