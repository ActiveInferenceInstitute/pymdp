#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the pymdp -> GNN specification bridge (`pymdp.gnn`).

The pinned-parser and executor round trips are opt-in: they run only when
``PYMDP_GNN_ROOT`` names a directory containing the ``gnn`` package (an
exported pinned tree; see ``pymdp.gnn.export_pinned_source``), and the ``gnn``
extra is installed (``uv sync --group test --extra gnn``). Without that they
skip cleanly instead of silently passing.
"""

import os
import pathlib
import sys
import tempfile
import unittest
import unittest.mock

import numpy as np
from jax import numpy as jnp
from jax import random as jr

from pymdp import gnn, utils
from pymdp.agent import Agent

GNN_SOURCE_ROOT = os.environ.get(gnn.GNN_SOURCE_ENV)
GNN_CHECKOUT = os.environ.get("PYMDP_GNN_CHECKOUT")


def build_agent(num_obs=3, num_states=3, num_actions=3, seed=0, num_timesteps=4):
    """Build a flat (single-modality, single-factor) seeded pymdp agent."""
    key = jr.PRNGKey(seed)
    k1, k2 = jr.split(key, 2)
    A = utils.random_A_array(k1, [num_obs], [num_states])
    B = utils.random_B_array(k2, [num_states], [num_actions])
    C = utils.list_array_uniform([[num_obs]])
    D = utils.list_array_uniform([[num_states]])
    return Agent(A=A, B=B, C=C, D=D, batch_size=1)


class TestEmission(unittest.TestCase):
    def test_spec_contains_canonical_sections_and_declarations(self):
        model = gnn.FlatPOMDP.from_agent(
            build_agent(), name="emission_check", num_timesteps=7
        )
        spec = model.to_gnn_spec()

        for section in gnn._SECTIONS:
            self.assertIn(f"## {section}", spec)
        for declaration in (
            "A[3,3,type=float]",
            "B[3,3,3,type=float]",
            "C[3,type=float]",
            "D[3,type=float]",
            "E[3,type=float]",
            "s[3,1,type=float]",
            "o[3,1,type=int]",
            "π[3,type=float]",
            "u[1,type=int]",
            "t[1,type=int]",
        ):
            self.assertIn(declaration, spec)
        self.assertIn(f"b_tensor_order: {gnn.GNN_B_ORDER}", spec)
        self.assertIn("ModelTimeHorizon=7", spec)
        self.assertIn("A=LikelihoodMatrix", spec)
        self.assertIn("B=TransitionMatrix", spec)
        self.assertIn("E=Habit", spec)
        self.assertIn("## ModelName\nemission_check", spec)
        self.assertIn(f"GNN {gnn.GNN_PINNED_RELEASE}", spec)

    def test_spec_omits_habit_prior_when_absent(self):
        model = gnn.FlatPOMDP.from_model(
            A=np.eye(2),
            B=np.stack([np.eye(2), np.eye(2)[:, ::-1]], axis=2),
            C=np.zeros(2),
            D=np.ones(2) / 2,
            name="no_habit",
        )
        spec = model.to_gnn_spec()
        self.assertNotIn("E[", spec)
        self.assertNotIn("E>π", spec)
        self.assertNotIn("E=Habit", spec)
        self.assertNotIn("\nE={", spec)

    def test_hand_checkable_matrices_are_exact(self):
        A = np.array([[1.0, 0.0], [0.0, 1.0]])
        B = np.array(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.0, 1.0], [1.0, 0.0]],
            ]
        )  # B[next, prev, action]
        model = gnn.FlatPOMDP.from_model(
            A=A, B=B, C=np.array([0.0, 1.0]), D=np.array([0.5, 0.5]), name="exact"
        )
        spec = model.to_gnn_spec()
        self.assertIn("A={\n  (1.0, 0.0),\n  (0.0, 1.0)\n}", spec)
        self.assertIn(
            "B={\n  ( (1.0, 0.0), (0.0, 1.0) ),\n  ( (0.0, 1.0), (1.0, 0.0) )\n}", spec
        )
        self.assertIn("C={(0.0, 1.0)}", spec)
        self.assertIn("D={(0.5, 0.5)}", spec)

    def test_single_action_transition_is_promoted(self):
        model = gnn.FlatPOMDP.from_model(
            A=np.array([[0.9, 0.1], [0.1, 0.9]]),
            B=np.array([[1.0, 0.0], [0.0, 1.0]]),
            C=np.zeros(2),
            D=np.ones(2) / 2,
            name="passive",
        )
        self.assertEqual(model.num_actions, 1)
        self.assertIn("B[2,2,1,type=float]", model.to_gnn_spec())
        self.assertIn("π[1,type=float]", model.to_gnn_spec())

    def test_from_agent_matches_agent_arrays(self):
        agent = build_agent()
        model = gnn.FlatPOMDP.from_agent(agent, name="agent_model")
        np.testing.assert_array_equal(model.A, np.asarray(agent.A[0])[0])
        np.testing.assert_array_equal(model.B, np.asarray(agent.B[0])[0])
        np.testing.assert_array_equal(model.C, np.asarray(agent.C[0])[0])
        np.testing.assert_array_equal(model.D, np.asarray(agent.D[0])[0])
        np.testing.assert_array_equal(model.E, np.asarray(agent.E)[0])

    def test_write_gnn_spec_creates_parent_and_file(self):
        model = gnn.FlatPOMDP.from_agent(build_agent(), name="written")
        with tempfile.TemporaryDirectory() as workspace:
            target = pathlib.Path(workspace) / "nested" / f"model{gnn.GNN_SPEC_SUFFIX}"
            written = gnn.write_gnn_spec(target, model)
            self.assertEqual(written, target)
            self.assertEqual(target.read_text(encoding="utf-8"), model.to_gnn_spec())

    def test_module_level_to_gnn_spec_accepts_agent_and_mapping(self):
        agent = build_agent()
        from_agent = gnn.to_gnn_spec(agent, name="via_agent")
        from_mapping = gnn.to_gnn_spec(
            {
                "A": np.asarray(agent.A[0])[0],
                "B": np.asarray(agent.B[0])[0],
                "C": np.asarray(agent.C[0])[0],
                "D": np.asarray(agent.D[0])[0],
                "name": "via_mapping",
            }
        )
        self.assertIn("via_agent", from_agent)
        self.assertIn("via_mapping", from_mapping)


class TestRefusals(unittest.TestCase):
    def test_multi_modality_agent_is_refused(self):
        key = jr.PRNGKey(3)
        k1, k2 = jr.split(key, 2)
        num_obs = [2, 3]
        num_states = [3]
        A = utils.random_A_array(k1, num_obs, num_states, A_dependencies=[[0], [0]])
        B = utils.random_B_array(k2, num_states, [3])
        agent = Agent(A=A, B=B, batch_size=1)
        with self.assertRaises(gnn.GNNBridgeError) as context:
            gnn.FlatPOMDP.from_agent(agent)
        self.assertIn("num_modalities=2", str(context.exception))

    def test_multi_factor_agent_is_refused(self):
        key = jr.PRNGKey(4)
        k1, k2 = jr.split(key, 2)
        num_obs = [4]
        num_states = [2, 3]
        A = utils.random_A_array(k1, num_obs, num_states, A_dependencies=[[0, 1]])
        B = utils.random_B_array(k2, num_states, [2, 3])
        agent = Agent(A=A, B=B, batch_size=1)
        with self.assertRaises(gnn.GNNBridgeError) as context:
            gnn.FlatPOMDP.from_agent(agent)
        self.assertIn("num_factors=2", str(context.exception))

    def test_multi_step_policy_prior_is_refused(self):
        agent = build_agent()
        model = gnn.FlatPOMDP.from_agent(agent)
        with self.assertRaises(gnn.GNNBridgeError) as context:
            gnn.FlatPOMDP(
                A=model.A,
                B=model.B,
                C=model.C,
                D=model.D,
                E=np.ones(9) / 9,
            )
        self.assertIn("habit prior", str(context.exception))

    def test_multi_entry_lists_are_refused(self):
        with self.assertRaises(gnn.GNNBridgeError) as context:
            gnn.FlatPOMDP.from_model(
                A=[np.eye(2), np.eye(2)],
                B=np.eye(2),
                C=np.zeros(2),
                D=np.ones(2) / 2,
            )
        self.assertIn("exactly one observation modality", str(context.exception))

    def test_inconsistent_shapes_are_refused(self):
        with self.assertRaises(gnn.GNNBridgeError):
            gnn.FlatPOMDP.from_model(
                A=np.eye(3), B=np.eye(2), C=np.zeros(3), D=np.ones(3) / 3
            )
        with self.assertRaises(gnn.GNNBridgeError):
            gnn.FlatPOMDP.from_model(
                A=np.eye(3), B=np.eye(3), C=np.zeros(2), D=np.ones(3) / 3
            )
        with self.assertRaises(gnn.GNNBridgeError):
            gnn.FlatPOMDP.from_model(
                A=np.eye(3), B=np.eye(3), C=np.zeros(3), D=np.ones(3) / 3, num_timesteps=0
            )

    def test_validation_requires_a_gnn_source(self):
        model = gnn.FlatPOMDP.from_agent(build_agent())
        with unittest.mock.patch.dict(os.environ):
            os.environ.pop(gnn.GNN_SOURCE_ENV, None)
            with self.assertRaises(gnn.GNNBridgeError) as context:
                gnn.validate_gnn_spec(model, model.to_gnn_spec(), gnn_source_root=None)
        self.assertIn(gnn.GNN_SOURCE_ENV, str(context.exception))


@unittest.skipUnless(
    GNN_SOURCE_ROOT, f"{gnn.GNN_SOURCE_ENV} is not set; skipping pinned-GNN tests"
)
class TestPinnedParserValidation(unittest.TestCase):
    """Round trips through the real GNN parser at the exported pinned tree."""

    def setUp(self):
        self.model = gnn.FlatPOMDP.from_agent(
            build_agent(num_obs=4, num_states=3, num_actions=2, seed=11),
            name="pinned_roundtrip",
            num_timesteps=6,
        )
        self.workspace = tempfile.TemporaryDirectory()
        self.spec_path = gnn.write_gnn_spec(
            pathlib.Path(self.workspace.name) / f"model{gnn.GNN_SPEC_SUFFIX}",
            self.model,
        )

    def tearDown(self):
        self.workspace.cleanup()

    def test_pinned_parser_round_trip(self):
        report = gnn.validate_gnn_spec(
            self.model,
            self.spec_path,
            gnn_source_root=GNN_SOURCE_ROOT,
            revision=gnn.GNN_PINNED_REVISION,
        )
        self.assertTrue(report.ok, report.describe())
        self.assertEqual(report.mismatches, ())
        self.assertEqual(report.parser_errors, ())
        self.assertEqual(report.model_name, "pinned_roundtrip")
        self.assertEqual(report.parsed_shapes["A"], (4, 3))
        self.assertEqual(report.parsed_shapes["B"], (3, 3, 2))
        self.assertEqual(report.parsed_shapes["C"], (1, 4))
        self.assertEqual(report.parsed_shapes["D"], (1, 3))
        self.assertEqual(report.parsed_shapes["E"], (1, 2))
        for section in gnn._SECTIONS:
            self.assertIn(section, report.sections)

    def test_validation_rejects_a_different_model(self):
        other = gnn.FlatPOMDP.from_agent(
            build_agent(num_obs=4, num_states=3, num_actions=2, seed=12),
            name="pinned_roundtrip",
        )
        report = gnn.validate_gnn_spec(
            other, self.spec_path, gnn_source_root=GNN_SOURCE_ROOT
        )
        self.assertFalse(report.ok)
        self.assertTrue(
            any("values differ" in mismatch for mismatch in report.mismatches),
            report.describe(),
        )

    def test_validation_detects_a_missing_section(self):
        tampered = self.spec_path.read_text(encoding="utf-8").replace(
            "## Connections", "## RemovedConnections"
        )
        report = gnn.validate_gnn_spec(
            self.model, tampered, gnn_source_root=GNN_SOURCE_ROOT
        )
        self.assertFalse(report.ok)
        self.assertTrue(
            any("Connections" in mismatch for mismatch in report.mismatches),
            report.describe(),
        )

    @unittest.skipUnless(
        GNN_CHECKOUT, "PYMDP_GNN_CHECKOUT is not set; skipping pinned-tree export test"
    )
    def test_export_pinned_source_is_reproducible(self):
        with tempfile.TemporaryDirectory() as workspace:
            exported = gnn.export_pinned_source(
                GNN_CHECKOUT, pathlib.Path(workspace) / "export"
            )
            self.assertEqual(exported.revision, gnn.GNN_PINNED_REVISION)
            self.assertTrue(
                (exported.path / "gnn" / "parsers" / "markdown_parser.py").is_file()
            )
            model = gnn.FlatPOMDP.from_agent(build_agent(), name="exported_source")
            report = gnn.validate_gnn_spec(
                model,
                model.to_gnn_spec(),
                gnn_source_root=exported.path,
                revision=exported.revision,
            )
            self.assertTrue(report.ok, report.describe())
            self.assertEqual(report.parser_revision, gnn.GNN_PINNED_REVISION)


@unittest.skipUnless(
    GNN_SOURCE_ROOT, f"{gnn.GNN_SOURCE_ENV} is not set; skipping GNN executor test"
)
class TestGNNExecutorRoundTrip(unittest.TestCase):
    """GNN renders and executes the emitted spec; beliefs must match pymdp."""

    def _beliefs_through_gnn(self, agent, model):
        """Render + execute the spec with GNN, returning its recorded rollout."""
        source_root = str(pathlib.Path(GNN_SOURCE_ROOT).resolve())
        sys.path.insert(0, source_root)
        try:
            from gnn.execute.pymdp.simulation import run_pymdp_simulation
            from gnn.render.pymdp.pymdp_renderer import (
                parse_gnn_markdown,
                render_gnn_to_pymdp,
            )

            with tempfile.TemporaryDirectory() as workspace:
                workspace = pathlib.Path(workspace)
                spec_path = gnn.write_gnn_spec(
                    workspace / f"model{gnn.GNN_SPEC_SUFFIX}", model
                )
                spec = parse_gnn_markdown(spec_path.read_text(encoding="utf-8"), spec_path)
                self.assertIsNotNone(spec, "the pinned parser rejected the emitted spec")
                rendered, message, _ = render_gnn_to_pymdp(
                    spec, workspace / "rendered_pymdp.py"
                )
                self.assertTrue(rendered, message)
                ok, results = run_pymdp_simulation(spec, workspace / "out")
                self.assertTrue(ok, results)
        finally:
            sys.path.remove(source_root)
        return results

    def test_beliefs_match_pymdp_inference(self):
        agent = build_agent(num_obs=4, num_states=3, num_actions=2, seed=21)
        model = gnn.FlatPOMDP.from_agent(agent, name="executor_roundtrip", num_timesteps=4)
        results = self._beliefs_through_gnn(agent, model)

        replay = build_agent(num_obs=4, num_states=3, num_actions=2, seed=21)
        prior = replay.D
        for step, (observation, action) in enumerate(
            zip(results["observations"], results["actions"])
        ):
            qs, _ = replay.infer_states(
                [jnp.array([observation], dtype=jnp.int32)],
                empirical_prior=prior,
                return_info=True,
            )
            native = np.asarray(qs[0][0, -1], dtype=np.float64)
            through_gnn = np.asarray(results["beliefs"][step], dtype=np.float64)
            np.testing.assert_allclose(
                through_gnn,
                native,
                atol=1e-6,
                err_msg=f"belief mismatch at step {step}",
            )
            prior = replay.update_empirical_prior(jnp.array([[action]]), qs)


if __name__ == "__main__":
    unittest.main()
