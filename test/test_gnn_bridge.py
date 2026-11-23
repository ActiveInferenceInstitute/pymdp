#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the pymdp -> GNN specification bridge (`pymdp.gnn`).

The pinned-parser and executor round trips are opt-in: they run only when
``PYMDP_GNN_ROOT`` names a directory containing the ``gnn`` package (an
exported pinned tree; see ``pymdp.gnn.export_pinned_source``), and the ``gnn``
extra is installed (``uv sync --group test --extra gnn``). Without that they
skip cleanly instead of silently passing.
"""

import io
import json
import os
import pathlib
import subprocess
import sys
import tarfile
import tempfile
import unittest
import unittest.mock

import numpy as np
from jax import numpy as jnp
from jax import random as jr

from pymdp import gnn, utils
from pymdp.agent import Agent
from pymdp.legacy.agent import Agent as LegacyAgent

GNN_SOURCE_ROOT = os.environ.get(gnn.GNN_SOURCE_ENV)
GNN_CHECKOUT = os.environ.get("PYMDP_GNN_CHECKOUT")
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


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

    def test_singleton_batched_passive_transition_is_promoted(self):
        for list_wrapped in (False, True):
            for batched_likelihood in (False, True):
                with self.subTest(
                    list_wrapped=list_wrapped,
                    batched_likelihood=batched_likelihood,
                ):
                    arrays = {
                        "A": np.eye(2)[None, ...] if batched_likelihood else np.eye(2),
                        "B": np.eye(2)[None, ...],
                        "C": np.zeros(2),
                        "D": np.ones(2) / 2,
                    }
                    if list_wrapped:
                        arrays = {key: [value] for key, value in arrays.items()}
                    model = gnn.FlatPOMDP.from_model(**arrays)
                    self.assertEqual(model.B.shape, (2, 2, 1))
                    self.assertEqual(model.num_actions, 1)
                    np.testing.assert_array_equal(model.B[:, :, 0], np.eye(2))
                    self.assertIn("B[2,2,1,type=float]", model.to_gnn_spec())

    def test_one_state_controlled_transition_retains_both_actions(self):
        for batched in (False, True):
            with self.subTest(batched=batched):
                B = np.ones((1, 1, 2))
                model = gnn.FlatPOMDP.from_model(
                    A=np.array([[0.5], [0.5]]),
                    B=B[None, ...] if batched else B,
                    C=np.zeros(2),
                    D=np.ones(1),
                    E=np.array([0.2, 0.8]),
                )
                self.assertEqual(model.B.shape, (1, 1, 2))
                self.assertEqual(model.num_actions, 2)
                np.testing.assert_array_equal(model.B, B)
                self.assertIn("B[1,1,2,type=float]", model.to_gnn_spec())

    def test_from_agent_matches_agent_arrays(self):
        agent = build_agent()
        model = gnn.FlatPOMDP.from_agent(agent, name="agent_model")
        np.testing.assert_array_equal(model.A, np.asarray(agent.A[0])[0])
        np.testing.assert_array_equal(model.B, np.asarray(agent.B[0])[0])
        np.testing.assert_array_equal(model.C, np.asarray(agent.C[0])[0])
        np.testing.assert_array_equal(model.D, np.asarray(agent.D[0])[0])
        np.testing.assert_array_equal(model.E, np.asarray(agent.E)[0])

    def test_flat_legacy_agent_object_arrays_remain_publishable(self):
        agent = LegacyAgent(
            A=np.eye(2),
            B=np.stack([np.eye(2), np.eye(2)[:, ::-1]], axis=2),
            C=np.array([0.0, 1.0]),
            D=np.array([0.6, 0.4]),
            E=np.array([0.7, 0.3]),
        )
        model = gnn.FlatPOMDP.from_agent(agent, name="legacy_agent")
        for key in ("A", "B", "C", "D"):
            with self.subTest(parameter=key):
                self.assertEqual(getattr(agent, key).dtype, object)
                np.testing.assert_array_equal(
                    getattr(model, key), getattr(agent, key)[0]
                )
        np.testing.assert_array_equal(model.E, agent.E)
        expected = model.to_gnn_spec()
        self.assertEqual(gnn.to_gnn_spec(agent, name="legacy_agent"), expected)
        with tempfile.TemporaryDirectory() as workspace:
            target = pathlib.Path(workspace) / f"legacy{gnn.GNN_SPEC_SUFFIX}"
            written = gnn.write_gnn_spec(target, agent, name="legacy_agent")
            self.assertEqual(written.read_text(encoding="utf-8"), expected)

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

    def test_from_agent_exports_habit_prior_in_action_order(self):
        for agent_class, asarray in ((Agent, jnp.asarray), (LegacyAgent, np.asarray)):
            with self.subTest(agent=agent_class.__name__):
                A = asarray(np.eye(2))
                B = asarray(np.stack([np.eye(2), np.eye(2)[:, ::-1]], axis=2))
                agent = agent_class(
                    A=[A] if agent_class is Agent else A,
                    B=[B] if agent_class is Agent else B,
                    E=asarray([0.9, 0.1]),
                    policies=asarray([[[1]], [[0]]]),
                )
                model = gnn.FlatPOMDP.from_agent(agent)
                np.testing.assert_allclose(model.E, [0.1, 0.9])

    def test_from_agent_refuses_policies_that_do_not_map_one_to_one_to_actions(self):
        for policies, reason in (
            ([[[0], [1]], [[1], [0]]], "one-step"),
            ([[[0]], [[0]]], "one policy per action"),
            ([[[0]]], "one policy per action"),
        ):
            with self.subTest(policies=policies):
                agent = Agent(
                    A=[jnp.eye(2)],
                    B=[jnp.stack([jnp.eye(2), jnp.eye(2)[:, ::-1]], axis=2)],
                    policies=jnp.asarray(policies),
                )
                with self.assertRaisesRegex(gnn.GNNBridgeError, reason):
                    gnn.FlatPOMDP.from_agent(agent)

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


class TestParsedParameterValidation(unittest.TestCase):
    """Payload-comparison unit tests, not evidence about the real parser."""

    def setUp(self):
        self.model = gnn.FlatPOMDP.from_model(
            A=np.array([[0.8, 0.3], [0.2, 0.7]]),
            B=np.array(
                [
                    [[0.9, 0.1], [0.2, 0.8]],
                    [[0.1, 0.9], [0.8, 0.2]],
                ]
            ),
            C=np.array([-1.0, 1.0]),
            D=np.array([0.6, 0.4]),
            E=np.array([0.7, 0.3]),
        )

    def _validate_payload(self, overrides=None):
        arrays = {
            key: getattr(self.model, key)
            for key in ("A", "B", "C", "D", "E")
        }
        parameters = {
            key: (array[None, ...] if array.ndim == 1 else array).tolist()
            for key, array in arrays.items()
        }
        parameters.update(overrides or {})
        parameters.update(
            num_hidden_states=self.model.num_states,
            num_obs=self.model.num_obs,
            num_actions=self.model.num_actions,
            num_timesteps=self.model.num_timesteps,
            b_tensor_order=gnn.GNN_B_ORDER,
        )
        payload = {
            "ok": True,
            "success": True,
            "model_name": self.model.name,
            "sections": list(gnn._SECTIONS),
            "parameters": [
                {"name": key, "value": value}
                for key, value in parameters.items()
            ],
            "variables": [
                {"name": key, "dimensions": list(array.shape)}
                for key, array in arrays.items()
            ],
        }
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=gnn._RESULT_MARKER + json.dumps(payload) + "\n",
            stderr="",
        )
        with tempfile.TemporaryDirectory() as workspace:
            source = pathlib.Path(workspace)
            parser = source / "gnn" / "parsers" / "markdown_parser.py"
            parser.parent.mkdir(parents=True)
            parser.write_text("", encoding="utf-8")
            with unittest.mock.patch.object(
                gnn.subprocess, "run", return_value=completed
            ) as runner:
                report = gnn.validate_gnn_spec(
                    self.model,
                    self.model.to_gnn_spec(),
                    gnn_source_root=source,
                )
            runner.assert_called_once()
        return report

    def test_actual_tensor_shapes_must_match_despite_equal_flat_values(self):
        for key, shape in (("A", (1, 4)), ("B", (1, 2, 4))):
            with self.subTest(parameter=key):
                source = getattr(self.model, key)
                repacked = source.reshape(shape)
                np.testing.assert_array_equal(repacked.ravel(), source.ravel())
                report = self._validate_payload({key: repacked.tolist()})
                self.assertFalse(report.ok, report.describe())
                self.assertEqual(report.parsed_shapes[key], shape)
                self.assertTrue(
                    any(
                        mismatch.startswith(f"{key} parses to shape {shape}")
                        for mismatch in report.mismatches
                    ),
                    report.describe(),
                )

    def test_vector_row_wrappers_and_plain_vectors_are_accepted(self):
        for wrapped in (False, True):
            with self.subTest(wrapped=wrapped):
                overrides = {
                    key: (
                        getattr(self.model, key)[None, ...]
                        if wrapped
                        else getattr(self.model, key)
                    ).tolist()
                    for key in ("C", "D", "E")
                }
                report = self._validate_payload(overrides)
                self.assertTrue(report.ok, report.describe())
                for key in ("C", "D", "E"):
                    self.assertEqual(
                        report.parsed_shapes[key], (1, 2) if wrapped else (2,)
                    )


class TestPinnedSourceExport(unittest.TestCase):
    """Export admission/extraction unit tests; Git output is a byte fixture."""

    @staticmethod
    def _archive_payload():
        buffer = io.BytesIO()
        content = b"# committed parser bytes\n"
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            member = tarfile.TarInfo("src/gnn/parsers/markdown_parser.py")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
        return buffer.getvalue(), content

    def test_export_accepts_absent_and_empty_destinations(self):
        archive, content = self._archive_payload()
        with tempfile.TemporaryDirectory() as workspace:
            workspace = pathlib.Path(workspace).resolve()
            for existing in (False, True):
                with self.subTest(existing=existing):
                    target = workspace / ("empty" if existing else "absent")
                    if existing:
                        target.mkdir()
                    responses = [
                        subprocess.CompletedProcess(
                            args=[],
                            returncode=0,
                            stdout=gnn.GNN_PINNED_REVISION + "\n",
                            stderr="",
                        ),
                        subprocess.CompletedProcess(
                            args=[], returncode=0, stdout=archive, stderr=b""
                        ),
                    ]
                    with unittest.mock.patch.object(
                        gnn.subprocess, "run", side_effect=responses
                    ) as git:
                        exported = gnn.export_pinned_source(workspace, target)
                    self.assertEqual(git.call_count, 2)
                    self.assertEqual(exported.path, target / "src")
                    self.assertEqual(exported.revision, gnn.GNN_PINNED_REVISION)
                    self.assertEqual(
                        (exported.path / "gnn/parsers/markdown_parser.py").read_bytes(),
                        content,
                    )
                    self.assertEqual(
                        [path.relative_to(target).as_posix() for path in target.rglob("*")
                         if path.is_file()],
                        ["src/gnn/parsers/markdown_parser.py"],
                    )

    def test_nonempty_destination_preserves_sentinel_and_stale_parser(self):
        with tempfile.TemporaryDirectory() as workspace:
            target = pathlib.Path(workspace) / "old-export"
            stale = target / "src" / "gnn" / "stale_parser.py"
            stale.parent.mkdir(parents=True)
            stale.write_bytes(b"old parser module must remain untouched\n")
            sentinel = target / "sentinel"
            sentinel.write_bytes(b"preserve old cache\n")
            with unittest.mock.patch.object(gnn.subprocess, "run") as git:
                with self.assertRaisesRegex(gnn.GNNBridgeError, "not empty"):
                    gnn.export_pinned_source(workspace, target)
            git.assert_not_called()
            self.assertEqual(stale.read_bytes(), b"old parser module must remain untouched\n")
            self.assertEqual(sentinel.read_bytes(), b"preserve old cache\n")
            self.assertEqual(
                {path.relative_to(target).as_posix() for path in target.rglob("*")
                 if path.is_file()},
                {"sentinel", "src/gnn/stale_parser.py"},
            )

    def test_files_and_symlinks_are_incompatible_destinations(self):
        with tempfile.TemporaryDirectory() as workspace:
            workspace = pathlib.Path(workspace)
            file = workspace / "file"
            file.write_bytes(b"preserve file\n")
            empty = workspace / "empty"
            empty.mkdir()
            link = workspace / "link"
            link.symlink_to(empty, target_is_directory=True)
            broken = workspace / "broken"
            broken.symlink_to(workspace / "absent", target_is_directory=True)
            for target in (file, link, broken):
                with self.subTest(target=target.name):
                    with unittest.mock.patch.object(gnn.subprocess, "run") as git:
                        with self.assertRaisesRegex(
                            gnn.GNNBridgeError, "absent or empty directory"
                        ):
                            gnn.export_pinned_source(workspace, target)
                    git.assert_not_called()
            self.assertEqual(file.read_bytes(), b"preserve file\n")
            self.assertEqual(list(empty.iterdir()), [])
            self.assertTrue(link.is_symlink())
            self.assertTrue(broken.is_symlink())


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

    def test_repacked_tensor_shapes_fail_with_matching_declarations_and_values(self):
        original = self.spec_path.read_text(encoding="utf-8")
        for key, shape, formatter in (
            ("A", (1, 12), gnn._format_matrix),
            ("B", (1, 3, 6), gnn._format_tensor),
        ):
            with self.subTest(parameter=key):
                source = getattr(self.model, key)
                repacked = source.reshape(shape)
                np.testing.assert_array_equal(repacked.ravel(), source.ravel())
                old = f"{key}={formatter(source)}"
                new = f"{key}={formatter(repacked)}"
                self.assertIn(old, original)
                tampered = original.replace(old, new, 1)
                declaration = f"{key}[{','.join(map(str, source.shape))},type=float]"
                self.assertIn(declaration, tampered)
                report = gnn.validate_gnn_spec(
                    self.model,
                    tampered,
                    gnn_source_root=GNN_SOURCE_ROOT,
                    revision=gnn.GNN_PINNED_REVISION,
                )
                self.assertFalse(report.ok, report.describe())
                self.assertEqual(report.parsed_shapes[key], shape)
                self.assertTrue(
                    any(
                        mismatch.startswith(f"{key} parses to shape {shape}")
                        for mismatch in report.mismatches
                    ),
                    report.describe(),
                )

    def test_passive_and_one_state_controlled_shapes_round_trip(self):
        models = (
            gnn.FlatPOMDP.from_model(
                A=np.eye(2), B=np.eye(2)[None, ...],
                C=np.zeros(2), D=np.ones(2) / 2,
            ),
            gnn.FlatPOMDP.from_model(
                A=np.array([[0.5], [0.5]]), B=np.ones((1, 1, 2)),
                C=np.zeros(2), D=np.ones(1),
            ),
        )
        for model in models:
            with self.subTest(shape=model.B.shape):
                report = gnn.validate_gnn_spec(
                    model, model.to_gnn_spec(), gnn_source_root=GNN_SOURCE_ROOT,
                    revision=gnn.GNN_PINNED_REVISION,
                )
                self.assertTrue(report.ok, report.describe())
                self.assertEqual(report.parsed_shapes["B"], model.B.shape)


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
            second_target = pathlib.Path(workspace) / "second-export"
            second_target.mkdir()
            second = gnn.export_pinned_source(GNN_CHECKOUT, second_target)
            self.assertEqual(second.revision, gnn.GNN_PINNED_REVISION)
            self.assertNotEqual(exported.path, second.path)
            first_files = {
                path.relative_to(exported.path): path.read_bytes()
                for path in exported.path.rglob("*") if path.is_file()
            }
            second_files = {
                path.relative_to(second.path): path.read_bytes()
                for path in second.path.rglob("*") if path.is_file()
            }
            self.assertEqual(first_files, second_files)
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
        GNN_CHECKOUT, "PYMDP_GNN_CHECKOUT is not set; skipping pinned-tree export test"
    )
    def test_export_refuses_reusing_materialized_tree_without_changing_bytes(self):
        with tempfile.TemporaryDirectory() as workspace:
            target = pathlib.Path(workspace) / "export"
            exported = gnn.export_pinned_source(GNN_CHECKOUT, target)
            sentinel = exported.path / "gnn" / "stale_parser.py"
            sentinel.write_bytes(b"stale module must not be silently retained or removed\n")
            before = {
                path.relative_to(target): path.read_bytes()
                for path in target.rglob("*") if path.is_file()
            }
            with self.assertRaisesRegex(gnn.GNNBridgeError, "not empty"):
                gnn.export_pinned_source(GNN_CHECKOUT, target)
            after = {
                path.relative_to(target): path.read_bytes()
                for path in target.rglob("*") if path.is_file()
            }
            self.assertEqual(before, after)


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


@unittest.skipUnless(
    GNN_SOURCE_ROOT, f"{gnn.GNN_SOURCE_ENV} is not set; skipping pin reader test"
)
class TestPinRecord(unittest.TestCase):
    """The pin is read by the shipped reader, never by a copy kept here."""

    def test_pin_agrees_with_the_bridge_constants(self):
        source_root = str(pathlib.Path(GNN_SOURCE_ROOT).resolve())
        sys.path.insert(0, source_root)
        try:
            from gnn.importers import PIN_SCHEMA, UPSTREAM_REPOSITORY, read_pin
        finally:
            sys.path.remove(source_root)

        pin = read_pin(REPO_ROOT / ".github" / "gnn-pin.json")
        self.assertEqual(pin.schema, PIN_SCHEMA)
        self.assertEqual(pin.repository, UPSTREAM_REPOSITORY)
        self.assertEqual(pin.commit, gnn.GNN_PINNED_REVISION)
        self.assertEqual(pin.tag, gnn.GNN_PINNED_RELEASE)


if __name__ == "__main__":
    unittest.main()
