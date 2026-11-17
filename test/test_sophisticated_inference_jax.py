#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Unit tests for sophisticated inference planning (JAX)."""

import unittest

import numpy as np
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

from pymdp import control, utils
from pymdp.agent import Agent
from pymdp.planning.si import si_policy_search


def _build_single_cue_model():
    num_locations = 4  # 0: start, 1: left, 2: right, 3: cue
    num_reward_states = 2  # 0: reward on left, 1: reward on right

    cue_obs = jnp.zeros((5, num_locations, num_reward_states), dtype=jnp.float32)
    reward_obs = jnp.zeros((3, num_locations, num_reward_states), dtype=jnp.float32)

    for loc in range(num_locations):
        for reward_state in range(num_reward_states):
            if loc == 0:
                cue_obs = cue_obs.at[0, loc, reward_state].set(1.0)
                reward_obs = reward_obs.at[0, loc, reward_state].set(1.0)
            elif loc == 3:
                cue_idx = 3 if reward_state == 0 else 4
                cue_obs = cue_obs.at[cue_idx, loc, reward_state].set(1.0)
                reward_obs = reward_obs.at[0, loc, reward_state].set(1.0)
            elif loc == 1:
                cue_obs = cue_obs.at[1, loc, reward_state].set(1.0)
                observation_idx = 1 if reward_state == 0 else 2
                reward_obs = reward_obs.at[observation_idx, loc, reward_state].set(1.0)
            elif loc == 2:
                cue_obs = cue_obs.at[2, loc, reward_state].set(1.0)
                observation_idx = 1 if reward_state == 1 else 2
                reward_obs = reward_obs.at[observation_idx, loc, reward_state].set(1.0)

    A = [cue_obs, reward_obs]
    A_dependencies = [[0, 1], [0, 1]]

    B_loc = utils.create_controllable_B(num_locations, num_locations)[0]
    B_reward = jnp.eye(num_reward_states, dtype=jnp.float32).reshape(
        num_reward_states, num_reward_states, 1
    )
    B = [B_loc, B_reward]
    B_dependencies = [[0], [1]]

    D = [
        jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        jnp.array([0.5, 0.5], dtype=jnp.float32),
    ]

    return A, B, A_dependencies, B_dependencies, D


def _build_dual_cue_model():
    num_locations = 5  # 0: start, 1: cue A, 2: cue B, 3: reward L, 4: reward R
    num_reward_states = 2
    num_distractor_states = 2

    loc_obs = jnp.eye(num_locations, dtype=jnp.float32)
    cue_a_obs = jnp.zeros((3, num_locations, num_reward_states), dtype=jnp.float32)
    cue_b_obs = jnp.zeros((3, num_locations, num_distractor_states), dtype=jnp.float32)
    reward_obs = jnp.zeros((3, num_locations, num_reward_states), dtype=jnp.float32)

    for loc in range(num_locations):
        for reward_state in range(num_reward_states):
            if loc == 1:
                cue_idx = 1 if reward_state == 0 else 2
                cue_a_obs = cue_a_obs.at[cue_idx, loc, reward_state].set(1.0)
            else:
                cue_a_obs = cue_a_obs.at[0, loc, reward_state].set(1.0)

            if loc == 3:
                observation_idx = 1 if reward_state == 0 else 2
                reward_obs = reward_obs.at[observation_idx, loc, reward_state].set(1.0)
            elif loc == 4:
                observation_idx = 1 if reward_state == 1 else 2
                reward_obs = reward_obs.at[observation_idx, loc, reward_state].set(1.0)
            else:
                reward_obs = reward_obs.at[0, loc, reward_state].set(1.0)

        for distractor_state in range(num_distractor_states):
            if loc == 2:
                cue_idx = 1 if distractor_state == 0 else 2
                cue_b_obs = cue_b_obs.at[cue_idx, loc, distractor_state].set(1.0)
            else:
                cue_b_obs = cue_b_obs.at[0, loc, distractor_state].set(1.0)

    A = [loc_obs, cue_a_obs, cue_b_obs, reward_obs]
    A_dependencies = [[0], [0, 1], [0, 2], [0, 1]]

    B_loc = utils.create_controllable_B(num_locations, num_locations)[0]
    B_reward = jnp.eye(num_reward_states, dtype=jnp.float32).reshape(
        num_reward_states, num_reward_states, 1
    )
    B_distractor = jnp.eye(num_distractor_states, dtype=jnp.float32).reshape(
        num_distractor_states, num_distractor_states, 1
    )
    B = [B_loc, B_reward, B_distractor]
    B_dependencies = [[0], [1], [2]]

    D = [
        jnp.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        jnp.array([0.5, 0.5], dtype=jnp.float32),
        jnp.array([0.5, 0.5], dtype=jnp.float32),
    ]

    return A, B, A_dependencies, B_dependencies, D


class TestSophisticatedInferenceJax(unittest.TestCase):
    def _run_si_search(self, agent, horizon):
        qs = jtu.tree_map(lambda x: jnp.expand_dims(x, -2), agent.D)
        search_fn = si_policy_search(
            horizon=horizon,
            max_nodes=512,
            max_branching=32,
            policy_prune_threshold=0.0,
            observation_prune_threshold=0.0,
            entropy_stop_threshold=-1.0,
            gamma=4.0,
            topk_obsspace=2,
        )
        q_pi, _ = search_fn(agent, qs=qs, rng_key=jr.PRNGKey(0))
        return q_pi, qs

    def _run_vanilla_search(self, agent):
        qs = jtu.tree_map(lambda x: jnp.expand_dims(x, -2), agent.D)
        q_pi, _ = agent.infer_policies(qs)
        return q_pi

    def test_si_accepts_costly_informative_cue(self):
        A, B, A_dependencies, B_dependencies, D = _build_single_cue_model()
        cue_cost = -2.0
        reward_value = 6.0
        punishment_value = -12.0

        C = [
            jnp.array([0.0, 0.0, 0.0, cue_cost, cue_cost], dtype=jnp.float32),
            jnp.array([0.0, reward_value, punishment_value], dtype=jnp.float32),
        ]

        agent = Agent(
            A,
            B,
            C=C,
            D=D,
            A_dependencies=A_dependencies,
            B_dependencies=B_dependencies,
            num_controls=[4, 1],
            policy_len=1,
        )

        q_pi, qs = self._run_si_search(agent, horizon=2)
        action_marginals = control.get_marginals(q_pi[0], agent.policies, agent.num_controls)

        cue_action = int(jnp.argmax(action_marginals[0]))
        self.assertEqual(cue_action, 3)

        cue_prob = float(action_marginals[0][3])
        left_prob = float(action_marginals[0][1])
        right_prob = float(action_marginals[0][2])
        self.assertGreater(cue_prob, left_prob)
        self.assertGreater(cue_prob, right_prob)

        qs_current = D
        cue_policy = [3, 0]
        qs_next = control.compute_expected_state(qs_current, B, cue_policy, B_dependencies)
        qo = control.compute_expected_obs(qs_next, A, A_dependencies)
        immediate_u = control.compute_expected_utility(qo, C)
        self.assertLess(float(immediate_u), 0.0)

        vanilla_agent = Agent(
            A,
            B,
            C=C,
            D=D,
            A_dependencies=A_dependencies,
            B_dependencies=B_dependencies,
            num_controls=[4, 1],
            policy_len=2,
            gamma=4.0,
            use_utility=True,
            use_states_info_gain=True,
        )
        q_pi_vanilla = self._run_vanilla_search(vanilla_agent)
        action_marginals_vanilla = control.get_marginals(
            q_pi_vanilla[0], vanilla_agent.policies, vanilla_agent.num_controls
        )
        stay_prob = float(action_marginals_vanilla[0][0])
        cue_prob = float(action_marginals_vanilla[0][3])
        self.assertLess(cue_prob, stay_prob)
        self.assertNotEqual(int(jnp.argmax(action_marginals_vanilla[0])), 3)

    def test_si_ignores_irrelevant_distractor_cue(self):
        A, B, A_dependencies, B_dependencies, D = _build_dual_cue_model()
        cue_cost = -2.0
        reward_value = 6.0
        punishment_value = -12.0

        C = [
            jnp.zeros(5, dtype=jnp.float32),
            jnp.array([0.0, cue_cost, cue_cost], dtype=jnp.float32),
            jnp.array([0.0, cue_cost, cue_cost], dtype=jnp.float32),
            jnp.array([0.0, reward_value, punishment_value], dtype=jnp.float32),
        ]

        agent = Agent(
            A,
            B,
            C=C,
            D=D,
            A_dependencies=A_dependencies,
            B_dependencies=B_dependencies,
            num_controls=[5, 1, 1],
            policy_len=1,
        )

        q_pi, _ = self._run_si_search(agent, horizon=2)
        action_marginals = control.get_marginals(q_pi[0], agent.policies, agent.num_controls)

        cue_a_prob = float(action_marginals[0][1])
        cue_b_prob = float(action_marginals[0][2])
        self.assertGreater(cue_a_prob, cue_b_prob)
        self.assertEqual(int(jnp.argmax(action_marginals[0])), 1)

        vanilla_agent = Agent(
            A,
            B,
            C=C,
            D=D,
            A_dependencies=A_dependencies,
            B_dependencies=B_dependencies,
            num_controls=[5, 1, 1],
            policy_len=2,
            gamma=4.0,
            use_utility=True,
            use_states_info_gain=True,
        )
        q_pi_vanilla = self._run_vanilla_search(vanilla_agent)
        action_marginals_vanilla = control.get_marginals(
            q_pi_vanilla[0], vanilla_agent.policies, vanilla_agent.num_controls
        )
        cue_a_prob = float(action_marginals_vanilla[0][1])
        cue_b_prob = float(action_marginals_vanilla[0][2])
        self.assertAlmostEqual(cue_a_prob, cue_b_prob, places=6)

    def _run_contract_search(self, agent, **overrides):
        options = {
            "horizon": 1,
            "max_nodes": 64,
            "max_branching": 5,
            "policy_prune_threshold": 0.0,
            "observation_prune_threshold": 0.0,
            "entropy_stop_threshold": -1.0,
            "prune_penalty": 1.0,
            "gamma": 1.0,
            "topk_obsspace": 2,
        }
        options.update(overrides)
        qs = jtu.tree_map(lambda q: jnp.expand_dims(q, -2), agent.D)
        q_pi, extras = si_policy_search(**options)(agent, qs=qs)
        tree = jtu.tree_map(lambda value: value[0], extras["tree"])
        return np.asarray(q_pi[0]), tree

    def _make_probability_agent(self, probabilities):
        return Agent(
            [jnp.asarray(p, dtype=jnp.float32)[:, None] for p in probabilities],
            [jnp.ones((1, 1, 1), dtype=jnp.float32)],
            C=[jnp.zeros(len(p), dtype=jnp.float32) for p in probabilities],
            D=[jnp.ones(1, dtype=jnp.float32)],
            A_dependencies=[[0] for _ in probabilities],
            B_dependencies=[[0]],
            num_controls=[1],
            policy_len=1,
        )

    def _make_value_agent(self, values):
        return Agent(
            [jnp.eye(2, dtype=jnp.float32)],
            [jnp.broadcast_to(jnp.eye(2, dtype=jnp.float32)[:, None, :], (2, 2, 2))],
            C=[jnp.asarray(values, dtype=jnp.float32)],
            D=[jnp.array([0.5, 0.5], dtype=jnp.float32)],
            A_dependencies=[[0]],
            B_dependencies=[[0]],
            num_controls=[2],
            policy_len=1,
        )

    def test_si_categorical_matches_discrete_branches_and_values(self):
        model = {
            "A": [jnp.array([[0.9, 0.1], [0.1, 0.9]], dtype=jnp.float32)],
            "B": [jnp.stack((jnp.eye(2), jnp.array([[0.0, 1.0], [1.0, 0.0]])), axis=-1)],
            "C": [jnp.array([0.0, 1.0], dtype=jnp.float32)],
            "D": [jnp.array([0.5, 0.5], dtype=jnp.float32)],
            "A_dependencies": [[0]],
            "B_dependencies": [[0]],
            "num_controls": [2],
            "policy_len": 1,
        }
        discrete_q, discrete_tree = self._run_contract_search(
            Agent(**model, categorical_obs=False), horizon=2
        )
        def preprocess(observations):
            return [jnp.eye(2, dtype=jnp.float32)[outcome] for outcome in observations]

        for preprocess_fn in (None, preprocess):
            with self.subTest(custom_preprocessor=preprocess_fn is not None):
                categorical_q, categorical_tree = self._run_contract_search(
                    Agent(**model, categorical_obs=True, preprocess_fn=preprocess_fn), horizon=2
                )
                np.testing.assert_allclose(categorical_q, discrete_q, atol=1e-6)
                for field in ("used", "policy", "observation", "children_indices"):
                    np.testing.assert_array_equal(
                        getattr(categorical_tree, field), getattr(discrete_tree, field)
                    )
                for field in ("neg_efe", "children_probs"):
                    np.testing.assert_allclose(
                        getattr(categorical_tree, field), getattr(discrete_tree, field), atol=1e-6
                    )
                for categorical_belief, discrete_belief in zip(categorical_tree.qs, discrete_tree.qs):
                    np.testing.assert_allclose(categorical_belief, discrete_belief, atol=1e-6)
                self.assertGreater(float(categorical_tree.neg_efe[0, 0]), 0.0)
                for tree in (discrete_tree, categorical_tree):
                    for policy_idx in np.asarray(tree.children_indices[0, :2]):
                        observation_nodes = np.asarray(tree.children_indices[policy_idx, :2])
                        self.assertTrue(np.all(observation_nodes >= 0))
                        np.testing.assert_array_equal(tree.observation[observation_nodes, 0], [0, 1])
                        np.testing.assert_allclose(tree.children_probs[policy_idx, :2], [0.5, 0.5])
                        np.testing.assert_allclose(
                            tree.qs[0][observation_nodes, 0, :],
                            [[0.9, 0.1], [0.1, 0.9]],
                            atol=1e-5,
                        )

    def test_si_custom_inference_receives_integer_outcomes_for_categorical_agent(self):
        agent = Agent(
            [jnp.array([[0.9, 0.1], [0.1, 0.9]], dtype=jnp.float32)],
            [jnp.eye(2, dtype=jnp.float32)[..., None]],
            C=[jnp.array([0.0, 1.0], dtype=jnp.float32)],
            D=[jnp.array([0.5, 0.5], dtype=jnp.float32)],
            categorical_obs=True,
            policy_len=1,
        )

        def infer_integer_outcome(unbatched_agent, observations, prior):
            likelihood = unbatched_agent.A[0][observations[0][0, 0]]
            posterior = prior[0] * likelihood
            return [posterior / jnp.sum(posterior, axis=-1, keepdims=True)]

        expected_q, expected_tree = self._run_contract_search(agent)
        actual_q, actual_tree = self._run_contract_search(agent, infer_fn=infer_integer_outcome)
        np.testing.assert_allclose(actual_q, expected_q, atol=1e-6)
        np.testing.assert_array_equal(actual_tree.observation, expected_tree.observation)
        np.testing.assert_allclose(actual_tree.qs[0], expected_tree.qs[0], atol=1e-5)
        np.testing.assert_allclose(actual_tree.neg_efe, expected_tree.neg_efe, atol=1e-6)

    def test_si_joint_budget_preserves_retained_weights_and_pruning(self):
        probabilities = ([0.7, 0.2, 0.1], [0.6, 0.3, 0.1])
        agent = self._make_probability_agent(probabilities)
        raw_masses = np.array([0.42, 0.21])
        retained_weights = raw_masses / (0.9 * 0.9)
        for threshold, expected_count in ((0.0, 2), (0.3, 1)):
            with self.subTest(observation_prune_threshold=threshold):
                q_pi, tree = self._run_contract_search(
                    agent, max_branching=2, observation_prune_threshold=threshold
                )
                policy_idx = int(tree.children_indices[0, 0])
                observation_nodes = np.asarray(tree.children_indices[policy_idx])
                observation_nodes = observation_nodes[observation_nodes >= 0]
                self.assertEqual(len(observation_nodes), expected_count)
                selected = np.asarray(tree.observation[observation_nodes])
                np.testing.assert_array_equal(selected, [[0, 0], [0, 1]][:expected_count])
                selected_raw_masses = np.array([
                    np.prod([p[outcome] for p, outcome in zip(probabilities, observation)])
                    for observation in selected
                ])
                np.testing.assert_allclose(
                    selected_raw_masses, raw_masses[:expected_count], atol=1e-7
                )
                weights = np.asarray(tree.children_probs[policy_idx])
                np.testing.assert_allclose(
                    weights[:expected_count], retained_weights[:expected_count], atol=1e-6
                )
                np.testing.assert_array_equal(weights[expected_count:], 0.0)
                np.testing.assert_allclose(q_pi, [1.0])
                self.assertLess(float(weights.sum()), 1.0)
                if expected_count == 2:
                    self.assertAlmostEqual(float(weights[0] / weights[1]), 2.0, places=6)

    def test_si_joint_budget_bounds_high_modality_selection(self):
        num_modalities = 24
        probabilities = [[0.7, 0.2, 0.1] for _ in range(num_modalities - 1)]
        probabilities.append([0.6, 0.3, 0.1])
        q_pi, tree = self._run_contract_search(
            self._make_probability_agent(probabilities), max_nodes=16, max_branching=2
        )
        policy_idx = int(tree.children_indices[0, 0])
        observation_nodes = np.asarray(tree.children_indices[policy_idx])
        self.assertTrue(np.all(observation_nodes >= 0))
        expected = np.zeros((2, num_modalities), dtype=np.int32)
        expected[1, -1] = 1
        np.testing.assert_array_equal(tree.observation[observation_nodes], expected)
        expected_weights = (0.7 / 0.9) ** (num_modalities - 1) * np.array([0.6, 0.3]) / 0.9
        np.testing.assert_allclose(
            tree.children_probs[policy_idx], expected_weights, rtol=2e-6, atol=1e-9
        )
        used_observations = (
            np.asarray(tree.used[:, 0])
            & (np.asarray(tree.horizon[:, 0]) == 1)
            & (np.asarray(tree.observation[:, 0]) >= 0)
        )
        self.assertEqual(int(used_observations.sum()), 2)
        self.assertEqual(int(np.asarray(tree.used).sum()), 4)
        np.testing.assert_allclose(q_pi, [1.0])

    def test_si_joint_budget_resolves_probability_ties_deterministically(self):
        _, tree = self._run_contract_search(
            self._make_probability_agent([[0.5, 0.5], [0.5, 0.5]]), max_branching=2
        )
        policy_idx = int(tree.children_indices[0, 0])
        observation_nodes = np.asarray(tree.children_indices[policy_idx])
        np.testing.assert_array_equal(tree.observation[observation_nodes], [[0, 0], [0, 1]])
        np.testing.assert_allclose(tree.children_probs[policy_idx], [0.25, 0.25])

    def test_si_padding_preserves_equal_policy_posterior_and_value(self):
        for capacity in (2, 5):
            with self.subTest(max_branching=capacity):
                q_pi, tree = self._run_contract_search(
                    self._make_value_agent([0.0, 0.0]), max_branching=capacity
                )
                np.testing.assert_allclose(q_pi, [0.5, 0.5], atol=1e-6)
                self.assertAlmostEqual(float(tree.neg_efe[0, 0]), 0.0, places=6)
                np.testing.assert_array_equal(tree.children_probs[0, 2:], 0.0)
                self.assertTrue(np.all(np.isfinite(tree.neg_efe)))
                self.assertTrue(np.all(np.isfinite(tree.children_probs)))

    def test_si_padding_preserves_unequal_policy_posterior_and_value(self):
        expected_q = np.exp([0.0, 1.0]) / np.exp([0.0, 1.0]).sum()
        for capacity in (2, 5):
            with self.subTest(max_branching=capacity):
                q_pi, tree = self._run_contract_search(
                    self._make_value_agent([0.0, 1.0]), max_branching=capacity
                )
                np.testing.assert_allclose(q_pi, expected_q, atol=1e-6)
                self.assertAlmostEqual(float(tree.neg_efe[0, 0]), expected_q[1], places=6)
                np.testing.assert_array_equal(tree.children_probs[0, 2:], 0.0)
                self.assertTrue(np.all(np.isfinite(tree.neg_efe)))
                self.assertTrue(np.all(np.isfinite(tree.children_probs)))

    def test_si_real_pruned_policy_retains_finite_penalty_mass(self):
        expected_q = np.exp([0.0, -1.0]) / np.exp([0.0, -1.0]).sum()
        for capacity in (2, 5):
            with self.subTest(max_branching=capacity):
                q_pi, tree = self._run_contract_search(
                    self._make_value_agent([0.0, -2.0]),
                    max_branching=capacity,
                    policy_prune_threshold=0.3,
                )
                np.testing.assert_allclose(q_pi, expected_q, atol=1e-6)
                self.assertEqual(int(tree.children_indices[0, 1]), -1)
                self.assertGreater(float(tree.children_probs[0, 1]), 0.0)
                self.assertAlmostEqual(float(tree.neg_efe[0, 0]), -expected_q[1], places=6)
                np.testing.assert_array_equal(tree.children_probs[0, 2:], 0.0)

    def test_si_all_pruned_policies_remain_finite_without_padding_mass(self):
        for capacity in (2, 5):
            with self.subTest(max_branching=capacity):
                q_pi, tree = self._run_contract_search(
                    self._make_value_agent([0.0, 0.0]),
                    horizon=2,
                    max_branching=capacity,
                    policy_prune_threshold=0.6,
                )
                np.testing.assert_allclose(q_pi, [0.5, 0.5], atol=1e-6)
                np.testing.assert_array_equal(tree.children_indices[0], -1)
                self.assertAlmostEqual(float(tree.neg_efe[0, 0]), -1.0, places=6)
                np.testing.assert_array_equal(tree.children_probs[0, 2:], 0.0)
                self.assertTrue(np.all(np.isfinite(tree.neg_efe)))
                self.assertTrue(np.all(np.isfinite(tree.children_probs)))

    def test_si_policy_leaves_remain_finite_without_padding_mass(self):
        for capacity in (2, 5):
            with self.subTest(max_branching=capacity):
                q_pi, tree = self._run_contract_search(
                    self._make_value_agent([0.0, 0.0]),
                    horizon=2,
                    max_branching=capacity,
                    observation_prune_threshold=1.0,
                )
                np.testing.assert_allclose(q_pi, [0.5, 0.5], atol=1e-6)
                self.assertTrue(np.all(np.asarray(tree.children_indices[0, :2]) >= 0))
                self.assertAlmostEqual(float(tree.neg_efe[0, 0]), -1.0, places=6)
                np.testing.assert_array_equal(tree.children_probs[0, 2:], 0.0)
                self.assertTrue(np.all(np.isfinite(tree.neg_efe)))
                self.assertTrue(np.all(np.isfinite(tree.children_probs)))


if __name__ == "__main__":
    unittest.main()
