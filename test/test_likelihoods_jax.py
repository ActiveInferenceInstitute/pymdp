"""Observed-action likelihood contracts; optional modeling runtime required."""

import numpy as np
import pytest
from jax import grad, nn
from jax import numpy as jnp

pytest.importorskip("numpyro")
from numpyro.infer.util import log_density

from pymdp.agent import Agent
from pymdp.likelihoods import aif_likelihood, evolve_trials


def _agent(probabilities, *, batch_size=1, controls=(2,), policies=None,
           sampling_mode="full", categorical_obs=False):
    return Agent(
        A=[jnp.ones((1, 1))],
        B=[jnp.ones((1, 1, n)) for n in controls],
        A_dependencies=[[0]], B_dependencies=[[f] for f in range(len(controls))],
        num_controls=list(controls), E=probabilities, policies=policies,
        batch_size=batch_size, sampling_mode=sampling_mode,
        categorical_obs=categorical_obs, use_utility=False,
        use_states_info_gain=False,
    )


def _data(actions, *, categorical=False):
    actions = jnp.asarray(actions, dtype=jnp.int32)
    observations = (jnp.ones(actions.shape[:-1] + (1,)) if categorical
                    else jnp.zeros(actions.shape[:-1], dtype=jnp.int32))
    return {"observations": [observations], "actions": actions}


def _density(agent, data):
    nb, nt, na = data["actions"].shape[:3]
    return log_density(aif_likelihood, (nb, nt, na, data, agent), {}, {})


@pytest.mark.parametrize("categorical", [False, True])
@pytest.mark.parametrize("sampling_mode", ["full", "marginal"])
def test_action_probability_likelihood_has_independent_trial_and_agent_axes(
    categorical, sampling_mode,
):
    agent = _agent(jnp.array([0.8, 0.2]), batch_size=2,
                   categorical_obs=categorical, sampling_mode=sampling_mode)
    data = _data([[[[0], [1]], [[1], [0]], [[0], [0]]]], categorical=categorical)
    value, trace = _density(agent, data)
    np.testing.assert_allclose(value, 4 * np.log(0.8) + 2 * np.log(0.2), atol=1e-6)
    site = trace["actions"]
    assert site["fn"].event_shape == ()
    assert site["fn"].log_prob(site["value"]).shape == (1, 3, 2)
    np.testing.assert_array_equal(site["value"], data["actions"][..., 0])
    np.testing.assert_array_equal(trace["action_vectors"]["value"], data["actions"])


def test_likelihood_maps_restricted_policy_action_tuples_not_factor_indices():
    policies = jnp.array([[[0, 1]], [[1, 0]]])
    agent = _agent(jnp.array([0.8, 0.2]), controls=(2, 2), policies=policies)
    data = _data([[[[0, 1]], [[1, 0]]]])
    value, trace = _density(agent, data)
    np.testing.assert_allclose(value, np.log(0.8) + np.log(0.2), atol=1e-6)
    np.testing.assert_array_equal(trace["actions"]["value"], [[[0], [1]]])


def test_marginal_likelihood_includes_independently_sampled_recombinations():
    policies = jnp.array([[[0, 1]], [[1, 0]]])
    agent = _agent(jnp.array([0.8, 0.2]), controls=(2, 2), policies=policies,
                   sampling_mode="marginal")
    value, _ = _density(agent, _data([[[[0, 0]], [[1, 1]]]]))
    np.testing.assert_allclose(value, 2 * np.log(0.16), atol=1e-6)


def test_likelihood_matches_shared_control_flattened_action_support():
    agent = Agent(
        A=[jnp.ones((1, 1))], B=[jnp.ones((1, 1, 2, 2))] * 2,
        A_dependencies=[[0]], B_dependencies=[[0], [1]],
        B_action_dependencies=[[0, 1], [0, 1]], num_controls=[2, 2],
        E=jnp.array([0.1, 0.2, 0.3, 0.4]), use_utility=False,
        use_states_info_gain=False, sampling_mode="full",
    )
    value, _ = _density(agent, _data([[[[3, 3]], [[1, 1]]]]))
    np.testing.assert_allclose(value, np.log(0.4) + np.log(0.2), atol=1e-6)


def test_out_of_support_tuple_is_not_silently_assigned_a_category():
    agent = _agent(jnp.array([0.8, 0.2]), controls=(2, 2),
                   policies=jnp.array([[[0, 1]], [[1, 0]]]))
    with pytest.raises(Exception, match="outside.*multi-action support"):
        value, _ = _density(agent, _data([[[[0, 0]]]]))
        value.block_until_ready()


@pytest.mark.parametrize("method", ["fpi", "mmp", "vmp"])
def test_blocks_reset_prior_but_trials_propagate_actions(method):
    agent = Agent(
        A=[jnp.array([[0.9, 0.1], [0.1, 0.9]])],
        B=[jnp.stack([jnp.eye(2), jnp.array([[0.0, 1.0], [1.0, 0.0]])], -1)],
        D=[jnp.array([0.9, 0.1])], C=[jnp.array([1.0, 0.0])],
        inference_algo=method, inference_horizon=1, use_states_info_gain=False,
    )
    data = _data([[[[1]], [[0]]], [[[1]], [[0]]]])
    _, trace = _density(agent, data)
    probs = trace["actions"]["fn"].probs
    np.testing.assert_allclose(probs[0], probs[1], atol=1e-6)
    assert not np.allclose(probs[0, 0], probs[0, 1])
    # Bayes filtering for these two supplied observations yields .9878049 and .1
    # for state zero; the identity/swap policies then differ in expected utility.
    state_zero = np.array([0.81 / 0.82, 0.1])
    utility_difference = 1.6 * state_zero - 0.8
    expected_action_zero = 1 / (1 + np.exp(-utility_difference))
    np.testing.assert_allclose(probs[0, :, 0, 0], expected_action_zero, atol=1e-6)


def test_likelihood_gradient_uses_probabilities_not_logits_twice():
    data = _data([[[[0]], [[0]], [[1]]]])

    def objective(logits):
        value, _ = _density(_agent(nn.softmax(logits)), data)
        return value

    derivative = grad(objective)(jnp.log(jnp.array([0.8, 0.2])))
    np.testing.assert_allclose(derivative, [-0.4, 0.4], atol=1e-6)


def test_evolve_trials_preserves_outcomes_alias_and_supplied_vectors():
    agent = _agent(jnp.array([0.8, 0.2]))
    data = _data([[[0]], [[1]]])
    data["outcomes"] = data.pop("observations")
    probs, observations, actions = evolve_trials(agent, data)
    np.testing.assert_allclose(probs[:, 0], [[0.8, 0.2], [0.8, 0.2]], atol=1e-6)
    np.testing.assert_array_equal(observations[0], data["outcomes"][0])
    np.testing.assert_array_equal(actions, data["actions"])


@pytest.mark.parametrize("missing", ["observations", "actions"])
def test_evolve_trials_requires_observed_inputs_without_sampling_fallback(missing):
    data = _data([[[0]]])
    data.pop(missing)
    with pytest.raises(KeyError, match=missing):
        evolve_trials(_agent(jnp.array([0.8, 0.2])), data)


def test_categorical_modalities_keep_distinct_observation_shapes():
    agent = Agent(
        A=[jnp.ones((1, 1)), jnp.array([[0.7], [0.3]])],
        B=[jnp.ones((1, 1, 2))], E=jnp.array([0.8, 0.2]),
        categorical_obs=True, use_utility=False, use_states_info_gain=False,
    )
    data = {
        "observations": [jnp.ones((1, 2, 1, 1)),
                         jnp.array([[[[1.0, 0.0]], [[0.0, 1.0]]]])],
        "actions": jnp.array([[[[0]], [[1]]]]),
    }
    value, trace = _density(agent, data)
    np.testing.assert_allclose(value, np.log(0.8) + np.log(0.2), atol=1e-6)
    for recorded, supplied in zip(trace["observations"]["value"], data["observations"]):
        np.testing.assert_array_equal(recorded, supplied)
