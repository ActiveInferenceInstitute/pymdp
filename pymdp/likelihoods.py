"""Observed-action likelihoods for current, batched pymdp agents."""

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpyro.distributions as dist
from jax import lax
from numpyro import deterministic, plate, sample
from numpyro.contrib.control_flow import scan


def evolve_trials(agent: Any, data: Any) -> Any:
    """Return action probabilities, observations and action vectors for one block.

    ``observations`` (or the existing ``outcomes`` alias) is a list of modality
    arrays with leading axes ``(trials, agents)``. ``actions`` has shape
    ``(trials, agents, factors)`` and uses the same integer tuple encoding as
    ``Agent.sample_action``. Both are required; this is not a generative model.
    Probability columns follow ``agent.unique_multiactions``.

    Every call starts at ``agent.D``. Each trial uses the agent's state-inference
    and empirical-prior APIs; no learning or inter-block state carry is added.

    Parameters
    ----------
    agent : Agent
        Current batched agent, including its action-support convention.
    data : dict
        Required observed inputs in the shapes described above.

    Returns
    -------
    tuple
        Joint action probabilities, modality observations, and supplied action
        vectors, all stacked along the trial axis.
    """
    observations = data.get("observations", data.get("outcomes"))
    if observations is None:
        raise KeyError("Expected `observations` in trial data.")
    if data.get("actions") is None:
        raise KeyError("Expected `actions` in trial data.")
    actions = jnp.asarray(data["actions"])
    if actions.ndim != 3 or actions.shape[1:] != (agent.batch_size, agent.num_factors):
        raise ValueError("`actions` must have shape (trials, agents, factors).")
    if not jnp.issubdtype(actions.dtype, jnp.integer):
        raise ValueError("`actions` must contain integer action vectors.")
    if len(observations) != agent.num_modalities or any(
        o.shape[:2] != actions.shape[:2] for o in observations
    ):
        raise ValueError("Each observation modality must have leading axes (trials, agents).")

    def step_fn(empirical_prior: Any, xs: dict[str, Any]) -> tuple[Any, tuple[Any, Any, Any]]:
        observations_t = xs["observations"]
        # Retain the explicit time axis required by sequence inference; one-step
        # inference also accepts this form and selects its single observation.
        qs = agent.infer_states([o[:, None, ...] for o in observations_t], empirical_prior)
        q_pi, _ = agent.infer_policies(qs)
        probs = agent.multiaction_probabilities(q_pi)
        actions_t = xs["actions"]
        empirical_prior = agent.update_empirical_prior(actions_t, qs)
        return empirical_prior, (probs, observations_t, actions_t)

    _, result = lax.scan(
        step_fn, agent.D, {"observations": observations, "actions": actions}
    )
    return result


def _action_indices(agent: Any, actions: Any) -> Any:
    """Map observed tuples to the probability support, without clipped misses."""
    matches = jnp.all(actions[..., None, :] == agent.unique_multiactions, axis=-1)
    indices = jnp.argmax(matches, axis=-1)
    return eqx.error_if(
        indices,
        jnp.any(~jnp.any(matches, axis=-1)),
        "Observed action vector is outside the agent's multi-action support.",
    )


def aif_likelihood(Nb: int, Nt: int, Na: int, data: Any, agent: Any) -> None:
    """Observe joint actions for independent experimental blocks.

    Data has leading axes ``(Nb, Nt, Na)``. Blocks independently reset to
    ``agent.D``. The ``actions`` sample site contains joint categorical indices;
    ``action_vectors`` preserves the corresponding supplied factor-wise tuples.
    Trial and agent axes are independent plates, not categorical event axes.

    Parameters
    ----------
    Nb, Nt, Na : int
        Number of blocks, trials per block, and agents respectively.
    data : dict
        Observations and action vectors with an additional leading block axis.
    agent : Agent
        Current batched agent; its batch size must equal ``Na``.

    Returns
    -------
    None
        Registers observed sample and deterministic sites with NumPyro.
    """
    if Na != agent.batch_size:
        raise ValueError("`Na` must equal the agent's batch size.")
    if data.get("actions") is None:
        raise KeyError("Expected `actions` in trial data.")
    if data["actions"].shape != (Nb, Nt, Na, agent.num_factors):
        raise ValueError("`actions` must have shape (Nb, Nt, Na, factors).")

    def step_fn(carry: Any, xs: Any) -> tuple[None, None]:
        probs, observations, actions = evolve_trials(agent, xs)
        deterministic("observations", observations)
        deterministic("action_vectors", actions)
        indices = _action_indices(agent, actions)
        with plate("num_agents", Na, dim=-1), plate("num_trials", Nt, dim=-2):
            sample("actions", dist.Categorical(probs=probs), obs=indices)
        return None, None

    scan(step_fn, None, data, length=Nb)
