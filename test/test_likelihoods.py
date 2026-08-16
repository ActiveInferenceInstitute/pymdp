"""Direct import coverage for the shipped NumPyro likelihood module."""

import numpyro

from pymdp import likelihoods


def test_likelihoods_imports_with_declared_numpyro_dependency():
    assert numpyro.__name__ == "numpyro"
    assert callable(likelihoods.evolve_trials)
    assert callable(likelihoods.aif_likelihood)
