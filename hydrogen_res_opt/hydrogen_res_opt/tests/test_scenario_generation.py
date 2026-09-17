"""Unit tests for scenario_generation.py."""

import numpy as np
import pytest

from hydrogen_res_opt.config import ScenarioParameters
from hydrogen_res_opt.exceptions import ScenarioGenerationError
from hydrogen_res_opt.scenario_generation import ScenarioGenerator, ScenarioTree
from hydrogen_res_opt.sets_indices import ModelSets


@pytest.fixture
def small_sets():
    return ModelSets(n_periods=48, n_scenarios=5)


@pytest.fixture
def small_params():
    return ScenarioParameters(n_scenarios=5, monte_carlo_pool_size=20, random_seed=1)


def test_synthetic_tree_has_expected_shape(small_sets, small_params):
    gen = ScenarioGenerator(small_sets, small_params)
    tree = gen.synthetic()
    assert tree.price.shape == (5, 48)
    assert tree.wind_cf.shape == (5, 48)
    assert tree.solar_cf.shape == (5, 48)
    assert tree.probabilities.shape == (5,)
    assert np.isclose(tree.probabilities.sum(), 1.0)


def test_synthetic_tree_bounds(small_sets, small_params):
    gen = ScenarioGenerator(small_sets, small_params)
    tree = gen.synthetic()
    assert (tree.wind_cf >= 0).all() and (tree.wind_cf <= 1).all()
    assert (tree.solar_cf >= 0).all() and (tree.solar_cf <= 1).all()


def test_expected_value_scenario_collapses_to_one(small_sets, small_params):
    gen = ScenarioGenerator(small_sets, small_params)
    tree = gen.synthetic()
    ev = tree.expected_value_scenario()
    assert ev.price.shape == (1, 48)
    assert ev.probabilities.shape == (1,)
    assert ev.probabilities[0] == pytest.approx(1.0)


def test_rejects_positive_wind_price_correlation():
    with pytest.raises(Exception):
        ScenarioParameters(wind_price_correlation=0.5)


def test_scenario_tree_validate_catches_bad_probabilities():
    tree = ScenarioTree(
        price=np.zeros((2, 3)),
        wind_cf=np.zeros((2, 3)),
        solar_cf=np.zeros((2, 3)),
        probabilities=np.array([0.5, 0.6]),  # sums to 1.1
    )
    with pytest.raises(ScenarioGenerationError):
        tree.validate()


def test_n_scenarios_must_be_less_than_pool(small_sets):
    # ScenarioParameters itself enforces pool_size >= n_scenarios at
    # construction time (fail fast, Section "config.py" design note).
    from hydrogen_res_opt.exceptions import DataValidationError

    with pytest.raises(DataValidationError):
        ScenarioParameters(n_scenarios=20, monte_carlo_pool_size=10, random_seed=1)
