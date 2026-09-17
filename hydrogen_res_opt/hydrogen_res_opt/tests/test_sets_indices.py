"""Unit tests for sets_indices.py."""

import pytest

from hydrogen_res_opt.exceptions import DataValidationError
from hydrogen_res_opt.sets_indices import ModelSets


def test_default_sets_are_valid():
    sets = ModelSets()
    assert "FRN" in sets.substations
    assert set(sets.remote_sites()) | set(sets.collocated_sites()) == set(sets.res_sites)


def test_is_remote_indicator():
    sets = ModelSets()
    assert sets.is_remote("remote1") is True
    assert sets.is_remote("collocated_FRN") is False


def test_empty_res_sites_rejected():
    with pytest.raises(DataValidationError):
        ModelSets(res_sites=[])


def test_missing_distance_entry_rejected():
    with pytest.raises(DataValidationError):
        ModelSets(
            res_sites=["remote1"],
            substations=["FRN", "YRJ"],
            distance_matrix={("remote1", "FRN"): 20.0},  # missing (remote1, YRJ)
        )


def test_collocated_zero_distance_default():
    sets = ModelSets()
    assert sets.distance_matrix[("collocated_FRN", "FRN")] == 0.0
