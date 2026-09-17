
"""Unit tests for config.py -- parameter validation and derived values."""

import pytest

from hydrogen_res_opt.config import (
    ElectrolyserParameters,
    InvestmentParameters,
    ModelConfig,
    ScenarioParameters,
)
from hydrogen_res_opt.exceptions import DataValidationError


def test_crf_matches_known_value():
    # Table 6: CRF = 0.0872 at r=6%, n=20 years.
    inv = InvestmentParameters(discount_rate=0.06, project_lifetime_years=20)
    assert inv.crf == pytest.approx(0.0872, abs=1e-3)


def test_investment_rejects_negative_cost():
    with pytest.raises(DataValidationError):
        InvestmentParameters(c_wind=-1.0)


def test_electrolyser_rejects_min_above_max():
    with pytest.raises(DataValidationError):
        ElectrolyserParameters(p_elec_min=25, p_elec_max=20.0)


def test_electrolyser_ramp_rate_derivation():
    e = ElectrolyserParameters(p_elec_max=20.0, ramp_rate_frac_of_pmax=0.15)
    assert e.ramp_rate_mw_per_h == pytest.approx(3.0)


def test_scenario_requires_negative_correlation():
    with pytest.raises(DataValidationError):
        ScenarioParameters(wind_price_correlation=0.1)


def test_model_config_defaults_are_valid():
    cfg = ModelConfig()
    assert cfg.investment.crf > 0
    assert cfg.electrolyser.p_elec_max > 0
    d = cfg.to_dict()
    assert "investment" in d and "electrolyser" in d
