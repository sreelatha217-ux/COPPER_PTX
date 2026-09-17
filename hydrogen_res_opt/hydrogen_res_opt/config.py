"""
config.py
=========
Typed parameter containers.

Every field maps 1:1 to a symbol in the paper's Tables 2, 3 and 6-10.
Default values are the "Typical Value" column from those tables so the
model is runnable out of the box for the HySynergy / DK1 case study;
override any of them from a YAML/JSON config file via
``ModelConfig.from_file`` for a different case study.

Design notes
------------
* ``frozen=True`` dataclasses -> parameters are immutable once loaded,
  preventing accidental in-place mutation deep inside the solve loop.
* ``__post_init__`` validation raises ``DataValidationError`` early
  (fail fast) rather than letting a bad parameter silently propagate
  into the MILP and produce a confusing solver-level failure.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .exceptions import DataValidationError
from .logging_config import get_logger

logger = get_logger(__name__)


def _require_positive(name: str, value: float) -> None:
    if value is None or value <= 0:
        raise DataValidationError(f"Parameter '{name}' must be > 0, got {value!r}")


def _require_range(name: str, value: float, lo: float, hi: float) -> None:
    if not (lo <= value <= hi):
        raise DataValidationError(
            f"Parameter '{name}'={value!r} outside expected range [{lo}, {hi}]"
        )


@dataclass(frozen=True)
class InvestmentParameters:
    """Table 2 / Table 6 -- investment cost & CAPEX parameters."""

    c_wind: float = 1.6e6          # cw  [EUR/MW]
    c_solar: float = 0.45e6        # cs  [EUR/MW]
    c_cable_collection: float = 0.15e6   # c^cable_k1  [EUR/km], K1 catalog
    c_cable_connection: float = 0.15e6   # c^cable_k2  [EUR/km], K2 catalog
    discount_rate: float = 0.06    # r   [-]  (5.0-6.0 % typical WACC)
    project_lifetime_years: int = 20  # n
    p_res_w_min: float = 0.0       # P^min_w  [MW]
    p_res_w_max: float = 300.0     # P^max_w  [MW]
    p_res_s_min: float = 0.0       # P^min_s  [MW]
    p_res_s_max: float = 300.0     # P^max_s  [MW]

    def __post_init__(self):
        _require_positive("c_wind", self.c_wind)
        _require_positive("c_solar", self.c_solar)
        _require_positive("c_cable_collection", self.c_cable_collection)
        _require_positive("c_cable_connection", self.c_cable_connection)
        _require_range("discount_rate", self.discount_rate, 0.0, 0.30)
        _require_positive("project_lifetime_years", self.project_lifetime_years)
        if self.p_res_w_max < self.p_res_w_min:
            raise DataValidationError("p_res_w_max must be >= p_res_w_min")
        if self.p_res_s_max < self.p_res_s_min:
            raise DataValidationError("p_res_s_max must be >= p_res_s_min")

    @property
    def crf(self) -> float:
        """Capital recovery factor, Eq. following Table 2:
        CRF = r(1+r)^n / [(1+r)^n - 1]
        """
        r, n = self.discount_rate, self.project_lifetime_years
        return (r * (1 + r) ** n) / ((1 + r) ** n - 1)


@dataclass(frozen=True)
class ElectrolyserParameters:
    """Table 2 / Table 7 -- electrolyser + compressor specification."""

    p_elec_min: float = 2.4        # P^min_elec [MW]  (20% of nominal, alkaline)
    p_elec_max: float = 20.0       # P^max_elec [MW]  (HySynergy Phase 1)
    efficiency_kg_per_mwh: float = 19.0    # eta [kg H2/MWh]
    ramp_rate_frac_of_pmax: float = 0.15   # Delta P^max as a fraction of P^max_elec
    annual_target_tonnes: float =  4.7 # D  [tonnes/yr] (870-1740 typical) 1740
    compression_kwh_per_kg: float = 1.67   # eta_compress [kWh/kg]

    def __post_init__(self):
        _require_positive("p_elec_max", self.p_elec_max)
        if not (0 <= self.p_elec_min <= self.p_elec_max):
            raise DataValidationError("Require 0 <= p_elec_min <= p_elec_max")
        _require_positive("efficiency_kg_per_mwh", self.efficiency_kg_per_mwh)
        _require_range("ramp_rate_frac_of_pmax", self.ramp_rate_frac_of_pmax, 0.0, 1.0)
        _require_positive("annual_target_tonnes", self.annual_target_tonnes)
        _require_positive("compression_kwh_per_kg", self.compression_kwh_per_kg)

    @property
    def ramp_rate_mw_per_h(self) -> float:
        """Delta P^max [MW/h], Table 2."""
        return self.ramp_rate_frac_of_pmax * self.p_elec_max

    @property
    def compression_efficiency_tonnes_per_mwh(self) -> float:
        """Convert eta_compress [kWh/kg] to an MWh/tonne consumption
        figure usable inside the storage-dynamics constraint C11."""
        return self.compression_kwh_per_kg  # kWh/kg == MWh/tonne numerically


@dataclass(frozen=True)
class GridParameters:
    """Table 8 -- grid / network transmission parameters (DistFlow, C14-C18)."""

    transformer_thermal_limit_mva: float = 46.0    # T^max_j [MVA] (per substation)
    cable_thermal_limit_mw: float = 25.0           # F^max_k [MW] (33 kV collector, typical)
    existing_branch_thermal_limit_mw: float = 55.0  # F^fixed_mn [MW]
    line_reactance_pu: float = 0.13                # X_ij [pu/km] -- model_builder scales by L[i,j]
    line_resistance_pu: float = 0.05                # R_ij [pu/km] (typical MV cable, not tabulated -> assumed) -- model_builder scales by L[i,j]
    v_base_kv: float = 33.0                         # V_base [kV]
    v_min_pu: float = 0.95                          # V^min [pu]
    v_max_pu: float = 1.05                          # V^max [pu]
    tso_tariff_eur_per_mwh: float = 15.60           # lambda^TSO (DK1, Energinet)
    dso_tariff_eur_per_mwh: float = 5.36            # lambda^DSO (DK1, Radius Elnet)

    def __post_init__(self):
        _require_positive("transformer_thermal_limit_mva", self.transformer_thermal_limit_mva)
        _require_positive("cable_thermal_limit_mw", self.cable_thermal_limit_mw)
        _require_positive("existing_branch_thermal_limit_mw", self.existing_branch_thermal_limit_mw)
        _require_positive("v_base_kv", self.v_base_kv)
        if not (0 < self.v_min_pu < self.v_max_pu):
            raise DataValidationError("Require 0 < v_min_pu < v_max_pu")


@dataclass(frozen=True)
class StorageParameters:
    """Table 9 -- hydrogen storage / demand parameters."""

    s_max_tonnes: float =  2      # S^max [tonnes] (HySynergy mid-scale tank) # 10.0
    baseline_demand_tonnes_per_h: float =  0.18  # d_t [tonnes/h] 0.18
    soc_initial_frac: float = 0.50    # SOC_ini [-]
    cyclic_tolerance_eps: float = 0.01  # epsilon for relaxed C13, Eq. (26)

    def __post_init__(self):
        _require_positive("s_max_tonnes", self.s_max_tonnes)
        _require_positive("baseline_demand_tonnes_per_h", self.baseline_demand_tonnes_per_h)
        _require_range("soc_initial_frac", self.soc_initial_frac, 0.0, 1.0)
        _require_range("cyclic_tolerance_eps", self.cyclic_tolerance_eps, 0.0, 1.0)


@dataclass(frozen=True)
class ScenarioParameters:
    """Table 10 -- stochastic scenario generation parameters."""

    n_scenarios: int = 2                       # |Omega|  50
    monte_carlo_pool_size: int = 500            # initial pool before fast-forward selection
    price_mean_eur_per_mwh: float = 51.76       # midpoint of 25.02-78.50 range
    price_low_eur_per_mwh: float = 25.02
    price_high_eur_per_mwh: float = 78.50
    wind_cf_mean: float = 0.42                  # Jammerbugt profile average CF
    solar_cf_low: float = 0.11
    solar_cf_high: float = 0.15
    wind_price_correlation: float = -0.35        # enforced < 0, Eq. (47); magnitude is a design choice
    random_seed: Optional[int] = 42

    def __post_init__(self):
        _require_positive("n_scenarios", self.n_scenarios)
        if self.monte_carlo_pool_size < self.n_scenarios:
            raise DataValidationError(
                "monte_carlo_pool_size must be >= n_scenarios for fast-forward selection"
            )
        if self.wind_price_correlation >= 0:
            raise DataValidationError(
                "wind_price_correlation must be < 0 (Eq. 47, negative wind-price correlation)"
            )
        _require_range("solar_cf_low", self.solar_cf_low, 0.0, 1.0)
        _require_range("solar_cf_high", self.solar_cf_high, 0.0, 1.0)


@dataclass
class ModelConfig:
    """Top-level configuration bundle passed into ``model_builder.build_model``."""

    investment: InvestmentParameters = field(default_factory=InvestmentParameters)
    electrolyser: ElectrolyserParameters = field(default_factory=ElectrolyserParameters)
    grid: GridParameters = field(default_factory=GridParameters)
    storage: StorageParameters = field(default_factory=StorageParameters)
    scenarios: ScenarioParameters = field(default_factory=ScenarioParameters)
    solver_name: str = "gurobi"           # "gurobi" / "cplex" in production, "cbc" for testing
    solver_time_limit_s: int = 7200    # 2 h per Section 17 implementation notes
    mip_gap: float = 0.01

    @classmethod
    def from_file(cls, path: str | Path) -> "ModelConfig":
        """Load configuration overrides from a JSON file, falling back
        to paper defaults for any field not present."""
        path = Path(path)
        if not path.exists():
            raise DataValidationError(f"Config file not found: {path}")
        try:
            raw: Dict[str, Any] = json.loads(path.read_text())
        except json.JSONDecodeError as err:
            raise DataValidationError(f"Malformed JSON config: {path}") from err

        cfg = cls()
        for section_name in ("investment", "electrolyser", "grid", "storage", "scenarios"):
            if section_name in raw:
                current = getattr(cfg, section_name)
                merged = {**asdict(current), **raw[section_name]}
                setattr(cfg, section_name, type(current)(**merged))
        for scalar in ("solver_name", "solver_time_limit_s", "mip_gap"):
            if scalar in raw:
                setattr(cfg, scalar, raw[scalar])
        logger.info("Loaded model configuration from %s", path)
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return {
            "investment": asdict(self.investment),
            "electrolyser": asdict(self.electrolyser),
            "grid": asdict(self.grid),
            "storage": asdict(self.storage),
            "scenarios": asdict(self.scenarios),
            "solver_name": self.solver_name,
            "solver_time_limit_s": self.solver_time_limit_s,
            "mip_gap": self.mip_gap,
        }
