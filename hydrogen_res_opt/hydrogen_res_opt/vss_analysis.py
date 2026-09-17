"""
vss_analysis.py
================
Implements Section 13 (Value of the Stochastic Solution) end to end:

    1. Solve the full stochastic model -> LCOH*_stoch          (Eq. 39)
    2. Collapse to expected-value parameters (Eqs. 43-44),
       solve the deterministic single-scenario model -> LCOH*_det
    3. VSS = LCOH*_det - LCOH*_stoch                            (Eq. 45)

Also implements the value-of-storage check mentioned alongside it:

    VoStorage = LCOH_no_storage - LCOH_10t_storage              (Eq. 46)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from .config import ModelConfig, StorageParameters
from .logging_config import get_logger
from .model_builder import build_model
from .postprocessing import compute_lcoh
from .scenario_generation import ScenarioTree
from .sets_indices import ModelSets
from .solver_runner import solve

logger = get_logger(__name__)


@dataclass
class VSSResult:
    lcoh_stochastic_eur_per_kg: float
    lcoh_deterministic_eur_per_kg: float
    vss_eur_per_kg: float

    @property
    def stochastic_modeling_justified(self) -> bool:
        """VSS > 0 (materially) justifies the added modeling complexity
        of the two-stage stochastic formulation over a deterministic
        expected-value approximation (Section 13)."""
        return self.vss_eur_per_kg > 1e-6


def compute_vss(
    sets: ModelSets,
    config: ModelConfig,
    scenario_tree: ScenarioTree,
    solver_kwargs: Optional[dict] = None,
) -> VSSResult:
    """Runs steps 1-3 of Section 13 and returns the VSS."""
    solver_kwargs = solver_kwargs or {}

    logger.info("VSS step 1/2: solving the full stochastic model (|Omega|=%d)",
                scenario_tree.price.shape[0])
    stoch_build = build_model(sets, config, scenario_tree)
    stoch_outcome = solve(stoch_build, **solver_kwargs)
    lcoh_stoch = compute_lcoh(stoch_outcome).lcoh_eur_per_kg

    logger.info("VSS step 2/2: solving the deterministic expected-value model")
    ev_tree = scenario_tree.expected_value_scenario()
    det_sets = replace(sets, n_scenarios=1)
    det_build = build_model(det_sets, config, ev_tree)
    det_outcome = solve(det_build, **solver_kwargs)
    lcoh_det = compute_lcoh(det_outcome).lcoh_eur_per_kg

    vss = lcoh_det - lcoh_stoch
    logger.info(
        "VSS = %.4f EUR/kg (det=%.4f, stoch=%.4f) -> stochastic modeling %s",
        vss, lcoh_det, lcoh_stoch,
        "justified" if vss > 1e-6 else "NOT clearly justified",
    )
    return VSSResult(
        lcoh_stochastic_eur_per_kg=lcoh_stoch,
        lcoh_deterministic_eur_per_kg=lcoh_det,
        vss_eur_per_kg=vss,
    )


@dataclass
class ValueOfStorageResult:
    lcoh_no_storage_eur_per_kg: float
    lcoh_with_storage_eur_per_kg: float
    value_of_storage_eur_per_kg: float


def compute_value_of_storage(
    sets: ModelSets,
    config: ModelConfig,
    scenario_tree: ScenarioTree,
    solver_kwargs: Optional[dict] = None,
) -> ValueOfStorageResult:
    """Eq. (46): VoStorage = LCOH_no_storage - LCOH_10t_storage.

    Re-solves with S_max = 0 (storage removed) and compares against the
    already-configured storage capacity (typically the 10-tonne
    HySynergy buffer, Table 9).
    """
    solver_kwargs = solver_kwargs or {}

    logger.info("VoStorage: solving with configured storage (S_max=%.1f t)",
                config.storage.s_max_tonnes)
    with_build = build_model(sets, config, scenario_tree)
    with_outcome = solve(with_build, **solver_kwargs)
    lcoh_with = compute_lcoh(with_outcome).lcoh_eur_per_kg

    logger.info("VoStorage: solving with S_max=0 (no storage)")
    no_storage_cfg = replace(
        config, storage=replace(config.storage, s_max_tonnes=1e-6)
    )
    no_build = build_model(sets, no_storage_cfg, scenario_tree)
    no_outcome = solve(no_build, **solver_kwargs)
    lcoh_no = compute_lcoh(no_outcome).lcoh_eur_per_kg

    value = lcoh_no - lcoh_with
    logger.info(
        "VoStorage = %.4f EUR/kg (no_storage=%.4f, with_storage=%.4f)",
        value, lcoh_no, lcoh_with,
    )
    return ValueOfStorageResult(
        lcoh_no_storage_eur_per_kg=lcoh_no,
        lcoh_with_storage_eur_per_kg=lcoh_with,
        value_of_storage_eur_per_kg=value,
    )
