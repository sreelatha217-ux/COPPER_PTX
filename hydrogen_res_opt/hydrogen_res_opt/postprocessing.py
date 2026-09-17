"""
postprocessing.py
==================
Extracts solved Pyomo variable values into pandas structures matching
the paper's "Expected Results Structure" table:

    Result                    -> function
    ------------------------------------------------------------
    Optimal RES mix per stage -> extract_res_mix()
    LCOH surface              -> (see breakeven_analysis.lcoh_surface)
    Break-even cable distance -> (see breakeven_analysis.py)
    Grid impact per stage     -> extract_grid_impact()
    Value of stochastic sol.  -> (see vss_analysis.py)
    Scenario sensitivity      -> scenario_sensitivity()
    Storage adequacy          -> storage_adequacy()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from .exceptions import HydrogenOptError
from .logging_config import get_logger
from .solver_runner import SolveOutcome

logger = get_logger(__name__)

try:
    import pyomo.environ as pyo
except ImportError:  # pragma: no cover
    pyo = None


def _val(x) -> float:
    """Safely extract a float from a Pyomo variable/expression, tolerant
    of unfixed/None values (returns nan instead of raising)."""
    try:
        v = pyo.value(x, exception=False)
        return float(v) if v is not None else float("nan")
    except Exception:  # noqa: BLE001
        return float("nan")


@dataclass
class LCOHResult:
    total_annualized_cost_eur: float
    annual_h2_target_tonnes: float
    lcoh_eur_per_kg: float


def compute_lcoh(outcome: SolveOutcome) -> LCOHResult:
    """Eq. (39): LCOH* = TC* / D  [EUR/kg H2] (D expressed in kg)."""
    m = outcome.build.model
    tc = outcome.objective_value
    if tc is None or np.isnan(tc):
        raise HydrogenOptError("Cannot compute LCOH: objective value is missing/NaN")
    d_kg = (
        outcome.build.config.electrolyser.annual_target_tonnes
        * outcome.build.config.annualization_factor
        * 1000.0
    )
    lcoh = tc / d_kg
    logger.info("LCOH = %.4f EUR/kg (TC=%.2f EUR, D=%.1f t/yr)", lcoh, tc, d_kg / 1000.0)
    return LCOHResult(total_annualized_cost_eur=tc, annual_h2_target_tonnes=d_kg / 1000.0,
                       lcoh_eur_per_kg=lcoh)


def extract_res_mix(outcome: SolveOutcome) -> pd.DataFrame:
    """'Optimal RES mix per stage': MW of wind vs solar, selected
    substation, cable type."""
    m = outcome.build.model
    rows = []
    p_w, p_s = _val(m.P_res_w), _val(m.P_res_s)
    for i in m.I:
        for j in m.J:
            if _val(m.x[i, j]) > 0.5:
                cable = next((k2 for k2 in m.K2 if _val(m.y[i, j, k2]) > 0.5), None)
                collector = None
                if i in m.I_remote:
                    collector = next(
                        (k1 for k1 in m.K1 if _val(m.z1[i, j, k1]) > 0.5), None
                    )
                rows.append({
                    "res_site": i,
                    "substation": j,
                    "connection_cable": cable,
                    "collector_cable": collector,
                    "wind_mw": p_w,
                    "solar_mw": p_s,
                    "distance_km": outcome.build.sets.distance_matrix[(i, j)],
                })
    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("extract_res_mix: no active (x[i,j]=1) connections found in the solution.")
    return df


def extract_grid_impact(outcome: SolveOutcome) -> Dict[str, pd.DataFrame]:
    """'Grid impact per stage': line loading %, transformer utilization,
    voltage profiles, averaged/maxed across scenarios."""
    m = outcome.build.model
    line_rows, xf_rows, volt_rows = [], [], []

    for i in m.I:
        for j in m.J:
            loadings = []
            for w in m.OMEGA:
                for t in m.T:
                    p_flow = _val(m.P_flow[i, j, w, t])
                    if not np.isnan(p_flow):
                        loadings.append(abs(p_flow) / max(m.F_max_new, 1e-6) * 100.0)
            if loadings:
                line_rows.append({
                    "res_site": i, "substation": j,
                    "mean_loading_pct": float(np.mean(loadings)),
                    "max_loading_pct": float(np.max(loadings)),
                })

    for j in m.J:
        loadings, volts = [], []
        for w in m.OMEGA:
            for t in m.T:
                inflow = sum(_val(m.P_flow[i, j, w, t]) for i in m.I) + _val(m.P_existing[j, w, t])
                if not np.isnan(inflow):
                    loadings.append(abs(inflow) / max(m.T_max, 1e-6) * 100.0)
                v_sq = _val(m.v[j, w, t])
                if not np.isnan(v_sq) and v_sq >= 0:
                    volts.append(np.sqrt(v_sq) / m.V_base)  # pu
        if loadings:
            xf_rows.append({
                "substation": j,
                "mean_transformer_utilization_pct": float(np.mean(loadings)),
                "max_transformer_utilization_pct": float(np.max(loadings)),
            })
        if volts:
            volt_rows.append({
                "substation": j,
                "mean_voltage_pu": float(np.mean(volts)),
                "min_voltage_pu": float(np.min(volts)),
                "max_voltage_pu": float(np.max(volts)),
            })

    return {
        "line_loading": pd.DataFrame(line_rows),
        "transformer_utilization": pd.DataFrame(xf_rows),
        "voltage_profile": pd.DataFrame(volt_rows),
    }


def storage_adequacy(outcome: SolveOutcome, binding_tol: float = 0.01) -> pd.DataFrame:
    """'Storage adequacy': whether the S_max buffer is binding, i.e.
    whether the storage-level trajectory touches its upper bound in any
    scenario (would indicate S_max should be increased for this stage)."""
    m = outcome.build.model
    s_max = outcome.build.config.storage.s_max_tonnes
    rows = []
    for w in m.OMEGA:
        levels = np.array([_val(m.s[w, t]) for t in m.T])
        levels = levels[~np.isnan(levels)]
        if levels.size == 0:
            continue
        max_level = float(levels.max())
        min_level = float(levels.min())
        binding = (s_max - max_level) <= binding_tol * s_max
        rows.append({
            "scenario": w,
            "max_storage_tonnes": max_level,
            "min_storage_tonnes": min_level,
            "s_max_tonnes": s_max,
            "binding": bool(binding),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        frac_binding = df["binding"].mean()
        logger.info(
            "Storage buffer binding in %.1f%% of scenarios (S_max=%.1f t)",
            frac_binding * 100, s_max,
        )
    return df


def scenario_sensitivity(outcome: SolveOutcome) -> pd.DataFrame:
    """'Scenario sensitivity': per-scenario realized cost / grid-import
    cost, to show how LCOH would move under each individual price
    trajectory (a decomposition of the expectation in Eq. 38, not a
    re-solve)."""
    m = outcome.build.model
    rows = []
    for w in m.OMEGA:
        import_cost = sum(
            _val(m.price[w, t]) * _val(m.g[w, t]) * m.delta_t for t in m.T
        )
        h2_produced = sum(_val(m.q[w, t]) for t in m.T)
        rows.append({
            "scenario": w,
            "probability": _val(m.pi[w]),
            "import_cost_eur": import_cost,
            "h2_produced_tonnes": h2_produced,
            "avg_price_eur_per_mwh": float(np.nanmean(
                [_val(m.price[w, t]) for t in m.T]
            )),
        })
    return pd.DataFrame(rows)


def full_results_bundle(outcome: SolveOutcome) -> Dict[str, object]:
    """Convenience wrapper returning every table in the paper's
    'Expected Results Structure' that can be derived from a single
    solved outcome (RES mix, grid impact, storage adequacy, scenario
    sensitivity, LCOH). LCOH surface / break-even distance / VSS
    require multiple solves and live in their own modules."""
    return {
        "lcoh": compute_lcoh(outcome),
        "res_mix": extract_res_mix(outcome),
        "grid_impact": extract_grid_impact(outcome),
        "storage_adequacy": storage_adequacy(outcome),
        "scenario_sensitivity": scenario_sensitivity(outcome),
    }
