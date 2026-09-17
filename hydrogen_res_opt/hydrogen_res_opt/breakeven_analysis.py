"""
breakeven_analysis.py
======================
Implements Section 12 (Break-Even Cable Distance Analysis) and the
"LCOH surface" result from the Expected Results table.

Break-even distance (Eqs. 41-42)
---------------------------------
Parametrically re-solves the MILP for a sweep of remote distances
L_remote in {5,10,...,50} km (fixing i=remote site, L_ij=L_remote),
and separately for the collocated sites, to find L* such that
LCOH_remote(L*) = LCOH_colloc.

LCOH surface
-------------
LCOH as a function of substation (FRN vs YRJ) and cable length --
implemented as a 2-D sweep reusing the same re-solve machinery.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import ModelConfig
from .exceptions import DataValidationError
from .logging_config import get_logger
from .model_builder import build_model
from .postprocessing import compute_lcoh
from .scenario_generation import ScenarioTree
from .sets_indices import ModelSets
from .solver_runner import solve

logger = get_logger(__name__)

DEFAULT_DISTANCE_SWEEP_KM: List[float] = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]


def _resolve_at_distance(
    sets: ModelSets,
    config: ModelConfig,
    scenario_tree: ScenarioTree,
    remote_site: str,
    substation: str,
    distance_km: float,
    solver_kwargs: Optional[dict] = None,
) -> float:
    """Re-solves the MILP with L[remote_site, substation] overridden to
    ``distance_km`` and that corridor's connection forced active, then
    returns the resulting LCOH."""
    solver_kwargs = solver_kwargs or {}
    new_distance_matrix = dict(sets.distance_matrix)
    new_distance_matrix[(remote_site, substation)] = distance_km
    swept_sets = replace(sets, distance_matrix=new_distance_matrix)

    build = build_model(
        swept_sets, config, scenario_tree,
        fix_first_stage={"x": {(remote_site, substation): 1}},
    )
    outcome = solve(build, **solver_kwargs)
    return compute_lcoh(outcome).lcoh_eur_per_kg


def breakeven_distance(
    sets: ModelSets,
    config: ModelConfig,
    scenario_tree: ScenarioTree,
    remote_site: str,
    collocated_site: str,
    substation: str,
    distance_sweep_km: Sequence[float] = DEFAULT_DISTANCE_SWEEP_KM,
    solver_kwargs: Optional[dict] = None,
) -> Dict[str, object]:
    """Section 12, Eqs. (41)-(42).

    Returns a dict with the swept LCOH_remote(L) curve, the (constant)
    LCOH_colloc, and the interpolated break-even distance L* (None if
    the sweep range never crosses LCOH_colloc).
    """
    if remote_site not in sets.remote_sites():
        raise DataValidationError(f"'{remote_site}' is not in I_remote")
    if collocated_site not in sets.collocated_sites():
        raise DataValidationError(f"'{collocated_site}' is not in I_colloc")

    logger.info("Break-even sweep: remote_site=%s substation=%s, %d distances",
                remote_site, substation, len(distance_sweep_km))

    lcoh_remote = []
    for dist in distance_sweep_km:
        lcoh = _resolve_at_distance(
            sets, config, scenario_tree, remote_site, substation, dist, solver_kwargs
        )
        lcoh_remote.append(lcoh)
        logger.info("  L_remote=%5.1f km -> LCOH=%.4f EUR/kg", dist, lcoh)

    logger.info("Solving collocated baseline (%s @ %s)", collocated_site, substation)
    colloc_build = build_model(
        sets, config, scenario_tree,
        fix_first_stage={"x": {(collocated_site, substation): 1}},
    )
    colloc_outcome = solve(colloc_build, **(solver_kwargs or {}))
    lcoh_colloc = compute_lcoh(colloc_outcome).lcoh_eur_per_kg

    breakeven_km = _interpolate_crossing(
        list(distance_sweep_km), lcoh_remote, lcoh_colloc
    )
    if breakeven_km is None:
        logger.warning(
            "No break-even distance found within sweep range [%s, %s] km "
            "(LCOH_remote never crosses LCOH_colloc=%.4f)",
            distance_sweep_km[0], distance_sweep_km[-1], lcoh_colloc,
        )
    else:
        logger.info("Break-even distance L* = %.2f km", breakeven_km)

    df = pd.DataFrame({
        "distance_km": list(distance_sweep_km),
        "lcoh_remote_eur_per_kg": lcoh_remote,
        "lcoh_colloc_eur_per_kg": [lcoh_colloc] * len(distance_sweep_km),
    })

    return {
        "curve": df,
        "lcoh_colloc_eur_per_kg": lcoh_colloc,
        "breakeven_distance_km": breakeven_km,
    }


def _interpolate_crossing(
    x: List[float], y_remote: List[float], y_flat: float
) -> Optional[float]:
    """Linear interpolation for the first x where y_remote crosses
    y_flat (LCOH_remote(L*) = LCOH_colloc, Eq. 42)."""
    diffs = [yr - y_flat for yr in y_remote]
    for idx in range(len(diffs) - 1):
        if diffs[idx] == 0:
            return x[idx]
        if diffs[idx] * diffs[idx + 1] < 0:
            x0, x1 = x[idx], x[idx + 1]
            d0, d1 = diffs[idx], diffs[idx + 1]
            frac = -d0 / (d1 - d0)
            return x0 + frac * (x1 - x0)
    if diffs and diffs[-1] == 0:
        return x[-1]
    return None


def lcoh_surface(
    sets: ModelSets,
    config: ModelConfig,
    scenario_tree: ScenarioTree,
    remote_site: str,
    distance_sweep_km: Sequence[float] = DEFAULT_DISTANCE_SWEEP_KM,
    solver_kwargs: Optional[dict] = None,
) -> pd.DataFrame:
    """'LCOH surface': LCOH as a function of substation (FRN vs. YRJ)
    and cable length -- a 2-D sweep over (substation, distance)."""
    rows = []
    for substation in sets.substations:
        for dist in distance_sweep_km:
            lcoh = _resolve_at_distance(
                sets, config, scenario_tree, remote_site, substation, dist, solver_kwargs
            )
            rows.append({"substation": substation, "distance_km": dist, "lcoh_eur_per_kg": lcoh})
            logger.info("  substation=%s L=%5.1f km -> LCOH=%.4f EUR/kg", substation, dist, lcoh)
    return pd.DataFrame(rows)
