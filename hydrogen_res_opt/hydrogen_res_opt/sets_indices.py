"""
sets_indices.py
================
Implements Table 1 (Sets and Indices) of the paper, plus the
remote/collocated indicator delta_i (Section 4 / C4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .exceptions import DataValidationError
from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ModelSets:
    """Container + validator for the MILP's index sets.

    Parameters
    ----------
    res_sites:
        I = {collocated_FRN, collocated_YRJ, remote1, remote2, ...}
    substations:
        J = {FRN, YRJ, ...}
    collection_cables:
        K1, catalog for RES -> PCC corridors (remote sites only)
    connection_cables:
        K2, catalog for PCC -> substation / collocated corridors
    n_periods:
        |T|, defaults to 8760 (one full year, hourly resolution)
    n_scenarios:
        |Omega|
    distance_matrix:
        L_ij [km], keyed (site, substation). Collocated sites must
        have L_ij = 0 for their own substation (Remark 1).
    """

    res_sites: List[str] = field(
        default_factory=lambda: ["collocated_FRN", "collocated_YRJ", "remote1", "remote2"]
    )
    substations: List[str] = field(default_factory=lambda: ["FRN", "YRJ"])
    collection_cables: List[str] = field(
        default_factory=lambda: ["cable_33kV_A", "cable_33kV_B"]
    )
    connection_cables: List[str] = field(
        default_factory=lambda: ["cable_33kV_A", "cable_33kV_B", "cable_132kV_A"]
    )
    n_periods: int = 24  #8760
    n_scenarios: int = 2   #50
    distance_matrix: Dict[Tuple[str, str], float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.res_sites:
            raise DataValidationError("res_sites (set I) must be non-empty")
        if not self.substations:
            raise DataValidationError("substations (set J) must be non-empty")
        if not self.collection_cables:
            raise DataValidationError("collection_cables (set K1) must be non-empty")
        if not self.connection_cables:
            raise DataValidationError("connection_cables (set K2) must be non-empty")

        if not self.distance_matrix:
            self.distance_matrix = self._default_distance_matrix()
        self._validate_distance_matrix()

    # ------------------------------------------------------------------
    def _default_distance_matrix(self) -> Dict[Tuple[str, str], float]:
        """Builds a plausible default L_ij: 0 km for a site's own
        substation if collocated, else a nominal remote distance."""
        matrix: Dict[Tuple[str, str], float] = {}
        for i in self.res_sites:
            for j in self.substations:
                if i.startswith("collocated") and i.endswith(j):
                    matrix[(i, j)] = 0.0
                elif i.startswith("collocated"):
                    # collocated to a *different* substation: treat as remote
                    matrix[(i, j)] = 15.0
                else:
                    matrix[(i, j)] = 20.0
        return matrix

    def _validate_distance_matrix(self) -> None:
        missing = [
            (i, j)
            for i in self.res_sites
            for j in self.substations
            if (i, j) not in self.distance_matrix
        ]
        if missing:
            raise DataValidationError(
                "distance_matrix is missing entries for site/substation pairs",
                context={"missing_pairs": missing},
            )
        for (i, j), dist in self.distance_matrix.items():
            if dist < 0:
                raise DataValidationError(
                    f"Negative distance L[{i},{j}] = {dist} km is not physical"
                )
            if i.startswith("collocated") and i.endswith(j) and dist != 0:
                logger.warning(
                    "Collocated site %s has nonzero distance %.2f km to its own "
                    "substation %s (Remark 1 expects 0 km).", i, dist, j
                )

    # ------------------------------------------------------------------
    def is_remote(self, site: str) -> bool:
        """delta_i indicator (Section 4 / C4): True for i in I_remote."""
        return not site.startswith("collocated")

    def remote_sites(self) -> List[str]:
        return [i for i in self.res_sites if self.is_remote(i)]

    def collocated_sites(self) -> List[str]:
        return [i for i in self.res_sites if not self.is_remote(i)]

    def scenario_ids(self) -> List[int]:
        return list(range(self.n_scenarios))

    def period_ids(self) -> List[int]:
        return list(range(self.n_periods))
