"""
scenario_generation.py
=======================
Implements Section 14 (Scenario Generation).

Historical DK1 spot price, wind (Jammerbugt) and solar (ERA5, Fredericia
coords) series are the intended real inputs; this module also exposes a
synthetic generator (Cholesky-correlated Monte Carlo + fast-forward
reduction) so the optimization pipeline is runnable and testable without
requiring a live data feed, and so unit tests are deterministic.

Key equation implemented
-------------------------
Eq. (47):  Corr(lambda^omega_t, CF^omega_w,t) < 0
    (negative wind-price correlation, enforced via Cholesky decomposition)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .config import ScenarioParameters
from .exceptions import ScenarioGenerationError
from .logging_config import get_logger
from .sets_indices import ModelSets

logger = get_logger(__name__)


@dataclass
class ScenarioTree:
    """Reduced scenario tree ready to feed into the Pyomo model.

    Shapes: all arrays are [n_scenarios, n_periods].
    """

    price: np.ndarray          # lambda^omega_t   [EUR/MWh]
    wind_cf: np.ndarray        # CF^omega_w,t      [-]
    solar_cf: np.ndarray       # CF^omega_s,t      [-]
    probabilities: np.ndarray  # pi^omega          [-], sums to 1

    def validate(self) -> None:
        n_scen, n_period = self.price.shape
        for name, arr in (("wind_cf", self.wind_cf), ("solar_cf", self.solar_cf)):
            if arr.shape != (n_scen, n_period):
                raise ScenarioGenerationError(
                    f"Shape mismatch: {name}.shape={arr.shape} != price.shape={self.price.shape}"
                )
        if self.probabilities.shape != (n_scen,):
            raise ScenarioGenerationError(
                f"probabilities shape {self.probabilities.shape} != ({n_scen},)"
            )
        if not np.isclose(self.probabilities.sum(), 1.0, atol=1e-6):
            raise ScenarioGenerationError(
                f"Scenario probabilities sum to {self.probabilities.sum():.6f}, expected 1.0"
            )
        if np.any(self.wind_cf < 0) or np.any(self.wind_cf > 1):
            raise ScenarioGenerationError("wind_cf must lie in [0, 1]")
        if np.any(self.solar_cf < 0) or np.any(self.solar_cf > 1):
            raise ScenarioGenerationError("solar_cf must lie in [0, 1]")
        if np.any(self.price < 0):
            logger.warning(
                "Negative DK1 spot prices present in scenario tree "
                "(physically plausible for DK1, but check inputs if unexpected)."
            )

    def expected_value_scenario(self) -> "ScenarioTree":
        """Collapses the tree to a single expected-value scenario,
        used by the VSS analysis (Section 13, Eqs. 43-44)."""
        price_ev = np.average(self.price, axis=0, weights=self.probabilities)
        wind_ev = np.average(self.wind_cf, axis=0, weights=self.probabilities)
        solar_ev = np.average(self.solar_cf, axis=0, weights=self.probabilities)
        return ScenarioTree(
            price=price_ev.reshape(1, -1),
            wind_cf=wind_ev.reshape(1, -1),
            solar_cf=solar_ev.reshape(1, -1),
            probabilities=np.array([1.0]),
        )


class ScenarioGenerator:
    """Generates (or ingests) joint price/CF scenarios.

    Two code paths:
    * ``from_historical_data`` -- the intended production path; ingests
      user-supplied historical arrays (Energinet/Center Denmark prices,
      Energinet wind, ERA5 solar) and performs fast-forward selection
      down to ``n_scenarios``.
    * ``synthetic`` -- Cholesky-correlated Monte Carlo generator used
      for development/testing when historical series are not attached.
    """

    def __init__(self, sets: ModelSets, params: ScenarioParameters):
        self.sets = sets
        self.params = params
        self._rng = np.random.default_rng(params.random_seed)

    # ------------------------------------------------------------------
    def synthetic(self) -> ScenarioTree:
        """Generates a Cholesky-correlated Monte Carlo pool and reduces
        it to ``n_scenarios`` via fast-forward selection.

        Enforces Eq. (47): Corr(price, wind_cf) < 0.
        """
        n_period = self.sets.n_periods
        pool_size = self.params.monte_carlo_pool_size
        rho = self.params.wind_price_correlation

        logger.info(
            "Generating synthetic scenario pool: pool_size=%d, n_periods=%d, "
            "target n_scenarios=%d, wind-price corr=%.2f",
            pool_size, n_period, self.params.n_scenarios, rho,
        )

        try:
            corr = np.array([[1.0, rho], [rho, 1.0]])
            chol = np.linalg.cholesky(corr)
        except np.linalg.LinAlgError as err:
            raise ScenarioGenerationError(
                "Correlation matrix is not positive-definite; "
                "wind_price_correlation must lie in (-1, 1)"
            ) from err

        price_pool = np.empty((pool_size, n_period))
        wind_pool = np.empty((pool_size, n_period))
        solar_pool = np.empty((pool_size, n_period))

        price_mu, price_sigma = self.params.price_mean_eur_per_mwh, 20.0
        wind_mu, wind_sigma = self.params.wind_cf_mean, 0.18
        solar_lo, solar_hi = self.params.solar_cf_low, self.params.solar_cf_high

        for s in range(pool_size):
            z = self._rng.standard_normal((n_period, 2))
            correlated = z @ chol.T
            price_pool[s] = np.clip(price_mu + price_sigma * correlated[:, 0], -50.0, 400.0)
            wind_pool[s] = np.clip(wind_mu + wind_sigma * correlated[:, 1], 0.0, 1.0)
            solar_pool[s] = np.clip(
                self._rng.uniform(solar_lo, solar_hi, size=n_period)
                * self._diurnal_solar_shape(n_period),
                0.0,
                1.0,
            )

        selected_idx, weights = self._fast_forward_selection(
            price_pool, self.params.n_scenarios
        )

        tree = ScenarioTree(
            price=price_pool[selected_idx],
            wind_cf=wind_pool[selected_idx],
            solar_cf=solar_pool[selected_idx],
            probabilities=weights,
        )
        tree.validate()

        empirical_corr = np.corrcoef(tree.price.ravel(), tree.wind_cf.ravel())[0, 1]
        if empirical_corr >= 0:
            logger.warning(
                "Empirical wind-price correlation is non-negative (%.3f); "
                "Eq. (47) intent may not hold on the reduced tree.", empirical_corr
            )
        else:
            logger.info("Empirical wind-price correlation on reduced tree: %.3f", empirical_corr)

        return tree

    # ------------------------------------------------------------------
    @staticmethod
    def _diurnal_solar_shape(n_period: int) -> np.ndarray:
        """Simple bell-shaped diurnal solar availability mask (0 at
        night, peak at midday) repeated over the horizon."""
        hours = np.arange(n_period) % 24
        shape = np.clip(np.sin(np.pi * (hours - 6) / 12), 0, None)
        return shape

    # ------------------------------------------------------------------
    def _fast_forward_selection(
        self, price_pool: np.ndarray, n_scenarios: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simplified fast-forward scenario reduction (Heitsch & Romisch,
        2003 [8]): greedily selects the scenario whose removal induces
        minimal probability-weighted Kantorovich distance to the rest of
        the pool, re-weighting survivors at each step.

        This is a tractable approximation suitable for pipeline testing;
        production runs should substitute a validated reduction library
        if higher fidelity is required.
        """
        pool_size = price_pool.shape[0]
        if n_scenarios >= pool_size:
            raise ScenarioGenerationError(
                f"n_scenarios ({n_scenarios}) must be < monte_carlo_pool_size ({pool_size})"
            )

        remaining = list(range(pool_size))
        weights = np.ones(pool_size) / pool_size

        # Precompute pairwise L2 distances between scenario price paths.
        #dist = np.linalg.norm(
        #    price_pool[:, None, :] - price_pool[None, :, :], axis=2
        #)
        # Precompute pairwise L2 distances between scenario price paths
        # without constructing a (pool_size, pool_size, n_period) array.
        #
        # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2*x_i.x_j

        price_pool = np.asarray(price_pool, dtype=np.float64)

        squared_norms = np.sum(price_pool * price_pool, axis=1)

        dist_squared = (
                squared_norms[:, None]
                + squared_norms[None, :]
                - 2.0 * (price_pool @ price_pool.T)
        )

        # Numerical round-off can produce tiny negative values.
        dist_squared = np.maximum(dist_squared, 0.0)

        dist = np.sqrt(dist_squared)

        selected: List[int] = []
        while len(remaining) > n_scenarios:
            # For each candidate to remove, find its nearest remaining
            # neighbour distance weighted by probability mass.
            best_remove, best_cost = None, np.inf
            for idx in remaining:
                others = [j for j in remaining if j != idx]
                nearest = min(others, key=lambda j: dist[idx, j])
                cost = weights[idx] * dist[idx, nearest]
                if cost < best_cost:
                    best_cost, best_remove, nearest_of_best = cost, idx, nearest
            weights[nearest_of_best] += weights[best_remove]
            weights[best_remove] = 0.0
            remaining.remove(best_remove)

        selected = sorted(remaining)
        final_weights = weights[selected]
        final_weights = final_weights / final_weights.sum()
        return np.array(selected), final_weights

    # ------------------------------------------------------------------
    def from_historical_data(
        self,
        price_series: np.ndarray,
        wind_cf_series: np.ndarray,
        solar_cf_series: np.ndarray,
    ) -> ScenarioTree:
        """Production path: ingest historical [n_years*8760] series and
        reduce to an |Omega|-scenario tree.

        Parameters mirror Section 14: Energinet/Center Denmark DK1
        prices (2021-2024), Energinet wind, ERA5 solar for Fredericia.
        """
        for name, series in (
            ("price_series", price_series),
            ("wind_cf_series", wind_cf_series),
            ("solar_cf_series", solar_cf_series),
        ):
            if series.ndim != 2:
                raise ScenarioGenerationError(
                    f"{name} must be 2-D [n_historical_scenarios, n_periods], "
                    f"got shape {series.shape}"
                )
        if not (price_series.shape == wind_cf_series.shape == solar_cf_series.shape):
            raise ScenarioGenerationError("Historical series shapes must match")

        selected_idx, weights = self._fast_forward_selection(
            price_series, self.params.n_scenarios
        )
        tree = ScenarioTree(
            price=price_series[selected_idx],
            wind_cf=wind_cf_series[selected_idx],
            solar_cf=solar_cf_series[selected_idx],
            probabilities=weights,
        )
        tree.validate()
        return tree

    # ------------------------------------------------------------------
    def convergence_check(self, candidate_sizes: Optional[List[int]] = None) -> Dict[int, float]:
        """Section 14: 're-solving at |Omega| in {20,30,50,100} and
        confirming LCOH stability'. This helper only performs the
        scenario-generation half (the solve/LCOH half lives in
        pipeline.py, which calls this then re-solves the MILP per size).
        Returns the empirical price-mean per candidate size as a cheap
        stability proxy ahead of a full re-solve.
        """
        candidate_sizes = candidate_sizes or [20, 30, 50, 100]
        results = {}
        for size in candidate_sizes:
            probe_params = ScenarioParameters(
                n_scenarios=size,
                monte_carlo_pool_size=max(self.params.monte_carlo_pool_size, size * 2),
                random_seed=self.params.random_seed,
            )
            probe_gen = ScenarioGenerator(self.sets, probe_params)
            tree = probe_gen.synthetic()
            results[size] = float(np.average(tree.price.mean(axis=1), weights=tree.probabilities))
        return results
