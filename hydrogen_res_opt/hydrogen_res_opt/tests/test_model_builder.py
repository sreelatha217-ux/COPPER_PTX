"""
Integration tests for model_builder.py / solver_runner.py / postprocessing.py.

These require Pyomo *and* a MILP solver (CBC by default) to actually
solve; both are skipped automatically (``pytest.importorskip`` /
``solver.available()`` check) in environments where they are not
installed, so this test module never breaks a plain ``pip install -e .``
CI job that only checks the pure-Python logic.
"""

from __future__ import annotations

import pytest

pyo = pytest.importorskip("pyomo.environ", reason="Pyomo not installed")

from hydrogen_res_opt.config import ElectrolyserParameters, ModelConfig, ScenarioParameters
from hydrogen_res_opt.model_builder import build_model
from hydrogen_res_opt.scenario_generation import ScenarioGenerator
from hydrogen_res_opt.sets_indices import ModelSets
from hydrogen_res_opt.solver_runner import _find_pulp_bundled_cbc, check_feasibility_preconditions, solve


def _cbc_is_available() -> bool:
    """Checks solver availability for tests.
    First checks if Gurobi is available and licensed,
    otherwise falls back to checking CBC.
    """
    # 1. Check Gurobi availability
    if pyo.SolverFactory("gurobi").available(exception_flag=False):
        return True

    # 2. Check system CBC availability
    if pyo.SolverFactory("cbc").available(exception_flag=False):
        return True

    # 3. Fallback check for PuLP's bundled CBC binary
    bundled = _find_pulp_bundled_cbc()
    return bundled is not None and pyo.SolverFactory("cbc", executable=bundled).available(
        exception_flag=False
    )


def _tiny_setup():
    """A deliberately tiny instance (2 sites, 2 substations, 24h, 2
    scenarios) so a full solve completes in well under a second."""
    sets = ModelSets(
        res_sites=["collocated_FRN", "remote1"],
        substations=["FRN", "YRJ"],
        collection_cables=["cable_33kV_A"],
        connection_cables=["cable_33kV_A"],
        n_periods=24,
        n_scenarios=2,
        distance_matrix={
            ("collocated_FRN", "FRN"): 0.0,
            ("collocated_FRN", "YRJ"): 15.0,
            ("remote1", "FRN"): 10.0,
            ("remote1", "YRJ"): 20.0,
        },
    )
    config = ModelConfig(
        electrolyser=ElectrolyserParameters(
            p_elec_min=0.5, p_elec_max=15.0, annual_target_tonnes=0.3
        ),
        scenarios=ScenarioParameters(n_scenarios=2, monte_carlo_pool_size=10, random_seed=7),
        solver_time_limit_s=30,
    )
    generator = ScenarioGenerator(sets, config.scenarios)
    tree = generator.synthetic()
    return sets, config, tree


def test_model_builds_without_error():
    sets, config, tree = _tiny_setup()
    build = build_model(sets, config, tree)
    assert build.model is not None
    assert len(list(build.model.I)) == 2
    assert len(list(build.model.OMEGA)) == 2


def test_feasibility_precheck_flags_undersized_transformer():
    sets, config, tree = _tiny_setup()
    from dataclasses import replace
    bad_config = replace(config, grid=replace(config.grid, transformer_thermal_limit_mva=0.1))
    build = build_model(sets, bad_config, tree)
    violations = check_feasibility_preconditions(build)
    assert any("F2" in v for v in violations)


@pytest.mark.skipif(
    not _cbc_is_available(),
    reason="CBC solver not available in this environment (checked PATH and pulp's bundled binary)",
)
def test_tiny_instance_solves_to_optimality():
    sets, config, tree = _tiny_setup()
    build = build_model(sets, config, tree)
    outcome = solve(build, max_retries=0)
    assert outcome.objective_value is not None
    assert outcome.objective_value >= 0

    from hydrogen_res_opt.postprocessing import compute_lcoh
    lcoh = compute_lcoh(outcome)
    assert lcoh.lcoh_eur_per_kg > 0
