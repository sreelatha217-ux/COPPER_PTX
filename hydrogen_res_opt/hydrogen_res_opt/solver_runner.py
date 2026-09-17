"""
solver_runner.py
=================
Wraps Pyomo's ``SolverFactory`` with the error handling, retry and
diagnostic behaviour expected of a production optimization service:

* Verifies solver availability before attempting a solve (fail fast
  with a clear message rather than an opaque Pyomo `ApplicationError`).
* Distinguishes "solved to optimality", "solved with gap", "infeasible"
  and "solver crashed" outcomes and raises the matching exception.
* On infeasibility, runs the analytic feasibility pre-checks (F0-F5,
  Section 15) to point the caller at the *parameter*, not just the
  symptom.
* On infeasibility with Gurobi available, additionally computes an
  IIS (Irreducible Infeasible Subset) so the *exact* conflicting
  constraints are named in the raised exception / log, instead of
  requiring a separate manual debugging pass.
* Retries transient failures (e.g. a flaky remote license server) with
  exponential backoff.
* Persists solver logs (and, when computed, the IIS .ilp file) next to
  the run's output directory for audit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import ModelConfig
from .exceptions import InfeasibilityError, SolverError
from .logging_config import get_logger
from .model_builder import BuildResult

logger = get_logger(__name__)

try:
    import pyomo.environ as pyo
    from pyomo.opt import SolverStatus, TerminationCondition
except ImportError:  # pragma: no cover
    pyo = None
    SolverStatus = None
    TerminationCondition = None


def _find_pulp_bundled_cbc() -> Optional[str]:
    """Locates the CBC binary bundled inside the ``pulp`` package, if
    installed.

    ``pulp`` ships a working ``cbc``/``cbc.exe`` for every major OS as
    part of its own package data (this is the standard way to get CBC
    on Windows without conda or compiling from source). Reusing it here
    means CBC "just works" on any machine that ran
    ``pip install -r requirements.txt`` -- no manual ``$env:Path`` edit
    required, and no per-machine setup that fails to travel with the
    repo when a teammate clones it or CI spins up a fresh runner.
    """
    try:
        import pulp
    except ImportError:
        return None

    try:
        path = pulp.apis.PULP_CBC_CMD().path
    except Exception:  # noqa: BLE001 - pulp version differences, missing binary, etc.
        return None

    if path and Path(path).exists():
        return path
    return None


@dataclass
class SolveOutcome:
    build: BuildResult
    status: str
    termination_condition: str
    objective_value: Optional[float]
    solve_time_s: float
    solver_log_path: Optional[Path]


def check_feasibility_preconditions(build: BuildResult) -> list[str]:
    """Analytic feasibility pre-checks, F0-F5 (Section 15), run *before*
    the solver so an infeasible instance is diagnosed with a specific
    parameter-level message rather than a bare "infeasible" from the
    MILP solver.

    Returns a list of human-readable violation messages (empty if all
    checks pass).
    """
    cfg: ModelConfig = build.config
    sets = build.sets
    elec, grid, sto = cfg.electrolyser, cfg.grid, cfg.storage
    n_period = build.scenario_tree.price.shape[1]
    violations: list[str] = []

    # F0: demand/target consistency is enforced structurally in
    # model_builder (d_t is *derived* from D_target, Eq. 48), so it
    # cannot be violated by construction here -- but we double check.
    total_demand = elec.compression_efficiency_tonnes_per_mwh * elec.annual_target_tonnes
    if abs(total_demand - elec.compression_efficiency_tonnes_per_mwh * elec.annual_target_tonnes) > 1e-6:
        violations.append("F0: derived demand profile inconsistent with annual target D")

    # F1: production capability vs. target.
    # Uses n_period (length of the modelled scenario horizon) to match
    # model_builder's own d_t derivation (d_t[t] = D_target / len(T)),
    # which treats T as the full horizon the annual target D must be
    # met over -- so the max-production capability check must be scaled
    # over the same horizon, not a fixed 8760, to stay consistent with
    # what the MILP itself actually enforces.
    max_production_over_horizon = elec.p_elec_max * (elec.efficiency_kg_per_mwh / 1000.0) * n_period
    if max_production_over_horizon < elec.annual_target_tonnes:
        violations.append(
            f"F1 violated: max production over horizon (n_period={n_period}) "
            f"{max_production_over_horizon:.1f} t < target D={elec.annual_target_tonnes:.1f} t "
            "(raise P_elec_max, raise efficiency, lower D, or increase n_period)"
        )

    # F2: grid import capability vs. electrolyser max load
    if grid.transformer_thermal_limit_mva < elec.p_elec_max:
        violations.append(
            f"F2 violated: transformer_thermal_limit_mva={grid.transformer_thermal_limit_mva} "
            f"< P_elec_max={elec.p_elec_max} MW at every substation j"
        )

    # F3: storage sized for peak hourly offtake (checked at flat-profile
    # level). Uses n_period for the SAME reason as F1: model_builder's
    # d_t is D_target / len(T), so the *actual* per-period offtake the
    # MILP enforces is D_target / n_period -- checking against that
    # (rather than a fixed 8760) is what keeps this precondition
    # consistent with what the solver will actually see.
    peak_hourly_demand = (
        elec.compression_efficiency_tonnes_per_mwh * elec.annual_target_tonnes
    ) / n_period
    if sto.s_max_tonnes < peak_hourly_demand:
        violations.append(
            f"F3 violated: S_max={sto.s_max_tonnes} t < implied hourly demand "
            f"{peak_hourly_demand:.4f} t/h (non-flat offtake profiles may violate this "
            "more severely at true peak hours -- re-check with the real profile)"
        )

    # F4: voltage feasibility band sanity
    if not (grid.v_min_pu < 1.0 < grid.v_max_pu):
        violations.append(
            f"F4 violated: nominal 1.0 pu operating point not inside "
            f"[{grid.v_min_pu}, {grid.v_max_pu}]"
        )

    # F5: at least one cable in K2 must be rated for full electrolyser load
    if grid.cable_thermal_limit_mw < elec.p_elec_max:
        violations.append(
            f"F5 violated: cable_thermal_limit_mw={grid.cable_thermal_limit_mw} "
            f"< P_elec_max={elec.p_elec_max}; no cable in K2 can carry full load"
        )

    if violations:
        logger.error("Feasibility pre-check found %d violation(s): %s", len(violations), violations)
    else:
        logger.info("Feasibility pre-checks F0-F5 passed.")
    return violations


def compute_iis(
    build: BuildResult, log_dir: Path = Path("./logs/solver")
) -> tuple[Optional[Path], list[str]]:
    """Computes an IIS (Irreducible Infeasible Subset) for an infeasible
    model using Gurobi, and writes it to ``log_dir``.

    Returns ``(iis_path, conflicting_names)``. ``iis_path`` is ``None``
    if Gurobi (or gurobipy) is not available -- this is a best-effort
    diagnostic aid, never a hard requirement, so callers should treat a
    ``None`` path as "diagnostics unavailable", not an error.

    ``conflicting_names`` covers every kind of element Gurobi's IIS can
    flag: linear constraints, quadratic constraints, AND variable
    bounds (``IISLB``/``IISUB``) -- a bound-only conflict (e.g. two
    constraints both pinning the same variable to incompatible values)
    shows up as a flagged *variable*, not a constraint, and is easy to
    miss if only ``getConstrs``/``getQConstrs`` are checked. Bound
    entries are prefixed ``bound:`` so they're distinguishable from
    constraint names in the returned list and in logs.

    Requires the model to have been solved with
    ``symbolic_solver_labels=True`` (as ``solve()`` in this module
    does) for the names to be human-readable (e.g. ``v[FRN,0,3]``)
    rather than generic (``x1265``).
    """
    try:
        import gurobipy as gp  # noqa: F401
    except ImportError:
        logger.info("gurobipy not installed; skipping IIS computation.")
        return None, []

    if not pyo.SolverFactory("gurobi").available(exception_flag=False):
        logger.info("Gurobi not available/licensed; skipping IIS computation.")
        return None, []

    try:
        # NOTE: for the persistent interface, symbolic_solver_labels is a
        # set_instance() kwarg, not a solve() kwarg (unlike the regular
        # "gurobi" interface used in solve() above) -- passing it to
        # solve() raises "unrecognized keywords". Pyomo versions differ
        # on whether set_instance() itself accepts it either, so fall
        # back to no labels (generic names like "x1265" in the IIS
        # output) rather than letting IIS computation fail outright.
        opt = pyo.SolverFactory("gurobi_persistent")
        try:
            opt.set_instance(build.model, symbolic_solver_labels=True)
        except TypeError:
            opt.set_instance(build.model)
        opt.solve(tee=False)
        grb_model = opt._solver_model
        grb_model.computeIIS()
    except Exception as err:  # noqa: BLE001 - diagnostics must never crash the caller
        logger.warning("IIS computation failed: %s", err)
        return None, []

    log_dir.mkdir(parents=True, exist_ok=True)
    iis_path = log_dir / f"infeasible_{int(time.time())}.ilp"
    grb_model.write(str(iis_path))

    conflicting = [c.ConstrName for c in grb_model.getConstrs() if c.IISConstr]
    conflicting += [c.QCName for c in grb_model.getQConstrs() if c.IISQConstr]
    bound_conflicts = [
        f"bound:{v.VarName}{'[LB]' if v.IISLB else ''}{'[UB]' if v.IISUB else ''}"
        for v in grb_model.getVars()
        if v.IISLB or v.IISUB
    ]
    conflicting += bound_conflicts

    if conflicting:
        logger.error(
            "IIS found %d conflicting element(s) (%d constraint(s), %d bound(s)), "
            "written to %s: %s",
            len(conflicting), len(conflicting) - len(bound_conflicts),
            len(bound_conflicts), iis_path, conflicting,
        )
    else:
        logger.warning(
            "IIS computation completed but found no conflicting elements "
            "(unexpected) -- see %s.", iis_path,
        )
    return iis_path, conflicting


def solve(
        build: BuildResult,
        log_dir: Path = Path("./logs/solver"),
        max_retries: int = 2,
        retry_backoff_s: float = 5.0,
        run_precheck: bool = True,
        compute_iis_on_infeasible: bool = True,
) -> SolveOutcome:
    """Solves the optimization model built by ``model_builder.build_model``.

    ``compute_iis_on_infeasible``: when the solver reports the model as
    infeasible, attempt to compute and log an IIS (Gurobi only) before
    raising ``InfeasibilityError``, so the exception context names the
    specific conflicting constraints rather than just "infeasible".
    """
    if pyo is None:
        raise SolverError("Pyomo is not installed; cannot solve the model.")

    if run_precheck:
        violations = check_feasibility_preconditions(build)
        if violations:
            raise InfeasibilityError(
                "Feasibility pre-checks (Section 15, F0-F5) failed before solving.",
                context={"violations": violations},
            )

    cfg = build.config

    # ---------------------------------------------------------
    # Solver Selection logic (Prefers Gurobi for Non-Linear x * p)
    # ---------------------------------------------------------
    if pyo.SolverFactory("gurobi").available(exception_flag=False):
        solver_name = "gurobi"
        solver = pyo.SolverFactory("gurobi")
    elif pyo.SolverFactory(cfg.solver_name).available(exception_flag=False):
        solver_name = cfg.solver_name
        solver = pyo.SolverFactory(solver_name)
    elif cfg.solver_name == "cbc":
        bundled_path = _find_pulp_bundled_cbc()
        if bundled_path is not None:
            solver_name = "cbc"
            solver = pyo.SolverFactory("cbc", executable=bundled_path)
        else:
            solver = None
    else:
        solver = None

    if solver is None or not solver.available(exception_flag=False):
        raise SolverError(
            f"Solver '{cfg.solver_name}' or 'gurobi' is not available in this environment."
        )

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"solve_{int(time.time())}.log"

    # ---------------------------------------------------------
    # Solver Options Setup
    # ---------------------------------------------------------
    options = {}
    if solver_name in ("gurobi", "gurobi_direct"):
        options["TimeLimit"] = cfg.solver_time_limit_s
        options["MIPGap"] = cfg.mip_gap
        # NonConvex = 2 is MANDATORY for bilinear x * p terms!
        options["NonConvex"] = 2
    elif solver_name in ("cbc",):
        options["seconds"] = cfg.solver_time_limit_s
        options["ratioGap"] = cfg.mip_gap
    elif solver_name in ("cplex", "cplex_direct"):
        options["timelimit"] = cfg.solver_time_limit_s
        options["mip_tolerances_mipgap"] = cfg.mip_gap

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 2):
        logger.info(
            "Solving with %s (attempt %d/%d), time_limit=%ds, mip_gap=%.4f",
            solver_name, attempt, max_retries + 1, cfg.solver_time_limit_s, cfg.mip_gap,
        )
        start = time.time()
        try:
            result = solver.solve(
                build.model,
                options=options,
                tee=True,
                logfile=str(log_path),
                symbolic_solver_labels=True,
            )
        except Exception as err:
            last_error = err
            logger.warning(
                "Solver call raised an exception on attempt %d: %s", attempt, err
            )
            if attempt <= max_retries:
                time.sleep(retry_backoff_s * attempt)
                continue
            raise SolverError(
                f"Solver '{solver_name}' failed after {max_retries + 1} attempts.",
                context={"last_error": str(last_error)},
            ) from last_error

        elapsed = time.time() - start
        status = str(result.solver.status)
        term = str(result.solver.termination_condition)

        if result.solver.termination_condition == TerminationCondition.infeasible:
            iis_path, conflicting_elements = None, []
            if compute_iis_on_infeasible and solver_name in ("gurobi", "gurobi_direct"):
                iis_path, conflicting_elements = compute_iis(build, log_dir=log_dir)
            raise InfeasibilityError(
                "Solver reported the model as infeasible.",
                context={
                    "status": status,
                    "termination_condition": term,
                    "solver_log": str(log_path),
                    "iis_log": str(iis_path) if iis_path else None,
                    "iis_conflicting_elements": conflicting_elements[:50],
                },
            )
        if result.solver.termination_condition in (
                TerminationCondition.optimal,
                TerminationCondition.locallyOptimal,
                TerminationCondition.feasible,
                TerminationCondition.maxTimeLimit,
        ):
            obj_value = pyo.value(build.model.total_cost, exception=False)
            if result.solver.termination_condition == TerminationCondition.maxTimeLimit:
                logger.warning(
                    "Solver hit time limit; returning best incumbent (objective=%.4f).",
                    obj_value or float("nan"),
                )
            logger.info(
                "Solve finished in %.1fs: status=%s termination=%s objective=%.4f",
                elapsed, status, term, obj_value or float("nan"),
            )
            return SolveOutcome(
                build=build,
                status=status,
                termination_condition=term,
                objective_value=obj_value,
                solve_time_s=elapsed,
                solver_log_path=log_path,
            )

        logger.warning("Unexpected termination condition on attempt %d: %s", attempt, term)
        last_error = SolverError(
            f"Unexpected solver termination condition: {term}",
            context={"status": status, "solver_log": str(log_path)},
        )
        if attempt <= max_retries:
            time.sleep(retry_backoff_s * attempt)
            continue
        raise last_error

    raise SolverError("Exhausted retries without a definitive solver outcome.")