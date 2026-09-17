"""
pipeline.py
===========
End-to-end orchestration, wiring every module together into the
sequence needed to reproduce the paper's "Expected Results Structure":

    1. Load config + sets                     (config.py, sets_indices.py)
    2. Generate/reduce scenarios                (scenario_generation.py)
    3. Solve the full stochastic MILP            (model_builder.py, solver_runner.py)
    4. Extract RES mix / grid impact / storage / scenario tables
                                                  (postprocessing.py)
    5. Compute VSS and value-of-storage          (vss_analysis.py)
    6. Sweep break-even distance / LCOH surface  (breakeven_analysis.py)
    7. (Optional) Validate the chosen design on the real PowerFactory
       network model                            (powerfactory_interface.py)

Each stage is independently callable (Agile "thin vertical slice"
philosophy): a developer can run stage 3 alone in a notebook while
stage 6 is still being built, and the CLI (`cli.py`) exposes each
stage as its own subcommand for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from .breakeven_analysis import breakeven_distance, lcoh_surface
from .config import ModelConfig
from .exceptions import HydrogenOptError
from .logging_config import configure_logging, get_logger
from .model_builder import build_model
from .postprocessing import full_results_bundle
from .powerfactory_interface import PowerFactoryConfig, PowerFactoryLoadFlow
from .scenario_generation import ScenarioGenerator, ScenarioTree
from .sets_indices import ModelSets
from .solver_runner import solve
from .vss_analysis import compute_value_of_storage, compute_vss
import time
from pyomo.util.model_size import build_model_size_report

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    outcome: object
    results: Dict[str, object]
    vss: Optional[object] = None
    value_of_storage: Optional[object] = None
    breakeven: Optional[Dict[str, object]] = None
    lcoh_surface: Optional[object] = None
    powerfactory_validation: Optional[object] = None


def run_full_pipeline(
    sets: Optional[ModelSets] = None,
    config: Optional[ModelConfig] = None,
    scenario_tree: Optional[ScenarioTree] = None,
    run_vss: bool = True,
    run_value_of_storage: bool = True,
    run_breakeven: bool = True,
    run_lcoh_surface: bool = False,
    remote_site_for_sweeps: str = "remote1",
    collocated_site_for_sweeps: str = "collocated_FRN",
    substation_for_sweeps: str = "FRN",
    powerfactory_config: Optional[PowerFactoryConfig] = None,
    output_dir: Path = Path("./outputs"),
    log_level: int = 20,  # logging.INFO
) -> PipelineResult:
    """Runs the full analysis pipeline for one electrolyser scaling
    stage (i.e. one ``config.electrolyser.p_elec_max`` / ``annual_target_tonnes``
    setting -- call this once per stage, e.g. 20/50/100/300 MW, to build
    the paper's per-stage result tables).

    Raises
    ------
    HydrogenOptError (or a subclass)
        Propagated from any stage; the pipeline does not swallow errors
        -- it logs the stage that failed and re-raises so the caller
        (CLI, notebook, CI) gets a clean stack trace and a non-zero exit
        code where relevant.
    """
    configure_logging(level=log_level)
    output_dir.mkdir(parents=True, exist_ok=True)

    sets = sets or ModelSets()
    config = config or ModelConfig()

    logger.info("=== Stage: scenario generation ===")
    if scenario_tree is None:
        generator = ScenarioGenerator(sets, config.scenarios)
        scenario_tree = generator.synthetic()

    logger.info("=== Stage: solve full stochastic MILP ===")

    try:
        # -------------------------------
        # Model construction
        # -------------------------------
        logger.info("Building optimisation model...")
        build_start = time.perf_counter()

        build = build_model(sets, config, scenario_tree)

        build_time = time.perf_counter() - build_start

        logger.info(
            "Model construction finished in %.2f seconds.",
            build_time
        )

        # -------------------------------
        # Model size
        # -------------------------------
        try:
            model = build.model if hasattr(build, "model") else build

            report = build_model_size_report(model)

            logger.info(
                "Model size: variables=%s, binary=%s, "
                "integer=%s, constraints=%s",
                report.activated.variables,
                report.activated.binary_variables,
                report.activated.integer_variables,
                report.activated.constraints,
            )

        except Exception:
            logger.exception(
                "Could not generate model size report."
            )

        # -------------------------------
        # Solver
        # -------------------------------
        logger.info("Starting MILP solver...")
        solve_start = time.perf_counter()

        outcome = solve(
            build,
            log_dir=output_dir / "solver_logs"
        )

        solve_time = time.perf_counter() - solve_start

        logger.info(
            "MILP solver completed successfully in %.2f seconds.",
            solve_time
        )

    except HydrogenOptError:
        logger.exception("Full stochastic MILP solve failed.")
        raise

    except Exception:
        logger.exception(
            "Unexpected error during model construction or solve."
        )
        raise

    logger.info("=== Stage: postprocessing / result extraction ===")
    results = full_results_bundle(outcome)
    logger.info("LCOH = %.4f EUR/kg", results["lcoh"].lcoh_eur_per_kg)

    result = PipelineResult(outcome=outcome, results=results)

    if run_vss:
        logger.info("=== Stage: Value of the Stochastic Solution ===")
        try:
            result.vss = compute_vss(sets, config, scenario_tree)
        except HydrogenOptError:
            logger.exception("VSS computation failed; continuing without it.")

    if run_value_of_storage:
        logger.info("=== Stage: Value of Storage ===")
        try:
            result.value_of_storage = compute_value_of_storage(sets, config, scenario_tree)
        except HydrogenOptError:
            logger.exception("Value-of-storage computation failed; continuing without it.")

    if run_breakeven:
        logger.info("=== Stage: break-even cable distance ===")
        try:
            result.breakeven = breakeven_distance(
                sets, config, scenario_tree,
                remote_site=remote_site_for_sweeps,
                collocated_site=collocated_site_for_sweeps,
                substation=substation_for_sweeps,
            )
        except HydrogenOptError:
            logger.exception("Break-even sweep failed; continuing without it.")

    if run_lcoh_surface:
        logger.info("=== Stage: LCOH surface ===")
        try:
            result.lcoh_surface = lcoh_surface(
                sets, config, scenario_tree, remote_site=remote_site_for_sweeps
            )
        except HydrogenOptError:
            logger.exception("LCOH surface sweep failed; continuing without it.")

    if powerfactory_config is not None:
        logger.info("=== Stage: PowerFactory grid validation ===")
        try:
            result.powerfactory_validation = validate_design_in_powerfactory(
                result, powerfactory_config
            )
        except HydrogenOptError:
            logger.exception(
                "PowerFactory validation failed; MILP results above are still valid, "
                "but the design has NOT been confirmed against the real network model."
            )

    _persist_results(result, output_dir)
    return result


def validate_design_in_powerfactory(
    pipeline_result: PipelineResult, pf_config: PowerFactoryConfig
):
    """Writes the MILP-selected RES setpoints into the real PowerFactory
    network model and runs a full (unbalanced) load flow to cross-check
    the simplified DistFlow results used inside the MILP.

    This closes the loop between the planning-stage optimizer
    (model_builder.py) and the operational-fidelity tool
    (powerfactory_interface.py) described in the module docstrings.
    """
    res_mix = pipeline_result.results["res_mix"]
    if res_mix.empty:
        raise HydrogenOptError("Cannot validate in PowerFactory: res_mix table is empty.")

    with PowerFactoryLoadFlow(pf_config) as pf_lf:
        wind_mw = float(res_mix["wind_mw"].iloc[0])
        solar_mw = float(res_mix["solar_mw"].iloc[0])
        # NOTE: generator naming convention must match the PF project;
        # adjust to the real ElmSym object names in your network model.
        pf_lf.set_generator_setpoint("Wind_Aggregate", p_mw=wind_mw)
        pf_lf.set_generator_setpoint("Solar_Aggregate", p_mw=solar_mw)

        pf_lf.configure()
        converged = pf_lf.run()
        if not converged:
            logger.warning(
                "PowerFactory load flow did not converge for the MILP-selected design "
                "(wind=%.1f MW, solar=%.1f MW). Grid constraints may be tighter in the "
                "real network than the MILP's simplified DistFlow representation captured.",
                wind_mw, solar_mw,
            )
        return pf_lf.results_dataframe()


def _persist_results(result: PipelineResult, output_dir: Path) -> None:
    """Writes every result table to CSV under ``output_dir`` for the
    manuscript's tables/figures pipeline."""
    res_mix = result.results.get("res_mix")
    if res_mix is not None and not res_mix.empty:
        res_mix.to_csv(output_dir / "res_mix.csv", index=False)

    grid_impact = result.results.get("grid_impact", {})
    for name, df in grid_impact.items():
        if df is not None and not df.empty:
            df.to_csv(output_dir / f"grid_impact_{name}.csv", index=False)

    storage_df = result.results.get("storage_adequacy")
    if storage_df is not None and not storage_df.empty:
        storage_df.to_csv(output_dir / "storage_adequacy.csv", index=False)

    scenario_df = result.results.get("scenario_sensitivity")
    if scenario_df is not None and not scenario_df.empty:
        scenario_df.to_csv(output_dir / "scenario_sensitivity.csv", index=False)

    if result.breakeven is not None:
        result.breakeven["curve"].to_csv(output_dir / "breakeven_curve.csv", index=False)

    if result.lcoh_surface is not None and not result.lcoh_surface.empty:
        result.lcoh_surface.to_csv(output_dir / "lcoh_surface.csv", index=False)

    logger.info("Result tables written to %s", output_dir.resolve())
