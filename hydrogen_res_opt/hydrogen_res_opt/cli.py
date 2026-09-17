"""
cli.py
======
Command-line entry point.

Examples
--------
Run the full pipeline for the 20 MW stage with defaults::

    python -m hydrogen_res_opt.cli run --stage-mw 20 --output-dir ./outputs/stage_20MW

Run just the break-even sweep::

    python -m hydrogen_res_opt.cli breakeven --remote-site remote1 --substation FRN

Run the scenario-count convergence check (Section 14)::

    python -m hydrogen_res_opt.cli scenario-convergence
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from .config import ElectrolyserParameters, ModelConfig
from .exceptions import HydrogenOptError
from .logging_config import configure_logging, get_logger
from .pipeline import run_full_pipeline
from .scenario_generation import ScenarioGenerator
from .sets_indices import ModelSets

logger = get_logger(__name__)

# Electrolyser scaling stages referenced throughout the paper (Section 1).
STAGE_MW_TO_TARGET_TONNES = {20: 4.56, 50: 1200, 100: 1500, 300: 1740}  #{20: 870, 50: 1200, 100: 1500, 300: 1740}


def _build_config_for_stage(stage_mw: int) -> ModelConfig:
    cfg = ModelConfig()
    if stage_mw not in STAGE_MW_TO_TARGET_TONNES:
        logger.warning(
            "Stage %d MW is not one of the paper's canonical stages %s; "
            "using the default annual target.",
            stage_mw, list(STAGE_MW_TO_TARGET_TONNES),
        )
        return cfg
    target = STAGE_MW_TO_TARGET_TONNES[stage_mw]
    cfg.electrolyser = replace(
        cfg.electrolyser, p_elec_max=float(stage_mw), annual_target_tonnes=float(target)
    )
    return cfg


def _cmd_run(args: argparse.Namespace) -> int:
    config = _build_config_for_stage(args.stage_mw)
    sets = ModelSets()
    try:
        run_full_pipeline(
            sets=sets,
            config=config,
            run_lcoh_surface=args.lcoh_surface,
            output_dir=Path(args.output_dir),
        )
    except HydrogenOptError as err:
        logger.error("Pipeline failed: %s", err)
        return 1
    return 0


def _cmd_breakeven(args: argparse.Namespace) -> int:
    from .breakeven_analysis import breakeven_distance

    config = ModelConfig()
    sets = ModelSets()
    generator = ScenarioGenerator(sets, config.scenarios)
    tree = generator.synthetic()
    try:
        result = breakeven_distance(
            sets, config, tree,
            remote_site=args.remote_site,
            collocated_site=args.collocated_site,
            substation=args.substation,
        )
    except HydrogenOptError as err:
        logger.error("Break-even sweep failed: %s", err)
        return 1
    print(result["curve"].to_string(index=False))
    print(f"\nBreak-even distance: {result['breakeven_distance_km']} km")
    return 0


def _cmd_scenario_convergence(args: argparse.Namespace) -> int:
    sets = ModelSets()
    config = ModelConfig()
    generator = ScenarioGenerator(sets, config.scenarios)
    results = generator.convergence_check()
    for size, mean_price in results.items():
        print(f"|Omega|={size:4d} -> mean scenario price = {mean_price:.2f} EUR/MWh")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hydrogen_res_opt",
        description="RES/substation/cable co-optimization for green hydrogen LCOH.",
    )
    parser.add_argument("--log-level", default="INFO",
                         choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run the full pipeline for one electrolyser stage.")
    p_run.add_argument("--stage-mw", type=int, default=20, choices=[20, 50, 100, 300])
    p_run.add_argument("--output-dir", type=str, default="./outputs")
    p_run.add_argument("--lcoh-surface", action="store_true",
                        help="Also compute the (expensive) 2-D LCOH surface sweep.")
    p_run.set_defaults(func=_cmd_run)

    p_break = sub.add_parser("breakeven", help="Run the break-even cable distance sweep.")
    p_break.add_argument("--remote-site", default="remote1")
    p_break.add_argument("--collocated-site", default="collocated_FRN")
    p_break.add_argument("--substation", default="FRN")
    p_break.set_defaults(func=_cmd_breakeven)

    p_conv = sub.add_parser(
        "scenario-convergence",
        help="Check LCOH-relevant stability across |Omega| in {20,30,50,100} (Section 14).",
    )
    p_conv.set_defaults(func=_cmd_scenario_convergence)

    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    level = getattr(__import__("logging"), args.log_level)
    configure_logging(level=level)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
