# hydrogen_res_opt

Python implementation of the two-stage stochastic MILP from:

> *Co-Optimization of RES Capacity, Substation Selection, and Cable Sizing
> for Green Hydrogen LCOH Under Stochastic Electricity Prices in
> Distribution Networks: A Danish DK1 Case Study*
> S. Aihloor Subramanyam, A. Anvari-Moghaddam, Department of Energy, Aalborg University.

It reproduces every row of the paper's **Expected Results Structure**
table, and integrates with a real DIgSILENT PowerFactory network model
for post-optimization grid validation.

| Result (paper)             | Where it's produced                                   |
|-----------------------------|--------------------------------------------------------|
| Optimal RES mix per stage   | `postprocessing.extract_res_mix`                       |
| LCOH surface                | `breakeven_analysis.lcoh_surface`                      |
| Break-even cable distance   | `breakeven_analysis.breakeven_distance`                |
| Grid impact per stage       | `postprocessing.extract_grid_impact`                   |
| Value of the stochastic sol.| `vss_analysis.compute_vss`                              |
| Scenario sensitivity        | `postprocessing.scenario_sensitivity`                  |
| Storage adequacy            | `postprocessing.storage_adequacy`                       |

## Architecture

```
hydrogen_res_opt/
    config.py                  Table 2-3, 6-10 parameters (validated dataclasses)
    exceptions.py               Typed exception hierarchy
    logging_config.py           Central structured logging
    sets_indices.py              Table 1 sets/indices + delta_i (C4)
    scenario_generation.py      Section 14: joint price/CF scenarios (Eq. 47)
    model_builder.py            Section 11: full Pyomo MILP (Eqs. 6-37)
    solver_runner.py            Solve orchestration + F0-F5 pre-checks (Section 15)
    postprocessing.py           KPI extraction -> pandas
    vss_analysis.py              Section 13: VSS (Eq. 45), VoStorage (Eq. 46)
    breakeven_analysis.py       Section 12: break-even distance (Eq. 42), LCOH surface
    powerfactory_interface.py    DIgSILENT PowerFactory load-flow wrapper
    grid_calibration.py          Calibrates GridParameters from real PF network/results data
    pipeline.py                  End-to-end orchestration
    cli.py                       `python -m hydrogen_res_opt.cli ...`
    tests/                        pytest unit + integration tests
```

Every constraint in `model_builder.py` is annotated in its module
docstring with the exact paper equation it implements (C1-C18,
non-anticipativity Eqs. 34-37, objective Eq. 38).

## Installation

```bash
pip install -r requirements.txt
# plus a MILP solver, e.g.:
conda install -c conda-forge coincbc      # open-source, good for dev/testing
# or configure a licensed Gurobi / CPLEX installation for production runs
# (Section 17: "Solver: Gurobi or CPLEX (commercial); CBC for testing")

pip install -e .    # optional editable install, exposes `hydrogen-res-opt` CLI
```

## Quick start

```python
from hydrogen_res_opt.config import ModelConfig
from hydrogen_res_opt.sets_indices import ModelSets
from hydrogen_res_opt.pipeline import run_full_pipeline

result = run_full_pipeline(
    sets=ModelSets(),
    config=ModelConfig(),
    output_dir="./outputs/stage_20MW",
)

print(result.results["lcoh"].lcoh_eur_per_kg)
print(result.results["res_mix"])
print(result.vss.vss_eur_per_kg)
print(result.breakeven["breakeven_distance_km"])
```

Or via the CLI:

```bash
python -m hydrogen_res_opt.cli run --stage-mw 20 --output-dir ./outputs/stage_20MW
python -m hydrogen_res_opt.cli breakeven --remote-site remote1 --substation FRN
python -m hydrogen_res_opt.cli scenario-convergence
```

## PowerFactory grid validation

`model_builder.py` uses a *simplified radial DistFlow* representation
(documented in its module docstring) so the MILP stays tractable at
planning-stage scale. Once a design is selected, validate it against
the real network topology:

```python
from hydrogen_res_opt.powerfactory_interface import PowerFactoryConfig, PowerFactoryLoadFlow

pf_cfg = PowerFactoryConfig(
    project_name=r"\LF15NS.IntUser\TREFOR RYTTERGÅRDEN(3).IntPrj",
    study_case_name="Study Case_Existing_RES",
)
with PowerFactoryLoadFlow(pf_cfg) as pf_lf:
    pf_lf.configure()
    converged = pf_lf.run()
    df = pf_lf.results_dataframe()
```

`pipeline.run_full_pipeline(..., powerfactory_config=pf_cfg)` chains
this automatically after a solve.

## Testing

```bash
pytest hydrogen_res_opt/tests -v
```

Model-building/solve tests use `pytest.importorskip` / a CBC
availability check so they skip gracefully (rather than failing CI)
on machines without Pyomo or a solver installed; pure-Python
config/sets/scenario-generation tests always run.

## Known simplifications / next steps for the manuscript

1. **DistFlow topology**: the MILP's C14-C18 use a simplified
   two-level radial graph (RES site -> substation -> external grid).
   The paper's full formulation supports an arbitrary graph `(i,j) in E`;
   extending `model_builder.py` to ingest a real branch list (e.g.
   exported from PowerFactory) is a natural next increment.
2. **Reactive power**: `Q_flow` is currently derived from `P_flow` via
   a fixed assumed power factor (0.95) since the manuscript does not
   supply a separate reactive-demand profile; replace with real data
   if available.
3. **Historical scenario data**: `scenario_generation.py` ships a
   Cholesky-correlated synthetic generator for development/testing.
   Point `ScenarioGenerator.from_historical_data` at the real
   Energinet/Center Denmark price series, Energinet wind, and ERA5
   solar series for the production run referenced in Section 14.
4. ~~**Line resistance `R_ij`**: not tabulated in Table 8...~~ **Fixed.**
   `line_resistance_pu` / `line_reactance_pu` are now per-km values,
   scaled by each corridor's actual distance `L[i,j]` inside
   `model_builder.C14_voltage_drop` (previously a flat value applied
   uniformly regardless of length). Calibrate both, plus
   `cable_thermal_limit_mw`, `transformer_thermal_limit_mva`, and
   `v_base_kv`, from a real PowerFactory network export via
   `grid_calibration.calibrate_from_network_export` -- see
   "PowerFactory grid parameter calibration" below.

## PowerFactory grid parameter calibration

Beyond the post-optimization *validation* load flow above,
`grid_calibration.py` calibrates `GridParameters` itself from real
PowerFactory data before you solve, so the MILP's DistFlow constraints
reflect your actual network rather than the paper's placeholder
defaults:

```python
from hydrogen_res_opt.config import ModelConfig
from hydrogen_res_opt.grid_calibration import calibrate_from_network_export, apply_calibration

config = ModelConfig()
calib = calibrate_from_network_export(
    config, "path/to/powerfactory_network_export"  # ElmLne/TypLne/ElmLnesec/ElmTr2/TypTr2 CSVs
)
config = apply_calibration(config, calib)
for note in calib.notes.values():
    print(note)
```

This needs a PowerFactory "export all sheets to CSV" of the *network
model* (line/transformer type objects), not a load-flow results
export -- `get_line_type_parameters()` / `get_transformer_ratings()`
on `PowerFactoryLoadFlow` pull the same data live if you don't have a
CSV export handy. A load-flow *results* CSV (as produced by
`save_results_csv`) can still calibrate `cable_thermal_limit_mw` alone,
via `grid_calibration.calibrate_grid_parameters` -- useful as an
operational/N-1-derated cross-check against the network export's
nameplate ratings.
