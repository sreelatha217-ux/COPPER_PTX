"""
hydrogen_res_opt
=================

Co-optimization of RES capacity, substation selection and cable sizing
for green-hydrogen LCOH under stochastic DK1 electricity prices, with
distribution-grid (DistFlow) constraints and PowerFactory-based grid
validation.

This package is an implementation of the two-stage stochastic MILP
described in:

    "Co-Optimization of RES Capacity, Substation Selection, and Cable
    Sizing for Green Hydrogen LCOH Under Stochastic Electricity Prices
    in Distribution Networks: A Danish DK1 Case Study"
    S. Aihloor Subramanyam, A. Anvari-Moghaddam, Aalborg University.

Package layout
---------------
config.py                  Typed parameter containers (Tables 2-3, 6-10)
exceptions.py               Project-specific exception hierarchy
logging_config.py           Central logging configuration
sets_indices.py              Sets/indices (Table 1) + validation
scenario_generation.py      Joint price / capacity-factor scenario trees
model_builder.py            Pyomo MILP model (objective + C1-C18)
solver_runner.py            Solve orchestration, retries, diagnostics
postprocessing.py           Result extraction, LCOH, KPI tables
vss_analysis.py              Value of the Stochastic Solution (Sec. 13)
breakeven_analysis.py       Break-even cable distance sweep (Sec. 12)
powerfactory_interface.py    DIgSILENT PowerFactory load-flow wrapper
pipeline.py                  End-to-end orchestration / Agile "runner"
cli.py                       Command line entry point

Each module is independently unit-testable and only depends on the
layers below it, so the codebase can be developed and tested
incrementally (Agile: thin vertical slices, one module = one
deployable increment).
"""

from .config import (
    InvestmentParameters,
    ElectrolyserParameters,
    GridParameters,
    StorageParameters,
    ScenarioParameters,
    ModelConfig,
)
from .sets_indices import ModelSets
from .exceptions import (
    HydrogenOptError,
    DataValidationError,
    InfeasibilityError,
    ScenarioGenerationError,
    SolverError,
    PowerFactoryConnectionError,
    PowerFactoryExecutionError,
)

__all__ = [
    "InvestmentParameters",
    "ElectrolyserParameters",
    "GridParameters",
    "StorageParameters",
    "ScenarioParameters",
    "ModelConfig",
    "ModelSets",
    "HydrogenOptError",
    "DataValidationError",
    "InfeasibilityError",
    "ScenarioGenerationError",
    "SolverError",
    "PowerFactoryConnectionError",
    "PowerFactoryExecutionError",
]

__version__ = "0.1.0"
