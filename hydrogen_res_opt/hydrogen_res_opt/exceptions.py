"""
exceptions.py
=============
Project-specific exception hierarchy.

Rationale
---------
Using a dedicated hierarchy (rather than letting raw ``KeyError`` /
``ValueError`` / third-party solver exceptions bubble up) means:

1. Callers can catch precisely what they expect
   (``except InfeasibilityError`` vs. ``except Exception``).
2. Every exception carries a stable ``.code`` for log aggregation /
   dashboards and an optional ``.context`` dict for structured
   diagnostics (scenario id, constraint name, PF object name, etc.).
3. Traceback chaining (`raise ... from err`) is used everywhere data
   crosses a module boundary, so the original stack is never lost.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class HydrogenOptError(Exception):
    """Base class for all errors raised by hydrogen_res_opt."""

    code: str = "HRO-000"

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        self.context = context or {}
        super().__init__(f"[{self.code}] {message} | context={self.context}")


class DataValidationError(HydrogenOptError):
    """Raised when input parameters/sets fail validation (Section 15, F0-F5)."""

    code = "HRO-100"


class ScenarioGenerationError(HydrogenOptError):
    """Raised when joint price/capacity-factor scenario generation fails."""

    code = "HRO-200"


class ModelBuildError(HydrogenOptError):
    """Raised when the Pyomo model fails to construct (missing set/param)."""

    code = "HRO-300"


class SolverError(HydrogenOptError):
    """Raised when the MILP solver terminates abnormally (not optimal)."""

    code = "HRO-400"


class InfeasibilityError(SolverError):
    """Raised when the solver reports infeasibility.

    Section 15 (F0-F5) enumerates the analytic feasibility pre-checks;
    this exception is raised either when a pre-check fails (fast,
    before the solver is even called) or when the solver itself
    reports infeasibility.
    """

    code = "HRO-401"


class PowerFactoryConnectionError(HydrogenOptError):
    """Raised when the DIgSILENT PowerFactory application/project/study
    case cannot be reached or activated."""

    code = "HRO-500"


class PowerFactoryExecutionError(HydrogenOptError):
    """Raised when a PowerFactory command object (e.g. ComLdf) fails
    to execute or returns a non-converged / error result."""

    code = "HRO-501"
