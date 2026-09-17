"""
powerfactory_interface.py
==========================
DIgSILENT PowerFactory load-flow wrapper.

This is a hardened, module-based refactor of the user-supplied
"PowerFactory Load Flow Runner" script. Behaviour is preserved
(same PowerFactory API calls, same result extraction), but it is now:

* Exception-safe: every PowerFactory API call that can fail is wrapped
  and re-raised as a typed exception (``PowerFactoryConnectionError`` /
  ``PowerFactoryExecutionError``) with the original traceback chained
  (`raise ... from err`), instead of letting raw ``AttributeError`` /
  COM exceptions propagate.
* Logged: every stage (connect, activate project, activate study case,
  configure, execute, extract) emits a structured log line instead of
  ``print``.
* Config-driven: connection details come from ``PowerFactoryConfig``
  instead of hard-coded paths, so the same module works across machines
  / PowerFactory versions / CI (where PowerFactory is simply absent).
* Context-manager friendly: supports ``with PowerFactoryLoadFlow(...) as pf:``
  so the project is always deactivated even if an exception is raised
  mid-run.
* Returns pandas DataFrames (in addition to the original list-of-dicts
  API) so results plug straight into ``postprocessing.py``'s grid-impact
  tables for cross-validation against the MILP's simplified DistFlow
  results.

Role in the pipeline
---------------------
The stochastic MILP in ``model_builder.py`` uses a *simplified* radial
DistFlow representation for tractable planning-stage optimization. This
module is the validation step: once the MILP proposes an RES mix /
substation / cable design, that design is written into the real
PowerFactory network model (bus assignments, cable types, generator
setpoints) and a full unbalanced AC load flow is run here to confirm
voltage, line and transformer limits hold on the *actual* network
topology -- see ``pipeline.py`` for how the two are chained together.
"""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .exceptions import PowerFactoryConnectionError, PowerFactoryExecutionError
from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PowerFactoryConfig:
    """Connection and study-case configuration.

    Replaces the hard-coded constants at the top of the original
    script; override any field to match your installation.
    """

    install_path: str = r"C:\Program Files\DIgSILENT\PowerFactory 2024 SP6"
    python_api_subpath: str = r"Python\3.12"
    project_name: str = r"\LF15NS.IntUser\TREFOR RYTTERGÅRDEN(3).IntPrj"
    study_case_name: Optional[str] = "Study Case_Existing_RES"
    operation_scenario_name: Optional[str] = "With newData_QS"
    grid_name: Optional[str] = "Grid_2025Baseline"
    headless: bool = True
    balanced_load_flow: bool = False  # False -> AC unbalanced (iopt_net=1)

    @property
    def python_api_path(self) -> str:
        return str(Path(self.install_path) / self.python_api_subpath)


def _import_powerfactory_module(config: PowerFactoryConfig):
    """Adds the PowerFactory Python API to ``sys.path`` and imports it.

    Isolated into its own function (rather than a module-level import)
    so that the rest of this module -- and the rest of the package --
    remains importable on machines without PowerFactory installed
    (e.g. CI, or a colleague's laptop doing pure MILP work). The
    import error is only raised when a ``PowerFactoryLoadFlow`` is
    actually instantiated.
    """
    api_path = config.python_api_path
    if not os.path.exists(api_path):
        raise PowerFactoryConnectionError(
            f"PowerFactory Python API folder not found: {api_path}. "
            "Check the installed PowerFactory version/service pack and the "
            "Python sub-version folder under '...\\Python\\'.",
            context={"install_path": config.install_path, "api_path": api_path},
        )

    if api_path not in sys.path:
        sys.path.append(api_path)

    # Python 3.8+ requires explicit DLL directory registration so
    # dependent DLLs (not just the .pyd) can be found.
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(config.install_path)
        except OSError as err:
            raise PowerFactoryConnectionError(
                f"Could not register PowerFactory DLL directory: {config.install_path}",
                context={"error": str(err)},
            ) from err

    try:
        import powerfactory as pf  # noqa: WPS433 (intentional deferred import)
    except ImportError as err:
        raise PowerFactoryConnectionError(
            "Failed to import the 'powerfactory' module even though the API "
            "folder exists. Verify the Python version running this process "
            "matches the PowerFactory-bundled Python (see python_api_subpath).",
            context={"api_path": api_path},
        ) from err
    return pf


class PowerFactoryLoadFlow:
    """Connects to PowerFactory, runs a load flow, and extracts results.

    Usage
    -----

    >>> cfg = PowerFactoryConfig(project_name=r' \LF15NS.IntUser\TREFOR RYTTERGÅRDEN(3).IntPrj')
    >>> with PowerFactoryLoadFlow(cfg) as pf_lf:
    ...     pf_lf.configure()
    ...     if pf_lf.run():
    ...         df = pf_lf.results_dataframe()
    """

    def __init__(self, config: PowerFactoryConfig):
        self.config = config
        self._pf_module = _import_powerfactory_module(config)
        self.app = None
        self.project = None
        self.study_case = None
        self.ldf = None
        self.res = None
        self._connect()

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------
    def __enter__(self) -> "PowerFactoryLoadFlow":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Connection / activation
    # ------------------------------------------------------------------
    def _connect(self) -> None:
        logger.info("Connecting to PowerFactory application...")
        self.app = self._pf_module.GetApplication()
        if self.app is None:
            raise PowerFactoryConnectionError(
                "Could not start/connect to the PowerFactory application."
            )

        if not self.config.headless:
            self.app.Show()

        self.app.ClearOutputWindow()
        self.app.EchoOff()

        self._activate_project()
        if self.config.study_case_name:
            self._activate_study_case(self.config.study_case_name)

        self.study_case = self.app.GetActiveStudyCase()
        if self.study_case is None:
            raise PowerFactoryConnectionError(
                "No active study case found. Either pass study_case_name explicitly, "
                "or activate one manually in the project before running this script."
            )

        self.ldf = self.app.GetFromStudyCase("ComLdf")
        if self.ldf is None:
            raise PowerFactoryConnectionError(
                "Could not retrieve ComLdf (load flow command) from the study case."
            )

        self.res = self.app.GetFromStudyCase("Myresult.ElmRes")
        if self.res is not None:
            self.res.Clear()

        logger.info(
            "Connected: project='%s' study_case='%s'",
            self.config.project_name, self.config.study_case_name,
        )

    def _activate_project(self) -> None:
        ret = self.app.ActivateProject(self.config.project_name)
        if ret != 0:
            available = self._list_available_projects()
            raise PowerFactoryConnectionError(
                f"Could not activate project '{self.config.project_name}' "
                f"(return code {ret}).",
                context={"available_projects": available},
            )
        self.project = self.app.GetActiveProject()
        if self.project is None:
            raise PowerFactoryConnectionError(
                "Project activation reported success but GetActiveProject() is None."
            )

    def _list_available_projects(self) -> List[str]:
        """Helper: lists projects visible under the current user, for
        diagnostics attached to a connection-failure exception."""
        user = self.app.GetCurrentUser()
        if user is None:
            logger.warning("Could not retrieve current PF user for project listing.")
            return []
        projects = user.GetContents("*.IntPrj", 1)  # 1 = recursive search
        names = [p.GetFullName() for p in projects]
        logger.info("Projects available under user '%s': %s", user.loc_name, names)
        return names

    def _activate_study_case(self, study_case_name: str) -> None:
        study_case_folder = self.app.GetProjectFolder("study")
        cases = study_case_folder.GetContents(f"{study_case_name}.IntCase")
        if not cases:
            raise PowerFactoryConnectionError(
                f"Study case '{study_case_name}' not found in project.",
                context={"project": self.config.project_name},
            )
        cases[0].Activate()

    # ------------------------------------------------------------------
    # Load flow configuration / execution
    # ------------------------------------------------------------------
    def configure(self) -> None:
        """balanced_load_flow=False -> AC unbalanced (iopt_net=1);
        True -> AC balanced (iopt_net=0)."""
        self.ldf.iopt_net = 0 if self.config.balanced_load_flow else 1
        logger.debug("Configured load flow: iopt_net=%d", self.ldf.iopt_net)

    def run(self) -> bool:
        """Executes the load flow.

        Returns
        -------
        bool
            True if converged.

        Raises
        ------
        PowerFactoryExecutionError
            If the ``ComLdf.Execute()`` call itself raises (COM error,
            licence issue, etc.) -- a *non-converged* result is NOT an
            exception (it's a normal, expected outcome the caller should
            branch on), only a hard execution failure is.
        """
        logger.info("Executing load flow...")
        try:
            error_code = self.ldf.Execute()
        except Exception as err:  # noqa: BLE001 - COM errors surface as generic Exception
            raise PowerFactoryExecutionError(
                "ComLdf.Execute() raised an exception.", context={"error": str(err)}
            ) from err

        converged = error_code == 0
        if not converged:
            logger.warning("Load flow did NOT converge (error code: %s)", error_code)
            self.app.PrintWarn(f"Load flow did NOT converge (error code: {error_code})")
        else:
            logger.info("Load flow converged successfully.")
        return converged

    # ------------------------------------------------------------------
    # Results extraction
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_get(obj, attr: str) -> Optional[Any]:
        try:
            return obj.GetAttribute(attr)
        except Exception:  # noqa: BLE001 - PF raises bare Exception for missing attrs
            logger.debug("Attribute '%s' unavailable on object '%s'", attr,
                         getattr(obj, "loc_name", "<unknown>"))
            return None

    def get_bus_voltages(self) -> List[Dict[str, Any]]:
        buses = self.app.GetCalcRelevantObjects("*.ElmTerm")
        return [
            {
                "name": b.loc_name,
                "u_pu": self._safe_get(b, "m:u"),
                "u_kv": self._safe_get(b, "m:U"),
                "angle_deg": self._safe_get(b, "m:phiu"),
            }
            for b in buses
        ]

    def get_line_loadings(self) -> List[Dict[str, Any]]:
        lines = self.app.GetCalcRelevantObjects("*.ElmLne")
        return [
            {
                "name": l.loc_name,
                "loading_pct": self._safe_get(l, "c:loading"),
                "p_from_mw": self._safe_get(l, "m:P:bus1"),
                "q_from_mvar": self._safe_get(l, "m:Q:bus1"),
            }
            for l in lines
        ]

    def get_generator_dispatch(self) -> List[Dict[str, Any]]:
        gens = self.app.GetCalcRelevantObjects("*.ElmSym")
        return [
            {
                "name": g.loc_name,
                "p_mw": self._safe_get(g, "m:P:bus1"),
                "q_mvar": self._safe_get(g, "m:Q:bus1"),
            }
            for g in gens
        ]

    def get_transformer_loadings(self) -> List[Dict[str, Any]]:
        trafos = self.app.GetCalcRelevantObjects("*.ElmTr2")
        return [
            {"name": t.loc_name, "loading_pct": self._safe_get(t, "c:loading")}
            for t in trafos
        ]

    # ------------------------------------------------------------------
    # Network *type* data (as opposed to load-flow *results*): needed to
    # calibrate the MILP's GridParameters (line R/X, transformer MVA
    # rating), since none of that is present in a results export -- see
    # ``grid_calibration.py``. Prefer ``grid_calibration
    # .calibrate_from_network_export`` against a PowerFactory "export
    # all sheets to CSV" of ElmLne/TypLne/ElmLnesec/ElmTr2/TypTr2 when
    # you have one; these two methods are the live-connection
    # equivalent for when you don't.
    # ------------------------------------------------------------------
    def get_line_type_parameters(self) -> List[Dict[str, Any]]:
        """Per-line R/X (ohm/km) and length, pulled from each
        ``ElmLne``'s type object (``typ_id``) rather than from
        load-flow results. This is what ``config.GridParameters
        .line_resistance_pu`` / ``.line_reactance_pu`` should be
        calibrated against (both are per-km; ``model_builder``
        multiplies them by each corridor's ``L[i,j]`` distance)."""
        lines = self.app.GetCalcRelevantObjects("*.ElmLne")
        rows = []
        for l in lines:
            typ = getattr(l, "typ_id", None)
            rows.append({
                "name": l.loc_name,
                "length_km": self._safe_get(l, "dline"),
                "r_ohm_per_km": self._safe_get(typ, "rline") if typ else None,
                "x_ohm_per_km": self._safe_get(typ, "xline") if typ else None,
                "rated_current_ka": self._safe_get(typ, "sline") if typ else None,
            })
        return rows

    def get_transformer_ratings(self) -> List[Dict[str, Any]]:
        """Each ``ElmTr2``'s nameplate MVA rating (``typ_id.strn``),
        needed to calibrate ``config.GridParameters
        .transformer_thermal_limit_mva`` -- transformer *results* rows
        only carry ``loading_pct``, not the absolute rating."""
        trafos = self.app.GetCalcRelevantObjects("*.ElmTr2")
        rows = []
        for t in trafos:
            typ = getattr(t, "typ_id", None)
            rows.append({
                "name": t.loc_name,
                "rated_mva": self._safe_get(typ, "strn") if typ else None,
            })
        return rows

    def results_dataframe(self) -> pd.DataFrame:
        """Combines bus/line/generator/transformer results into one
        tidy DataFrame -- the pandas-native counterpart to
        ``save_results_csv`` below, and the format expected by
        ``postprocessing.extract_grid_impact`` for cross-validation."""
        rows: List[Dict[str, Any]] = []

        for b in self.get_bus_voltages():
            rows.append({"result_type": "bus_voltage", "name": b["name"], **b})
        for l in self.get_line_loadings():
            rows.append({"result_type": "line_loading", "name": l["name"], **l})
        for g in self.get_generator_dispatch():
            rows.append({"result_type": "generator_dispatch", "name": g["name"], **g})
        for t in self.get_transformer_loadings():
            rows.append({"result_type": "transformer_loading", "name": t["name"], **t})

        return pd.DataFrame(rows)

    def save_results_csv(self, filepath: Path) -> Path:
        """Extracts bus, line, generator, and transformer results and
        saves them to a single CSV file (preserves the original
        script's on-disk schema for backward compatibility)."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "result_type", "name", "u_pu", "u_kv", "angle_deg",
            "loading_pct", "p_mw", "q_mvar", "p_from_mw", "q_from_mvar",
        ]
        rows = []
        for b in self.get_bus_voltages():
            rows.append({"result_type": "bus_voltage", "name": b["name"],
                         "u_pu": b["u_pu"], "u_kv": b["u_kv"], "angle_deg": b["angle_deg"],
                         "loading_pct": "", "p_mw": "", "q_mvar": "",
                         "p_from_mw": "", "q_from_mvar": ""})
        for l in self.get_line_loadings():
            rows.append({"result_type": "line_loading", "name": l["name"],
                         "u_pu": "", "u_kv": "", "angle_deg": "",
                         "loading_pct": l["loading_pct"], "p_mw": "", "q_mvar": "",
                         "p_from_mw": l["p_from_mw"], "q_from_mvar": l["q_from_mvar"]})
        for g in self.get_generator_dispatch():
            rows.append({"result_type": "generator_dispatch", "name": g["name"],
                         "u_pu": "", "u_kv": "", "angle_deg": "", "loading_pct": "",
                         "p_mw": g["p_mw"], "q_mvar": g["q_mvar"],
                         "p_from_mw": "", "q_from_mvar": ""})
        for t in self.get_transformer_loadings():
            rows.append({"result_type": "transformer_loading", "name": t["name"],
                         "u_pu": "", "u_kv": "", "angle_deg": "",
                         "loading_pct": t["loading_pct"], "p_mw": "", "q_mvar": "",
                         "p_from_mw": "", "q_from_mvar": ""})

        try:
            with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        except OSError as err:
            raise PowerFactoryExecutionError(
                f"Failed writing results CSV to {filepath}", context={"error": str(err)}
            ) from err

        logger.info("Results saved to: %s", filepath.resolve())
        return filepath

    # ------------------------------------------------------------------
    # Optimization-loop convenience (write decision variables into PF)
    # ------------------------------------------------------------------
    def set_generator_setpoint(
        self, gen_name: str, p_mw: Optional[float] = None, q_mvar: Optional[float] = None
    ) -> bool:
        """Writes a generator's dispatch setpoint before a load flow --
        the hook an outer optimization loop (scipy/Pyomo callback) uses
        to test an MILP-proposed RES setpoint against the real network."""
        gens = self.app.GetCalcRelevantObjects("*.ElmSym")
        for gen in gens:
            if gen.loc_name == gen_name:
                if p_mw is not None:
                    gen.pgini = p_mw
                if q_mvar is not None:
                    gen.qgini = q_mvar
                logger.debug("Set generator '%s' setpoint: P=%s MW, Q=%s MVAr",
                             gen_name, p_mw, q_mvar)
                return True
        logger.warning("Generator '%s' not found; setpoint not applied.", gen_name)
        self.app.PrintWarn(f"Generator '{gen_name}' not found.")
        return False

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Deactivates the project cleanly. Safe to call multiple times."""
        if self.project is not None:
            try:
                self.project.Deactivate()
                logger.info("Project deactivated.")
            except Exception as err:  # noqa: BLE001
                logger.warning("Error while deactivating project: %s", err)
            finally:
                self.project = None
