"""
grid_calibration.py
====================
Bridges *real* PowerFactory load-flow results (either a live
``PowerFactoryLoadFlow.results_dataframe()`` or a previously exported
CSV in that same schema) back into the MILP's ``GridParameters``, and
re-validates a solved MILP design against those real results.

Why this is a separate module (and not just more of
``powerfactory_interface.py``)
------------------------------------------------------------------
``powerfactory_interface.py`` is the *live connection* layer (it needs
a running PowerFactory instance). This module works purely on the
*data* PowerFactory produces -- a pandas DataFrame with columns
``result_type, name, u_pu, u_kv, angle_deg, loading_pct, p_mw, q_mvar,
p_from_mw, q_from_mvar`` -- so it runs anywhere (CI, a laptop with no
PowerFactory licence, etc.), and so a previously-saved CSV can be used
without needing to reconnect to PowerFactory at all.

What can honestly be calibrated from a *results* export
----------------------------------------------------------
A load-flow *results* CSV (voltages, loadings, P/Q at one end of a
branch) does NOT contain a line's per-km R/X or a transformer's
nameplate MVA rating -- those live on the line/transformer *type*
object in PowerFactory (``ElmLne.typ_id.rline`` /
``.xline``, ``ElmTr2.typ_id.strn``), not on the result attributes this
project currently extracts. So from a results-only export we can
calibrate:

    * ``cable_thermal_limit_mw``  -- via S = sqrt(P^2+Q^2),
      S_rated = S / (loading_pct/100), for every ``line_loading`` row
      that has non-null P, Q and loading_pct.

We can NOT (yet) calibrate from a results-only export:

    * ``transformer_thermal_limit_mva`` -- transformer_loading rows in
      this project's schema only carry loading_pct, no P/Q/S.
    * ``line_resistance_pu`` / ``line_reactance_pu`` -- not present in
      load-flow *results* at all; needs the type object's data.
    * ``existing_branch_thermal_limit_mw`` -- even if the MVA rating
      were available, there's no mapping yet from a named PF line/
      transformer to the abstract MILP index it represents
      ("substation -> external grid" vs "RES site -> substation").

See ``powerfactory_interface.get_line_type_parameters`` /
``get_transformer_ratings`` (added alongside this module) for the
methods that *would* need to run live against PowerFactory to fill
those two remaining gaps.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import GridParameters, ModelConfig
from .exceptions import DataValidationError
from .logging_config import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = {
    "result_type", "name", "u_pu", "u_kv", "angle_deg",
    "loading_pct", "p_mw", "q_mvar", "p_from_mw", "q_from_mvar",
}

# PowerFactory "export all sheets to CSV" naming convention -- one CSV
# per object class, columns suffixed with the DIgSILENT type code
# (e.g. 'rline(r)' = real, 'typ_id(p)' = pointer/reference).
_NETWORK_EXPORT_FILES = {
    "ElmLne": "ElmLne.csv",       # line elements: loc_name, typ_id, dline (length, km)
    "TypLne": "TypLne.csv",       # line types: rline/xline (ohm/km), uline (kV), sline (rated current, kA)
    "ElmLnesec": "ElmLnesec.csv",  # line *sections* for composite lines (no typ_id on the parent ElmLne)
    "ElmTr2": "ElmTr2.csv",       # 2-winding transformers: loc_name, typ_id
    "TypTr2": "TypTr2.csv",       # transformer types: strn (rated MVA), utrn_h/utrn_l (kV)
}


def load_results_csv(path: str | Path) -> pd.DataFrame:
    """Loads a PowerFactory results export (the schema produced by
    ``PowerFactoryLoadFlow.save_results_csv`` /
    ``.results_dataframe()``) and validates its shape.

    Tolerates a leading BOM in the header (``\\ufeffresult_type``),
    which some PowerFactory/Excel export paths add.
    """
    path = Path(path)
    if not path.exists():
        raise DataValidationError(f"PowerFactory results CSV not found: {path}")

    df = pd.read_csv(path)
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise DataValidationError(
            f"PowerFactory results CSV is missing expected columns: {sorted(missing)}",
            context={"path": str(path), "found_columns": list(df.columns)},
        )

    counts = df["result_type"].value_counts().to_dict()
    logger.info("Loaded PF results from %s: %s", path, counts)
    if counts.get("generator_dispatch", 0) == 0:
        logger.warning(
            "No 'generator_dispatch' rows in %s -- RES-mix cross-validation "
            "against the MILP's P_res_w/P_res_s will not be possible from "
            "this export.", path,
        )
    return df


@dataclass
class GridCalibrationResult:
    """What was actually derived from the PF export, plus what still
    needs a live PowerFactory pull (so the caller/report is explicit
    about provenance rather than silently keeping paper-default
    guesses)."""

    calibrated_grid: GridParameters
    cable_rating_samples_mva: pd.Series
    cable_thermal_limit_mw_old: float
    cable_thermal_limit_mw_new: float
    still_uncalibrated: Dict[str, str]


def calibrate_grid_parameters(
    config: ModelConfig,
    pf_results: pd.DataFrame,
    cable_rating_percentile: float = 50.0,
) -> GridCalibrationResult:
    """Returns a new ``ModelConfig`` whose ``grid.cable_thermal_limit_mw``
    is replaced by a value derived from real PowerFactory line loadings,
    instead of the paper's typical-value guess.

    Parameters
    ----------
    cable_rating_percentile:
        Which percentile of the fleet's implied per-line MVA rating to
        adopt as the single scalar ``cable_thermal_limit_mw`` (the MILP
        uses one thermal limit for every K2-catalog cable). Default is
        the median -- conservative choices should pass a lower
        percentile (e.g. 10-25) instead.
    """
    lines = pf_results[pf_results["result_type"] == "line_loading"].copy()
    lines = lines.dropna(subset=["p_from_mw", "q_from_mvar", "loading_pct"])
    lines = lines[lines["loading_pct"] > 0]  # avoid div-by-zero on unloaded lines

    if lines.empty:
        raise DataValidationError(
            "No usable 'line_loading' rows (with p_from_mw, q_from_mvar, "
            "loading_pct all present and loading_pct > 0) found in the "
            "PF results -- cannot calibrate cable_thermal_limit_mw."
        )

    s_mva = np.sqrt(lines["p_from_mw"] ** 2 + lines["q_from_mvar"] ** 2)
    implied_rating_mva = s_mva / (lines["loading_pct"] / 100.0)
    implied_rating_mva.index = lines["name"]

    new_limit = float(np.percentile(implied_rating_mva, cable_rating_percentile))
    old_limit = config.grid.cable_thermal_limit_mw

    logger.info(
        "Calibrated cable_thermal_limit_mw: %.2f -> %.2f MW "
        "(p%d of %d lines; range %.1f-%.1f MVA)",
        old_limit, new_limit, cable_rating_percentile, len(implied_rating_mva),
        implied_rating_mva.min(), implied_rating_mva.max(),
    )

    new_grid = replace(config.grid, cable_thermal_limit_mw=new_limit)

    still_uncalibrated = {
        "transformer_thermal_limit_mva": (
            "transformer_loading rows in this export carry loading_pct only "
            "(no P/Q/S) -- pull ElmTr2.typ_id.strn live to calibrate this."
        ),
        "line_resistance_pu / line_reactance_pu": (
            "not present in load-flow results at all -- pull "
            "ElmLne.typ_id.rline / .xline live to calibrate this."
        ),
        "existing_branch_thermal_limit_mw": (
            "no mapping yet from a named PF line/transformer to the MILP's "
            "abstract 'substation -> external grid' branch -- supply a "
            "name_map to identify which rows correspond to it."
        ),
    }

    return GridCalibrationResult(
        calibrated_grid=new_grid,
        cable_rating_samples_mva=implied_rating_mva,
        cable_thermal_limit_mw_old=old_limit,
        cable_thermal_limit_mw_new=new_limit,
        still_uncalibrated=still_uncalibrated,
    )


def apply_calibration(config: ModelConfig, calibration: GridCalibrationResult) -> ModelConfig:
    """Returns a new ``ModelConfig`` with the calibrated grid parameters
    applied (leaves everything else in ``config`` untouched)."""
    return replace(config, grid=calibration.calibrated_grid)


# ---------------------------------------------------------------------
# Network *type* data (line R/X, transformer nameplate MVA), from a
# PowerFactory "export all sheets to CSV" of the network model itself
# -- not a load-flow results export. This is what closes the two gaps
# ``calibrate_grid_parameters`` (above) explicitly could not fill.
# ---------------------------------------------------------------------
@dataclass
class NetworkExportCalibrationResult:
    """Everything derived from the raw network-object export, plus the
    per-line/per-substation tables so the caller can audit the numbers
    rather than trust a single scalar blindly."""

    calibrated_grid: GridParameters
    lines: pd.DataFrame                # one row per named ElmLne, with r/x/length/rating
    transformers_by_substation: pd.Series  # summed nameplate MVA per substation prefix
    v_base_kv_observed: float
    v_base_kv_config_before: float
    notes: Dict[str, str]


def _load_network_export(directory: str | Path) -> Dict[str, pd.DataFrame]:
    directory = Path(directory)
    dfs = {}
    for key, filename in _NETWORK_EXPORT_FILES.items():
        path = directory / filename
        if not path.exists():
            raise DataValidationError(
                f"Network export is missing '{filename}' (expected at {path}). "
                "This should be a PowerFactory 'export all sheets to CSV' of the "
                "network model, not a load-flow results export.",
            )
        dfs[key] = pd.read_csv(path)
    return dfs


def _line_r_x_length(
    elm_lne: pd.DataFrame, typ_lne: pd.DataFrame, elm_lnesec: pd.DataFrame
) -> pd.DataFrame:
    """Per named line: total R (ohm), total X (ohm), total length (km),
    nominal voltage (kV), and thermal current rating (kA).

    Handles both simple lines (a single ``typ_id`` directly on the
    ``ElmLne`` row) and composite lines built from multiple
    ``ElmLnesec`` sections (each with its own type and sub-length) --
    PowerFactory represents a line with intermediate joints/cable
    changes this way, and about half the lines in a typical DSO export
    are composite.
    """
    typ_lne = typ_lne.set_index("FID(a:40)")

    def _for_line(row) -> pd.Series:
        fid = row["FID(a:40)"]
        if pd.notna(row["typ_id(p)"]):
            t = typ_lne.loc[row["typ_id(p)"]]
            return pd.Series({
                "r_ohm": t["rline(r)"] * row["dline(r)"],
                "x_ohm": t["xline(r)"] * row["dline(r)"],
                "length_km": row["dline(r)"],
                "u_kv": t["uline(r)"],
                "i_rated_ka": t["sline(r)"],
            })
        secs = elm_lnesec[elm_lnesec["fold_id(p)"] == fid]
        if secs.empty:
            return pd.Series({
                "r_ohm": np.nan, "x_ohm": np.nan, "length_km": np.nan,
                "u_kv": np.nan, "i_rated_ka": np.nan,
            })
        r = x = length = 0.0
        u_kvs, i_rateds = [], []
        for _, s in secs.iterrows():
            t = typ_lne.loc[s["typ_id(p)"]]
            r += t["rline(r)"] * s["dline(r)"]
            x += t["xline(r)"] * s["dline(r)"]
            length += s["dline(r)"]
            u_kvs.append(t["uline(r)"])
            i_rateds.append(t["sline(r)"])  # weakest section caps the whole line's rating
        return pd.Series({
            "r_ohm": r, "x_ohm": x, "length_km": length,
            "u_kv": u_kvs[0], "i_rated_ka": min(i_rateds),
        })

    result = elm_lne.apply(_for_line, axis=1)
    lines = pd.concat([elm_lne[["loc_name(a:80)"]].rename(columns={"loc_name(a:80)": "name"}), result], axis=1)
    lines = lines.dropna(subset=["r_ohm", "x_ohm", "length_km"])
    lines["r_ohm_per_km"] = lines["r_ohm"] / lines["length_km"]
    lines["x_ohm_per_km"] = lines["x_ohm"] / lines["length_km"]
    lines["rated_mva"] = np.sqrt(3) * lines["u_kv"] * lines["i_rated_ka"]
    return lines.set_index("name")


def calibrate_from_network_export(
    config: ModelConfig,
    network_export_dir: str | Path,
    s_base_mva: float = 100.0,
    cable_rating_percentile: float = 25.0,
    transformer_rating_percentile: float = 50.0,
) -> NetworkExportCalibrationResult:
    """Calibrates ``GridParameters`` from a PowerFactory network-object
    export (``ElmLne``/``TypLne``/``ElmLnesec``/``ElmTr2``/``TypTr2``
    CSVs -- e.g. from PowerFactory's "export all sheets to CSV"), which
    carries the two things a load-flow *results* export cannot:
    per-line R/X and transformer nameplate MVA.

    Parameters
    ----------
    s_base_mva:
        Base apparent power for the per-unit conversion of line R/X.
        100 MVA is the standard power-system convention; change this
        only if the rest of your MILP's pu system uses something else.
    cable_rating_percentile:
        Which percentile of the fleet's *nameplate* line rating (from
        cable current rating x nominal voltage) to adopt as
        ``cable_thermal_limit_mw``. Defaults to the 25th percentile
        (conservative -- most cables can do more than this), since
        nameplate ratings run well above what a `results`-CSV-derived
        operational limit would show (see ``calibrate_grid_parameters``
        for that alternative, operational-limit-based calibration).
    transformer_rating_percentile:
        Which percentile of per-substation aggregated transformer MVA
        (summing parallel units at the same substation) to adopt as
        ``transformer_thermal_limit_mva``.

    Notes
    -----
    The MILP's ``model_builder._c14_voltage_drop`` now scales
    ``R_line``/``X_line`` by each corridor's actual distance
    ``L[i,j]`` (fixed from the earlier flat-scalar simplification --
    see the README's "next steps" list), so the returned
    ``line_resistance_pu`` / ``line_reactance_pu`` here are true
    per-km values, ready to be multiplied by ``L[i,j]`` inside the
    model exactly as ``config.py``'s field comments now say.
    """
    dfs = _load_network_export(network_export_dir)

    lines = _line_r_x_length(dfs["ElmLne"], dfs["TypLne"], dfs["ElmLnesec"])
    if lines.empty:
        raise DataValidationError("No lines with resolvable R/X/length found in the network export.")

    v_base_observed = float(lines["u_kv"].mode().iloc[0])
    if not np.isclose(v_base_observed, config.grid.v_base_kv, rtol=0.05):
        logger.warning(
            "Network export's nominal line voltage (%.1f kV) does not match "
            "config.grid.v_base_kv (%.1f kV) -- the MILP's DistFlow base "
            "voltage assumption looks wrong for this network and should be "
            "updated, not just R/X/ratings.", v_base_observed, config.grid.v_base_kv,
        )

    z_base_ohm = v_base_observed ** 2 / s_base_mva
    r_pu = float(lines["r_ohm_per_km"].median() / z_base_ohm)
    x_pu = float(lines["x_ohm_per_km"].median() / z_base_ohm)

    cable_limit_mw = float(np.percentile(lines["rated_mva"], cable_rating_percentile))

    # Transformers: aggregate parallel units at the same substation
    # (e.g. "EGE-KT22" + "EGE-KT23" -> substation "EGE"), since the
    # MILP's T_max represents total substation transformer capacity.
    tr2 = dfs["ElmTr2"][["loc_name(a:80)", "typ_id(p)"]].copy()
    typ_tr2 = dfs["TypTr2"].set_index("FID(a:40)")
    tr2["rated_mva"] = tr2["typ_id(p)"].map(typ_tr2["strn(r)"])
    tr2["substation"] = tr2["loc_name(a:80)"].str.split("-").str[0]
    per_substation_mva = tr2.groupby("substation")["rated_mva"].sum().sort_values()
    transformer_limit_mva = float(np.percentile(per_substation_mva, transformer_rating_percentile))

    logger.info(
        "Network-export calibration: v_base=%.0fkV (was %.0fkV), "
        "R_line=%.5fpu/km X_line=%.5fpu/km (S_base=%.0fMVA), "
        "cable_thermal_limit_mw=%.1f (p%d of %d lines' nameplate MVA), "
        "transformer_thermal_limit_mva=%.1f (p%d of %d substations)",
        v_base_observed, config.grid.v_base_kv, r_pu, x_pu, s_base_mva,
        cable_limit_mw, cable_rating_percentile, len(lines),
        transformer_limit_mva, transformer_rating_percentile, len(per_substation_mva),
    )

    new_grid = replace(
        config.grid,
        v_base_kv=v_base_observed,
        line_resistance_pu=r_pu,
        line_reactance_pu=x_pu,
        cable_thermal_limit_mw=cable_limit_mw,
        transformer_thermal_limit_mva=transformer_limit_mva,
    )

    notes = {
        "RX_are_per_km": (
            f"line_resistance_pu={r_pu:.5f} and line_reactance_pu={x_pu:.5f} are "
            f"per-km values -- model_builder.C14_voltage_drop multiplies them by "
            f"each corridor's L[i,j] distance, so a 20 km remote corridor now sees "
            f"a correspondingly larger voltage drop than a 2 km collocated one."
        ),
        "cable_rating_is_nameplate_not_operational": (
            f"cable_thermal_limit_mw={cable_limit_mw:.1f} MW is derived from cable "
            f"nameplate current rating x voltage, at the p{cable_rating_percentile:.0f}. "
            f"This may be materially higher than an operational/N-1-derated limit -- "
            f"cross-check against calibrate_grid_parameters()'s results-CSV-based "
            f"figure if you have one, and pick whichever the MILP should represent."
        ),
    }

    return NetworkExportCalibrationResult(
        calibrated_grid=new_grid,
        lines=lines,
        transformers_by_substation=per_substation_mva,
        v_base_kv_observed=v_base_observed,
        v_base_kv_config_before=config.grid.v_base_kv,
        notes=notes,
    )


# ---------------------------------------------------------------------
# Re-validation: MILP's own grid_impact table vs. real PF results
# ---------------------------------------------------------------------
def compare_line_loading(
    mip_line_loading: pd.DataFrame,
    pf_results: pd.DataFrame,
    name_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Compares the MILP's ``postprocessing.extract_grid_impact()['line_loading']``
    table (indexed by abstract ``(res_site, substation)`` pairs) against
    real PowerFactory line loadings (indexed by named lines like
    ``'BDV-RYT'``).

    Parameters
    ----------
    name_map:
        Required to make this meaningful: maps
        ``"{res_site}->{substation}"`` (as it appears in
        ``mip_line_loading``) to the corresponding PowerFactory line
        ``name``. Without it, this function can only report the two
        tables side by side -- it cannot line up specific rows,
        because the MILP's site/substation IDs are abstract planning
        labels with no fixed correspondence to real named network
        elements.

    Returns
    -------
    A DataFrame with the MILP's simplified-DistFlow loading % next to
    PowerFactory's real loading % for every mapped corridor, plus the
    absolute and relative difference.
    """
    pf_lines = pf_results[pf_results["result_type"] == "line_loading"].set_index("name")

    if not name_map:
        logger.warning(
            "compare_line_loading called without a name_map -- returning "
            "the two loading tables unmerged; supply name_map={'site->sub': "
            "'PF_LINE_NAME'} to get an aligned comparison."
        )
        return pd.DataFrame({
            "mip_corridors": [f"{r.res_site}->{r.substation}" for r in mip_line_loading.itertuples()],
            "mip_mean_loading_pct": mip_line_loading["mean_loading_pct"].values,
        }).merge(
            pf_lines[["loading_pct"]].reset_index().rename(
                columns={"name": "pf_line", "loading_pct": "pf_loading_pct"}
            ),
            how="cross",
        )

    rows = []
    for r in mip_line_loading.itertuples():
        corridor = f"{r.res_site}->{r.substation}"
        pf_name = name_map.get(corridor)
        if pf_name is None or pf_name not in pf_lines.index:
            logger.warning("No PF match for corridor '%s' (mapped name '%s')", corridor, pf_name)
            continue
        pf_loading = float(pf_lines.loc[pf_name, "loading_pct"])
        rows.append({
            "corridor": corridor,
            "pf_line_name": pf_name,
            "mip_mean_loading_pct": r.mean_loading_pct,
            "mip_max_loading_pct": r.max_loading_pct,
            "pf_loading_pct": pf_loading,
            "abs_diff_pct_points": r.mean_loading_pct - pf_loading,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("No corridors could be matched between the MILP and PF results.")
    return df
