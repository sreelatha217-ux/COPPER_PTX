"""Generate publication-ready tables and figures from stage comparison outputs.

Run from the project root with::

    .venv/bin/python -m hydrogen_res_opt.journal_results

The generated PDF files are vector figures suitable for journal submission;
PNG copies are included for previews and slide decks.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STAGES = (20, 300)
CASES = {
    "Grid-only": "grid_only",
    "Compulsory RES": "res",
}
REPRESENTATIVE_TARGET_TONNES = {20: 4.56, 300: 1740.0 / 365.0}
ANNUALIZATION_FACTOR = 365.0


def _objective_value(folder: Path) -> float:
    values = []
    for log_path in sorted((folder / "solver_logs").glob("*.log")):
        text = log_path.read_text(errors="replace")
        matches = re.findall(r"Objective value:\s*([-+]?\d+(?:\.\d+)?)", text)
        if matches:
            values.append(float(matches[-1]))
    if not values:
        raise FileNotFoundError(f"No solver objective found in {folder / 'solver_logs'}")
    return values[0]


def _read_case(root: Path, stage_mw: int, case_name: str, suffix: str) -> dict:
    folder = root / f"compare_{stage_mw}mw_{suffix}"
    objective = _objective_value(folder)
    annual_target = REPRESENTATIVE_TARGET_TONNES[stage_mw] * ANNUALIZATION_FACTOR
    mix_path = folder / "res_mix.csv"
    mix = pd.read_csv(mix_path) if mix_path.exists() else pd.DataFrame()
    scenario = pd.read_csv(folder / "scenario_sensitivity.csv")
    storage = pd.read_csv(folder / "storage_adequacy.csv")
    voltage = pd.read_csv(folder / "grid_impact_voltage_profile.csv")
    line = pd.read_csv(folder / "grid_impact_line_loading.csv")
    return {
        "stage_mw": stage_mw,
        "case": case_name,
        "annual_cost_eur": objective,
        "annual_h2_tonnes": annual_target,
        "lcoh_eur_per_kg": objective / (annual_target * 1000.0),
        "res_capacity_mw": float(mix[["wind_mw", "solar_mw"]].sum(axis=1).max())
        if not mix.empty
        else 0.0,
        "res_site": ", ".join(mix["res_site"].astype(str)) if not mix.empty else "None",
        "max_line_loading_pct": float(line["max_loading_pct"].max()),
        "min_voltage_pu": float(voltage["min_voltage_pu"].min()),
        "max_storage_tonnes": float(storage["max_storage_tonnes"].max()),
        "storage_binding_scenarios": int(storage["binding"].sum()),
        "mean_import_cost_eur": float(scenario["import_cost_eur"].mean()),
    }


def _save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_paper_artifacts(root: Path, output_dir: Path, summary: pd.DataFrame) -> None:
    """Write paper tables/figures supported by the saved comparison outputs."""
    res_cases = []
    investment_rows = []
    grid_rows = []
    for stage in STAGES:
        for case, suffix in CASES.items():
            folder = root / f"compare_{stage}mw_{suffix}"
            mix_path = folder / "res_mix.csv"
            if mix_path.exists():
                mix = pd.read_csv(mix_path)
                mix.insert(0, "case", case)
                mix.insert(0, "stage_mw", stage)
                investment_rows.append(mix)
            voltage = pd.read_csv(folder / "grid_impact_voltage_profile.csv")
            loading = pd.read_csv(folder / "grid_impact_line_loading.csv")
            grid_rows.append({
                "stage_mw": stage,
                "case": case,
                "min_voltage_pu": voltage["min_voltage_pu"].min(),
                "max_voltage_pu": voltage["max_voltage_pu"].max(),
                "max_line_loading_pct": loading["max_loading_pct"].max(),
            })

    if investment_rows:
        pd.concat(investment_rows, ignore_index=True).to_csv(
            output_dir / "table_R2_investment_decisions.csv", index=False
        )
    summary.to_csv(output_dir / "table_R5_scaling_and_policy.csv", index=False)
    pd.DataFrame(grid_rows).to_csv(output_dir / "table_R3_grid_impact.csv", index=False)

    res_summary = summary[summary["case"] == "Compulsory RES"]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    width = 0.35
    x = np.arange(len(STAGES))
    for offset, (case, color) in enumerate({"Grid-only": "#adb5bd", "Compulsory RES": "#007f5f"}.items()):
        values = summary[summary["case"] == case].set_index("stage_mw").loc[list(STAGES)]
        ax.bar(x + (offset - 0.5) * width, values["res_capacity_mw"], width,
               label=case, color=color)
    ax.set_xticks(x, [f"{stage} MW" for stage in STAGES])
    ax.set_ylabel("Installed RES capacity (MW)")
    ax.set_title("Optimal RES capacity under alternative policies")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    _save_figure(fig, output_dir, "figure_R1_res_capacity_mix")

    grid = pd.DataFrame(grid_rows)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for case, color in {"Grid-only": "#6c757d", "Compulsory RES": "#007f5f"}.items():
        values = grid[grid["case"] == case]
        ax.plot(values["stage_mw"], values["max_line_loading_pct"], marker="o",
                linewidth=2.0, label=case, color=color)
    ax.set(xlabel="Electrolyser stage (MW)", ylabel="Maximum line loading (%)",
           xticks=list(STAGES), title="Grid loading under alternative RES policies")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    _save_figure(fig, output_dir, "figure_R3_loading_comparison")

    for stage in STAGES:
        folder = root / f"compare_{stage}mw_res"
        curve_path = folder / "breakeven_curve.csv"
        if curve_path.exists():
            curve = pd.read_csv(curve_path)
            fig, ax = plt.subplots(figsize=(7.0, 4.2))
            ax.plot(curve["distance_km"], curve["lcoh_remote_eur_per_kg"],
                    marker="o", label="Remote RES", color="#bc4749")
            ax.plot(curve["distance_km"], curve["lcoh_colloc_eur_per_kg"],
                    linestyle="--", label="Collocated RES", color="#007f5f")
            ax.set(xlabel="Remote cable distance (km)", ylabel="LCOH (EUR/kg H2)",
                   title=f"Break-even distance sweep ({stage} MW)")
            ax.grid(axis="y", alpha=0.25)
            ax.legend(frameon=False)
            _save_figure(fig, output_dir, f"figure_R6_breakeven_{stage}mw")

    (output_dir / "results_section.tex").write_text(_latex_results_section(summary))
    (output_dir / "results_data_availability.md").write_text(
        "# Results data availability\n\n"
        "Generated from saved comparison outputs.\n\n"
        "Available: 20/300 MW policy comparison, investment decisions, grid loading, "
        "voltage summaries, storage adequacy, scenario sensitivity, and break-even curves.\n\n"
        "Pending: 50/100/150/200 MW scaling, historical price-wind validation, five-run "
        "scenario-reduction error bars, held-out out-of-sample validation, and solver-statistics "
        "table. These require additional model runs or saved raw scenario/solver metadata.\n"
    )


def _latex_results_section(summary: pd.DataFrame) -> str:
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            f"{int(row.stage_mw)} & {row.case} & {row.lcoh_eur_per_kg:.3f} & "
            f"{row.res_capacity_mw:.1f} & {row.max_line_loading_pct:.2f} & "
            f"{row.min_voltage_pu:.3f} \\\\" 
        )
    return (
        "% Generated by hydrogen_res_opt.journal_results.\n"
        "% Replace pending subsections only after the corresponding experiments are run.\n"
        "\\section{Results}\n"
        "\\subsection{Base-case solution and policy comparison}\n"
        "Table~\\ref{tab:policy-comparison} reports the available 20 and 300~MW results. "
        "All values use the same representative-horizon annualisation convention.\n\n"
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Available LCOH and grid metrics under alternative RES policies.}\n"
        "\\label{tab:policy-comparison}\n"
        "\\begin{tabular}{r l r r r r}\n\\toprule\n"
        "Stage (MW) & Policy & LCOH (EUR/kg) & RES (MW) & Max loading (\\%) & Min voltage (p.u.) \\\\ \\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n\n"
        "The compulsory-RES policy requires at least one RES connection and RES nameplate "
        "capacity equal to the electrolyser rating. The grid-only case is retained as a "
        "counterfactual and must not be interpreted as a green-hydrogen design.\n\n"
        "\\subsection{Grid impact}\n"
        "The corrected voltage formulation stores squared voltage in kV$^2$ and reports "
        "post-processed magnitudes in per unit. The saved voltage and loading figures are "
        "therefore directly comparable with the stated operating limits.\n\n"
        "\\subsection{Stochastic value, scaling, and validation}\n"
        "Scenario sensitivity and storage adequacy are available in the corresponding CSV files. "
        "The historical-correlation, out-of-sample, multi-stage scaling, and solver-statistics "
        "subsections remain pending because those raw experiments are not present in the current output set.\n"
    )


def generate_results(root: Path, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        _read_case(root, stage, case, suffix)
        for stage in STAGES
        for case, suffix in CASES.items()
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "journal_summary.csv", index=False)
    _write_paper_artifacts(root, output_dir, summary)

    style = {"Grid-only": "#6c757d", "Compulsory RES": "#007f5f"}
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for case in CASES:
        values = summary[summary["case"] == case]
        ax.plot(values["stage_mw"], values["lcoh_eur_per_kg"], marker="o",
                linewidth=2.2, markersize=7, label=case, color=style[case])
    ax.set(xlabel="Electrolyser stage (MW)", ylabel="LCOH (EUR/kg H2)",
           xticks=list(STAGES), title="Levelised cost of hydrogen by RES policy")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    _save_figure(fig, output_dir, "figure_1_lcoh_policy_comparison")

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8), sharey=True)
    x = np.arange(len(STAGES))
    width = 0.34
    for offset, (case, color) in enumerate(style.items()):
        values = summary[summary["case"] == case].set_index("stage_mw").loc[list(STAGES)]
        axes[0].bar(x + (offset - 0.5) * width, values["res_capacity_mw"], width,
                    label=case, color=color)
        axes[1].bar(x + (offset - 0.5) * width, values["max_line_loading_pct"], width,
                    label=case, color=color)
    axes[0].set_title("Installed RES capacity")
    axes[0].set_ylabel("MW")
    axes[1].set_title("Maximum line loading")
    axes[1].set_ylabel("% of cable limit")
    for ax in axes:
        ax.set_xticks(x, [f"{stage} MW" for stage in STAGES])
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    _save_figure(fig, output_dir, "figure_2_res_and_grid_impacts")

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for case, color in style.items():
        values = summary[summary["case"] == case]
        ax.plot(values["stage_mw"], values["annual_cost_eur"] / 1e6, marker="o",
                linewidth=2.2, markersize=7, label=case, color=color)
    ax.set(xlabel="Electrolyser stage (MW)", ylabel="Annualised cost (million EUR)",
           xticks=list(STAGES), title="Annualised system cost by RES policy")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    _save_figure(fig, output_dir, "figure_3_annualised_cost_comparison")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/journal_results"))
    args = parser.parse_args(argv)
    summary = generate_results(args.root, args.output_dir)
    print(summary.to_string(index=False))
    print(f"\nWrote tables and figures to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())