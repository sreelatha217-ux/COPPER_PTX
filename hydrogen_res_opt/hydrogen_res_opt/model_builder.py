"""
model_builder.py
=================
Builds the two-stage stochastic MILP of Section 11 ("Complete Model
Summary") as a Pyomo ``ConcreteModel``.

Equation -> constraint-rule mapping
------------------------------------
Objective        Eq. (38)            -> ``_objective_rule``
C1                Eq. (7)             -> ``_c1_single_connection``
C2 (corrected)    Eqs. (8)-(9)        -> ``_c2a_cable_requires_link`` / ``_c2b_one_cable_if_linked``
C3                Eqs. (10)-(11)      -> RES capacity bounds (Var bounds, not a rule)
C4                Eq. (12)            -> ``sets.is_remote`` (delta_i, a fixed parameter)
Two-corridor CAPEX Eq. (6)            -> ``_objective_rule`` CAPEX term
C5                Eq. (13)            -> ``_c5_energy_balance``
C6                Eq. (14)            -> ``_c6_res_availability``
C7                Eq. (15)            -> Var bounds on ``p``
C8                Eqs. (16)-(17)      -> ``_c8_ramp_up`` / ``_c8_ramp_down``
C9                Eq. (18)            -> ``_c9_hydrogen_production``
C10               Eqs. (19)-(20)      -> ``_c10_annual_target``
C11               Eq. (21)            -> ``_c11_storage_dynamics``
C12               Eq. (22)            -> Var bounds on ``s``
C13 / relaxed     Eqs. (23), (26)     -> ``_c13_cyclic_storage``
C14 (DistFlow)    Eqs. (27)-(29)      -> ``_c14_active_balance`` / ``_c14_reactive_balance`` / ``_c14_voltage_drop_upper`` / ``_c14_voltage_drop_lower``
C15               Eq. (30)            -> Var bounds on ``v``
C16               Eq. (31)            -> ``_c16_new_corridor_thermal``
C17               Eq. (32)            -> Var bounds on ``P_existing``
C18               Eq. (33)            -> ``_c18_transformer_limit``
Non-anticipativity Eqs. (34)-(37)      -> implicit (first-stage vars carry no omega index)

Network topology simplification
--------------------------------
The paper's DistFlow constraints (C14-C18) are written for a general
radial graph (i, j) in E, but the manuscript does not supply an
explicit edge list -- that is expected to come from the real PowerFactory
network model (see ``powerfactory_interface.py``, which pulls buses,
lines and transformers directly from PowerFactory).

For the *planning-stage* MILP (whose purpose is RES sizing / substation
selection / cable choice, not full network operation), this module
therefore represents the grid as a two-level radial topology:

    RES site i --(new corridor, cable type k)--> substation j --(existing
    backbone branch)--> external grid / TSO connection point

which is exactly the level of detail the investment variables (x, y, z)
act on. Once a design is selected, ``powerfactory_interface.py`` should
be used to re-validate that design against the *full* network model in
PowerFactory (real topology, meshed sections, unbalanced load flow) --
that is the intended division of labour between this planning MILP and
the operational load-flow tool.

Known fix -- C14_voltage_drop (big-M gating on x[i,j])
-------------------------------------------------------
``v[j, w, t]`` (the squared voltage at substation j) does not carry an
``i`` index, but every RES site ``i`` that could potentially route
power to substation ``j`` was previously given its own *equality*
constraint pinning ``v[j, w, t]`` to a value computed from that site's
own corridor flow and distance ``L[i, j]``. Since multiple sites ``i``
share the same substation ``j`` (``m.I x m.J`` is a dense product, not
just the connected pairs), this produced several simultaneous
equalities on the *same* variable -- one per site, each demanding a
different value of ``v[j, w, t]`` -- most of which are mutually
inconsistent (e.g. the equation from an *unconnected* site collapses
to ``v[j,w,t] == V_base**2`` because ``P_flow[i,j,w,t]`` is forced to
0 by ``C14_active_balance`` when ``x[i,j]=0``, which is only
compatible with a *connected* site's equation if that site's flow is
also forced to 0 -- i.e. it silently forced grid import to 0 on every
corridor, which produces a hard infeasibility as soon as available RES
generation can't meet electrolyser demand in every period/scenario).

The fix relaxes the equality into a big-M-gated upper/lower pair: the
voltage-drop relationship only *binds* for the site ``i`` actually
connected to substation ``j`` (``x[i,j] = 1``); for every other site
sharing that substation, the pair is relaxed wide enough to be a
no-op. See ``_c14_voltage_drop_upper`` / ``_c14_voltage_drop_lower``
below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .config import ModelConfig
from .exceptions import ModelBuildError
from .logging_config import get_logger
from .scenario_generation import ScenarioTree
from .sets_indices import ModelSets

logger = get_logger(__name__)

try:
    import pyomo.environ as pyo
except ImportError as _imp_err:  # pragma: no cover - exercised only without pyomo installed
    pyo = None
    _PYOMO_IMPORT_ERROR = _imp_err
else:
    _PYOMO_IMPORT_ERROR = None


def _require_pyomo() -> None:
    if pyo is None:
        raise ModelBuildError(
            "Pyomo is not installed. Install it with "
            "`pip install pyomo` (and a MILP solver such as CBC/Gurobi/CPLEX) "
            "before calling build_model()."
        ) from _PYOMO_IMPORT_ERROR


@dataclass
class BuildResult:
    """Wraps the constructed model together with the inputs used to
    build it, so downstream modules (solver_runner, postprocessing)
    don't need to re-derive them."""

    model: "pyo.ConcreteModel"
    sets: ModelSets
    config: ModelConfig
    scenario_tree: ScenarioTree


def build_model(
    sets: ModelSets,
    config: ModelConfig,
    scenario_tree: ScenarioTree,
    fix_first_stage: Optional[dict] = None,
) -> BuildResult:
    """Construct the full two-stage stochastic MILP.

    Parameters
    ----------
    sets:
        Validated :class:`ModelSets` (Table 1).
    config:
        Validated :class:`ModelConfig` bundling Tables 2-3, 6-10.
    scenario_tree:
        A validated :class:`ScenarioTree` (price / wind CF / solar CF /
        probabilities), typically from ``ScenarioGenerator``.
    fix_first_stage:
        Optional dict of first-stage values to fix, e.g.
        ``{"P_res_w": 120.0, "x": {("remote1", "FRN"): 1}}``. Used by
        ``breakeven_analysis`` and ``vss_analysis`` to re-solve
        restricted/deterministic variants without duplicating the
        whole model definition.

    Returns
    -------
    BuildResult
    """
    _require_pyomo()
    scenario_tree.validate()

    inv, elec, grid, sto = config.investment, config.electrolyser, config.grid, config.storage
    n_scen, n_period = scenario_tree.price.shape

    if n_scen != sets.n_scenarios:
        logger.warning(
            "scenario_tree has %d scenarios but sets.n_scenarios=%d; "
            "using the scenario_tree's actual count.", n_scen, sets.n_scenarios
        )
    if n_period != sets.n_periods:
        logger.warning(
            "scenario_tree has %d periods but sets.n_periods=%d; "
            "using the scenario_tree's actual count.", n_period, sets.n_periods
        )

    m = pyo.ConcreteModel(name="RES_Substation_Cable_LCOH_MILP")

    # ------------------------------------------------------------ Sets
    m.I = pyo.Set(initialize=sets.res_sites)
    m.J = pyo.Set(initialize=sets.substations)
    m.K1 = pyo.Set(initialize=sets.collection_cables)          # remote-only collector cables
    m.K2 = pyo.Set(initialize=sets.connection_cables)          # PCC/collocated -> substation
    m.T = pyo.RangeSet(0, n_period - 1)
    m.OMEGA = pyo.RangeSet(0, n_scen - 1)
    m.I_remote = pyo.Set(initialize=sets.remote_sites(), within=m.I)
    m.I_colloc = pyo.Set(initialize=sets.collocated_sites(), within=m.I)

    # --------------------------------------------------------- Params
    m.L = pyo.Param(m.I, m.J, initialize=lambda mm, i, j: sets.distance_matrix[(i, j)])
    m.pi = pyo.Param(m.OMEGA, initialize=lambda mm, w: float(scenario_tree.probabilities[w]))
    m.price = pyo.Param(
        m.OMEGA, m.T, initialize=lambda mm, w, t: float(scenario_tree.price[w, t])
    )
    m.CF_w = pyo.Param(
        m.OMEGA, m.T, initialize=lambda mm, w, t: float(scenario_tree.wind_cf[w, t])
    )
    m.CF_s = pyo.Param(
        m.OMEGA, m.T, initialize=lambda mm, w, t: float(scenario_tree.solar_cf[w, t])
    )
    m.delta_t = 1.0  # hourly resolution

    # Investment / capex
    m.c_w = inv.c_wind
    m.c_s = inv.c_solar
    m.c_cable_k1 = grid_cable_cost = inv.c_cable_collection
    m.c_cable_k2 = inv.c_cable_connection
    m.CRF = inv.crf

    # Electrolyser
    m.P_elec_min = elec.p_elec_min
    m.P_elec_max = elec.p_elec_max
    m.eta = elec.efficiency_kg_per_mwh / 1000.0  # kg/MWh -> tonnes/MWh
    m.ramp_max = elec.ramp_rate_mw_per_h
    m.D_target = elec.annual_target_tonnes
    m.eta_compress = elec.compression_efficiency_tonnes_per_mwh

    # Storage
    m.S_max = sto.s_max_tonnes
    m.soc_init_frac = sto.soc_initial_frac
    m.eps_cyclic = sto.cyclic_tolerance_eps

    # Demand shape (F0, Eq. 48): flat offtake profile unless overridden.
    m.d_t = pyo.Param(
        m.T,
        initialize=lambda mm, t: (m.eta_compress * m.D_target) / (m.eta_compress * len(m.T)),
    )

    # Grid
    m.T_max = grid.transformer_thermal_limit_mva
    m.F_max_new = grid.cable_thermal_limit_mw
    m.F_fixed = grid.existing_branch_thermal_limit_mw
    m.R_line = grid.line_resistance_pu  # pu PER KM -- scaled by L[i,j] in C14_voltage_drop
    m.X_line = grid.line_reactance_pu   # pu PER KM -- scaled by L[i,j] in C14_voltage_drop
    m.V_base = grid.v_base_kv
    m.V_min_sq = grid.v_min_pu ** 2
    m.V_max_sq = grid.v_max_pu ** 2

    # ---------------------------------------------- First-stage variables
    m.P_res_w = pyo.Var(domain=pyo.NonNegativeReals, bounds=(inv.p_res_w_min, inv.p_res_w_max))
    m.P_res_s = pyo.Var(domain=pyo.NonNegativeReals, bounds=(inv.p_res_s_min, inv.p_res_s_max))
    m.x = pyo.Var(m.I, m.J, domain=pyo.Binary)             # C1/C2
    m.y = pyo.Var(m.I, m.J, m.K2, domain=pyo.Binary)       # Corridor-2 / collocated cable choice
    m.z1 = pyo.Var(m.I_remote, m.J, m.K1, domain=pyo.Binary)  # Corridor-1 (remote collector) cable choice

    # --------------------------------------------------- Second-stage
    m.p = pyo.Var(m.OMEGA, m.T, domain=pyo.NonNegativeReals, bounds=(0, elec.p_elec_max))
    m.g = pyo.Var(m.OMEGA, m.T, domain=pyo.NonNegativeReals)
    m.r = pyo.Var(m.OMEGA, m.T, domain=pyo.NonNegativeReals)
    m.q = pyo.Var(m.OMEGA, m.T, domain=pyo.NonNegativeReals)
    m.s = pyo.Var(m.OMEGA, m.T, domain=pyo.NonNegativeReals, bounds=(0, sto.s_max_tonnes))

    # Simplified radial DistFlow variables: one branch per (i, j) corridor
    # (RES site -> substation) plus one aggregate existing branch per
    # substation (substation -> external grid / TSO connection point).
    m.P_flow = pyo.Var(m.I, m.J, m.OMEGA, m.T, domain=pyo.Reals)
    m.Q_flow = pyo.Var(m.I, m.J, m.OMEGA, m.T, domain=pyo.Reals)
    m.P_existing = pyo.Var(
        m.J, m.OMEGA, m.T, domain=pyo.Reals,
        bounds=(-grid.existing_branch_thermal_limit_mw, grid.existing_branch_thermal_limit_mw),
    )
    m.v = pyo.Var(
        m.J, m.OMEGA, m.T, domain=pyo.NonNegativeReals,
        bounds=(m.V_min_sq, m.V_max_sq),
    )

    # ============================================================
    # Objective -- Eq. (38)
    # ============================================================
    def _objective_rule(mm):
        capex_res = mm.c_w * mm.P_res_w + mm.c_s * mm.P_res_s
        capex_cable_k2 = sum(
            mm.c_cable_k2 * mm.L[i, j] * mm.y[i, j, k2]
            for i in mm.I for j in mm.J for k2 in mm.K2
        )
        capex_cable_k1 = sum(
            mm.c_cable_k1 * mm.L[i, j] * mm.z1[i, j, k1]
            for i in mm.I_remote for j in mm.J for k1 in mm.K1
        )
        capex = capex_res + capex_cable_k2 + capex_cable_k1
        expected_import_cost = sum(
            mm.pi[w] * mm.price[w, t] * mm.g[w, t] * mm.delta_t
            for w in mm.OMEGA for t in mm.T
        )
        return mm.CRF * capex + expected_import_cost

    m.total_cost = pyo.Objective(rule=_objective_rule, sense=pyo.minimize)

    # ============================================================
    # C1 -- Eq. (7): each site connects to at most one substation
    # ============================================================
    def _c1_single_connection(mm, i):
        return sum(mm.x[i, j] for j in mm.J) <= 1
    m.C1_single_connection = pyo.Constraint(m.I, rule=_c1_single_connection)

    # ============================================================
    # C2 corrected -- Eqs. (8)-(9): cable selection logic (K2 catalog)
    # ============================================================
    def _c2a_cable_requires_link(mm, i, j, k2):
        return mm.y[i, j, k2] <= mm.x[i, j]
    m.C2a_cable_requires_link = pyo.Constraint(m.I, m.J, m.K2, rule=_c2a_cable_requires_link)

    def _c2b_one_cable_if_linked(mm, i, j):
        return sum(mm.y[i, j, k2] for k2 in mm.K2) == mm.x[i, j]
    m.C2b_one_cable_if_linked = pyo.Constraint(m.I, m.J, rule=_c2b_one_cable_if_linked)

    # Corridor-1 (collector) cable logic mirrors C2 for remote sites only,
    # activated by delta_i = 1 (C4); this operationalises Eq. (6).
    def _c2c_collector_requires_link(mm, i, j, k1):
        return mm.z1[i, j, k1] <= mm.x[i, j]
    m.C2c_collector_requires_link = pyo.Constraint(
        m.I_remote, m.J, m.K1, rule=_c2c_collector_requires_link
    )

    def _c2d_one_collector_if_linked(mm, i, j):
        return sum(mm.z1[i, j, k1] for k1 in mm.K1) == mm.x[i, j]
    m.C2d_one_collector_if_linked = pyo.Constraint(
        m.I_remote, m.J, rule=_c2d_one_collector_if_linked
    )

    # C3 (RES capacity bounds) is enforced directly via Var bounds above.

    # ============================================================
    # C5 -- Eq. (13): electrolyser energy balance
    # ============================================================
    def _c5_energy_balance(mm, w, t):
        return mm.p[w, t] == mm.r[w, t] + mm.g[w, t]
    m.C5_energy_balance = pyo.Constraint(m.OMEGA, m.T, rule=_c5_energy_balance)

    # ============================================================
    # C6 -- Eq. (14): RES availability bound
    # ============================================================
    def _c6_res_availability(mm, w, t):
        return mm.r[w, t] <= mm.CF_w[w, t] * mm.P_res_w + mm.CF_s[w, t] * mm.P_res_s
    m.C6_res_availability = pyo.Constraint(m.OMEGA, m.T, rule=_c6_res_availability)

    # C7 (electrolyser operating bounds) enforced via Var bounds on `p`.

    # ============================================================
    # C8 -- Eqs. (16)-(17): ramp-rate limits
    # ============================================================
    def _c8_ramp_up(mm, w, t):
        if t == mm.T.first():
            return pyo.Constraint.Skip
        return mm.p[w, t] - mm.p[w, t - 1] <= mm.ramp_max
    m.C8_ramp_up = pyo.Constraint(m.OMEGA, m.T, rule=_c8_ramp_up)

    def _c8_ramp_down(mm, w, t):
        if t == mm.T.first():
            return pyo.Constraint.Skip
        return mm.p[w, t - 1] - mm.p[w, t] <= mm.ramp_max
    m.C8_ramp_down = pyo.Constraint(m.OMEGA, m.T, rule=_c8_ramp_down)

    # ============================================================
    # C9 -- Eq. (18): hydrogen production
    # ============================================================
    def _c9_hydrogen_production(mm, w, t):
        return mm.q[w, t] == mm.eta * mm.p[w, t] * mm.delta_t
    m.C9_hydrogen_production = pyo.Constraint(m.OMEGA, m.T, rule=_c9_hydrogen_production)

    # ============================================================
    # C10 -- Eq. (19): per-scenario production floor (kept as a
    # redundant numerical safety check per the F0 discussion, Section 15)
    # ============================================================
    def _c10_annual_target(mm, w):
        return sum(mm.q[w, t] for t in mm.T) >= mm.D_target / len(mm.OMEGA)
    m.C10_annual_target = pyo.Constraint(m.OMEGA, rule=_c10_annual_target)

    # ============================================================
    # C11 -- Eq. (21): buffer storage dynamics
    # ============================================================
    def _c11_storage_dynamics(mm, w, t):
        if t == mm.T.first():
            return pyo.Constraint.Skip
        return mm.s[w, t] == mm.s[w, t - 1] + mm.eta_compress * mm.q[w, t] - mm.d_t[t]
    m.C11_storage_dynamics = pyo.Constraint(m.OMEGA, m.T, rule=_c11_storage_dynamics)

    # C12 (storage bounds) enforced via Var bounds on `s`.

    # ============================================================
    # C13 -- relaxed cyclic boundary, Eq. (26) (tolerance band around
    # the strict equality Eq. (23); set eps_cyclic=0 for the strict form)
    # ============================================================
    def _c13_cyclic_lower(mm, w):
        t0, tN = mm.T.first(), mm.T.last()
        s0 = mm.soc_init_frac * mm.S_max
        return -mm.eps_cyclic * mm.S_max <= mm.s[w, tN] - s0
    m.C13_cyclic_lower = pyo.Constraint(m.OMEGA, rule=_c13_cyclic_lower)

    def _c13_cyclic_upper(mm, w):
        t0, tN = mm.T.first(), mm.T.last()
        s0 = mm.soc_init_frac * mm.S_max
        return mm.s[w, tN] - s0 <= mm.eps_cyclic * mm.S_max
    m.C13_cyclic_upper = pyo.Constraint(m.OMEGA, rule=_c13_cyclic_upper)

    # ============================================================
    # C14 -- Eqs. (27)-(29): linearized DistFlow (simplified radial
    # topology, see module docstring)
    # ============================================================
    def _c14_active_balance(mm, i, j, w, t):
        """Active power on corridor (i,j): electrolyser load attributed
        to substation j is served net of any RES output routed through
        that same corridor, scaled by the connection decision x[i,j]."""
        return mm.P_flow[i, j, w, t] == mm.x[i, j] * (mm.p[w, t] - mm.r[w, t]) / max(len(mm.I), 1)
    m.C14_active_balance = pyo.Constraint(m.I, m.J, m.OMEGA, m.T, rule=_c14_active_balance)

    # C14_voltage_drop -- FIXED (big-M gated on x[i,j]).
    #
    # v[j, w, t] carries no `i` index, but this rule is evaluated once
    # per (i, j) pair. Previously this was a hard equality, so every
    # RES site i that could reach substation j (regardless of whether
    # x[i,j]=1) imposed its own equation on the SAME v[j,w,t] variable
    # -- these are almost never mutually consistent once |I| > 1 sites
    # share a substation (see module docstring for the failure mode),
    # producing a structural infeasibility on any non-trivial instance.
    #
    # Fix: only the connected site (x[i,j]=1) should determine v[j,w,t];
    # for every other site sharing substation j, the relationship is
    # relaxed to a no-op via a big-M term. M is sized to the full
    # feasible voltage-squared range so the relaxed side never binds.
    def _c14_voltage_drop_upper(mm, i, j, w, t):
        M = mm.V_max_sq - mm.V_min_sq + mm.V_base ** 2
        return mm.v[j, w, t] <= mm.V_base ** 2 - 2 * mm.L[i, j] * (
            mm.R_line * mm.P_flow[i, j, w, t] + mm.X_line * mm.Q_flow[i, j, w, t]
        ) + M * (1 - mm.x[i, j])
    m.C14_voltage_drop_upper = pyo.Constraint(m.I, m.J, m.OMEGA, m.T, rule=_c14_voltage_drop_upper)

    def _c14_voltage_drop_lower(mm, i, j, w, t):
        M = mm.V_max_sq - mm.V_min_sq + mm.V_base ** 2
        return mm.v[j, w, t] >= mm.V_base ** 2 - 2 * mm.L[i, j] * (
            mm.R_line * mm.P_flow[i, j, w, t] + mm.X_line * mm.Q_flow[i, j, w, t]
        ) - M * (1 - mm.x[i, j])
    m.C14_voltage_drop_lower = pyo.Constraint(m.I, m.J, m.OMEGA, m.T, rule=_c14_voltage_drop_lower)

    def _c14_reactive_balance(mm, i, j, w, t):
        # Reactive load assumed proportional to active load (fixed power
        # factor pf=0.95 -> tan(phi) ~= 0.33) absent a supplied Q profile.
        return mm.Q_flow[i, j, w, t] == 0.33 * mm.P_flow[i, j, w, t]
    m.C14_reactive_balance = pyo.Constraint(m.I, m.J, m.OMEGA, m.T, rule=_c14_reactive_balance)

    # ============================================================
    # C16 -- Eq. (31): thermal limits, new cable corridors
    # ============================================================
    def _c16_new_corridor_upper(mm, i, j, w, t):
        cap = sum(mm.y[i, j, k2] for k2 in mm.K2) * mm.F_max_new
        return mm.P_flow[i, j, w, t] <= cap
    m.C16_new_corridor_upper = pyo.Constraint(m.I, m.J, m.OMEGA, m.T, rule=_c16_new_corridor_upper)

    def _c16_new_corridor_lower(mm, i, j, w, t):
        cap = sum(mm.y[i, j, k2] for k2 in mm.K2) * mm.F_max_new
        return -cap <= mm.P_flow[i, j, w, t]
    m.C16_new_corridor_lower = pyo.Constraint(m.I, m.J, m.OMEGA, m.T, rule=_c16_new_corridor_lower)

    # C17 (existing-branch thermal limit) enforced via Var bounds on
    # `P_existing` (Eq. 32) -- fixed rating, independent of x_ij as
    # required by the corrected formulation.

    # ============================================================
    # C18 -- Eq. (33): transformer thermal limit
    # ============================================================
    def _c18_transformer_limit(mm, j, w, t):
        inflow = sum(mm.P_flow[i, j, w, t] for i in mm.I) + mm.P_existing[j, w, t]
        return inflow <= mm.T_max
    m.C18_transformer_limit = pyo.Constraint(m.J, m.OMEGA, m.T, rule=_c18_transformer_limit)

    def _c18_transformer_limit_lower(mm, j, w, t):
        inflow = sum(mm.P_flow[i, j, w, t] for i in mm.I) + mm.P_existing[j, w, t]
        return -mm.T_max <= inflow
    m.C18_transformer_limit_lower = pyo.Constraint(m.J, m.OMEGA, m.T, rule=_c18_transformer_limit_lower)

    # ============================================================
    # Optional first-stage fixing (used by breakeven / VSS analyses)
    # ============================================================
    if fix_first_stage:
        _apply_first_stage_fixing(m, fix_first_stage)

    logger.info(
        "Built MILP: |I|=%d |J|=%d |K1|=%d |K2|=%d |T|=%d |Omega|=%d",
        len(sets.res_sites), len(sets.substations), len(sets.collection_cables),
        len(sets.connection_cables), n_period, n_scen,
    )
    return BuildResult(model=m, sets=sets, config=config, scenario_tree=scenario_tree)


def _apply_first_stage_fixing(m: "pyo.ConcreteModel", fixed: dict) -> None:
    """Fixes first-stage variables to given values, e.g. for the
    break-even-distance sweep (fix which site/substation pair is
    active) or the deterministic VSS run."""
    if "P_res_w" in fixed:
        m.P_res_w.fix(fixed["P_res_w"])
    if "P_res_s" in fixed:
        m.P_res_s.fix(fixed["P_res_s"])
    for (i, j), val in fixed.get("x", {}).items():
        m.x[i, j].fix(val)
    for (i, j, k2), val in fixed.get("y", {}).items():
        m.y[i, j, k2].fix(val)
    for (i, j, k1), val in fixed.get("z1", {}).items():
        m.z1[i, j, k1].fix(val)
    logger.debug("Applied first-stage fixing: %s", fixed)