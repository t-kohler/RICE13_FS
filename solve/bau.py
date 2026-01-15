# RICE13_FS/solve/bau.py
"""
BAU solver (baseline) for the RICE-2013_FS IAM.

Design:
  - Time grid is fixed at 10-year periods inside the model.
    `tstep` is retained only for legacy compatibility and calendar-year mapping
    in the μ step policy below.
  - S is fixed to BAU saving rates (params.bau_saving_rates).
  - μ is fixed to a step policy: 0 before backstop_switch_year, 1 afterwards.
  - Utility can be 'crra' or 'fs'. The objective is population-weighted in both
    cases. (It does not affect BAU decisions because S and μ are fixed, but it
    keeps the objective consistent with planner/Nash/coallition solve structures.)
  - Returns a strict discounted-payoff pipeline: the solution must include 'disc'
    because downstream payoff/stability logic and coalition cache entries are
    stored/compared in discounted lifetime units.

Outputs
-------
dict
  Keys for 2D variables are (region, period). 1D variables are keyed by period.
  Includes REQUIRED 'disc', and attaches normalized S_exogenous for spec/FP.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

import logging
import pyomo.environ as pe
from pyomo.opt import TerminationCondition as TC, SolverStatus as SS

from RICE13_FS.core import model_builder
from RICE13_FS.common import utils as _U
from RICE13_FS.common.utils import (
    _collect_1d,
    _collect_2d,
    build_ipopt,
    normalize_exogenous_S,
    build_solution_spec_id,
    build_solution_fingerprint,
    back_out_carbon_tax,
    attach_inequality_series,
)

logger = logging.getLogger(__name__)


def solve_bau(
    params,
    T: int,
    tstep: int,
    solver_opts: Dict[str, Any],
    diagnostics_dir: Path,
    *,
    utility: str = "crra",
    population_weight_envy_guilt: bool = True,
) -> Dict[str, Any]:
    """
    Solve the BAU baseline with:
      - S fixed to BAU saving rates
      - μ fixed to a step policy (0 before backstop_switch_year, 1 afterwards)

    Parameters
    ----------
    params : Params
    T : int
        Number of modeled periods (1..T).
    tstep : int
        Legacy-only; used only for calendar-year mapping in the μ step policy.
    solver_opts : dict
        {'executable': '/path/to/ipopt', 'options': {'tol':..., 'max_iter':...}}
    diagnostics_dir : Path
        Where IPOPT logs and any diagnostics are written.
    utility : {'crra','fs'}
        Utility to attach (objective only; BAU decisions are fixed).
    population_weight_envy_guilt : bool
        FS envy/guilt weighting switch (only relevant if utility=='fs').

    Returns
    -------
    dict
        Solution dict with keys like 'K','S','mu','C','Y', ... (2D),
        climate/SLR (1D), REQUIRED 'disc', and metadata for cache/spec IDs.
    """
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    log_path = diagnostics_dir / f"ipopt_bau_{utility}.log"

    # --- Build model with S fixed to BAU saving rates
    exog_S = params.bau_saving_rates
    m = model_builder.build_model(
        params=params,
        T=T,
        utility=utility,
        exogenous_S=exog_S,
        population_weight_envy_guilt=population_weight_envy_guilt,
    )

    # --- Fix μ per BAU step policy: 0 until switch year, then 1
    for r in m.REGIONS:
        for t in m.T:
            # remove bounds so fix() never conflicts
            m.mu[r, t].setlb(None)
            m.mu[r, t].setub(None)
            year = int(params.base_year) + int(t) * int(tstep)
            m.mu[r, t].fix(1.0 if year >= int(params.backstop_switch_year) else 0.0)
    logger.info(
        "BAU μ-policy: μ=0 for years < %d, μ=1 for ≥ %d.",
        int(params.backstop_switch_year), int(params.backstop_switch_year)
    )

    # --- Population-weighted discounted objective (structure only; choices are fixed by policy)
    if utility not in ("crra", "fs"):
        raise ValueError(f"Unknown utility for BAU: {utility!r}")
    m.OBJ = pe.Objective(
        expr=sum(m.disc[r, t] * m.U[r, t] for r in m.REGIONS for t in m.T),
        sense=pe.maximize,
    )

    # --- Solve
    ipopt = build_ipopt(solver_opts, log_path)
    res = ipopt.solve(m, tee=(logging.getLogger().getEffectiveLevel() <= logging.DEBUG))
    term = res.solver.termination_condition
    stat = res.solver.status
    optimal = bool(stat in (SS.ok, SS.warning) and term in (TC.optimal, TC.locallyOptimal))
    if not optimal:
        logger.warning(
            "BAU solve finished with status=%s, termination=%s",
            stat, res.solver.termination_condition
        )

    # --- Collect results
    sol: Dict[str, Any] = {}

    # 2D (region, t): core econ + utility
    for name in ("K", "S", "I", "Q", "Y", "mu", "AB", "D", "C", "E_ind", "U"):
        d = _collect_2d(m, name)
        sol[name] = {(r, int(t)): float(v) for (r, t), v in d.items()}

    # FS summaries exist only when utility == 'fs'; collect as 2D (region,t)
    if utility == "fs":
        for name in ("FS_envy_avg", "FS_guilt_avg"):
            if hasattr(m, name):
                d = _collect_2d(m, name)
                sol[name] = {(r, int(t)): float(v) for (r, t), v in d.items()}

    # 1D (t): climate & carbon cycle & SLR
    for name in (
        "E_tot", "M_at", "M_up", "M_lo", "T_at", "T_lo", "F",
        "slr", "slr_TE", "gsic_remain", "gsic_melt", "gsic_cum",
        "gis_remain", "gis_melt", "gis_cum",
        "ais_remain", "ais_melt", "ais_cum",
    ):
        sol[name] = _collect_1d(m, name)

    # REQUIRED: discount factors for strict discounted-payoff pipeline
    # (coalition store/export expects discounted lifetime payoff rows)
    sol["disc"] = {(r, int(t)): float(pe.value(m.disc[r, t])) for r in m.REGIONS for t in m.T}

    # Meta & spec/fingerprint (attach normalized exogenous S for spec matching)
    sol["solver_status"] = str(stat)
    sol["termination"] = str(term)
    sol["optimal"] = optimal
    sol["utility"] = utility
    sol["S_source"] = "bau"
    sol["S_exogenous"] = normalize_exogenous_S(exog_S, params.countries, T)
    if _U.DIAGNOSTICS_ON:
        sol["logfile"] = str(log_path)

    spec_id = build_solution_spec_id(
        utility=utility,
        T=T,
        countries=params.countries,
        population_weight_envy_guilt=population_weight_envy_guilt,
        exogenous_S=sol.get("S_exogenous"),
        negishi_use=False,
        negishi_weights=None,
    )
    sol["spec_id"] = spec_id
    sol["fingerprint"] = build_solution_fingerprint(
        mode="bau", coalition_vec=None, spec_id=spec_id
    )

    back_out_carbon_tax(sol, params)

    # Attach global inequality diagnostics (Gini, Atkinson) based on
    # population-weighted per-capita consumption C/L for each period.
    regions = list(params.countries)
    periods = list(range(1, T + 1))
    attach_inequality_series(sol, params=params, regions=regions, periods=periods)    

    return sol
