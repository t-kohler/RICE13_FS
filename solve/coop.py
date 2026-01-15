# RICE13_FS/solve/coop.py
"""
Cooperative (planner) solver for the RICE-2013_FS IAM.

Design:
- Utility can be 'crra' or 'fs'. The period objective inside the model is already
  population-weighted; FS envy/guilt averaging across countries follows the flag
  `population_weight_envy_guilt` (population-weighted shares vs simple fractions).
- Sea level rise is endogenous (warm-started from params).
- Saving rates S can be optimized (default) or EXOGENOUS if a DataFrame is
  provided via `exogenous_S` (regions x period, columns MUST include 1..T).
- Returns discounted-ready solutions: `solution["disc"][(r,t)]` is always present.
- Optional Negishi weights apply a time/region weight W[r,t] to the planner objective
  (validated lightly here; normalization is enforced upstream).
- FS discount alignment can override discount factors via `discount_series`.
  If a non-default series is used for FS, `disc_tag` must be passed so cache keys and
  exports remain deterministic.

Notes:
- The model grid is fixed at 10-year periods. `tstep` is legacy-only: retained for
  call-site compatibility and calendar-year mapping in downstream helpers.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import logging
import pandas as pd
import pyomo.environ as pe
from pyomo.opt import TerminationCondition as TC, SolverStatus as SS
import math

from RICE13_FS.core import model_builder
from RICE13_FS.common import utils as _U
from RICE13_FS.common.utils import (
    build_ipopt,
    _collect_1d,
    _collect_2d,
    print_most_violated_constraints,
    normalize_exogenous_S,
    build_solution_spec_id,
    build_solution_fingerprint,
    back_out_carbon_tax,
    attach_inequality_series
)
from RICE13_FS.analysis.negishi import _crra_mu, fs_negishi_mu


logger = logging.getLogger(__name__)


def solve_planner(
    params,
    T: int,
    tstep: int,  # legacy-only; not used by the builder (fixed 10-year grid)
    *,
    utility: str,
    solver_opts: Dict[str, Any],
    diagnostics_dir: Path,
    exogenous_S: Optional[pd.DataFrame],
    population_weight_envy_guilt: bool,
    # --- Negishi controls ---
    negishi_weights: Optional[pd.DataFrame],
    negishi_use: bool,
    # --- Discounting (FS only) ---
    # discount_series may be either:
    #   - global: {t -> disc_t}, applied to all regions, or
    #   - regional: {(r,t) -> disc_rt}, if constructed upstream.
    discount_series: Optional[Dict[Any, float]] = None,
    disc_tag: Optional[str] = None,                      # identity tag for discount series (goes into spec_id) 
) -> Dict[str, Any]:
    """
    Solve the cooperative planner problem (CRRA or FS).

    Parameters
    ----------
    params : Params
        Data and warm-starts (see data_loader).
    T : int
        Number of modeled periods (1..T).
    tstep : int
        Legacy-only. The model grid is fixed at 10-year periods; `tstep` is not used by
        the model builder and exists for compatibility / year mapping in helpers.    utility : {'crra','fs'}
        Utility to attach in the model.
    solver_opts : dict
        IPOPT options, e.g., {'executable': '/path/ipopt', 'options': {'tol':..., 'max_iter':...}}
    diagnostics_dir : Path
        Directory for IPOPT logs and debug artifacts.
    exogenous_S : DataFrame or None
        If provided, fixes S[r,t] to these values (regions x periods). Columns must include 1..T.
    population_weight_envy_guilt : bool
        For FS only: if True, compute envy/guilt as population-weighted shares among others;
        else simple unweighted fractions. Ignored for CRRA.
    negishi_weights : DataFrame or None
        If applying Negishi in the planner objective, provide a DataFrame with index=regions,
        columns=1..T, column-wise sums = 1 (enforced upstream).
    negishi_use : bool
        If True, multiply the objective by the provided Negishi weights W[r,t]. If True but
        negishi_weights is None, a ValueError is raised.
    discount_series : dict or None
        Optional discount factors to override the default geometric path from rho.
        Supported shapes:
          - global: {t -> disc_t}, applied uniformly to all regions
          - regional: {(region, t) -> disc_rt}, applied region-by-region (built upstream)
        Intended usage is FS-only discount alignment; CRRA typically uses the default path.
    disc_tag : str or None
        Optional identity tag for the discount series (e.g., "disc:data", "disc:file:<sha8>",
        "disc:one_pass:planner:<sha8>"). Included in the solution spec_id for cache safety.

    Returns
    -------
    dict
        Solution dict with keys like 'K','S','I','Q','Y','mu','AB','D','C','E_ind','U', ...
        Includes meta fields: 'utility', 'S_source', 'logfile', REQUIRED 'disc', and spec/fingerprint ids.
    """
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    s_mode = "exoS" if exogenous_S is not None else "optS"
    log_path = diagnostics_dir / f"ipopt_planner_{utility}_{s_mode}.log"


    exoS = normalize_exogenous_S(exogenous_S, params.countries, T) if exogenous_S is not None else None
    m = model_builder.build_model(
        params=params,
        T=T,
        utility=utility,
        exogenous_S=exoS,
        population_weight_envy_guilt=population_weight_envy_guilt,
        discount_series=discount_series,
    )
    
    # --- Objective: discounted utility, optionally Negishi-weighted
    if utility not in ("crra", "fs"):
        raise ValueError(f"Unknown utility: {utility!r}")

    if negishi_use and negishi_weights is None:
        raise ValueError("negishi_use=True but negishi_weights is None.")

    # Light validation when applying weights
    use_weights = bool(negishi_use and negishi_weights is not None)
    if use_weights:
        missing_r = [str(r) for r in m.REGIONS if str(r) not in negishi_weights.index]
        missing_t = [int(t) for t in m.T if int(t) not in negishi_weights.columns]
        if missing_r:
            raise ValueError(f"Negishi weights missing regions: {missing_r}")
        if missing_t:
            raise ValueError(f"Negishi weights missing periods: {missing_t}")

    obj_expr = sum(
        (float(negishi_weights.at[str(r), int(t)]) if use_weights else 1.0)
        * m.disc[r, t] * m.U[r, t]
        for r in m.REGIONS for t in m.T
    )
    m.OBJ = pe.Objective(expr=obj_expr, sense=pe.maximize)

    # --- Solve
    ipopt = build_ipopt(solver_opts, log_path)
    res = ipopt.solve(m, tee=logger.isEnabledFor(logging.DEBUG))
    term = res.solver.termination_condition
    stat = res.solver.status
    optimal = bool(stat in (SS.ok, SS.warning) and term in (TC.optimal, TC.locallyOptimal))
    if not optimal:
        logger.warning("Planner solve finished with status=%s, termination=%s", stat, term)

    # --- Collect results (2D: regions×periods)
    sol: Dict[str, Any] = {}
    for name in ("K", "S", "I", "Q", "Y", "mu", "AB", "D", "C", "E_ind", "U"):
        sol[name] = _collect_2d(m, name)

    if utility == "fs":
        sol["FS_envy_avg"]  = _collect_2d(m, "FS_envy_avg")
        sol["FS_guilt_avg"] = _collect_2d(m, "FS_guilt_avg")

    # 1D climate/SLR components
    for name in (
        "E_tot", "M_at", "M_up", "M_lo", "T_at", "T_lo", "F",
        "slr", "slr_TE", "gsic_remain", "gsic_melt", "gsic_cum",
        "gis_remain", "gis_melt", "gis_cum",
        "ais_remain", "ais_melt", "ais_cum",
    ):
        sol[name] = _collect_1d(m, name)

    # --- Add discount factors explicitly for downstream discounted sums
    sol["disc"] = {(r, int(t)): float(pe.value(m.disc[r, t])) for r in m.REGIONS for t in m.T}

    # Meta & spec/fingerprint
    sol["solver_status"] = str(stat)
    sol["termination"] = str(term)
    sol["optimal"] = optimal
    sol["utility"] = utility
    sol["S_source"] = "exogenous" if exogenous_S is not None else "optimal"
    if _U.DIAGNOSTICS_ON:
        sol["logfile"] = str(log_path)
    if exogenous_S is not None:
        sol["S_exogenous"] = normalize_exogenous_S(exogenous_S, params.countries, T)
    if use_weights:
        sol["negishi_weights"] = True  # marker for exporters (_planner_name adds "_N")

    spec_id = build_solution_spec_id(
        utility=utility,
        T=T,
        countries=params.countries,
        population_weight_envy_guilt=population_weight_envy_guilt,
        exogenous_S=sol.get("S_exogenous"),
        negishi_use=use_weights,
        negishi_weights=(negishi_weights if use_weights else None),
        disc_tag=disc_tag if utility == "fs" else None,
    )
    sol["spec_id"] = spec_id
    sol["fingerprint"] = build_solution_fingerprint(mode="planner", coalition_vec=None, spec_id=spec_id)

    if logger.isEnabledFor(logging.DEBUG):
        print_most_violated_constraints(m)

    back_out_carbon_tax(sol, params)

    # ------------------------------------------------------------------
    # Back out SCC along the cooperative planner path
    #
    # We compute SCC in:
    #   - welfare units (same numéraire as the planner objective), and
    #   - money units (divide by ∂W/∂C_{r,t} under the run's utility/weights).
    # This is diagnostic post-processing; it does not affect the solve.
    # ------------------------------------------------------------------
    # 1) Shadow value of net GDP: λ_Y(r,t) = ∂W/∂Y_{r,t}
    lambda_Y: Dict[tuple, float] = {}
    for r in m.REGIONS:
        r_key = str(r)
        for t in m.T:
            t_key = int(t)
            c = m.Y_eq[r, t]
            try:
                val = m.dual[c]
            except KeyError:
                val = None
            lambda_Y[(r_key, t_key)] = float(val) if val is not None else float("nan")
    #sol["lambda_Y"] = lambda_Y

    # 2) SCC in welfare units (same numéraire as planner objective)
    carbon_tax = sol.get("carbon_tax", {})
    mu = sol.get("mu", {})
    SCC_welfare: Dict[tuple, float] = {}

    # Reuse Negishi flag; same notion as in the objective
    use_weights = bool(negishi_use and negishi_weights is not None)
    regions_list = [str(r) for r in m.REGIONS]

    for r in m.REGIONS:
        r_key = str(r)
        for t in m.T:
            t_key = int(t)
            key = (r_key, t_key)
            lam = lambda_Y.get(key, float("nan"))
            tau = carbon_tax.get(key)
            mu_rt = mu.get(key)

            if tau is None or mu_rt is None:
                SCC_welfare[key] = float("nan")
                continue
            # Skip boundary cases where FOC does not pin down SCC
            if mu_rt <= 1e-6 or mu_rt >= 1.0 - 1e-6:
                SCC_welfare[key] = float("nan")
                continue
            if not math.isfinite(lam):
                SCC_welfare[key] = float("nan")
                continue

            # λ_Y (utility per $) × τ (k$ per tC) → SCC in welfare units per tC
            SCC_welfare[key] = lam * tau

    #sol["SCC_welfare"] = SCC_welfare

    # 3) SCC in money units, using each region's own consumption as numéraire
    #    SCC^money_{r,t} = SCC^welfare_{r,t} / (∂W/∂C_{r,t})
    #    with ∂W/∂C_{r,t} = w_{r,t} * disc_{r,t} * (∂U_{r,t}/∂C_{r,t})
    SCC_money: Dict[tuple, float] = {}
    social_MU_C: Dict[tuple, float] = {}

    for r in m.REGIONS:
        r_key = str(r)
        # Region-specific curvature for CRRA
        if utility == "crra":
            eta_r = float(params.crra_eta[r_key])

        for t in m.T:
            t_key = int(t)
            key = (r_key, t_key)
            scc_w = SCC_welfare.get(key, float("nan"))
            if not math.isfinite(scc_w):
                SCC_money[key] = float("nan")
                continue

            # Negishi weight in the planner objective
            if use_weights:
                w_rt = float(negishi_weights.at[r_key, t_key])
            else:
                w_rt = 1.0

            disc_rt = float(m.disc[r, t])
            C_rt = sol["C"][key]
            L_rt = float(params.L.at[r_key, t_key])

            # Marginal utility of aggregate consumption dU/dC_{r,t}
            if utility == "crra":
                # _crra_mu ∝ (C/L)^(-η); objective uses U = L * u(c_pc_th), c_pc_th=(C/L)*1000
                mu_proxy = _crra_mu(C_rt, L_rt, eta_r)
                dU_dC = (1000.0 ** (1.0 - eta_r)) * mu_proxy
            elif utility == "fs":
                # fs_negishi_mu ∝ 1 + α share_rich - β share_poor; FS utility uses payoff_pc=(C/L)*1000
                mu_proxy = fs_negishi_mu(
                    r=r_key,
                    t=t_key,
                    params=params,
                    regions=regions_list,
                    C=sol["C"],
                    L=params.L,
                    population_weight_envy_guilt=population_weight_envy_guilt,
                )
                dU_dC = 1000.0 * mu_proxy
            else:
                dU_dC = float("nan")

            if not (w_rt > 0.0 and disc_rt > 0.0 and math.isfinite(dU_dC) and dU_dC > 0.0):
                SCC_money[key] = float("nan")
                continue

            # ∂W/∂C_{r,t} = w_{r,t} * disc_{r,t} * dU/dC_{r,t}
            marginal_utility_weighted = w_rt * disc_rt * dU_dC
            social_MU_C[key] = marginal_utility_weighted
            SCC_money[key] = scc_w / marginal_utility_weighted

    sol["SCC_money"] = SCC_money
    #sol["social_MU_C"] = social_MU_C

    # 4) Global SCC in money units, using an equal-per-capita lump-sum transfer
    #
    # Conceptually: SCC^welfare_t is the marginal change in the planner's
    # objective from a 1-ton pulse of emissions in period t. To express this
    # in "dollars per ton" under a hypothetical uniform per-capita transfer,
    # we divide SCC^welfare_t by the social marginal utility of giving every
    # person in the world one more unit of consumption in period t:
    #
    #   MU_global_pc,t = sum_r (dW/dC_{r,t}) * L_{r,t}
    #
    # with dW/dC_{r,t} taken from `social_MU_C` above.
    SCC_global_money_pc: Dict[int, float] = {}
    for t in m.T:
        t_key = int(t)

        # Pick the (common) welfare SCC for this period from any region
        scc_w_t = None
        for r in m.REGIONS:
            r_key = str(r)
            val = SCC_welfare.get((r_key, t_key), float("nan"))
            if math.isfinite(val):
                scc_w_t = val
                break

        if scc_w_t is None:
            SCC_global_money_pc[t_key] = float("nan")
            continue

        # Social marginal utility of a 1-unit per-capita lump-sum to everyone
        mu_global_pc = 0.0
        L_world = 0.0
        
        for r in m.REGIONS:
            r_key = str(r)
            key = (r_key, t_key)
            mu_c_weighted = social_MU_C.get(key)
            if mu_c_weighted is None or not math.isfinite(mu_c_weighted):
                continue
            L_rt = float(params.L.at[r_key, t_key])
            mu_global_pc += mu_c_weighted * L_rt
            L_world += L_rt
            
        avg_mu_global = mu_global_pc / L_world
        if avg_mu_global <= 0.0 or not math.isfinite(mu_global_pc):
            SCC_global_money_pc[t_key] = float("nan")
        else:
            SCC_global_money_pc[t_key] = scc_w_t / avg_mu_global

    # SCC in money units under a world-wide, equal-per-capita lump-sum numéraire.
    # Inherits utility curvature + (optional) Negishi weights from the planner objective.

    sol["SCC_global_money_pc"] = SCC_global_money_pc

    # Attach global inequality diagnostics (Gini, Atkinson) based on
    # population-weighted per-capita consumption C/L for each period.
    regions = list(params.countries)
    periods = list(range(1, T + 1))
    attach_inequality_series(sol, params=params, regions=regions, periods=periods)

    return sol