# RICE13_FS/solve/noncoop.py
"""
Noncooperative (Nash) solver via iterative best responses (Gauss–Seidel).

Design notes
------------
- Fixed 10-year model grid: we do not pass `tstep` into `model_builder`.
  `tstep` is retained here as legacy-only for calendar-year mapping in μ seeding.
- Strict discounted pipeline: the returned solution always includes
  `solution["disc"][(r,t)]` for downstream discounted payoffs/stability.
- Spec/compat: `build_solution_spec_id` no longer carries `tstep`.
- Convergence guard: require at least one successful best-response update per
  outer iteration (tracked via `made_update`) to avoid false convergence.
- Convergence metric: primary check is the μ-residual to the unrelaxed best response;
  relaxed profile Δ is tracked as a diagnostic.
- Exogenous-S support: if provided, S remains fixed for the active region too.

Returns
-------
dict
  Full solution dict (2D econ, 1D climate/SLR, FS summaries if fs-utility),
  plus metadata: 'iterations', 'converged', 'max_delta', 'logfiles',
  REQUIRED `disc`, and spec/fingerprint identifiers.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging

import pyomo.environ as pe
from pyomo.opt import TerminationCondition as TC, SolverStatus as SS
import pandas as pd

from RICE13_FS.core import model_builder
from RICE13_FS.common import utils as _U
from RICE13_FS.common.utils import (
    has_converged_multi,
    get_max_delta,
    plot_nonconv_diag,
    clean,  # clamp helper (lower, upper)
    normalize_exogenous_S,
    build_ipopt,
    build_solution_spec_id, build_solution_fingerprint,
    _collect_1d, _collect_2d,
    fix_profiles_except,
    final_evaluation_setup, assert_exogenous_S_fixed,
    back_out_carbon_tax,
    attach_inequality_series,
)

logger = logging.getLogger(__name__)


# ------------------------- Seeding helpers -------------------------

def _init_profiles_from_solution(
    regions: List[str],
    periods: List[int],
    sol: Optional[Dict[str, Dict[tuple, float]]],
    bau_S: pd.DataFrame,
    base_year: int,
    backstop_switch_year: int,
    tstep: int,
    saving_seed: Optional[pd.DataFrame] = None,  # saving_rate_t.csv
    mu_seed: Optional[pd.DataFrame] = None,      # mu_init.csv
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """
    Initialize S[r][i] and μ[r][i] lists for all regions and periods.
    If `sol` provided, take S/μ from it (raw); otherwise seed S from BAU and
    μ by year threshold: μ=1 if year >= backstop_switch_year else 0.1.
    (The 0.1 floor matches historical initialization and avoids pathological
    IPOPT starts at exactly zero abatement.)
    """
    S_prof: Dict[str, List[float]] = {r: [0.0] * len(periods) for r in regions}
    mu_prof: Dict[str, List[float]] = {r: [0.0] * len(periods) for r in regions}

    if sol is not None and (sol.get("seed_kind") == "data"):
        sol = None
    if sol is not None and "S" in sol and "mu" in sol:
        for r in regions:
            for i, t in enumerate(periods):
                S_prof[r][i]  = float(sol["S"].get((r, t), 0.0))
                mu_prof[r][i] = float(sol["mu"].get((r, t), 0.0))
        return S_prof, mu_prof

    for r in regions:
        for i, t in enumerate(periods):
            if (saving_seed is not None) and (r in saving_seed.index) and (t in saving_seed.columns):
                S_prof[r][i] = float(saving_seed.at[r, t])
            else:
                S_prof[r][i] = float(bau_S.at[r, t])
            if (mu_seed is not None) and (r in mu_seed.index) and (t in mu_seed.columns):
                mu_prof[r][i] = float(mu_seed.at[r, t])
            else:
                year = base_year + t * tstep
                mu_prof[r][i] = 1.0 if year >= backstop_switch_year else 0.1
    return S_prof, mu_prof


def _objective_for_region(m: pe.ConcreteModel, r_star: str) -> pe.Objective:
    """Discounted intertemporal utility of the active region r_star."""
    return pe.Objective(expr=sum(m.disc[r_star, t] * m.U[r_star, t] for t in m.T), sense=pe.maximize)


# ------------------------- Main solver -------------------------

def solve_nash(
    params,
    T: int,
    tstep: int,
    *,
    utility: str,  # 'crra' or 'fs'
    solver_opts: Dict[str, Any],
    diagnostics_dir: Path,
    initial_solution: Optional[Dict[str, Any]],
    exogenous_S: Optional[pd.DataFrame],
    population_weight_envy_guilt: bool,
    max_iter: int,
    tol: float,
    relax: float,
    ignore_last_k_periods: int,
    region_order: Optional[List[str]] = None,
    # --- Discounting (FS only) ---
    # This Nash solver accepts a *global* per-period series {t -> disc_t}.
    # Regional discount alignment (e.g., one_pass/two_pass) is constructed upstream
    # and handled in the coalition solver / orchestrator layer.
    discount_series: Optional[Dict[int, float]] = None,  # {t -> disc_t}, t=1..T
    disc_tag: Optional[str] = None,                      # identity tag for discount series (goes into spec_id)
) -> Dict[str, Any]:
    """
    Iterative best-response Nash solver (discounted).
    """
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    regions = list(params.countries)
    periods = list(range(1, T + 1))
    if region_order is None:
        region_order = regions[:]

    # Initialize profiles from seed or BAU
    S_prof, mu_prof = _init_profiles_from_solution(
        regions, periods, initial_solution, params.bau_saving_rates,
        params.base_year, params.backstop_switch_year, tstep
    )

    ipopt_logs: List[Path] = []
    diag_evo: List[dict] = []
    last_success = False

    for k in range(1, max_iter + 1):
        logger.info("Nash iteration %d / %d", k, max_iter)

        prev_S = {r: S_prof[r][:] for r in regions}
        prev_mu = {r: mu_prof[r][:] for r in regions}
        made_update = False  # track if any region successfully updated this outer iter

        # Window for residual diagnostics and convergence (respect ignore_last_k_periods)
        n_total = len(periods)
        k_ignored = int(ignore_last_k_periods or 0)
        if k_ignored < 0:
            k_ignored = 0
        if k_ignored >= n_total and n_total > 0:
            k_ignored = n_total - 1
        n_use = max(0, n_total - k_ignored)

        # Track μ-residual to the *unrelaxed* best response vs iteration baseline (prev_mu).
        # This is the primary convergence metric; relaxed Δ is diagnostic-only below.
        max_resid_iter: float = float("inf")
        resid_updated: bool = False

        for r_star in region_order:
            log_path = diagnostics_dir / f"ipopt_nash_{utility}_it{k:03d}_{r_star}.log"
            ipopt = build_ipopt(solver_opts, log_path)

            # Normalize exogenous S once so model_builder always sees regions×(1..T)
            exoS = normalize_exogenous_S(exogenous_S, params.countries, T) if exogenous_S is not None else None

            # Fixed 10-year grid: do NOT pass tstep into builder (tstep is legacy-only).
            m = model_builder.build_model(
                params=params,
                T=T,
                utility=utility,
                exogenous_S=exoS,
                population_weight_envy_guilt=population_weight_envy_guilt,
                discount_series=discount_series,
            )

            fix_profiles_except(
                m, regions, periods, S_prof, mu_prof,
                active_region=r_star,
                respect_exogenous_S=(exogenous_S is not None),
            )
            m.OBJ = _objective_for_region(m, r_star)
            # Seed the active region with previous-iteration profiles (good IPOPT start)
            for i, t in enumerate(periods):
                s_lb = float(m.S[r_star, t].lb) if m.S[r_star, t].lb is not None else 0.0
                s_ub = float(m.S[r_star, t].ub) if m.S[r_star, t].ub is not None else 1.0
                mu_lb = float(m.mu[r_star, t].lb) if m.mu[r_star, t].lb is not None else 0.0
                mu_ub = float(m.mu[r_star, t].ub) if m.mu[r_star, t].ub is not None else 1.0
                m.S[r_star, t].value  = clean(S_prof[r_star][i],  lower=s_lb,  upper=s_ub)
                m.mu[r_star, t].value = clean(mu_prof[r_star][i], lower=mu_lb, upper=mu_ub)


            try:
                res = ipopt.solve(m, tee=logger.isEnabledFor(logging.DEBUG))
                stat = res.solver.status
                term = res.solver.termination_condition
                if not (stat in (SS.ok, SS.warning) and term in (TC.optimal, TC.locallyOptimal)):
                    logger.warning("Best response for %s ended with status=%s, term=%s", r_star, stat, term)
                    raise RuntimeError(f"{stat}/{term}")

                # Update active profiles (relaxed; clamp to model bounds)
                for i, t in enumerate(periods):
                    s_lb = float(m.S[r_star, t].lb) if m.S[r_star, t].lb is not None else 0.0
                    s_ub = float(m.S[r_star, t].ub) if m.S[r_star, t].ub is not None else 1.0
                    mu_lb = float(m.mu[r_star, t].lb) if m.mu[r_star, t].lb is not None else 0.0
                    mu_ub = float(m.mu[r_star, t].ub) if m.mu[r_star, t].ub is not None else 1.0
                    br_S  = clean(float(pe.value(m.S[r_star, t])),  lower=s_lb,  upper=s_ub)
                    br_mu = clean(float(pe.value(m.mu[r_star, t])), lower=mu_lb, upper=mu_ub)

                    # μ-residual to best response vs iteration baseline (prev_mu)
                    if i < n_use:
                        delta_resid = abs(br_mu - prev_mu[r_star][i])
                        if not resid_updated:
                            max_resid_iter = delta_resid
                            resid_updated = True
                        elif delta_resid > max_resid_iter:
                            max_resid_iter = delta_resid

                    S_prof[r_star][i]  = relax * br_S  + (1 - relax) * S_prof[r_star][i]
                    mu_prof[r_star][i] = relax * br_mu + (1 - relax) * mu_prof[r_star][i]
                made_update = True
            except Exception as e:
                logger.warning("Skipping update for %s in iter %d due to IPOPT failure: %s", r_star, k, e)

            if _U.DIAGNOSTICS_ON:
                ipopt_logs.append(log_path)

        # Check relaxed-change Δ on both μ and S (diagnostic; optionally ignore tail window)
        conv_delta = has_converged_multi(
            prevs=[prev_mu, prev_S],
            currs=[mu_prof, S_prof],
            tols=[tol, tol],
            ignore_last_k_periods=ignore_last_k_periods,
        )
        max_d = max(
            get_max_delta(prev_mu, mu_prof, ignore_last_k_periods),
            get_max_delta(prev_S, S_prof, ignore_last_k_periods),
        )
        # Residual-based convergence: require at least one updated residual AND
        # max residual <= tol, plus at least one successful update this iteration.
        conv_resid = bool(resid_updated and (max_resid_iter <= tol))
        conv = bool(conv_resid and made_update)

        diag_evo.append({
            "iteration": k,
            # 'max_delta' records the μ-residual to the unrelaxed best response (primary metric).
            "max_delta": float(max_resid_iter if resid_updated else float("inf")),
            # Diagnostic: relaxed-change max Δ over μ and S.
            "max_delta_relaxed": float(max_d),
        })
        logger.info(
            "After iteration %d: max residual = %.3e (tol=%.3e) | max Δ(relaxed) = %.3e | any update=%s → %s",
            k,
            max_resid_iter if resid_updated else float("inf"),
            tol,
            max_d,
            made_update,
            "CONVERGED" if conv else "continue",
        )
        if not made_update:
            logger.error("All best-responses failed at iter %d → aborting (avoid fake convergence).", k)
            break
        if conv:
            last_success = True
            break

    converged = (len(diag_evo) > 0 and diag_evo[-1]["max_delta"] <= tol) and last_success

    # Plot convergence diagnostics (best-effort)
    try:
        plot_nonconv_diag(
            diag_evo,
            diagnostics_dir / "nash_max_delta.png",
            title="Nash max residual (μ best-response)",
        )
    except Exception as e:
        logger.debug("Could not write convergence plot: %s", e)

    # Final evaluation with fixed profiles to collect a consistent solution portfolio
    logger.info("Final evaluation with fixed profiles (collecting full solution).")
    exoS = normalize_exogenous_S(exogenous_S, params.countries, T) if exogenous_S is not None else None
    
    # Final evaluation — builder fixes S when exogenous_S is used
    m_final = model_builder.build_model(
        params=params,
        T=T,
        utility=utility,
        exogenous_S=exoS,  # if not None, builder fixes S internally
        population_weight_envy_guilt=population_weight_envy_guilt,
        discount_series=discount_series,
    )
    
    if exoS is not None:
        assert_exogenous_S_fixed(m_final, exoS, regions, periods)
    final_evaluation_setup(
        m_final, regions, mu_prof, S_prof,
        exogenous_S_used=(exoS is not None),
    )


    ipopt_final_path = diagnostics_dir / "ipopt_nash_final.log"
    ipopt_final = build_ipopt(solver_opts, ipopt_final_path)
    if _U.DIAGNOSTICS_ON:
        ipopt_logs.append(ipopt_final_path)
    try:
        res_final = ipopt_final.solve(m_final, tee=False)
        stat_final = res_final.solver.status
        term_final = res_final.solver.termination_condition
        optimal_final = bool(stat_final in (SS.ok, SS.warning) and term_final in (TC.optimal, TC.locallyOptimal))
        if not optimal_final:
            logger.warning("Nash final evaluation finished with status=%s, term=%s", stat_final, term_final)
    except Exception as e:
        logger.error("Final evaluation failed: %s", e)
        raise RuntimeError("Nash final evaluation failed") from e

    # Collect solution arrays
    sol: Dict[str, Any] = {}
    for name in ("K", "S", "I", "Q", "Y", "mu", "AB", "D", "C", "E_ind", "U"):
        sol[name] = _collect_2d(m_final, name)
    if utility == "fs":
        sol["FS_envy_avg"]  = _collect_2d(m_final, "FS_envy_avg")
        sol["FS_guilt_avg"] = _collect_2d(m_final, "FS_guilt_avg")
    for name in (
        "E_tot", "M_at", "M_up", "M_lo", "T_at", "T_lo", "F",
        "slr", "slr_TE", "gsic_remain", "gsic_melt", "gsic_cum",
        "gis_remain", "gis_melt", "gis_cum",
        "ais_remain", "ais_melt", "ais_cum",
    ):
        sol[name] = _collect_1d(m_final, name)

    # REQUIRED for discounted payoffs downstream (coalition cache/export uses discounted lifetime payoffs)
    sol["disc"] = {(r, int(t)): float(pe.value(m_final.disc[r, t])) for r in m_final.REGIONS for t in m_final.T}

    # If S was exogenous, attach normalized copy for cache/spec compatibility
    if exogenous_S is not None:
        sol["S_exogenous"] = normalize_exogenous_S(exogenous_S, params.countries, T)

    # Meta & identifiers
    sol["solver_status"] = str(stat_final)
    sol["termination"] = str(term_final)
    sol["optimal"] = bool(optimal_final)
    sol["mode"] = f"nash_{utility}"
    sol["utility"] = utility
    sol["S_source"] = "exogenous" if exogenous_S is not None else "optimal"
    sol["iterations"] = len(diag_evo)
    sol["converged"] = bool(converged)
    sol["max_delta"] = float(diag_evo[-1]["max_delta"]) if diag_evo else None
    sol["logfiles"] = [str(p) for p in ipopt_logs]

    spec_id = build_solution_spec_id(
        utility=utility,
        T=T,
        countries=params.countries,
        population_weight_envy_guilt=population_weight_envy_guilt,
        exogenous_S=sol.get("S_exogenous"),
        negishi_use=False,
        negishi_weights=None,
        disc_tag=(disc_tag if utility == "fs" else None),
    )
    sol["spec_id"] = spec_id
    sol["fingerprint"] = build_solution_fingerprint(mode="nash", coalition_vec=None, spec_id=spec_id)

    back_out_carbon_tax(sol, params)

    # Attach global inequality diagnostics (Gini, Atkinson) based on
    # population-weighted per-capita consumption C/L for each period.
    regions = list(params.countries)
    periods = list(range(1, T + 1))
    attach_inequality_series(sol, params=params, regions=regions, periods=periods)

    return sol
