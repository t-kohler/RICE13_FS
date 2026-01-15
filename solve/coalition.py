# RICE13_FS/solve/coalition.py
"""
Coalition Nash solver for RICE-2013_FS.

Game:
  • Inside the coalition, members maximize their joint discounted objective (planner-style).
  • Outside the coalition, each non-member maximizes their own discounted objective.
  • Iterate best responses: Coalition block ↔ each singleton non-member.

Result:
  • A Nash equilibrium between "the coalition" and "the set of non-members".
  • Optionally, neighbor coalitions (remove one member; add one non-member) for stability tests.

Notes:
  • Utility: 'crra' or 'fs'. FS objective is population-weighted inside model_builder.
  • FS envy/guilt weighting across regions follows `population_weight_envy_guilt`.
  • Exogenous S is supported: if provided, S stays fixed for all regions; only μ moves.
  • Robust to IPOPT failures: skip profile updates for failing blocks. If *all* blocks fail
    in an iteration, abort to avoid fake convergence.
  • Convergence is assessed using a μ-residual to the *unrelaxed* best response (primary),
    with relaxed profile deltas tracked only for diagnostics.
  • Periods are model periods 1..T. Calendar years are only used for μ seeding/backstop logic.
  • Discount alignment (FS optional): an injected `discount_series` must come with a stable
    `disc_tag` so cache keys and exports remain deterministic.

Reuse (neighbors only):
  • Internal neighbors that are singletons (sum(vec)==1) may reuse a precomputed Nash result
    iff the full spec matches (utility, countries/order, S policy, Negishi), i.e., same spec_id.
  • External neighbors that are grand coalitions (sum(vec)==N) may reuse a precomputed Planner
    iff the full spec matches (same spec_id).
  • Endogenous/optimal S is never reused across specs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Iterable
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
    clean,  # clamp helper (lower/upper)
    normalize_exogenous_S,
    build_ipopt,
    build_solution_spec_id, build_solution_fingerprint,
    _collect_1d, _collect_2d,
    fix_profiles,
    unfix_controls, 
    final_evaluation_setup,
    assert_exogenous_S_fixed,
    back_out_carbon_tax,
    attach_inequality_series,
)

logger = logging.getLogger(__name__)


# ------------------- Coalition profile helpers -------------------

def _init_profiles_from_seed(
    regions: List[str],
    periods: List[int],
    seed: Optional[Dict[str, Dict[tuple, float]]],
    bau_S: pd.DataFrame,
    base_year: int,
    backstop_switch_year: int,
    tstep: int,
    saving_seed: Optional[pd.DataFrame] = None,  # saving_rate_t.csv (CRRA planner S)
    mu_seed: Optional[pd.DataFrame] = None,      # mu_init.csv (strictly positive)
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """
    Initialize S[r][i] and μ[r][i] lists.

    Priority:
      1) If a real seed solution is provided, use it literally.
      2) Else prefer saving_rate_t.csv and mu_init.csv.
      3) Else fall back to BAU S and Excel-style μ step (pre-backstop ~ 0, post-backstop 1).

    Notes:
      - Treat legacy “data” seeds (or all-zero S/μ) as *no seed*.
      - Keep μ inside model bounds: use a tiny positive floor before backstop.
    """
    # Keep this in sync with your model's mu lower bound
    MU_FLOOR = 1e-8

    def _looks_empty_seed(s: Optional[Dict[str, Dict[tuple, float]]]) -> bool:
        if not s:
            return False
        try:
            S_empty = (not s.get("S")) or all(abs(float(v)) <= 1e-14 for v in s.get("S", {}).values())
            mu_empty = (not s.get("mu")) or all(abs(float(v)) <= 1e-14 for v in s.get("mu", {}).values())
            return S_empty and mu_empty
        except Exception:
            return False

    S_prof: Dict[str, List[float]] = {r: [0.0] * len(periods) for r in regions}
    mu_prof: Dict[str, List[float]] = {r: [0.0] * len(periods) for r in regions}

    # Ignore legacy “data” seeds or accidental all-zero blobs
    if seed and (seed.get("seed_kind") == "data" or _looks_empty_seed(seed)):
        seed = None

    # 1) Use a real seed if present
    if seed and "S" in seed and "mu" in seed:
        for r in regions:
            for i, t in enumerate(periods):
                S_prof[r][i]  = float(seed["S"].get((r, t), 0.0))
                mu_prof[r][i] = float(seed["mu"].get((r, t), MU_FLOOR))
        return S_prof, mu_prof

    # 2) Prefer CSV seeds (saving_rate_t, mu_init); 3) else BAU/backstop
    for r in regions:
        for i, t in enumerate(periods):
            # S: CRRA planner schedule if available, else BAU
            if (saving_seed is not None) and (r in saving_seed.index) and (t in saving_seed.columns):
                S_val = saving_seed.at[r, t]
            else:
                S_val = bau_S.at[r, t]
            # clamp S into [0,1] defensively
            S_prof[r][i] = float(min(1.0, max(0.0, float(S_val))))

            # μ: positive mu_init if available, else Excel-style step but within bounds
            if (mu_seed is not None) and (r in mu_seed.index) and (t in mu_seed.columns):
                mu_val = float(mu_seed.at[r, t])
                mu_prof[r][i] = float(min(1.0, max(MU_FLOOR, mu_val)))
            else:
                year = base_year + t * tstep
                mu_prof[r][i] = 1.0 if year >= backstop_switch_year else MU_FLOOR

    return S_prof, mu_prof



def _obj_coalition(
    m: pe.ConcreteModel,
    coalition: Iterable[str],
    *,
    use_negishi: bool,
    negishi_weights: Optional[pd.DataFrame] = None,
) -> pe.Objective:
    """
    Sum of members' discounted intertemporal utilities.
    If use_negishi: multiply each member's (r,t) utility by W[r,t] (as in planner).
    """
    if use_negishi:
        if negishi_weights is None:
            raise ValueError("negishi_use=True but negishi_weights is None.")
        missing_r = [str(r) for r in m.REGIONS if str(r) not in negishi_weights.index]
        missing_t = [int(t) for t in m.T if int(t) not in negishi_weights.columns]
        if missing_r or missing_t:
           raise ValueError(f"Negishi weights missing regions={missing_r} or periods={missing_t}")
        expr = sum(
            float(negishi_weights.at[str(r), int(t)]) * m.disc[r, t] * m.U[r, t]
            for r in coalition for t in m.T
        )
    else:
        expr = sum(m.disc[r, t] * m.U[r, t] for r in coalition for t in m.T)
    return pe.Objective(expr=expr, sense=pe.maximize)


def _obj_singleton(m: pe.ConcreteModel, r_star: str) -> pe.Objective:
    """Singleton region discounted objective (no Negishi — mirrors Nash)."""
    return pe.Objective(expr=sum(m.disc[r_star, t] * m.U[r_star, t] for t in m.T), sense=pe.maximize)


def _evaluate_fixed(
    params, T, tstep, utility, population_weight_envy_guilt,
    S_prof, mu_prof, solver_opts, diagnostics_dir, *,
    exogenous_S_df=None,
    discount_series=None,
):
    """
    Evaluate full model with all S, μ fixed to the provided profiles; collect outputs.
    Also returns 'disc[(r,t)]' for discounted downstream use.
    """
    # Fixed 10-year grid: do NOT pass tstep into the builder.
    # If exogenous_S_df is provided, the builder is the single source of truth for fixed S.
    m = model_builder.build_model(
        params=params, T=T, utility=utility,
        # If exogenous_S_df is provided, the builder fixes S internally (single source of truth)
        exogenous_S=exogenous_S_df,
        population_weight_envy_guilt=population_weight_envy_guilt,
        # Optional: injected discount series (global or regional)
        discount_series=discount_series,
    )
    regions = list(m.REGIONS)
    periods = list(m.T)
    # Optional invariant: if exogenous S is used, ensure it is actually fixed by the builder
    if exogenous_S_df is not None:
        assert_exogenous_S_fixed(m, exogenous_S_df, regions, periods)

    # Standardized final-evaluation pattern:
    # - Always fix μ to profile
    # - Fix S to profile only when endogenous (i.e., exogenous_S_df is None)
    final_evaluation_setup(
        m, regions, mu_prof, S_prof,
        exogenous_S_used=(exogenous_S_df is not None),
    )

    res = build_ipopt(solver_opts, diagnostics_dir / "ipopt_coalition_final.log").solve(
        m, tee=logger.isEnabledFor(logging.DEBUG)
    )
    if str(res.solver.termination_condition).lower() not in ("optimal", "locally optimal"):
        raise RuntimeError(f"Coalition final evaluation ended with {res.solver.status}/{res.solver.termination_condition}")

    sol: Dict[str, Any] = {}
    for name in ("K", "S", "I", "Q", "Y", "mu", "AB", "D", "C", "E_ind", "U"):
        sol[name] = _collect_2d(m, name)
    if utility == "fs":
        # parity with coop/noncoop exports
        sol["FS_envy_avg"] = _collect_2d(m, "FS_envy_avg")
        sol["FS_guilt_avg"] = _collect_2d(m, "FS_guilt_avg")
    for name in ("E_tot", "M_at", "M_up", "M_lo", "T_at", "T_lo", "F",
                 "slr", "slr_TE", "gsic_remain", "gsic_melt", "gsic_cum",
                 "gis_remain", "gis_melt", "gis_cum",
                 "ais_remain", "ais_melt", "ais_cum"):
        sol[name] = _collect_1d(m, name)

    # REQUIRED downstream for discounted payoffs
    sol["disc"] = {(r, int(t)): float(pe.value(m.disc[r, t])) for r in m.REGIONS for t in m.T}

    # Record the exogenous S used (normalized) so cache/spec checks can compare exactly
    if exogenous_S_df is not None:
        sol["S_exogenous"] = exogenous_S_df.copy()

    # Coalition final evaluation is strict above, so this is always optimal
    sol["solver_status"] = str(res.solver.status)
    sol["termination"] = str(res.solver.termination_condition)
    sol["optimal"] = True

    back_out_carbon_tax(sol, params)

    # Attach global inequality diagnostics (Gini, Atkinson) based on
    # population-weighted per-capita consumption C/L for each period.
    # Use the same regions / periods as in this final evaluation.
    attach_inequality_series(sol, params=params, regions=regions, periods=periods)

    return sol


def _ipopt_block_succeeded_for_nash(stat, term) -> bool:
    """
    Decide if a block solve is 'good enough' for Gauss–Seidel updates.

    We accept:
      - solver.status in {ok, warning}, and
      - termination_condition in {optimal, locallyOptimal, maxIterations}.
    Everything else is treated as a hard failure for this iteration.
    """
    return (stat in (SS.ok, SS.warning)) and (term in (TC.optimal, TC.locallyOptimal, TC.maxIterations))
 

# ------------------- Coalition Nash core -------------------

def solve_coalition_game(
    params,
    T: int,
    tstep: int,
    coalition_vec: List[int],
    *,
    utility: str,                           # 'crra' or 'fs'
    solver_opts: Dict[str, Any],
    diagnostics_dir: Path,
    population_weight_envy_guilt: bool = True,
    initial_solution: Optional[Dict[str, Any]] = None,
    exogenous_S: Optional[pd.DataFrame] = None,
    negishi_use: bool = False,
    negishi_weights: Optional[pd.DataFrame] = None,
    max_iter: int,
    tol: float,
    relax: float,
    ignore_last_k_periods: int = 0,
    # Optional FS discounting controls: discount series + identity tag
    discount_series: Optional[Dict[Any, float]] = None,
    disc_tag: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Iterative best-response between {coalition as planner} and {non-members as singletons}.

    Returns a dict with:
      - 'solution': full evaluated solution dict (includes 'disc[(r,t)]')
      - 'iterations': int
      - 'converged': bool
      - 'max_delta': float (μ-based residual to best response on last iteration)
      - 'vector': coalition vector (0/1 list)
      - 'logfiles': list of IPOPT logs from subproblems + final eval
    """
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    # STRICT: if FS uses an explicit discount series, require a disc_tag (no silent fallbacks).
    if utility == "fs" and (discount_series is not None) and not disc_tag:
        raise ValueError(
            "FS coalition run received a discount_series but no disc_tag. "
            "Aligned/file FS discounting must pass a deterministic disc_tag."
        )
    regions = list(params.countries)
    if len(coalition_vec) != len(regions):
        raise ValueError("coalition_vec length must equal number of regions.")
    coalition = [r for bit, r in zip(coalition_vec, regions) if bit == 1]
    outsiders = [r for bit, r in zip(coalition_vec, regions) if bit == 0]
    periods = list(range(1, T + 1))

    S_prof, mu_prof = _init_profiles_from_seed(
        # S, μ initial profiles (can be overridden by exogenous S below)
        regions, periods, initial_solution, params.bau_saving_rates,
        params.base_year, params.backstop_switch_year, tstep,
        saving_seed=getattr(params, "savings_init", None),
        mu_seed=getattr(params, "mu_init", None),
    )
    # IMPORTANT: if S is exogenous, every place we "fix S" must use that exogenous path.
    # Precompute normalized exogenous S once and force S_prof to match it
    exoS = normalize_exogenous_S(exogenous_S, params.countries, T) if exogenous_S is not None else None
    if exoS is not None:
        for r in regions:
            for i, t in enumerate(periods):
                # ensure S_prof drives .fix(...) with the exact exogenous planner values
                S_prof[r][i] = float(exoS.at[r, t])

    diag_evo: List[dict] = []
    logfiles: List[str] = []

    converged_flag = False
    for k in range(1, max_iter + 1):
        logger.info("Coalition Nash iteration %d / %d", k, max_iter)
        if utility == "fs" and discount_series is not None and disc_tag:
            logger.debug("FS coalition iteration with discount tag: %s", disc_tag)
        prev_S = {r: S_prof[r][:] for r in regions}
        prev_mu = {r: mu_prof[r][:] for r in regions}

        # Window for convergence diagnostics and residuals (respect ignore_last_k_periods)
        n_total = len(periods)
        k_ignored = int(ignore_last_k_periods or 0)
        if k_ignored < 0:
            k_ignored = 0
        if k_ignored >= n_total and n_total > 0:
            # keep at least one period to avoid empty slices
            k_ignored = n_total - 1
        n_use = max(0, n_total - k_ignored)

        # μ-based residual to best response for this GS iteration.
        # Initialize to +inf so that, if residuals are never updated for some reason,
        # we cannot accidentally claim convergence.
        max_resid_iter: float = float("inf")
        resid_updated: bool = False

        # ---- (A) Coalition best response (joint planner for its members)
        success_this_iter = False
        logA = diagnostics_dir / f"ipopt_coalition_{utility}_it{k:03d}_COAL.log"
        ipoptA = build_ipopt(solver_opts, logA)

        # Fixed 10-year grid: do NOT pass tstep into the builder
        mA = model_builder.build_model(
            params=params, T=T, utility=utility,
            exogenous_S=exoS,
            population_weight_envy_guilt=population_weight_envy_guilt,
            # Thread through optional discount series (FS alignment)
            discount_series=discount_series,
        )
        # Fix outsiders to their current profiles
        fix_profiles(mA, S_prof, mu_prof, lock=outsiders, periods=periods)
        # Coalition members: unfix μ; unfix S only if S is not exogenous
        unfix_controls(
            mA, coalition,
            unfix_S=(exogenous_S is None),
            unfix_mu=True,
        )
        try:
            mA.OBJ = _obj_coalition(mA, coalition, use_negishi=negishi_use, negishi_weights=negishi_weights)
            # Seed coalition members with previous profiles (good IPOPT start)
            for r in coalition:
                for i, t in enumerate(periods):
                    s_lb = float(mA.S[r, t].lb) if mA.S[r, t].lb is not None else 0.0
                    s_ub = float(mA.S[r, t].ub) if mA.S[r, t].ub is not None else 1.0
                    mu_lb = float(mA.mu[r, t].lb) if mA.mu[r, t].lb is not None else 0.0
                    mu_ub = float(mA.mu[r, t].ub) if mA.mu[r, t].ub is not None else 1.0
                    mA.S[r, t].value  = clean(S_prof[r][i],  lower=s_lb,  upper=s_ub)
                    mA.mu[r, t].value = clean(mu_prof[r][i], lower=mu_lb, upper=mu_ub)

            resA = ipoptA.solve(mA, tee=logger.isEnabledFor(logging.DEBUG))
            statA = resA.solver.status
            termA = resA.solver.termination_condition
            if not _ipopt_block_succeeded_for_nash(statA, termA):
                logger.warning("Coalition block ended with status=%s, term=%s", statA, termA)
                raise RuntimeError(f"{statA}/{termA}")
            if termA == TC.maxIterations:
                logger.warning(
                    "Coalition block hit maxIterations at iter %d; using last iterate for profiles.", k
                )
            # Otherwise (optimal/locallyOptimal), behavior is as before: just update profiles.

            # Update coalition profiles (relaxed) and track μ-residual vs iteration baseline (prev_mu)
            for r in coalition:
                for i, t in enumerate(periods):
                    s_lb = float(mA.S[r, t].lb) if mA.S[r, t].lb is not None else 0.0
                    s_ub = float(mA.S[r, t].ub) if mA.S[r, t].ub is not None else 1.0
                    mu_lb = float(mA.mu[r, t].lb) if mA.mu[r, t].lb is not None else 0.0
                    mu_ub = float(mA.mu[r, t].ub) if mA.mu[r, t].ub is not None else 1.0
                    br_S = clean(float(pe.value(mA.S[r, t])), lower=s_lb, upper=s_ub)
                    br_mu = clean(float(pe.value(mA.mu[r, t])), lower=mu_lb, upper=mu_ub)

                    # Residual uses μ* (br_mu) vs μ_old (prev_mu), NOT the relaxed update.
                    if i < n_use:
                        delta_resid = abs(br_mu - prev_mu[r][i])
                        if not resid_updated:
                            max_resid_iter = delta_resid
                            resid_updated = True
                        elif delta_resid > max_resid_iter:
                            max_resid_iter = delta_resid

                    S_prof[r][i] = relax * br_S + (1 - relax) * S_prof[r][i]
                    mu_prof[r][i] = relax * br_mu + (1 - relax) * mu_prof[r][i]
                success_this_iter = True
        except Exception as e:
            logger.warning("Coalition block failed at iter %d (keeping previous profiles): %s", k, e)
        if _U.DIAGNOSTICS_ON:
            logfiles.append(str(logA))

        # ---- (B) Outsiders' best responses (singletons, sequential)
        for r_star in outsiders:
            logB = diagnostics_dir / f"ipopt_coalition_{utility}_it{k:03d}_{r_star}.log"
            ipoptB = build_ipopt(solver_opts, logB)
            mB = model_builder.build_model(
                params=params, T=T, utility=utility,
                exogenous_S=exoS,
                population_weight_envy_guilt=population_weight_envy_guilt,
                # NEW: outsiders see the same discount series
                discount_series=discount_series,
            )
            # Fix everyone except r_star
            lock = [r for r in regions if r != r_star]
            fix_profiles(mB, S_prof, mu_prof, lock=lock, periods=periods)
            # Unfix controls for r_star: μ always; S only if not exogenous
            unfix_controls(mB, [r_star], unfix_S=(exogenous_S is None), unfix_mu=True)

            try:
                mB.OBJ = _obj_singleton(mB, r_star)
                # Seed outsider with previous profile (good IPOPT start)
                for i, t in enumerate(periods):
                    s_lb = float(mB.S[r_star, t].lb) if mB.S[r_star, t].lb is not None else 0.0
                    s_ub = float(mB.S[r_star, t].ub) if mB.S[r_star, t].ub is not None else 1.0
                    mu_lb = float(mB.mu[r_star, t].lb) if mB.mu[r_star, t].lb is not None else 0.0
                    mu_ub = float(mB.mu[r_star, t].ub) if mB.mu[r_star, t].ub is not None else 1.0
                    mB.S[r_star, t].value  = clean(S_prof[r_star][i],  lower=s_lb,  upper=s_ub)
                    mB.mu[r_star, t].value = clean(mu_prof[r_star][i], lower=mu_lb, upper=mu_ub)

                resB = ipoptB.solve(mB, tee=logger.isEnabledFor(logging.DEBUG))
                statB = resB.solver.status
                termB = resB.solver.termination_condition
                if not _ipopt_block_succeeded_for_nash(statB, termB):
                    logger.warning("Singleton %s ended with status=%s, term=%s", r_star, statB, termB)
                    raise RuntimeError(f"{statB}/{termB}")
                if termB == TC.maxIterations:
                    logger.warning(
                        "Singleton %s hit maxIterations at iter %d; using last iterate for profiles.",
                        r_star,
                        k,
                    )
                # Otherwise (optimal/locallyOptimal), behavior is as before: just update profiles.
                # Update r_star profile (relaxed) and track μ-residual vs iteration baseline (prev_mu)
                for i, t in enumerate(periods):
                    s_lb = float(mB.S[r_star, t].lb) if mB.S[r_star, t].lb is not None else 0.0
                    s_ub = float(mB.S[r_star, t].ub) if mB.S[r_star, t].ub is not None else 1.0
                    mu_lb = float(mB.mu[r_star, t].lb) if mB.mu[r_star, t].lb is not None else 0.0
                    mu_ub = float(mB.mu[r_star, t].ub) if mB.mu[r_star, t].ub is not None else 1.0
                    br_S  = clean(float(pe.value(mB.S[r_star, t])),  lower=s_lb,  upper=s_ub)
                    br_mu = clean(float(pe.value(mB.mu[r_star, t])), lower=mu_lb, upper=mu_ub)

                    # μ-residual to best response for this singleton block
                    if i < n_use:
                        delta_resid = abs(br_mu - prev_mu[r_star][i])
                        if not resid_updated:
                            max_resid_iter = delta_resid
                            resid_updated = True
                        elif delta_resid > max_resid_iter:
                            max_resid_iter = delta_resid

                    S_prof[r_star][i]  = relax * br_S  + (1 - relax) * S_prof[r_star][i]
                    mu_prof[r_star][i] = relax * br_mu + (1 - relax) * mu_prof[r_star][i]
                success_this_iter = True
            except Exception as e:
                logger.warning("Singleton %s failed at iter %d (keeping previous profile): %s", r_star, k, e)
            if _U.DIAGNOSTICS_ON:
                logfiles.append(str(logB))

        # ---- Convergence metrics:
        # Primary: μ-residual to the unrelaxed best response (max_resid_iter).
        # Secondary/diagnostic: relaxed profile deltas (max_d).
        #
        # NOTE: conv_delta is currently diagnostic-only; convergence uses conv_resid below.
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
        
        
        # --- Per-region diagnostics (per iteration)
        # We track μ deltas because exogenous-S runs imply ΔS ≈ 0 by construction.
        # Respect ignore_last_k_periods by truncating the tail window (n_use).
        per_region: Dict[str, Dict[str, Any]] = {}

        for r in regions:
            if n_use > 0:
                mu_deltas = [
                    float(abs(mu_prof[r][i] - prev_mu[r][i]))
                    for i in range(n_use)
               ]
                mu_max = max(mu_deltas) if mu_deltas else 0.0
            else:
                mu_deltas = []
                mu_max = 0.0

            per_region[r] = {
                "mu_max_delta": mu_max,
                "mu_deltas": mu_deltas,  # per-period Δμ for this iteration (up to ignored tail)
            }

        diag_evo.append({
            "iteration": k,
            # Primary convergence metric: μ-residual to best response (unrelaxed).
            "max_delta": float(max_resid_iter if resid_updated else float("inf")),
            #  Diagnostic: relaxed-change max Δ over μ and S.
            "max_delta_relaxed": float(max_d),
            "per_region": per_region,
        })

        # Residual-based convergence: require at least one updated residual
        # and that the max residual be below tol, plus at least one successful block.
        conv_resid = bool(resid_updated and (max_resid_iter <= tol))
        conv = bool(conv_resid and success_this_iter)
         
        # More informative status string for the per-iteration log
        if conv:
            status_str = "CONVERGED"
        elif not success_this_iter:
            status_str = "ABORT (no successful block solves)"
        elif k == max_iter:
            status_str = "STOP (max_iter reached)"
        else:
            status_str = "continue"

        logger.info(
            "After iteration %d: max residual = %.3e (tol=%.3e) | max Δ(relaxed) = %.3e | any update=%s → %s",
            k,
            max_resid_iter if resid_updated else float("inf"),
            tol,
            max_d,
            success_this_iter,
            status_str,
        )

        if not success_this_iter:
            logger.error("All singleton solves failed at iter %d → aborting (avoid fake convergence).", k)
            break
        if conv:
            converged_flag = True
            break

    # If we exhausted the Gauss–Seidel iterations without convergence, emit a clear warning.
    if not converged_flag:
        last_resid = float(diag_evo[-1]["max_delta"]) if diag_evo else float("nan")
        last_delta_relaxed = float(diag_evo[-1].get("max_delta_relaxed", float("nan"))) if diag_evo else float("nan")
        logger.warning(
            "Coalition Nash did not converge within max_iter=%d iterations "
            "(last max residual = %.3e, tol=%.3e, last max Δ(relaxed) = %.3e).",
            max_iter, last_resid, tol, last_delta_relaxed,
        )

    # Final evaluation with locked profiles to collect a consistent full solution (incl. disc)
    try:
        plot_nonconv_diag(
            diag_evo,
            diagnostics_dir / "coalition_max_delta.png",
            title="Coalition Nash max residual (μ best-response)",
        )
    except Exception:
        pass

    try:
        sol = _evaluate_fixed(
            params, T, tstep, utility, population_weight_envy_guilt,
            S_prof, mu_prof,
            solver_opts=solver_opts,
            diagnostics_dir=diagnostics_dir,
            exogenous_S_df=exoS,
            # Ensure final evaluation matches the same discounting
            discount_series=discount_series,
         )
    except Exception as e:
        logger.error("Final coalition evaluation failed: %s", e)
        raise RuntimeError("Coalition final evaluation failed") from e

    # Build spec_id without tstep (time grid is fixed; matches planner/BAU cache keys)
    spec_id = build_solution_spec_id(
        utility=utility,
        T=T,
        countries=regions,
        population_weight_envy_guilt=population_weight_envy_guilt,
        exogenous_S=sol.get("S_exogenous"),
        negishi_use=negishi_use,
        negishi_weights=negishi_weights,
        # Make coalition cache keys discount-aware for FS runs
        disc_tag=(disc_tag if utility == "fs" else None),
    )
    sol["spec_id"] = spec_id
    # Optional: record discount tag for diagnostics/tracing (pass-through from caller).
    # No fallback here: exporters will error if they require a mode suffix and disc_tag is missing.
    if utility == "fs" and disc_tag is not None:
        sol["disc_tag"] = disc_tag
    sol["utility"] = utility.lower()
    sol["fingerprint"] = build_solution_fingerprint(
        mode="coalition", coalition_vec=coalition_vec, spec_id=spec_id
    )

    return {
        "vector": coalition_vec,
        "solution": sol,
        "iterations": len(diag_evo),
        "converged": bool(converged_flag),
        "max_delta": float(diag_evo[-1]["max_delta"]) if diag_evo else None,  # μ-residual at last iteration
        "logfiles": (logfiles + [str(diagnostics_dir / "coalition_max_delta.png")]) if _U.DIAGNOSTICS_ON else logfiles,
        "fingerprint": sol["fingerprint"],
    }


# ------------------- Coalition neighborhood + parsing -------------------

def parse_coalition_spec(spec: str, regions: List[str]) -> List[int]:
    """
    Parse a coalition spec into a 0/1 vector aligned with `regions`.

    Valid forms:
      - "GRAND"              → all regions are members
      - "US,EU,JAP"          → exactly those region codes (order-insensitive)
      - "101001..."          → bitstring of length len(regions), 1=member, 0=non-member

    Anything else (e.g., unknown region codes, wrong-length/invalid bitstrings) raises ValueError.
    """
    if not isinstance(spec, str):
        raise ValueError("Coalition spec must be a string.")
    s = spec.strip()
    if s.upper() == "GRAND":
        return [1] * len(regions)

    # Accept exact-length 0/1 bitstrings
    if set(s) <= {"0", "1"} and len(s) == len(regions):
        return [1 if ch == "1" else 0 for ch in s]

    tokens = [tok.strip() for tok in s.split(",") if tok.strip()]
    if not tokens:
        raise ValueError("Empty coalition spec (expected 'GRAND', comma-separated region codes, or a 0/1 bitstring).")
    unmatched = [tok for tok in tokens if tok not in regions]
    if unmatched:
        raise ValueError(f"Invalid region(s) in coalition spec: {unmatched}. Valid regions: {regions}")
    return [1 if r in tokens else 0 for r in regions]


def list_internal_neighbors(base_vec: List[int]) -> List[List[int]]:
    """All coalitions obtained by removing exactly one current member from base_vec."""
    out: List[List[int]] = []
    for i, bit in enumerate(base_vec):
        if bit == 1:
            v = base_vec.copy()
            v[i] = 0
            out.append(v)
    return out


def list_external_neighbors(base_vec: List[int]) -> List[List[int]]:
    """All coalitions obtained by adding exactly one current non-member to base_vec."""
    out: List[List[int]] = []
    for i, bit in enumerate(base_vec):
        if bit == 0:
            v = base_vec.copy()
            v[i] = 1
            out.append(v)
    return out


def coalition_vec_to_member_string(vec: List[int], regions: List[str]) -> str:
    """
    Turn a coalition vector into a readable member string for filenames/labels.
    Examples:
      ALL members → "GRAND"
      None        → "NONE" (should not appear in typical use)
      Subset      → "US_EU_JAP"
    """
    members = [r for r, bit in zip(regions, vec) if bit == 1]
    if len(members) == len(regions):
        return "GRAND"
    return "_".join(members) if members else "NONE"


# ------------------- Suite runner (base + neighbors) -------------------

def solve_coalition(
    params,
    T: int,
    tstep: int,
    coalition_spec: str,
    *,
    utility: str,
    solver_opts: Dict[str, Any],
    diagnostics_dir: Path,
    population_weight_envy_guilt: bool,
    initial_solution: Optional[Dict[str, Any]],
    exogenous_S: Optional[pd.DataFrame],
    negishi_use: bool,
    negishi_weights: Optional[pd.DataFrame],
    coalition_check_internal: bool,
    coalition_check_external: bool,
    max_iter: int,
    tol: float,
    relax: float,
    ignore_last_k_periods: int,
    # Hints to reuse precomputed solutions for special neighbors
    reuse_hints: Optional[Dict[str, Any]],
    # FS discounting (optional): regional series + identity tag
    discount_series: Optional[Dict[Any, float]] = None,
    disc_tag: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Compute base coalition Nash equilibrium, then optional 1-step neighbors
    (internal removals, external additions), returned in order:
      [internals...] + [base] + [externals...]

    reuse_hints (optional):
      {
        "planner_solution": <dict>,  # planner of matching utility (solution dict)
        "planner_S": <pd.DataFrame or None>,  # exact exog S used for that planner (None => endogenous)
        "nash_solution": <dict>,     # nash of matching utility (solution dict)
        "nash_S": <pd.DataFrame or None>,     # exact exog S used for that nash (None => endogenous)
      }
    """
    regions = list(params.countries)
    base_vec = parse_coalition_spec(coalition_spec, regions=regions)
    N = len(regions)

    results: List[Dict[str, Any]] = []

    def _wrap(vec: List[int], sol_dict: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "vector": vec,
            "solution": sol_dict,
            "iterations": 0,
            "converged": True,
            "max_delta": 0.0,
            "logfiles": [],
        }

    # Compute the run's spec_id once (matches what solve_coalition_game attaches)
    exoS_norm = normalize_exogenous_S(exogenous_S, params.countries, T) if exogenous_S is not None else None
    run_spec_id = build_solution_spec_id(
        utility=utility,
        T=T,
        countries=regions,
        population_weight_envy_guilt=population_weight_envy_guilt,
        exogenous_S=exoS_norm,
        negishi_use=negishi_use,
        negishi_weights=negishi_weights,
        # Make reuse matching discount-aware for FS runs
        disc_tag=(disc_tag if utility == "fs" else None),
    )

    # Internal neighbors first (optional)
    if coalition_check_internal:
        for v in list_internal_neighbors(base_vec):
            # Reuse path: singleton internal neighbor → Nash (if full spec matches)
            if sum(v) == 1 and reuse_hints is not None:
                ns = reuse_hints.get("nash_solution")
                if ns is not None and ns.get("spec_id") == run_spec_id:
                    logger.info("Reusing precomputed Nash result for singleton internal neighbor: %s", v)
                    results.append(_wrap(v, ns))
                    continue

            logger.info("Solving internal neighbor (drop one member): %s", v)
            res = solve_coalition_game(
                params, T, tstep, v,
                utility=utility,
                solver_opts=solver_opts,
                diagnostics_dir=diagnostics_dir / "neighbors" / "internal",
                population_weight_envy_guilt=population_weight_envy_guilt,
                initial_solution=initial_solution,
                exogenous_S=exogenous_S,
                negishi_use=negishi_use,
                negishi_weights=negishi_weights,
                max_iter=max_iter, tol=tol, relax=relax, ignore_last_k_periods=ignore_last_k_periods,
                discount_series=discount_series,
                disc_tag=disc_tag,
            )
            results.append(res)

    # Base coalition
    logger.info("Solving base coalition: %s", base_vec)
    base_res = solve_coalition_game(
        params, T, tstep, base_vec,
        utility=utility,
        solver_opts=solver_opts,
        diagnostics_dir=diagnostics_dir / "base",
        population_weight_envy_guilt=population_weight_envy_guilt,
        initial_solution=initial_solution,
        exogenous_S=exogenous_S,
        negishi_use=negishi_use,
        negishi_weights=negishi_weights,
        max_iter=max_iter, tol=tol, relax=relax, ignore_last_k_periods=ignore_last_k_periods,
        discount_series=discount_series,
        disc_tag=disc_tag,
    )
    results.append(base_res)

    # External neighbors (optional)
    if coalition_check_external:
        for v in list_external_neighbors(base_vec):
            # Reuse path: grand-coalition external neighbor → Planner (if full spec matches)
            if sum(v) == N and reuse_hints is not None:
                ps = reuse_hints.get("planner_solution")
                if ps is not None and ps.get("spec_id") == run_spec_id:
                    logger.info("Reusing precomputed Planner result for grand-coalition external neighbor: %s", v)
                    results.append(_wrap(v, ps))
                    continue

            logger.info("Solving external neighbor (add one member): %s", v)
            res = solve_coalition_game(
                params, T, tstep, v,
                utility=utility,
                solver_opts=solver_opts,
                diagnostics_dir=diagnostics_dir / "neighbors" / "external",
                population_weight_envy_guilt=population_weight_envy_guilt,
                initial_solution=initial_solution,
                exogenous_S=exogenous_S,
                negishi_use=negishi_use,
                negishi_weights=negishi_weights,
                max_iter=max_iter, tol=tol, relax=relax, ignore_last_k_periods=ignore_last_k_periods,
                discount_series=discount_series,
                disc_tag=disc_tag,
            )
            results.append(res)

    return results
