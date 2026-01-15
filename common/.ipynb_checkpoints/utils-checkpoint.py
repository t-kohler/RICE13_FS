"""
Shared utilities for RICE13_FS:

- Numeric helpers (clamp, safe division)
- Period ↔ year mapping (decadal only)
- Exogenous S normalization (regions × periods 1..T)
- IPOPT builder
- Discounted payoff row construction (exports use this)
- Result collectors (1D/2D)
- Convergence checks
- Coalition targeting (vectors & neighbors)
- Fingerprints / spec IDs (decadal-only; optional Negishi digest)
- Lightweight diagnostics plotting

All logging goes through `logging`; nothing prints by default.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple, Iterable, Sequence, Mapping
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing as mp
from pathlib import Path
from collections import deque
import logging
import sys
import json, hashlib
import ast
import numpy as np
import re
import pandas as pd
import pyomo.environ as pe
from pyomo.environ import value
from dataclasses import dataclass
from copy import deepcopy

from RICE13_FS.analysis.negishi import fs_negishi_mu

# Use a headless backend so CLI runs don't require a display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

DECADAL_STEP = 10  # years per period

# ------------------------------------------------------------------
# Global diagnostics switch (set once by cli.py after config load)
# Meaning here: controls *to-disk* diagnostics only
#   True  -> write IPOPT *.log files and convergence PNGs
#   False -> skip IPOPT logs and convergence PNGs
DIAGNOSTICS_ON: bool = False
# ------


# -----------------------------
# Numeric helpers
# -----------------------------
def clean(val: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp `val` into [lower, upper]."""
    if val < lower:
        return lower
    if val > upper:
        return upper
    return val


def safe_div(num: float, den: float, default: float = float("nan")) -> float:
    """
    Safe division that returns `default` (NaN by default) when denominator is zero.
    Handy for per-capita series to avoid ZeroDivisionError.
    """
    if den == 0:
        return default
    return num / den


def years_from_periods(periods, base_year, tstep):
    """
    Map model period index/indices to calendar year(s).
    Accepts:
      - iterable of ints, e.g. [1, 2, 3]
      - single int T → expands to 1..T
    """
    # normalize 'periods' to an iterable of ints
    if isinstance(periods, int):
        seq = range(1, int(periods) + 1)
    else:
        seq = [int(p) for p in periods]
    # normalize step/base_year
    s = int(tstep)
    by = int(base_year)
    return [by + p * s for p in seq]

# -----------------------------
# DataFrame normalization utils
# -----------------------------
def normalize_exogenous_S(df: pd.DataFrame, countries: Sequence[str], T: int) -> pd.DataFrame:
    """
    Normalize an exogenous savings schedule to shape (regions x periods) with
    integer period columns 1..T and index exactly == countries (same order).

    Accepts common variants:
      - regions in columns (auto-transpose)
      - string columns like '1','t1','T1','period_1'
      - 0..T-1 columns (auto-shift to 1..T)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("exogenous_S must be a pandas DataFrame (region x period).")
    S = df.copy()

    # If regions are on columns, transpose
    if all(r in S.columns for r in countries) and not all(r in S.index for r in countries):
        S = S.T

    # Coerce columns to ints
    def _to_int(c):
        s = str(c).strip().lower()
        s = re.sub(r'^(t|period[_\s-]*)', '', s)
        return int(s)

    try:
        S.columns = [_to_int(c) for c in S.columns]
    except Exception as e:
        raise ValueError(f"exogenous_S columns must be periods; got {list(S.columns)[:5]}…") from e

    # Shift 0..T-1 → 1..T if detected
    if set(S.columns) == set(range(0, T)):
        S.columns = [c + 1 for c in S.columns]

    # Require 1..T present
    missing = [t for t in range(1, T + 1) if t not in S.columns]
    if missing:
        raise KeyError(f"exogenous_S missing period columns: {missing[:10]}… (need 1..{T})")

    # Regions must match
    if not all(r in S.index for r in countries):
        missing_r = [r for r in countries if r not in S.index]
        raise KeyError(f"exogenous_S missing regions: {missing_r}")

    # Reindex & reorder to canonical layout
    S = S.loc[list(countries), list(range(1, T + 1))]
    return S


# -----------------------------
# Solver helpers (centralized)
# -----------------------------
# -----------------------------------
# Coalition S-resolution helper for exporter/consumers
# -----------------------------------
def _solution_S_to_df(S_obj: Any, countries: Sequence[str], T: int) -> pd.DataFrame:
    """Convert an internal S object (DF or dict[(r,t)]->float) to a canonical DataFrame.
    Columns are integer periods 1..T; index is exactly `countries` in that order."""
    if isinstance(S_obj, pd.DataFrame):
        return normalize_exogenous_S(S_obj, countries, T)
    # dict-like path
    df = pd.DataFrame(index=list(countries), columns=list(range(1, int(T)+1)), dtype=float)
    for r in countries:
        for t in range(1, int(T)+1):
            df.loc[r, t] = float(S_obj.get((r, t), np.nan))
    return normalize_exogenous_S(df, countries, T)


def _S_solution_to_df(S_map: dict, countries: List[str], T: int) -> pd.DataFrame:
    df = pd.DataFrame(0.0, index=[str(c) for c in countries], columns=list(range(1, int(T)+1)))
    # Accept keys as tuples (region, period) or strings convertible
    for r in countries:
        rs = str(r)
        for t in range(1, int(T)+1):
            df.at[rs, t] = float(S_map.get((rs, int(t)), S_map.get((r, int(t)), 0.0)))
    return df


def resolve_coalition_S_for_export(
    *,
    s_mode: str,
    params: Any,
    countries: Sequence[str],
    T: int,
    store: Any | None = None,
    negishi_use: bool = False,
    negishi_weights: Optional[pd.DataFrame] = None,
    population_weight_envy_guilt: bool = False,
    planner_disc_tag: Optional[str] = None,
    exoS_df: Optional[pd.DataFrame] = None,
) -> Optional[pd.DataFrame]:
    """Return the exogenous S DataFrame implied by `s_mode` for coalition export/lookup.

    Modes:
      - 'optimal'        -> returns None
      - 'bau'            -> returns params.bau_saving_rates (normalized copy)
      - 'file'/'exogenous' -> requires `exoS_df` (already loaded), returns normalized DF
      - 'planner_crra'   -> fetch GRAND planner-CRRA from cache and extract S
      - 'planner_fs'     -> fetch GRAND planner-FS (requires the exact `planner_disc_tag`) and extract S

    NOTE: We avoid CSV I/O here; pass `exoS_df` if you need 'file'.
    """
    mode = str(s_mode or "").strip().lower()
    if mode == "optimal":
        return None
    if mode == "bau":
        return normalize_exogenous_S(getattr(params, "bau_saving_rates").copy(), countries, int(T))
    if mode in ("file", "exogenous", "exog"):
        if exoS_df is None:
            raise ValueError("resolve_coalition_S_for_export(mode='file') requires exoS_df")
        return normalize_exogenous_S(exoS_df, countries, int(T))
    if mode not in ("planner_crra", "planner_fs"):
        raise ValueError(f"Unknown coalition S_mode: {s_mode!r}")

    if store is None:
        raise ValueError("resolve_coalition_S_for_export(planner_*) requires a cache store handle")

    util_src = "crra" if mode == "planner_crra" else "fs"
    # Build the GRAND planner spec-id exactly like the solver
    sid_pl = build_solution_spec_id(
        utility=util_src,
        T=int(T),
        countries=countries,
        population_weight_envy_guilt=(util_src == "fs" and bool(population_weight_envy_guilt)),
        exogenous_S=None,
        negishi_use=bool(negishi_use),
        negishi_weights=negishi_weights,
        disc_tag=(planner_disc_tag if util_src == "fs" else None),
    )
    grand_vec = tuple([1] * len(countries))
    hit = store.get(grand_vec, sid_pl)
    sol = (hit or {}).get("solution") if hit else None
    if not (sol and ("S" in sol)):
        raise RuntimeError(f"Planner S not found in cache for {mode}; looked under spec_id={sid_pl}")
    return _solution_S_to_df(sol["S"], countries, int(T))


def build_ipopt(options: Optional[Dict[str, Any]] = None, log_path: Optional[Path] = None) -> pe.SolverFactory:
    """
    Create an IPOPT solver with provided options and (optional) log file.
    Ensures the parent directory for the log exists (Windows-safe).

    Example:
        opt = build_ipopt({"tol": 1e-8, "max_iter": 10000}, diagnostics_dir/"ipopt.log")
        opt.solve(m, tee=False)
    """
    exec_path = None
    flat_opts: Dict[str, Any] = {}
    if options:
        exec_path = options.get("executable")
        if "options" in options and isinstance(options["options"], dict):
            flat_opts.update(options["options"])
        else:
            flat_opts.update({k: v for k, v in options.items() if k != "executable"})

    ipopt = pe.SolverFactory("ipopt", executable=exec_path) if exec_path else pe.SolverFactory("ipopt")
    for k, v in flat_opts.items():
        ipopt.options[k] = v
    # only attach an IPOPT output file when diagnostics are ON
    if DIAGNOSTICS_ON and (log_path is not None):
        p = Path(log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        ipopt.options["output_file"] = str(p)

    # Defaults that are robust for medium-sized RICE solves
    ipopt.options["linear_solver"] = "mumps"
    ipopt.options["mumps_mem_percent"] = 4000
    
    # Let Ipopt handle scaling of the KKT system (doesn’t change the optimum)
    ipopt.options["nlp_scaling_method"] = "gradient-based"
    ipopt.options["nlp_scaling_max_gradient"] = 100.0        # tame huge grads
    ipopt.options["nlp_scaling_min_value"] = 1e-8            # avoid tiny scales
    
    # Globalization more stable when late-period gradients are tiny
    #ipopt.options["line_search_method"] = "cg-penalty"
    #ipopt.options["alpha_for_y"] = "dual"

    # ---- Barrier & step strategy ----
    #ipopt.options["mu_strategy"] = "adaptive"  # keep
    #ipopt.options["mu_init"] = 1e-1            # good when many bounds are active
    # keep the default filter line-search; do NOT enable cg-penalty or alpha_for_y

    # Many bound-active variables (μ can be 1): keep iterates slightly interior
    #ipopt.options["bound_push"] = 1e-3
    #ipopt.options["bound_frac"] = 1e-3
    #ipopt.options["bound_relax_factor"] = 1e-8
    #ipopt.options["honor_original_bounds"] = "yes"
    
    #ipopt.options["hessian_approximation"] = "limited-memory"
    #ipopt.options["limited_memory_max_history"] = 20  # mild, stable default


    if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
        ipopt.options["print_timing_statistics"] = "yes"
        # ipopt.options["print_level"] = 5  # uncomment if you want verbose iteration logs

    return ipopt


def payoff_row_discounted(solution: dict, regions: List[str], periods: List[int]) -> List[float]:
    """
    Return one payoff row: for each region r, sum_t disc[(r,t)] * U[(r,t)].
    Strict: requires both 'U' and 'disc' in solution with (region, period) keys.
    """
    if "U" not in solution or "disc" not in solution:
        raise ValueError("Solution dict must contain 'U' and 'disc' with keys (region, period).")
    return [
        float(sum(solution["disc"][(r, t)] * solution["U"][(r, t)] for t in periods))
        for r in regions
    ]


def _read_disc_csv(path: Path, regions: list[str], T: int) -> dict[tuple[str, int], float]:
    """
    Read a regional discount grid from CSV with columns: region, t, disc.
    Returns {(region, t) -> disc}. Unknown regions/periods are ignored.
    """
    df = pd.read_csv(path)
    series: dict[tuple[str, int], float] = {}
    regset = {str(r) for r in regions}
    for _, row in df.iterrows():
        r = str(row["region"])
        t = int(row["t"])
        if r in regset and 1 <= t <= int(T):
            series[(r, t)] = float(row["disc"])
    # Fill any missing entries with 1.0 (neutral) to be defensive.
    for r in regions:
        for t in range(1, int(T) + 1):
            series.setdefault((str(r), int(t)), 1.0)
    return series

# ---------------------------------------------------------------------------
# Robust optimality detection
# ---------------------------------------------------------------------------
def _norm_status_text(s: str) -> str:
    """
    Normalize a status string into a compact, comparison-friendly token.
    Example: "Solved_To_Acceptable_Level" -> "solved_to_acceptable_level"
             "Locally Optimal"           -> "locally_optimal"
    """
    s = (s or "").strip()
    s = s.replace("-", "_")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-zA-Z0-9_]+", "", s)
    return s.lower()

def _coerce_status_mapping(status_obj):
    """
    Best-effort: take a status object (string | dict | JSON string | other)
    and return a flat mapping of candidate status-related fields (all lower-cased).
    """
    if status_obj is None:
        return {}
    # JSON-ish string?
    if isinstance(status_obj, str):
        s = status_obj.strip()
        if (s.startswith("{") and s.endswith("}")):
            try:
                status_obj = json.loads(s)
            except Exception:
                # Some logs store repr(dict); fall back to literal_eval
                try:
                    status_obj = ast.literal_eval(s)
                except Exception:
                    # Treat as a plain status string
                    return {"status": _norm_status_text(s)}
        else:
            return {"status": _norm_status_text(s)}
    # Dict-like?
    if isinstance(status_obj, dict):
        flat = {}
        for k, v in status_obj.items():
            key = str(k).lower()
            if isinstance(v, str):
                flat[key] = _norm_status_text(v)
            else:
                flat[key] = v
        return flat
    # Fallback: stringify
    return {"status": _norm_status_text(str(status_obj))}

def is_solution_optimal(sol: dict) -> bool:
    """
    Robustly decide whether a solution is 'optimal enough' to proceed or cache.
    Accepts:
      - sol['optimal'] == True
      - or 'solver_status' / 'status' / 'termination_condition' (string/dict/JSON)
        in any of these OK states:
          * optimal / optimal_termination / optimal_solution_found
          * locally_optimal / local_optimal / locallyoptimal
          * solve_succeeded / solved_to_acceptable_level / converged_to_acceptable_point
      - or a numeric return_code == 0 in a nested solver meta dict.
    """
    if not isinstance(sol, dict):
        return False

    # Explicit boolean beats everything
    if bool(sol.get("optimal", False)):
        return True

    # Common places we’ve seen status recorded
    candidates = []
    for k in (
        "solver_status", "status", "termination_condition", "termination",
        "ipopt_status", "ipopt", "meta", "solver"
    ):
        if k in sol and sol[k] is not None:
            candidates.append(sol[k])
    # Some solutions embed a nested 'solution'
    if "solution" in sol and isinstance(sol["solution"], dict):
        for k in ("solver_status", "status", "termination_condition", "termination", "ipopt_status", "ipopt", "meta", "solver"):
            if k in sol["solution"] and sol["solution"][k] is not None:
                candidates.append(sol["solution"][k])

    ACCEPT = {
        "optimal", "optimal_termination", "optimal_solution_found",
        "locally_optimal", "local_optimal", "locallyoptimal",
        "solve_succeeded", "solved_to_acceptable_level", "converged_to_acceptable_point"
    }

    for obj in candidates:
        mp = _coerce_status_mapping(obj)
        # Direct string-style status in 'status' / 'termination_condition' / similar
        for key in ("status", "termination_condition", "termination", "solver", "ipopt_status", "ipopt"):
            val = mp.get(key)
            if isinstance(val, str) and val in ACCEPT:
                return True
        # Sometimes a code == 0 indicates success
        for key in ("return_code", "code", "status_code", "rc"):
            try:
                if int(mp.get(key)) == 0:
                    return True
            except Exception:
                pass

    return False


# -----------------------------------
# FS discount series builders (regional)
#   - one-pass: CRRA per-capita growth and CRRA ranks (FS MU index)
#   - two-pass: CRRA growth and FS(1) ranks
# -----------------------------------
def _build_fs_discount_series_one_pass(
    *,
    anchor_crra_sol: dict,
    params,
    regions: List[str],
    periods: List[int],
    population_weight_envy_guilt: bool,
) -> Dict[Tuple[str, int], float]:

    """
    Construct **regional** FS discount series d_{r,t} by matching the *regional* SDF:
        (d_{r,t+1}/d_{r,t}) = (1/(1+ρ_r))^Δ * ( (c_{r,t+1}/c_{r,t})^(-η_r) ) / ( MU^{FS}_{r,t+1} / MU^{FS}_{r,t} )
    where c_{r,t} = C_{r,t}/L_{r,t} from the CRRA anchor, and MU^{FS} uses CRRA ranks.
    """
    C_map = anchor_crra_sol["C"]
    d: Dict[Tuple[str, int], float] = {}
    for r in regions:
        beta_r = (1.0 + float(params.rho[r])) ** (-10.0)
        d[(r, 1)] = float(beta_r)  # base alignment with geometric
        eta_r = float(params.crra_eta[r])
        for t in periods[:-1]:
            C_rt  = float(C_map[(r, t)])
            C_rtp = float(C_map[(r, t + 1)])
            L_rt  = float(params.L.at[r, t])
            L_rtp = float(params.L.at[r, t + 1])
            c_rt  = C_rt / max(L_rt, 1e-12)
            c_rtp = C_rtp / max(L_rtp, 1e-12)
            # FS MU index from CRRA ranks (strict ranks; ties=0), respecting population_weight_envy_guilt
            MU_rt  = fs_negishi_mu(r=r, t=int(t),     params=params, regions=regions,
                                   C=C_map, L=params.L, population_weight_envy_guilt=population_weight_envy_guilt)
            MU_rtp = fs_negishi_mu(r=r, t=int(t + 1), params=params, regions=regions,
                                   C=C_map, L=params.L, population_weight_envy_guilt=population_weight_envy_guilt)
            num = beta_r * ((c_rtp / max(c_rt, 1e-16)) ** (-eta_r))
            den = (MU_rtp / MU_rt) if MU_rt > 0 else 1.0
            step = float(num / den) if den > 0 else float(beta_r)
            if not (step > 0.0) or not pd.notna(step):
                step = float(beta_r)
            d[(r, t + 1)] = d[(r, t)] * step
    return d

def _build_fs_discount_series_two_pass(
    *,
    anchor_crra_sol: dict,
    fs_baseline_sol: dict,
    params,
    regions: List[str],
    periods: List[int],
    population_weight_envy_guilt: bool,
) -> Dict[Tuple[str, int], float]:
    """
    Two-pass (regional): use CRRA per-capita growth and FS(0) ranks for MU^{FS}.
    """
    C_map = anchor_crra_sol["C"]
    # Accept either a flat solution dict or a {"solution": {...}} wrapper for FS(1)
    if "C" not in fs_baseline_sol and isinstance(fs_baseline_sol, dict) and "solution" in fs_baseline_sol:
        inner = fs_baseline_sol.get("solution")
        if isinstance(inner, dict):
            fs_baseline_sol = inner
    C0 = fs_baseline_sol["C"]
    d: Dict[Tuple[str, int], float] = {}
    for r in regions:
        beta_r = (1.0 + float(params.rho[r])) ** (-10.0)
        d[(r, 1)] = float(beta_r)
        eta_r = float(params.crra_eta[r])
        for t in periods[:-1]:
            C_rt  = float(C_map[(r, t)])
            C_rtp = float(C_map[(r, t + 1)])
            L_rt  = float(params.L.at[r, t])
            L_rtp = float(params.L.at[r, t + 1])
            c_rt  = C_rt / max(L_rt, 1e-12)
            c_rtp = C_rtp / max(L_rtp, 1e-12)
            MU_rt  = fs_negishi_mu(r=r, t=int(t),     params=params, regions=regions,
                                   C=C0, L=params.L, population_weight_envy_guilt=population_weight_envy_guilt)
            MU_rtp = fs_negishi_mu(r=r, t=int(t + 1), params=params, regions=regions,
                                   C=C0, L=params.L, population_weight_envy_guilt=population_weight_envy_guilt)
            num = beta_r * ((c_rtp / max(c_rt, 1e-16)) ** (-eta_r))
            den = (MU_rtp / MU_rt) if MU_rt > 0 else 1.0
            step = float(num / den) if den > 0 else float(beta_r)
            if not (step > 0.0) or not pd.notna(step):
                step = float(beta_r)
            d[(r, t + 1)] = d[(r, t)] * step
    return d

# -----------------------------------
# Use results to back out the carbon tax that implements them
# -----------------------------------
def back_out_carbon_tax(sol: dict, params) -> None:
    """
    τ_{r,t} = backstpr_{r,t} * μ_{r,t}^{th2(r) - 1}
    - Writes to sol["carbon_tax"][(r,t)].
    - Units = 2005 k$/tC (same as params.backstpr from your data sheet).
    """
    mu = sol.get("mu", {})
    out: Dict[Tuple[str, int], float] = {}
    if not mu:
        sol["carbon_tax"] = out
        return

    for (r, t), m in mu.items():
        th2 = float(params.th_2[r])
        b   = float(params.backstpr.at[r, t])  # built from pback, decays, =0 from year>=2250
        expo = th2 - 1.0
        if expo < 0.0:
            expo = 0.0  # safety if someone miscalibrates
        # minimal supporting tax; if b==0 → τ==0
        tau = b * (float(m) ** expo) if b > 0.0 else 0.0
        out[(r, t)] = tau

    sol["carbon_tax"] = out


# -----------------------------
# Inequality helpers (Gini, Atkinson)
# -----------------------------

# We hard-code the Atkinson inequality-aversion parameter ε = 1.5 here,
# as a purely diagnostic choice (separate from any CRRA curvature used
# in the welfare function). This keeps the definition stable across
# regimes (CRRA, FS) and scenarios.
ATKINSON_EPSILON_DEFAULT: float = 1.5


def gini_index(values: Sequence[float], weights: Optional[Sequence[float]] = None) -> float:
    """
    Population-weighted Gini coefficient for a discrete distribution.

    Parameters
    ----------
    values : sequence of float
        Per-capita quantities (e.g. consumption per capita by region) for a
        single period. Must be non-negative and comparable across entries.
    weights : sequence of float, optional
        Population or frequency weights for each entry. If None, all entries
        receive equal weight.

    Returns
    -------
    float
        Gini coefficient in [0, 1], or NaN if it cannot be computed.

    Notes
    -----
    - Implements the standard area-under-the-Lorenz-curve definition:
          G = 1 - 2 * ∫_0^1 L(p) dp
      using a trapezoidal approximation after sorting by values.
    - Designed for cross-sectional use (one period at a time).
    """
    v = np.asarray(values, dtype=float)
    if weights is None:
        w = np.ones_like(v, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != v.shape:
            raise ValueError("gini_index: 'values' and 'weights' must have the same shape.")

    # Keep only finite, positively-weighted entries
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v = v[mask]
    w = w[mask]

    n = v.size
    if n == 0:
        return float("nan")

    w_sum = w.sum()
    if w_sum <= 0:
        return float("nan")

    # Sort by value (ascending)
    order = np.argsort(v)
    v = v[order]
    w = w[order]

    # Normalize weights to sum to 1 (population shares)
    w_rel = w / w_sum
    # Income shares (value * weight, normalized)
    income = v * w_rel
    inc_sum = income.sum()
    if inc_sum <= 0:
        # All-zero income → perfectly equal (but trivial); return 0
        return 0.0
    inc_rel = income / inc_sum

    # Lorenz curve: cumulative population vs cumulative income
    cum_pop = np.concatenate(([0.0], np.cumsum(w_rel)))
    cum_inc = np.concatenate(([0.0], np.cumsum(inc_rel)))

    # Area under Lorenz curve via trapezoidal rule
    B = float(np.trapz(cum_inc, cum_pop))
    return float(1.0 - 2.0 * B)


def atkinson_index(
    values: Sequence[float],
    weights: Optional[Sequence[float]] = None,
    epsilon: float = ATKINSON_EPSILON_DEFAULT,
) -> float:
    """
    Population-weighted Atkinson index for a discrete distribution.

    Parameters
    ----------
    values : sequence of float
        Per-capita quantities (strictly positive), e.g. consumption per capita
        by region for a single period.
    weights : sequence of float, optional
        Population or frequency weights for each entry. If None, all entries
        receive equal weight.
    epsilon : float, default = ATKINSON_EPSILON_DEFAULT (1.5)
        Inequality-aversion parameter ε > 0. We use a single global ε, not
        region-specific, as per the standard Atkinson definition.

    Returns
    -------
    float
        Atkinson index A(ε) in [0, 1], or NaN if it cannot be computed.

    Notes
    -----
    - For ε ≠ 1:
          c* = [ Σ w_i * c_i^(1-ε) ]^(1/(1-ε))
      and A = 1 - c* / mean(c).
    - For ε → 1, we use the log-limit:
          c* = exp( Σ w_i * log c_i )
      and A = 1 - c* / mean(c).
    - Designed for cross-sectional use (one period at a time), with weights
      interpreted as population shares.
    """
    v = np.asarray(values, dtype=float)
    if weights is None:
        w = np.ones_like(v, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != v.shape:
            raise ValueError("atkinson_index: 'values' and 'weights' must have the same shape.")

    # Must be strictly positive for logs/powers
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0) & (v > 0)
    v = v[mask]
    w = w[mask]

    n = v.size
    if n == 0:
        return float("nan")

    w_sum = w.sum()
    if w_sum <= 0:
        return float("nan")
    w_rel = w / w_sum

    mean = float(np.sum(w_rel * v))
    if mean <= 0.0:
        return float("nan")

    eps = float(epsilon)
    if abs(eps - 1.0) < 1e-12:
        # Limit case ε → 1: use weighted geometric mean
        c_eq = float(np.exp(np.sum(w_rel * np.log(v))))
    else:
        term = float(np.sum(w_rel * v ** (1.0 - eps)))
        if term <= 0.0:
            return float("nan")
        c_eq = float(term ** (1.0 / (1.0 - eps)))

    return float(1.0 - c_eq / mean)


def attach_inequality_series(
    sol: Optional[Dict[str, Any]],
    *,
    params,
    regions: List[str],
    periods: List[int],
) -> None:
    """
    Compute and attach global inequality time series (Gini, Atkinson) to a solution dict.

    - Cross-sectional, period-by-period.
    - Uses per-capita consumption (C/L) and population weights L.
    - Atkinson uses a single global ε (hard-coded in utils.atkinson_index).

    The results are stored as:
        sol["gini"]     = {t -> G_t}
        sol["atkinson"] = {t -> A_t}
    so that results.output_format() treats them as 1D global series.
    """
    if not sol:
        return
    if "C" not in sol:
        return
    L = getattr(params, "L", None)
    if L is None:
        return

    C_map = sol["C"]

    # Normalize C to a regions×periods DataFrame
    if isinstance(C_map, pd.DataFrame):
        C_df = C_map.reindex(index=regions, columns=periods)
    else:
        C_df = pd.DataFrame(index=regions, columns=periods, dtype=float)
        # Expect keys like (region, t); be defensive about types.
        for (key, v) in C_map.items():
            if not isinstance(key, tuple) or len(key) != 2:
                continue
            r, t = key
            r = str(r)
            try:
                t_int = int(t)
            except Exception:
                continue
            if r in C_df.index and t_int in C_df.columns:
                C_df.at[r, t_int] = float(v)

    gini_by_t: Dict[int, float] = {}
    atk_by_t: Dict[int, float] = {}

    for t in periods:
        # L has columns 0..T; solver periods are 1..T.
        if t not in C_df.columns or t not in L.columns:
            continue
        C_t = C_df[t]
        L_t = L.loc[regions, t]
        mask = C_t.notna() & L_t.notna() & (L_t > 0) & (C_t > 0)
        if not mask.any():
            continue

        c_pc = (C_t[mask] / L_t[mask]).astype(float)
        w = L_t[mask].astype(float)

        vals = c_pc.to_numpy(dtype=float)
        weights = w.to_numpy(dtype=float)
        gini_by_t[t] = gini_index(vals, weights)
        atk_by_t[t] = atkinson_index(vals, weights)

    if gini_by_t:
        sol["gini"] = {int(k): float(v) for k, v in gini_by_t.items()}
    if atk_by_t:
        sol["atkinson"] = {int(k): float(v) for k, v in atk_by_t.items()}


# -----------------------------------
# Collect model results
# -----------------------------------
def _collect_2d(m: pe.ConcreteModel, name: str) -> Dict[tuple, float]:
    """Collect a 2D Var/Expression over (region, period) into {(r,t): value}."""
    out: Dict[tuple, float] = {}
    if not hasattr(m, name):
        return out
    comp = getattr(m, name)
    for r in m.REGIONS:
        for t in m.T:
            out[(r, int(t))] = pe.value(comp[r, t])
    return out


def _collect_1d(m: pe.ConcreteModel, name: str) -> Dict[int, float]:
    """Collect a 1D Var/Expression over (T) into {t: value}."""
    out: Dict[int, float] = {}
    if not hasattr(m, name):
        return out
    comp = getattr(m, name)
    for t in m.T:
        out[int(t)] = pe.value(comp[t])
    return out


# ---- Stability (single source of truth) -------------------------------------

@dataclass
class StabilityResult:
    internally_stable: bool
    externally_stable: bool
    fully_stable: bool
    leavers: List[str]
    joiners: List[str]

# -----------------------------------
# Convergence checks for Nash/iterates
# -----------------------------------
def has_converged(
    prev: Dict[Any, List[float]],
    curr: Dict[Any, List[float]],
    tol: float,
    ignore_last_k_periods: int = 0,
) -> bool:
    """
    Return True iff all |curr - prev| <= tol (optionally ignoring the last period).
    prev/curr map an index (e.g. region) -> list over periods.
    """
    n_total = len(next(iter(prev.values())))
    k = int(ignore_last_k_periods or 0)
    if k < 0:
        k = 0
    if k >= n_total:
        # ignore all but 0 → treat as 0 to preserve a valid check window
        k = n_total - 1 if n_total > 0 else 0
    n_periods = n_total - k

    max_delta = 0.0
    max_region = None
    max_period = None

    for r in curr:
        for i in range(n_periods):
            delta = abs(curr[r][i] - prev[r][i])
            if delta > max_delta:
                max_delta = delta
                max_region = r
                max_period = i
            if delta > tol:
                logger.debug(
                    "No convergence: region=%s, period=%s, delta=%.3e (tol=%.3e)",
                    r, i, delta, tol
                )
                return False

    logger.debug(
        "Convergence checked: max delta=%.3e (region=%s, period=%s), tol=%.3e",
        max_delta, str(max_region), str(max_period), tol
    )
    return True


def get_max_delta(
    prev: Dict[Any, List[float]],
    curr: Dict[Any, List[float]],
    ignore_last_k_periods: int = 0,
) -> float:
    """Return the maximum absolute difference across all indices/periods."""
    n_total = len(next(iter(prev.values())))
    k = int(ignore_last_k_periods or 0)
    if k < 0:
        k = 0
    if k >= n_total:
        k = n_total - 1 if n_total > 0 else 0
    n_periods = n_total - k
    max_delta = max(
        abs(curr[r][i] - prev[r][i])
        for r in curr
        for i in range(n_periods)
    )
    logger.debug("Max delta across all: %.3e", max_delta)
    return max_delta


def has_converged_multi(
    prevs: List[Dict[Any, List[float]]],
    currs: List[Dict[Any, List[float]]],
    tols: List[float],
    ignore_last_k_periods: int = 0,
) -> bool:
    """
    Check convergence for multiple variables in lockstep.
    Returns True only if every variable meets its tolerance.
    """
    for idx, (prev, curr, tol) in enumerate(zip(prevs, currs, tols)):
        if not has_converged(prev, curr, tol, ignore_last_k_periods):
            logger.debug("Variable index %d did not converge (tol=%.3e).", idx, tol)
            return False
    logger.debug("All variables converged.")
    return True


def print_most_violated_constraints(model, max_print: int = 20, tol: float = 1e-7) -> None:
    """Print up to `max_print` constraints with violation > tol."""
    viols = []
    for constr in model.component_objects(pe.Constraint, active=True):
        for idx in constr:
            c = constr[idx]
            if c.body is None:
                continue
            lb = value(c.lower) if c.lower is not None else None
            ub = value(c.upper) if c.upper is not None else None
            val = value(c.body)
            lv = max(0, lb - val) if lb is not None else 0
            uv = max(0, val - ub) if ub is not None else 0
            v = max(lv, uv)
            if v > tol:
                viols.append((v, constr.name, idx, val, lb, ub))
    viols.sort(reverse=True)
    if not viols:
        print("No constraint violations above tolerance.")
        return
    print(f"\nTop {min(len(viols), max_print)} most violated constraints (tol={tol}):")
    for v in viols[:max_print]:
        print(f"Constraint {v[1]}{v[2]}: violation={v[0]:.3g}, value={v[3]:.5g}, lower={v[4]}, upper={v[5]}")


# ---------------------------------------
# Coalition targeting (vectors & neighbors)
# ---------------------------------------
def _canon_vec(vec, *, N: int | None = None) -> tuple[int, ...]:
    """Canonical coalition vector key: tuple of 0/1 ints."""
    if isinstance(vec, tuple):
        v = tuple(int(x) for x in vec)
    elif isinstance(vec, list):
        v = tuple(int(x) for x in vec)
    else:
        v = tuple(int(x) for x in list(vec))
    if N is not None and len(v) != N:
        raise ValueError(f"Coalition vector length {len(v)} != {N}")
    return v

def _gray_order(n: int) -> List[Tuple[int, ...]]:
    """Return N-bit Gray code as tuples of 0/1 in MSB→LSB order."""
    out: List[Tuple[int, ...]] = []
    for i in range(1 << n):
        g = i ^ (i >> 1)
        out.append(tuple((g >> j) & 1 for j in range(n - 1, -1, -1)))
    return out

def _order_by_size_gray(vectors: List[Tuple[int, ...]]) -> List[Tuple[int, ...]]:
    """Bucket by Hamming weight; within each bucket, Gray order."""
    if not vectors:
        return []
    N = len(vectors[0])
    present = {tuple(v) for v in vectors}
    gray = _gray_order(N)
    ordered: List[Tuple[int, ...]] = []
    for k in range(1, N + 1):
        bucket = [v for v in gray if sum(v) == k and v in present]
        ordered.extend(bucket)
    return ordered

def _trim_seed(sol: dict, regions: Sequence[str], periods: Sequence[int]) -> dict:
    """Extract only S and mu as {(region, period): value}."""
    S = sol.get("S") or {}
    mu = sol.get("mu") or {}
    return {
        "S": {(str(r), int(t)): float(S.get((str(r), int(t)), 0.0)) for r in regions for t in periods},
        "mu": {(str(r), int(t)): float(mu.get((str(r), int(t)), 0.0)) for r in regions for t in periods},
    }

def _full_seed_for(
    vec: Tuple[int, ...],
    regions: Sequence[str],
    periods: Sequence[int],
    params,
    tstep: int,
    seed_grand: Optional[dict],
    seed_singleton: Dict[str, Optional[dict]],
    solved_local: Dict[Tuple[int, ...], dict],
) -> dict:
    """
    Build a complete seed respecting membership:
      - insiders: from GRAND (or latest coalition containing r), else BAU/backstop
      - outsiders: from their singleton(r), else BAU/backstop
    """
    R = list(regions)
    T = list(periods)
    S_out: Dict[Tuple[str, int], float] = {}
    mu_out: Dict[Tuple[str, int], float] = {}

    # index solved coalitions by member region
    coal_with_r: Dict[str, List[dict]] = {}
    for key, sol in solved_local.items():
        for r, bit in zip(R, key):
            if bit == 1:
                coal_with_r.setdefault(r, []).append(sol)

    for r, bit in zip(R, vec):
        for t in T:
            # BAU/backstop fallback (used only when anchors missing)
            bau_s = float(params.bau_saving_rates.at[r, t])
            year = int(params.base_year) + int(tstep) * int(t)
            bau_mu = 1.0 if year >= int(params.backstop_switch_year) else 0.0
            s_val, mu_val = bau_s, bau_mu

            if bit == 1:
                # member region: prefer GRAND, else any coalition that includes r
                if seed_grand is not None:
                    s_val = float(seed_grand["S"][(r, t)])
                    mu_val = float(seed_grand["mu"][(r, t)])
                elif r in coal_with_r and coal_with_r[r]:
                    latest = coal_with_r[r][-1]
                    s_val = float(latest["S"][(r, t)])
                    mu_val = float(latest["mu"][(r, t)])
            else:
                # outsider region: prefer its singleton
                seed_r = seed_singleton.get(r)
                if seed_r is not None:
                    s_val = float(seed_r["S"][(r, t)])
                    mu_val = float(seed_r["mu"][(r, t)])

            S_out[(r, t)] = s_val
            mu_out[(r, t)] = mu_val

    return {"S": S_out, "mu": mu_out}

# -----------------------------
# WINDOWS-SAFE WORKER FUNCTION
# -----------------------------
def _solve_task_module_level(args):
    """
    Module-level worker to be picklable on Windows (spawn).
    Returns (v_key, solution_dict, converged_bool)
    """
    
    # Make INFO logs (coalition iteration progress) visible from workers
    _setup_worker_logging(logging.INFO)
    
    (v_key, seed, regions, periods, params, tstep, utility, solver_opts,
     diagnostics_dir, exogenous_S, population_weight_envy_guilt,
     max_iter_nash, tol_mu_nash, relax, ignore_last_k_periods,
     negishi_use, negishi_weights,
     # --- FS pairing knobs (may be None/False for unpaired calls) ---
     pair_fs_in_worker, fs_disc_enabled, fs_disc_mode, fs_disc_file,
     exogenous_S_fs, negishi_use_fs, negishi_weights_fs, population_weight_envy_guilt_fs,
     fs_seed_proxy, bau_sol, fs_negishi_source) = args

    # Local imports here to avoid circulars when workers import the module
    from RICE13_FS.solve.coalition import solve_coalition_game, coalition_vec_to_member_string
    from RICE13_FS.solve.noncoop import solve_nash
    from RICE13_FS.analysis.negishi import compute_negishi_weights_from_bau_fs_after_disc

    s_bits = sum(v_key)
    is_singleton = (s_bits == 1)
    mask = "".join(str(int(b)) for b in v_key)
    label = coalition_vec_to_member_string(list(v_key), regions)
    if is_singleton:
        who = regions[list(v_key).index(1)]
        logger.info("START SINGLETON Nash for %s (%s; vec=%s).", who, utility.upper(), mask)
    else:
        logger.info("START coalition game for %s (vec=%s → %s).", utility.upper(), mask, label)
    if is_singleton:
        who_idx = list(v_key).index(1)
        who = regions[who_idx]
        sol = solve_nash(
            params=params, T=len(periods), tstep=tstep,
            utility=utility, solver_opts=solver_opts,
            diagnostics_dir=diagnostics_dir / f"nash_singletons_{utility.lower()}",
            initial_solution=seed, exogenous_S=exogenous_S,
            population_weight_envy_guilt=(population_weight_envy_guilt if utility == "fs" else False),
            max_iter=max_iter_nash, tol=tol_mu_nash, relax=relax,
            ignore_last_k_periods=ignore_last_k_periods,
            region_order=[who] + [r for r in regions if r != who],
        )
        return v_key, sol, bool(sol.get("converged", True)), []
    else:
        T = len(periods)

        # FS-only coalition solves with discounting-from-file should still work
        # even when CRRA coalitions are disabled (no pairing). Handle that case
        # explicitly here before falling back to the generic coalition solver +
        # optional CRRA→FS pairing.
        if (
            str(utility).lower() == "fs"
            and bool(fs_disc_enabled)
            and str(fs_disc_mode or "").strip().lower() == "file"
        ):
            if not fs_disc_file:
                raise ValueError("fs_disc_mode='file' requires fs_disc_file for FS coalitions.")

            d_file = _read_disc_csv(Path(fs_disc_file), regions, int(T))
            disc_series_final = d_file
            disc_tag_final = f"disc:file:regional:{digest_regional_series(d_file, regions, T)}"

            res = solve_coalition_game(
                params=params, T=T, tstep=tstep,
                coalition_vec=list(v_key), utility="fs",
                solver_opts=solver_opts, diagnostics_dir=diagnostics_dir / "coalitions",
                population_weight_envy_guilt=bool(population_weight_envy_guilt),
                initial_solution=seed, exogenous_S=exogenous_S,
                negishi_use=negishi_use, negishi_weights=negishi_weights,
                max_iter=max_iter_nash, tol=tol_mu_nash, relax=relax,
                ignore_last_k_periods=ignore_last_k_periods,
                discount_series=disc_series_final, disc_tag=disc_tag_final,
            )
            sol = res["solution"] if isinstance(res, dict) else res
            conv = bool(res.get("converged", True)) if isinstance(res, dict) else True
            if isinstance(sol, dict):
                sol.setdefault("disc_tag", disc_tag_final)
            return v_key, sol, conv, []

        # Default: generic coalition solve (CRRA or FS with geometric/data discounts).
        res = solve_coalition_game(
            params=params, T=T, tstep=tstep,
            coalition_vec=list(v_key), utility=utility,
            solver_opts=solver_opts, diagnostics_dir=diagnostics_dir / "coalitions",
            population_weight_envy_guilt=(population_weight_envy_guilt if utility == "fs" else False),
            initial_solution=seed, exogenous_S=exogenous_S,
            negishi_use=negishi_use, negishi_weights=negishi_weights,
            max_iter=max_iter_nash, tol=tol_mu_nash, relax=relax,
            ignore_last_k_periods=ignore_last_k_periods,
        )
        sol = res["solution"] if isinstance(res, dict) else res
        conv = bool(res.get("converged", True)) if isinstance(res, dict) else True

        # --- Optional: run the paired FS for the *same coalition* inside this worker
        fs_payload: List[dict] = []
        if (utility == "crra") and bool(pair_fs_in_worker) and bool(conv):
            try:
                fs_neg_source = str(fs_negishi_source or "").strip().lower()
                # Prepare mask/tag helpers
                mask = "".join(str(int(b)) for b in v_key)
                T = len(periods)
                # Announce FS coalition solve more descriptively depending on fs_disc_mode
                if fs_disc_mode == "one_pass":
                    logger.info(
                        "START coalition game for FS (vec=%s → %s) — first stage: one-pass discounting.",
                        mask, label
                    )
                elif fs_disc_mode == "two_pass":
                    logger.info(
                        "START coalition game for FS (vec=%s → %s) — first stage: one-pass baseline (stage 1).",
                        mask, label
                    )
                else:
                    logger.info(
                        "START coalition game for FS (vec=%s → %s) — single-stage mode: %s.",
                        mask, label, fs_disc_mode
                    )

                # Prefer a parent-prepared FS seed (exact or neighbor) if provided.
                def _seed_from_proxy(vk, fallback):
                    try:
                        return (fs_seed_proxy or {}).get(tuple(vk)) or fallback
                    except Exception:
                        return fallback

                def _seed_from_solution(sln: dict) -> dict:
                    return {"S": dict(sln.get("S", {})), "mu": dict(sln.get("mu", {}))}

                # Resolve discount series per mode
                disc_series_final = None
                disc_tag_final = "disc:data"

                if not fs_disc_enabled or str(fs_disc_mode).strip().lower() == "off":
                    # data/geometric → no special series
                    disc_series_final = None
                    disc_tag_final = "disc:data"
                    seed_final = _seed_from_proxy(
                        v_key,
                        fs_seed_for(v_key, store=None, regions=regions, periods=periods,
                                    preferred_disc_tag=None, strict_disc_match=False)
                    )
                elif str(fs_disc_mode).strip().lower() == "file":
                    d_file = _read_disc_csv(Path(fs_disc_file), regions, int(T))
                    disc_series_final = d_file
                    disc_tag_final = f"disc:file:regional:{digest_regional_series(d_file, regions, T)}"
                    seed_final = _seed_from_proxy(
                        v_key,
                        fs_seed_for(v_key, store=None, regions=regions, periods=periods,
                                    preferred_disc_tag=disc_tag_final, strict_disc_match=False)
                    )
                elif str(fs_disc_mode).strip().lower() == "one_pass":
                    d1 = _build_fs_discount_series_one_pass(
                        anchor_crra_sol=sol, params=params, regions=regions, periods=periods,
                        population_weight_envy_guilt=bool(population_weight_envy_guilt_fs),
                    )
                    disc_series_final = d1
                    dig1 = digest_regional_series(d1, regions, T)
                    disc_tag_final = f"disc:one_pass:coalition:regional:{dig1}:mask={mask}"
                    seed_final = _seed_from_proxy(
                        v_key,
                        fs_seed_for(v_key, store=None, regions=regions, periods=periods,
                                    preferred_disc_tag=disc_tag_final, strict_disc_match=False)
                    )
                else:
                    # two_pass
                    d1 = _build_fs_discount_series_one_pass(
                        anchor_crra_sol=sol, params=params, regions=regions, periods=periods,
                        population_weight_envy_guilt=bool(population_weight_envy_guilt_fs),
                    )
                    dig1 = digest_regional_series(d1, regions, T)
                    tag1 = f"disc:one_pass:coalition:regional:{dig1}:mask={mask}"
                    # Stage 1: FS(1) baseline (seed from neighbor/data)
                    seed_nb = _seed_from_proxy(
                        v_key,
                        fs_seed_for(v_key, store=None, regions=regions, periods=periods,
                                    preferred_disc_tag=tag1, strict_disc_match=False)
                    )
                    # FS(1) one-pass baseline
                    # Stage-1 FS baseline: for fs_after_disc we deliberately avoid using
                    # Negishi in FS(1); it’s only there to generate the discount path.
                    use_negishi_fs1 = bool(negishi_use_fs) and fs_neg_source in {"bau", "file"}
                    fs1_weights = negishi_weights_fs if use_negishi_fs1 else None
                    fs1 = solve_coalition_game(
                        params=params, T=T, tstep=tstep, coalition_vec=list(v_key), utility="fs",
                        solver_opts=solver_opts, diagnostics_dir=diagnostics_dir / "coalitions",
                        population_weight_envy_guilt=bool(population_weight_envy_guilt_fs),
                        initial_solution=seed_nb, exogenous_S=exogenous_S_fs,
                        negishi_use=use_negishi_fs1, negishi_weights=fs1_weights,
                        max_iter=max_iter_nash, tol=tol_mu_nash, relax=relax,
                        ignore_last_k_periods=ignore_last_k_periods,
                        discount_series=d1, disc_tag=tag1,
                    )
                    
                    # Unwrap to inner solution if needed for subsequent use
                    fs1_sol = fs1["solution"] if (isinstance(fs1, dict) and isinstance(fs1.get("solution"), dict)) else fs1
                    # Log FS(1) baseline outcome and caching intent
                    if fs1 and is_solution_optimal(fs1):
                        logger.info("FS(1) baseline optimal (vec=%s → %s); disc_tag=%s — not caching intermediate.", mask, label, tag1)
                        # Do NOT cache FS(1) baseline during two_pass; proceed to stage-2 only.
                        # Stage 2: FS(2) discounts from FS(1) ranks
                        d2 = _build_fs_discount_series_two_pass(
                            anchor_crra_sol=sol, fs_baseline_sol=fs1_sol, params=params,
                            regions=regions, periods=periods, population_weight_envy_guilt=bool(population_weight_envy_guilt_fs),
                        )
                        dig2 = digest_regional_series(d2, regions, T)
                        disc_series_final = d2
                        disc_tag_final = f"disc:two_pass:coalition:regional:{dig2}:mask={mask}"
                        seed_final = _seed_from_solution(fs1_sol)
                    else:
                        # If FS(1) fails, skip FS(2)
                        logger.warning("FS(1) baseline not optimal for vec=%s → %s; skipping stage-2 two-pass.", mask, label)
                        disc_series_final = None
                        disc_tag_final = None
                        seed_final = None

                # Final FS solve for modes off/file/one_pass, or two_pass when fs1 succeeded
                if disc_tag_final is not None:
                    # Log explicit “second stage” for two-pass FS
                    if fs_disc_mode == "two_pass":
                        logger.info(
                            "START coalition game for FS (vec=%s → %s) — second stage: two-pass discounting.",
                            mask, label
                        )
                    elif fs_disc_mode == "one_pass":
                        logger.info(
                            "START coalition game for FS (vec=%s → %s) — final one-pass solve.",
                            mask, label
                        )

                    # Decide which Negishi weights to use for the FINAL FS solve:
                    # - fs_after_disc → compute coalition-specific Negishi using the
                    #   coalition’s FS discount path and the BAU solution.
                    # - otherwise → fall back to the global FS Negishi weights.
                    if (
                        bool(negishi_use_fs)
                        and fs_neg_source == "fs_after_disc"
                        and disc_series_final is not None
                        and bau_sol is not None
                    ):
                        # Optional CSV dump of coalition-specific FS-after-disc Negishi weights.
                        # Only when diagnostics are enabled; otherwise we compute weights
                        # in-memory only and do not touch the filesystem.
                        output_path = None
                        if DIAGNOSTICS_ON and diagnostics_dir is not None:
                            # Example:
                            #   <diagnostics_dir>/negishi_fs_after_disc/
                            #       negishi_weights_fs_after_disc_EU_JAP_OHI.csv
                            weights_dir = diagnostics_dir / "negishi_fs_after_disc"
                            filename = f"negishi_weights_fs_after_disc_{label}.csv"
                            output_path = weights_dir / filename

                        neg_weights_final = compute_negishi_weights_from_bau_fs_after_disc(
                            params=params,
                            bau_sol=bau_sol,
                            fs_disc=disc_series_final,
                            population_weight_envy_guilt=bool(population_weight_envy_guilt_fs),
                            output_path=output_path,
                        )
                    else:
                        neg_weights_final = negishi_weights_fs

                    fs_final = solve_coalition_game(
                        params=params, T=T, tstep=tstep,
                        coalition_vec=list(v_key), utility="fs",
                        solver_opts=solver_opts, diagnostics_dir=diagnostics_dir / "coalitions",
                        population_weight_envy_guilt=bool(population_weight_envy_guilt_fs),
                        initial_solution=seed_final, exogenous_S=exogenous_S_fs,
                        negishi_use=bool(negishi_use_fs), negishi_weights=neg_weights_final,
                        max_iter=max_iter_nash, tol=tol_mu_nash, relax=relax,
                        ignore_last_k_periods=ignore_last_k_periods,
                        discount_series=disc_series_final, disc_tag=disc_tag_final,
                    )                

                    if fs_final:
                        fs_final.setdefault("disc_tag", disc_tag_final)
                        if is_solution_optimal(fs_final):

                            logger.info("FS final optimal (vec=%s → %s); disc_tag=%s — returning to parent for caching.",
                                        mask, label, disc_tag_final)
                        else:
                            logger.warning("FS final returned but not optimal (vec=%s → %s); will be dropped by cache guard.",
                                           mask, label)
                        fs_payload.append({
                            "v_key": v_key, "solution": fs_final, "converged": bool(fs_final.get("converged", True))
                        })
                    else:
                        logger.warning("FS final solve failed for vec=%s → %s; nothing to cache.", mask, label)
            except Exception as e:
                logger.exception("Paired FS solve failed for coalition vec=%s due to: %s", v_key, e)

        return v_key, sol, conv, fs_payload

def _setup_worker_logging(level: int = logging.INFO) -> None:
    """
    Ensure worker processes emit INFO logs (e.g., coalition iteration progress).
    On Windows 'spawn', logging config doesn't propagate; set up a StreamHandler.
    """
    root = logging.getLogger()
    if not root.handlers:
        h = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | [%(processName)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        h.setFormatter(fmt)
        h.setLevel(level)
        root.addHandler(h)
    root.setLevel(level)


def neighbors_of(vec: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    """All Hamming-distance-1 neighbors of a coalition vector."""
    out: List[Tuple[int, ...]] = []
    for i, bit in enumerate(vec):
        v = list(vec)
        v[i] = 1 - int(bit)
        out.append(tuple(v))
    return out


def compute_target_vectors(
    *,
    regions: Sequence[str],
    base_vector: Optional[Tuple[int, ...]],
    want_neighbors: bool,
    mega_run: bool,
) -> List[Tuple[int, ...]]:
    """
    Decide which coalition vectors must be materialized.
    - mega_run=True  → all non-empty coalitions
    - else           → [base] (+ neighbors if requested)
    Order is stable and de-duplicated.
    """
    n = len(regions)
    if mega_run:
        vecs = []
        for k in range(1, 2 ** n):  # all non-empty subsets
            bits = tuple(int(b) for b in f"{k:0{n}b}")
            vecs.append(bits)
        return vecs

    vecs: List[Tuple[int, ...]] = []
    if base_vector is not None:
        vecs.append(base_vector)
        if want_neighbors:
            vecs.extend(neighbors_of(base_vector))
    # dedupe preserving order
    seen: set = set()
    out: List[Tuple[int, ...]] = []
    for v in vecs:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


# --------------------------------------------------------------------
# Shared fix/unfix helpers used by both coalition and noncoop solvers
# --------------------------------------------------------------------

def fix_profiles(
    m: pe.ConcreteModel,
    S_prof,
    mu_prof,
    *,
    lock: Iterable[str],
    periods,
) -> None:
    """
    Fix S and μ for all regions in `lock` to their (clamped) profile values.
    """
    for r in lock:
        for i, t in enumerate(periods):
            s_lb = float(m.S[r, t].lb) if m.S[r, t].lb is not None else 0.0
            s_ub = float(m.S[r, t].ub) if m.S[r, t].ub is not None else 1.0
            mu_lb = float(m.mu[r, t].lb) if m.mu[r, t].lb is not None else 0.0
            mu_ub = float(m.mu[r, t].ub) if m.mu[r, t].ub is not None else 1.0
            m.S[r, t].fix(clean(S_prof[r][i], lower=s_lb, upper=s_ub))
            m.mu[r, t].fix(clean(mu_prof[r][i], lower=mu_lb, upper=mu_ub))


def unfix_controls(
    m: pe.ConcreteModel,
    unlock: Iterable[str],
    *,
    unfix_S: bool,
    unfix_mu: bool,
) -> None:
    """
    Unfix S and/or μ for the listed regions.
    """
    for r in unlock:
        for t in m.T:
            if unfix_S and m.S[r, t].fixed:
                m.S[r, t].unfix()
            if unfix_mu and m.mu[r, t].fixed:
                m.mu[r, t].unfix()

def final_evaluation_setup(
    m,
    regions: List[str],
    mu_prof: Dict[str, List[float]],
    S_prof: Dict[str, List[float]],
    *,
    exogenous_S_used: bool,
) -> None:
    """Final evaluation convention:
       - Always fix μ to profile
       - Fix S to profile only when endogenous (exogenous_S_used == False)
       Re-creates OBJ to avoid stale components if this block runs multiple times."""
    # fix μ to profile (clamped to bounds)
    for r in regions:
        for i, t in enumerate(list(m.T)):
            mu_lb = float(m.mu[r, t].lb) if m.mu[r, t].lb is not None else 0.0
            mu_ub = float(m.mu[r, t].ub) if m.mu[r, t].ub is not None else 1.0
            v = mu_prof[r][i]
            m.mu[r, t].fix(v if mu_lb <= v <= mu_ub else max(mu_lb, min(mu_ub, v)))

    # fix S iff endogenous
    if not exogenous_S_used:
        for r in regions:
            for i, t in enumerate(list(m.T)):
                s_lb = float(m.S[r, t].lb) if m.S[r, t].lb is not None else 0.0
                s_ub = float(m.S[r, t].ub) if m.S[r, t].ub is not None else 1.0
                v = S_prof[r][i]
                m.S[r, t].fix(v if s_lb <= v <= s_ub else max(s_lb, min(s_ub, v)))

    # (re)create objective cleanly
    if hasattr(m, "OBJ"):
        m.del_component(m.OBJ)
    m.OBJ = pe.Objective(
        expr=sum(m.disc[r, t] * m.U[r, t] for r in regions for t in m.T),
        sense=pe.maximize,
    )

def assert_exogenous_S_fixed(m, exoS_df, regions: List[str], periods: List[int], tol: float = 1e-10) -> None:
    """Defense-in-depth: S must be fixed to exogenous path when exogenous_S is used."""
    for r in regions:
        for i, t in enumerate(periods):
            if not m.S[r, t].fixed:
                raise AssertionError("S must be fixed when exogenous_S is provided")
            v = float(pe.value(m.S[r, t]))
            target = float(exoS_df.at[r, t])
            if abs(v - target) > tol:
                raise AssertionError(f"Exogenous S mismatch at (r={r}, t={t}): {v} vs {target}")

def fix_profiles_except(
    m: pe.ConcreteModel,
    regions: List[str],
    periods: List[int],
    S_prof: Dict[str, List[float]],
    mu_prof: Dict[str, List[float]],
    *,
    active_region: str,
    respect_exogenous_S: bool,
) -> None:
    """
    Fix S, μ for all regions except `active_region` to their profile values.
    For the active region:
      - if `respect_exogenous_S` is False, unfix S (so it can move);
      - always unfix μ (active chooses μ).
    Values are clamped to the model's variable bounds.
    """
    for r in regions:
        for i, t in enumerate(periods):
            s_lb = float(m.S[r, t].lb) if m.S[r, t].lb is not None else 0.0
            s_ub = float(m.S[r, t].ub) if m.S[r, t].ub is not None else 1.0
            mu_lb = float(m.mu[r, t].lb) if m.mu[r, t].lb is not None else 0.0
            mu_ub = float(m.mu[r, t].ub) if m.mu[r, t].ub is not None else 1.0
            if r != active_region:
                m.S[r, t].fix(clean(S_prof[r][i],  lower=s_lb,  upper=s_ub))
                m.mu[r, t].fix(clean(mu_prof[r][i], lower=mu_lb, upper=mu_ub))
            else:
                if (not respect_exogenous_S) and m.S[r, t].fixed:
                    m.S[r, t].unfix()
                if m.mu[r, t].fixed:
                    m.mu[r, t].unfix()

# ---------------------------------------
# Fingerprints / spec identity
# ---------------------------------------
_NUM_KEYS_ROUND = 12  # normalize floats so 1e-6 == 0.000001


def _norm_for_fp(x: Any) -> Any:
    if isinstance(x, float):
        return float(f"{x:.{_NUM_KEYS_ROUND}g}")
    if isinstance(x, dict):
        return {k: _norm_for_fp(v) for k, v in sorted(x.items(), key=lambda kv: kv[0])}
    if isinstance(x, (list, tuple)):
        return [_norm_for_fp(v) for v in x]
    return x


def normalize_fingerprint(fp: Dict[str, Any]) -> Dict[str, Any]:
    """Round floats & sort keys to canonicalize fingerprint representation."""
    return _norm_for_fp(dict(fp))


def _hash_df_robust(df: pd.DataFrame, digits: int = 12) -> str:
    """Stable hash of a DF using labels and values rounded to `digits`."""
    idx = tuple(str(x) for x in df.index.tolist())
    cols = tuple(int(c) if str(c).isdigit() else str(c) for c in df.columns.tolist())
    vals = np.round(df.to_numpy(dtype=float), digits).flatten()
    h = hashlib.sha1()
    h.update(repr(idx).encode("utf-8"))
    h.update(repr(cols).encode("utf-8"))
    h.update(vals.tobytes())  # already rounded
    return h.hexdigest()


def digest_series(series_1_to_T: Sequence[float] | Mapping[int, float], *, decimals: int = 12) -> str:
    """
    Short, order-aware digest for a per-period series (1..T).
    Accepts either a list-like of length T or a dict {t -> value}.
    """
    if isinstance(series_1_to_T, dict) or isinstance(series_1_to_T, Mapping):
        items = sorted((int(k), float(v)) for k, v in series_1_to_T.items())
        payload = ",".join(f"{t}:{v:.{decimals}g}" for t, v in items)
    else:
        vals = [float(v) for v in series_1_to_T]
        payload = ",".join(f"{t+1}:{vals[t]:.{decimals}g}" for t in range(len(vals)))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]


def digest_regional_series(
    series_rt: Mapping[Tuple[str, int], float],
    regions: Sequence[str],
    T: int,
    *, decimals: int = 12
) -> str:
    """
    Short, order-aware digest for a per-region discount grid {(region, t) -> d_{r,t}}.
    Canonical order: regions in provided order, t=1..T.
    """
    parts: List[str] = []
    for r in regions:
        for t in range(1, int(T) + 1):
            v = float(series_rt.get((str(r), int(t)), 1.0))
            parts.append(f"{r}|{t}|{v:.{decimals}g}")
    payload = ";".join(parts).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:8]

def tag_from_digest(mode: str, digest: str, vec: Sequence[int] | None = None) -> str:
    """
    Stable tag for FS discount series used in spec ids and cache keys.
    Examples:
      - "disc:data"
      - "disc:file:<hash8>"
      - "disc:one_pass:<hash8>:mask=1010"
      - "disc:two_pass:<hash8>:mask=1110"
    """
    m = str(mode).strip().lower()
    if m == "data" or m == "off":
        return "disc:data"
    if m == "file":
        return f"disc:file:{digest}"
    mask = vec_to_bitmask(vec) if vec is not None else "None"
    return f"disc:{m}:{digest}:mask={mask}"

def tag_from_series(mode: str, series_rt: Mapping[Tuple[str, int], float],
                    regions: Sequence[str], T: int, vec: Sequence[int] | None = None) -> str:
    """
    Convenience wrapper: compute digest and return tag in one go.
    """
    d = digest_regional_series(series_rt, regions, T)
    return tag_from_digest(mode, d, vec)

def _store_has_latest_fs(store) -> bool:
    """Return True iff the CoalitionStore implements get_latest_fs(vec)."""
    return hasattr(store, "get_latest_fs") and callable(getattr(store, "get_latest_fs"))

def _try_get_latest_fs(store, vec):
    """Helper: call store.get_latest_fs(vec) if available; else return None."""
    if _store_has_latest_fs(store):
        return store.get_latest_fs(vec)
    return None

def make_data_seed(regions: Sequence[str], periods: Sequence[int]) -> dict:
    """
    Minimal 'data' seed: S,mu zeros; solvers that implement a 'data' seed
    convention can recognize this and fill Excel-style μ and BAU S as needed.
    """
    S = {(str(r), int(t)): 0.0 for r in regions for t in periods}
    mu = {(str(r), int(t)): 0.0 for r in regions for t in periods}
    return {"S": S, "mu": mu, "seed_kind": "data"}

def fs_seed_for(
    vec: Sequence[int],
    *, store,
    regions: Sequence[str],
    periods: Sequence[int],
    preferred_disc_tag: str | None = None,
    strict_disc_match: bool = False,
) -> Optional[dict]:
    """
    FS warm-start policy for coalitions (best → worst):
      1) Exact same coalition (if an FS result exists for it).
      2) Any **FS Hamming-1 neighbor**.
      3) Canonical 'data' seed.

    When strict_disc_match=True, steps (1) and (2) require sol['disc_tag'] == preferred_disc_tag.
    """
    vkey = _canon_vec(vec, N=len(regions))

    # (1) exact same coalition, latest FS (if store supports it)
    hit_exact = _try_get_latest_fs(store, vkey)
    if hit_exact and (hit_exact.get("solution") or {}).get("utility") == "fs":
        sol = hit_exact["solution"]
        if not strict_disc_match or preferred_disc_tag is None or str(sol.get("disc_tag")) == str(preferred_disc_tag):
            return _trim_seed(sol, regions, periods)

    # (2) Hamming-1 FS neighbor
    for nb in neighbors_of(vkey):
        hit = _try_get_latest_fs(store, nb)
        if hit and (hit.get("solution") or {}).get("utility") == "fs":
            sol = hit["solution"]
            if strict_disc_match and preferred_disc_tag is not None:
                if str(sol.get("disc_tag")) != str(preferred_disc_tag):
                    continue
            return _trim_seed(sol, regions, periods)

    # Fallback: 'data' seed (lets the solver build Excel-style μ and BAU S)
    return make_data_seed(regions, periods)


def same_exogenous_S(a: Optional[pd.DataFrame], b: Optional[pd.DataFrame], tol: float = 1e-12) -> bool:
    """
    True iff both DataFrames exist and (a) have identical labels and (b) are equal within `tol`.
    Fast path: if hashes (rounded to ~machine precision) match, accept immediately.
    We deliberately do NOT treat None as equal to anything (endogenous S never reuses).
    """
    if a is None or b is None:
        return False
    try:
        if a.shape != b.shape:
            return False
        if a.index.tolist() != b.index.tolist():
            return False
        if a.columns.tolist() != b.columns.tolist():
            return False
        if _hash_df_robust(a) == _hash_df_robust(b):
            return True
        diff = (a - b).abs()
        return bool(diff.le(tol).all().all())
    except Exception:
        return False


def _s_mode_tag(s_mode: str, s_file: Optional[str]) -> str:
    """
    Canonicalize a single S-mode into a stable tag.
    Supported: optimal | bau | file | exogenous | planner_crra | planner_fs
    """
    mode = str(s_mode or "").strip().lower()
    f = s_file
    if mode in ("exogenous", "exog", "file"):
        if not f:
            raise KeyError("S_mode='file' requires a file path.")
        return f"exogenous:file:{Path(f).resolve()}"
    if mode == "bau":
        return "exogenous:bau"
    if mode in ("planner_crra", "planner_fs"):
        return f"planner:{mode.split('_', 1)[1]}"
    # passthrough (e.g., "optimal")
    return mode


def build_config_fingerprint(config: Dict[str, Any], regions: Sequence[str]) -> Dict[str, Any]:
    """
    Inputs-only fingerprint (no defaults; raise if required keys are missing).
    Decadal-only (no tstep). Updated for split coalition config (CRRA/FS).
    """
    # Coalition S-mode tags per utility
    s_mode_crra = _s_mode_tag(
        config.get("coalition_crra_S_mode", "optimal"),
        config.get("coalition_crra_S_file") or config.get("S_file"),
    )
    s_mode_fs = _s_mode_tag(
        config.get("coalition_fs_S_mode", "optimal"),
        config.get("coalition_fs_S_file") or config.get("S_file"),
    )

    fp = {
        "regions": list(regions),
        "T": int(config["T"]),
        # Coalition utility toggles instead of a single 'coalition_utility'
        "coalitions": {
            "run_crra": bool(config.get("run_coalition_crra", False)),
            "run_fs": bool(config.get("run_coalition_fs", False)),
            "S_mode_crra": s_mode_crra,
            "S_mode_fs": s_mode_fs,
        },
        "negishi_use": bool(config.get("negishi_use", False)),
        "negishi_source": str(config.get("negishi_source", "")),
        "solver": {
            "max_iter_nash": int(config["max_iter_nash"]),
            "tol_mu_nash": float(config["tol_mu_nash"]),
            "relax": float(config["nash_relax"]),
        },
        "namespace": str(config.get("cache_namespace", "")),
    }

    # ---- FS discounting knobs only (no series content here) ----
    # population_weight_envy_guilt also affects FS MU shares; include as a knob
    fs_enabled = bool(config.get("fs_disc_enabled", False))
    fs_mode = str(config.get("fs_disc_mode", "off")).strip().lower()
    fs_path = "None"
    if fs_mode == "file":
        p = config.get("fs_disc_file")
        if p:
            fs_path = str(Path(p).resolve())
    fp["fs_discounting"] = {
        "enabled": fs_enabled,
        "mode": fs_mode,
        "file": fs_path,
        "population_weight_envy_guilt": bool(config.get("population_weight_envy_guilt", False)),
    }

    return normalize_fingerprint(fp)

def stamp_identity(sol: dict, *, utility: str, spec_id: str, disc_tag: str | None = None) -> dict:
    """Ensure solution carries identity fields required by cache/exporter."""
    u = (utility or "").lower()
    sol["utility"] = u
    if u == "fs":
        sol["disc_tag"] = disc_tag or sol.get("disc_tag") or "disc:data"
    else:
        sol.pop("disc_tag", None)
    sol["spec_id"] = spec_id
    return sol


def vec_to_bitmask(vec: Sequence[int]) -> str:
    """Convert 0/1 vector to bitmask string, e.g., (1,0,1) → '101'."""
    return "".join("1" if int(b) else "0" for b in vec)


def df_signature_canonical(
    df: Optional[pd.DataFrame],
    countries: Sequence[str],
    T: int,
    *, decimals: int = 8
) -> str:
    """Canonical, label-aware signature of a (regions×1..T) DF (or 'None')."""
    if df is None:
        return "None"
    S = normalize_exogenous_S(df, countries, T)
    return _hash_df_robust(S, digits=decimals)


def _negishi_digest_from_df(weights: Optional[pd.DataFrame],
                            countries: Sequence[str],
                            T: int) -> str:
    """
    Short, label-aware digest of Negishi weights (or 'None').
    Uses the same canonicalization as exogenous S (regions × 1..T).
    """
    if weights is None:
        return "None"
    W = normalize_exogenous_S(weights, countries, T)
    return _hash_df_robust(W, digits=8)


def build_solution_spec_id(
    *,
    utility: str,                    # 'crra' | 'fs'
    T: int,
    countries: Sequence[str],
    population_weight_envy_guilt: bool,
    exogenous_S: Optional[pd.DataFrame] = None,
    negishi_use: bool = False,
    negishi_digest: Optional[str] = None,
    negishi_weights: Optional[pd.DataFrame] = None,  # optional convenience if you don't precompute the digest
    disc_tag: Optional[str] = None,                  # optional: discount series identity for FS
) -> str:
    """
    Spec identity for *compatibility* (what must match to compare/reuse payoffs).
    Decadal-only; no tstep. Includes optional Negishi digest if negishi_use=True.
    For FS utility, optionally embeds a discount-series tag to prevent cross-reuse
    across different discount factor series.
    """
    util = str(utility).strip().lower()
    T = int(T)
    fsw = bool(population_weight_envy_guilt) if util == "fs" else False
    countries_norm = [str(c) for c in list(countries)]
    s_tag = df_signature_canonical(exogenous_S, countries_norm, T)

    # Prefer explicit digest; otherwise compute from provided weights
    if negishi_use:
        w_tag = negishi_digest or _negishi_digest_from_df(negishi_weights, countries_norm, T)
    else:
        w_tag = "None"

    # Discount tag: only meaningful for FS; CRRA ignores (set to "None")
    if util == "fs":
        d_tag = str(disc_tag or "disc:data")  # default geometric from data unless overridden
    else:
        d_tag = "None"


    spec = {
        "util": util,
        "T": T,
        "fsw": fsw,
        "S": s_tag,
        "neg": bool(negishi_use),
        "W": w_tag,
        "countries": countries_norm,
        "disc": d_tag,
    }
    blob = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def build_solution_fingerprint(
    *,
    mode: str,                       # 'bau' | 'planner' | 'nash' | 'coalition'
    coalition_vec: Optional[Sequence[int]],
    spec_id: str
) -> str:
    """Full fingerprint (adds mode + coalition bitmask to the spec id)."""
    coal = vec_to_bitmask(coalition_vec) if coalition_vec is not None else "None"
    return f"{mode}|{spec_id}|coal={coal}"


def fetch_or_solve_coalitions(
    *,
    regions: List[str],
    periods: List[int],
    utility: str,                                  # "crra" | "fs"
    vectors: List[Tuple[int, ...]],
    store,                                         # CoalitionStore | None
    params,
    tstep: int,                                    # <-- pass in explicitly
    solver_opts: Dict[str, Any],
    exogenous_S: Optional[pd.DataFrame],
    negishi_use: bool,
    negishi_weights: Optional[pd.DataFrame],
    population_weight_envy_guilt: bool,            # required; no default
    max_iter_nash: int,
    tol_mu_nash: float,
    relax: float,
    ignore_last_k_periods: int,
    diagnostics_dir: Path,
    workers: int = 1,
    planner_anchor_spec_id: str | None = None,
    nash_anchor_spec_id: str | None = None,
    # ---- Pair CRRA→FS in the SAME worker (optional) ----
    pair_fs_in_worker: bool = False,
    fs_disc_enabled: bool | None = None,
    fs_disc_mode: Optional[str] = None,
    fs_disc_file: Optional[str] = None,
    exogenous_S_fs: Optional[pd.DataFrame] = None,
    negishi_use_fs: bool | None = None,
    negishi_weights_fs: Optional[pd.DataFrame] = None,
    population_weight_envy_guilt_fs: bool | None = None,
    failures_path: Path | None = None,
    bau_sol: Optional[Dict[str, Any]] = None,
    fs_negishi_source: Optional[str] = None,
) -> None:
    """
    Ensure each coalition vector in `vectors` exists in cache; if missing, solve and put.
    Side-effect only (populates the cache). Cache contract requires discounted payoffs.
    """
    # Local imports to avoid cyclic dependencies across subpackages
    from RICE13_FS.solve.coalition import coalition_vec_to_member_string
    from RICE13_FS.solve.coop import solve_planner
    from RICE13_FS.solve.noncoop import solve_nash

    # ---- basic dims / helpers
    T = len(periods)
    N = len(regions)

    def _canon(v) -> Tuple[int, ...]:
        t = tuple(int(b) for b in v)
        if len(t) != N:
            raise ValueError(f"Coalition vector length {len(t)} != N={N}")
        return t

    grand = _canon([1] * N)
    singletons = {_canon([1 if j == i else 0 for j in range(N)]) for i in range(N)}

    # Normalize exogenous S once for spec_id/caching decisions
    exoS_norm = normalize_exogenous_S(exogenous_S, regions, T) if exogenous_S is not None else None

    # Build the run's spec_id once; this gates cache reuse
    spec_id = build_solution_spec_id(
        utility=utility,
        T=T,
        countries=regions,
        population_weight_envy_guilt=(population_weight_envy_guilt if utility == "fs" else False),
        exogenous_S=exoS_norm,
        negishi_use=bool(negishi_use),
        negishi_weights=negishi_weights,
    )

    # 0) Partition into hits/misses (and write-through hits as before)
    misses: List[Tuple[int, ...]] = []
    for vec in vectors:
        v_key = _canon(vec)  # canonical tuple key
        mask = "".join(str(int(b)) for b in v_key)
        label = coalition_vec_to_member_string(list(v_key), regions)

        # shape flags
        s_bits = sum(int(b) for b in v_key)
        is_singleton = (v_key in singletons)
        is_empty = (s_bits == 0)

        # --- cache probe
        hit = None
        if store is not None:
            hit = store.get(v_key, spec_id)
            if hit is not None and hit.get("solution") is not None:
                # (0) reject non-converged cached entries
                meta = dict(hit.get("meta") or {})
                if meta.get("converged") is False:
                    logger.info("Bypassing cache for %s (vec=%s → %s): cached entry marked non-converged.",
                                "EMPTY" if is_empty else ("SINGLETON" if is_singleton else "coalition"),
                                mask, label)
                    hit = None
                # (1) exogenous S mismatch → invalidate
                if exoS_norm is not None and not same_exogenous_S(exoS_norm, hit["solution"].get("S_exogenous")):
                    logger.info(
                        "Bypassing cache for %s (vec=%s → %s): exogenous S mismatch with cached entry.",
                        "EMPTY" if is_empty else ("SINGLETON" if is_singleton else "coalition"),
                        mask, label
                    )
                    hit = None

                # (2) non-optimal cached solution → invalidate (robust)
                elif not is_solution_optimal(hit["solution"]):
                    logger.info("Bypassing cache for %s (vec=%s → %s): cached solution is non-optimal.",
                                "EMPTY" if is_empty else ("SINGLETON" if is_singleton else "coalition"),
                                mask, label)
                    hit = None

        if hit is not None:
            # If we're in CRRA + paired-FS mode with FS discounting enabled, make sure
            # an FS result for this vector exists under the current FS mode. If not,
            # treat this as a miss so the worker re-runs CRRA and then FS.
            if (str(utility).lower() == "crra") and bool(pair_fs_in_worker) and bool(fs_disc_enabled):
                fs_mode = str(fs_disc_mode or "off").strip().lower()
                mode_to_prefix = {
                    "off":      "disc:data",
                    "file":     "disc:file:",
                    "one_pass": "disc:one_pass:coalition:",
                    "two_pass": "disc:two_pass:coalition:",
                }
                prefix = mode_to_prefix.get(fs_mode, "disc:data")
                fs_seen = False
                try:
                    hit_fs = _try_get_latest_fs(store, v_key)
                    if hit_fs and (hit_fs.get("solution") or {}).get("utility") == "fs":
                        tag = str(hit_fs["solution"].get("disc_tag", "")).lower()
                        if tag.startswith(prefix):
                            fs_seen = True
                except Exception:
                    logger.debug("FS presence scan failed; conservatively scheduling FS.", exc_info=True)
                    fs_seen = False
                if not fs_seen:
                    logger.info("Scheduling %s despite CRRA cache hit — FS (%s) is missing.", label, fs_mode)
                    misses.append(v_key)
                    continue
            # If we got a compatible hit, log with context
            kind = "EMPTY" if is_empty else ("SINGLETON" if is_singleton else "coalition")
            logger.info("Cache hit for %s %s (vec=%s → %s) — skipping solve.",
                        utility.upper(), kind, mask, label)
            # Ensure presence under current namespace (idempotent put). Only mirror if optimal (robust).
            if is_solution_optimal(hit["solution"]):
                payoff2 = payoff_row_discounted(hit["solution"], regions, periods)
                if store is not None:
                    sid = (hit.get("solution") or {}).get("spec_id")
                    if not sid:
                        raise ValueError("Cache hit missing 'spec_id' in solution; cannot write to store.")
                    store.put(vec=v_key, spec_id=sid, label=label,
                              payoff=payoff2, solution=hit["solution"],
                              meta=dict(hit.get("meta") or {}))
        else:
            misses.append(v_key)

    if not misses:
        return

    # 1) Solve anchors first (sequential): GRAND then SINGLETONS (if missing)
    solved_local: Dict[Tuple[int, ...], dict] = {}
    def _put_and_record(v_key, label, sol, S_tag="coalition", *, converged: bool | None = None):
        """
        Store a finished coalition solution in the cache and remember its trimmed seed locally.
        """
        if not v_key or not any(int(x) for x in v_key):
            return
        mask = "".join(str(int(b)) for b in v_key)
        try:
            # Unwrap accidental {"solution": {...}}
            if isinstance(sol, dict) and "optimal" not in sol and "solution" in sol and isinstance(sol["solution"], dict):
                sol = sol["solution"]
    
            # 1. Utility flavor: must be stamped by the solver, no guessing.
            sol_utility_raw = str(sol.get("utility", "") or "").lower()
            if sol_utility_raw not in ("crra", "fs"):
                raise ValueError(
                    f"Coalition solution missing valid 'utility' field "
                    f"(expected 'crra' or 'fs', got {sol.get('utility', None)!r}) "
                    f"for vec={mask} ({label}). "
                    "Solvers must set sol['utility'] before caching."
                )
            util_final = sol_utility_raw           # 'crra' or 'fs', lower-case
            util_for_log = util_final.upper()      # 'CRRA' or 'FS', for messages
    
            # 2. Optimality guard
            if sol and not is_solution_optimal(sol):
                logger.warning(
                    "Not caching %s (vec=%s → %s): solver did not report optimal termination.",
                    util_for_log,
                    mask,
                    label,
                )
                try:
                    if failures_path:
                        failures_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(failures_path, "a", encoding="utf-8") as fh:
                            fh.write(f"\t{util_for_log}\t{mask}\t{label}\tnonoptimal\n")
                except Exception:
                    logger.debug("Could not append to failures.txt", exc_info=True)
                return
    
            if store is not None and isinstance(sol, dict) and sol:
                # 3. Discount tag: only relevant for FS; CRRA ignores it.
                disc_raw = str(sol.get("disc_tag", "") or "")
                if util_final == "fs":
                    # Ensure we always carry an explicit discount tag for spec_id/export.
                    disc_final = disc_raw or "disc:data"
                else:
                    disc_final = None
    
                # 4. Build spec_id and stamp identity
                sid_final = build_solution_spec_id(
                    utility=util_final,   # lower-case
                    T=T,
                    countries=regions,
                    population_weight_envy_guilt=(population_weight_envy_guilt if util_final == "fs" else False),
                    exogenous_S=exoS_norm,
                    negishi_use=bool(negishi_use),
                    negishi_weights=negishi_weights,
                    disc_tag=disc_final,
                )
                sol = stamp_identity(sol, utility=util_final, spec_id=sid_final, disc_tag=disc_final)
    
                # 5. Convergence guard
                conv_flag = (bool(converged) if converged is not None else bool(sol.get("converged", True)))
                if conv_flag is False:
                    flavor_for_fail = util_for_log
                    logger.error(
                        "Nash loop did not converge (max_iter reached). "
                        "Skipping cache write for %s (S_tag=%s, vec=%s, utility=%s).",
                        label,
                        S_tag,
                        mask,
                        flavor_for_fail,
                    )
                    try:
                        if failures_path:
                            failures_path.parent.mkdir(parents=True, exist_ok=True)
                            with open(failures_path, "a", encoding="utf-8") as fh:
                                fh.write(f"\t{flavor_for_fail}\t{mask}\t{label}\tnonconverged\n")
                    except Exception:
                        logger.debug("Could not append to failures.txt", exc_info=True)
                    return
    
                # 6. Required fields and cache write
                if "disc" not in sol:
                    raise ValueError("Coalition solution missing 'disc' entries (discounted utilities).")
                sid = sol.get("spec_id")
                if not sid:
                    raise ValueError(
                        "Fresh coalition solution missing 'spec_id'; "
                        "solvers must tag spec_id before caching."
                    )
    
                payoff = payoff_row_discounted(sol, regions, periods)
                meta = {
                    "converged": conv_flag,
                    "max_delta": float(sol.get("max_delta", 0.0) or 0.0),
                    "S_tag": S_tag,
                    "utility": str(sol.get("utility", "")).lower(),
                    "disc_tag": str(sol.get("disc_tag", "")).lower(),
                }
                logger.info(
                    "Caching %s (%s) vec=%s → %s | spec_id=%s | disc_tag=%s",
                    util_for_log,
                    S_tag,
                    mask,
                    label,
                    sid,
                    sol.get("disc_tag", None),
                )
                store.put(vec=v_key, spec_id=sid, label=label, payoff=payoff, solution=sol, meta=meta)
                logger.info("Cached OK: %s vec=%s → %s | spec_id=%s", util_for_log, mask, label, sid)
        finally:
            # Always remember a trimmed seed locally
            solved_local[v_key] = _trim_seed(sol, regions, periods) if isinstance(sol, dict) else None



    # GRAND — strict: try exact current spec, then exactly the provided planner anchor spec; otherwise solve
    seed_grand = None
    hitG = store.get(grand, spec_id) if store is not None else None
    if hitG and hitG.get("solution") and is_solution_optimal(hitG["solution"]):
        solG = hitG["solution"]
        solved_local[grand] = _trim_seed(solG, regions, periods)
        if grand in misses:
            misses.remove(grand)
    elif planner_anchor_spec_id:
        hitG2 = store.get(grand, planner_anchor_spec_id)
        if hitG2 and hitG2.get("solution") and is_solution_optimal(hitG2["solution"]):
            logger.info("GRAND: mirroring existing Planner from anchor spec_id into current spec_id.")
            solG = hitG2["solution"]
            payoffG = payoff_row_discounted(solG, regions, periods)
            store.put(vec=grand, spec_id=spec_id, label="GRAND",
                      payoff=payoffG, solution=solG,
                      meta=dict(hitG2.get("meta") or {}, S_tag="grand_mirrored"))
            solved_local[grand] = _trim_seed(solG, regions, periods)
            if grand in misses:
                misses.remove(grand)
        else:
            logger.info("Solving GRAND planner for %s (anchor; no strict planner anchor hit).", utility.upper())
            solG = solve_planner(
                params=params, T=T, tstep=tstep,
                utility=utility, solver_opts=solver_opts,
                diagnostics_dir=diagnostics_dir / "grand",
                exogenous_S=exogenous_S,
                population_weight_envy_guilt=(population_weight_envy_guilt if utility == "fs" else False),
                negishi_weights=negishi_weights,
                negishi_use=negishi_use,
            )
            _put_and_record(grand, "GRAND", solG, S_tag="grand", converged=True)
            if grand in misses:
                misses.remove(grand)
    seed_grand = solved_local.get(grand)

    # SINGLETONS — mirror existing Nash singleton strictly into current spec_id; solve only if nothing exists
    seed_singleton: Dict[str, Optional[dict]] = {r: None for r in regions}
    for i, r in enumerate(regions):
        vec1 = tuple(1 if j == i else 0 for j in range(N))

        # (1) Exact spec_id hit?
        hit_exact = store.get(vec1, spec_id) if store is not None else None
        if hit_exact and hit_exact.get("solution") and hit_exact["solution"].get("optimal", True):
            sol = hit_exact["solution"]
            solved_local[vec1] = _trim_seed(sol, regions, periods)
            seed_singleton[r] = solved_local[vec1]
            if vec1 in misses:
                misses.remove(vec1)
            continue

        # (2) Mirror strictly from the provided Nash anchor spec_id (no scanning)
        if nash_anchor_spec_id:
            hitN = store.get(vec1, nash_anchor_spec_id)
            if hitN and hitN.get("solution") and hitN["solution"].get("optimal", True):
                logger.info("Singleton %s: mirroring existing Nash from anchor spec_id into current spec_id.", r)
                sol = hitN["solution"]
                payoff = payoff_row_discounted(sol, regions, periods)
                store.put(
                    vec=vec1, spec_id=spec_id, label=r,
                    payoff=payoff, solution=sol,
                    meta=dict(hitN.get("meta") or {}, S_tag="nash_singleton_mirrored"),
                )
                solved_local[vec1] = _trim_seed(sol, regions, periods)
                seed_singleton[r] = solved_local[vec1]
                if vec1 in misses:
                    misses.remove(vec1)
                continue

        # (3) Nothing cached or mirrorable → solve once
        if vec1 in misses:
            logger.info("Solving SINGLETON Nash for %s (anchor, %s).", r, utility.upper())
            sol1 = solve_nash(
                params=params, T=T, tstep=tstep,
                utility=utility, solver_opts=solver_opts,
                diagnostics_dir=diagnostics_dir / f"nash_singletons_{utility.lower()}",
                initial_solution=None,
                exogenous_S=exogenous_S,
                population_weight_envy_guilt=(population_weight_envy_guilt if utility == "fs" else False),
                max_iter=max_iter_nash, tol=tol_mu_nash, relax=relax,
                ignore_last_k_periods=ignore_last_k_periods,
                region_order=[r] + [rr for rr in regions if rr != r],
            )
            _put_and_record(vec1, r, sol1, S_tag="nash_singleton")
            misses.remove(vec1)
            seed_singleton[r] = solved_local[vec1]

    if not misses:
        return

    # 2) Order remaining misses by size + Gray
    ordered = _order_by_size_gray(list(misses))

    # 3) Solve in waves; only the parent writes to the store
    workers = int(max(1, workers))

    # --- Prepare a lightweight FS seed proxy for workers (exact-or-neighbor FS) ---
    # Only when we are pairing CRRA→FS inside workers AND FS discounting is enabled.
    fs_seed_proxy: dict | None = None
    if (str(utility).lower() == "crra") and bool(pair_fs_in_worker) and bool(fs_disc_enabled):
        fs_seed_proxy = {}
        # Build for all tasks we are about to submit (misses only)
        def _best_fs_seed_for(vk):
            # (1) Exact latest FS
            hit = _try_get_latest_fs(store, vk)
            if hit and (hit.get("solution") or {}).get("utility") == "fs" and (hit["solution"].get("optimal", True)):
                return _trim_seed(hit["solution"], regions, periods)
            # (2) Any Hamming-1 FS neighbor
            for nb in neighbors_of(vk):
                hn = _try_get_latest_fs(store, nb)
                if hn and (hn.get("solution") or {}).get("utility") == "fs" and (hn["solution"].get("optimal", True)):
                    return _trim_seed(hn["solution"], regions, periods)
            # (3) Fallback: 'data' seed
            return make_data_seed(regions, periods)

        for vk in ordered:
            try:
                fs_seed_proxy[tuple(vk)] = _best_fs_seed_for(tuple(vk))
            except Exception:
                fs_seed_proxy[tuple(vk)] = make_data_seed(regions, periods)


    if workers == 1:
        # sequential (still uses seeds + ordering)
        for v_key in ordered:
            # log which coalition we’re about to solve
            mask = "".join(str(int(b)) for b in v_key)
            label = coalition_vec_to_member_string(list(v_key), regions)
            s_bits = sum(v_key)
            if s_bits == 1:
                who = regions[list(v_key).index(1)]
                logger.info("Solving SINGLETON Nash for %s (%s; vec=%s).", who, utility.upper(), mask)
            else:
                logger.info("Solving coalition game for %s (vec=%s → %s).", utility.upper(), mask, label)

            seed = _full_seed_for(v_key, regions, periods, params, tstep, seed_grand, seed_singleton, solved_local)
            args = (v_key, seed, regions, periods, params, tstep, utility, solver_opts,
                    diagnostics_dir, exogenous_S, population_weight_envy_guilt,
                    max_iter_nash, tol_mu_nash, relax, ignore_last_k_periods,
                    negishi_use, negishi_weights,
                    # FS pairing knobs
                    bool(pair_fs_in_worker), bool(fs_disc_enabled), fs_disc_mode, fs_disc_file,
                    exogenous_S_fs, bool(negishi_use_fs), negishi_weights_fs, bool(population_weight_envy_guilt_fs),
                    fs_seed_proxy, bau_sol, fs_negishi_source)
            res = _solve_task_module_level(args)
            # Unpack including optional FS payload
            if isinstance(res, tuple) and len(res) >= 3:
                v_key, sol, converged = res[0], res[1], res[2]
                fs_payload = (res[3] if len(res) > 3 else []) or []
            else:
                v_key, sol = res
                converged = bool((sol or {}).get("converged", True))
                fs_payload = []
            _put_and_record(v_key, label, sol, converged=converged)
            # Cache any FS results returned by the worker (parent remains single writer)
            for item in fs_payload:
                v_fs = item["v_key"]
                label_fs = coalition_vec_to_member_string(list(v_fs), regions)
                _put_and_record(v_fs, label_fs, item["solution"], converged=bool(item.get("converged", True)))

    else:
        # Stream tasks to a single long-lived pool: keep at most `prefill` in flight,
        # and submit a new coalition as soon as one finishes. This avoids idle workers
        # at wave boundaries and lets later tasks use fresher seeds.
        queue = deque(ordered)
        prefill = workers  # keep at most `workers` tasks queued to maximize seed freshness

        def submit_one(v):
            # log which coalition goes out to the pool
            mask = "".join(str(int(b)) for b in v)
            label = coalition_vec_to_member_string(list(v), regions)
            s_bits = sum(v)
            if s_bits == 1:
                who = regions[list(v).index(1)]
                logger.info("Queueing SINGLETON Nash for %s (%s; vec=%s).", who, utility.upper(), mask)
            else:
                logger.info("Queueing coalition game for %s (vec=%s → %s).", utility.upper(), mask, label)

            seed = _full_seed_for(v, regions, periods, params, tstep, seed_grand, seed_singleton, solved_local)
            args = (v, seed, regions, periods, params, tstep, utility, solver_opts,
                    diagnostics_dir, exogenous_S, population_weight_envy_guilt,
                    max_iter_nash, tol_mu_nash, relax, ignore_last_k_periods,
                    negishi_use, negishi_weights,
                    # FS pairing knobs
                    bool(pair_fs_in_worker), bool(fs_disc_enabled), fs_disc_mode, fs_disc_file,
                    exogenous_S_fs, bool(negishi_use_fs), negishi_weights_fs, bool(population_weight_envy_guilt_fs),
                    fs_seed_proxy, bau_sol, fs_negishi_source)
            fut = ex.submit(_solve_task_module_level, args)
            in_flight[fut] = v

        # Use spawn on Linux to avoid inheriting pre-fork locks/FDs from BLAS/SQLite.
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            # Optional: ensure worker logs always get configured
            initializer=_setup_worker_logging,
            initargs=(logging.INFO,)
        ) as ex:
            in_flight = {}

            # prefill
            for _ in range(min(prefill, len(queue))):
                submit_one(queue.popleft())

            # true streaming: refill immediately when ANY future completes
            while in_flight:
                done, _ = wait(set(in_flight.keys()), return_when=FIRST_COMPLETED)
                for fut in done:
                    v_key = in_flight.pop(fut)
                    res = fut.result()
                    if isinstance(res, tuple) and len(res) >= 3:
                        v_key, sol, converged = res[0], res[1], res[2]
                        fs_payload = (res[3] if len(res) > 3 else []) or []
                    else:
                        _, sol = res
                        converged = bool((sol or {}).get("converged", True))
                        fs_payload = []
                    # log completion
                    mask_done = "".join(str(int(b)) for b in v_key)
                    label_done = coalition_vec_to_member_string(list(v_key), regions)
                    logger.info("Finished coalition (%s; vec=%s → %s).", utility.upper(), mask_done, label_done)
                    _put_and_record(v_key, label_done, sol, converged=converged)
                    for item in fs_payload:
                        v_fs = item["v_key"]
                        label_fs = coalition_vec_to_member_string(list(v_fs), regions)
                        _put_and_record(v_fs, label_fs, item["solution"], converged=bool(item.get("converged", True)))

                    if queue:
                        submit_one(queue.popleft())

# ---------------------------------------
# Diagnostics: plot convergence metric by iteration
# ---------------------------------------
def plot_nonconv_diag(diag_data: Iterable[dict], diag_path: Path, title: str) -> None:
    """
    Given a sequence of {'iteration': int, 'max_delta': float}, write a PNG to diag_path.
    The 'max_delta' field is interpreted as the chosen convergence metric
    (e.g., μ-based max residual to best response).
    """

    if not DIAGNOSTICS_ON:
        return    

    iterations = [e['iteration'] for e in diag_data]
    deltas = [e['max_delta'] for e in diag_data]

    plt.figure()
    plt.plot(iterations, deltas, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Max residual")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(diag_path)
    plt.close()
    logger.info("Saved diagnostic plot to %s", diag_path)
