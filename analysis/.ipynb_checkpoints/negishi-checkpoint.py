"""
Negishi weights from a BAU solution (CRRA or FS).

- CRRA marginal utility (population-weighted objective):
    MU_{r,t} ∝ (C_{r,t} / L_{r,t})^{-eta_r}
- FS marginal utility proxy (population-weighted objective; per-capita baseline):
    MU_{r,t} = 1 + alpha_r * share_richer_{r,t} - beta_r * share_poorer_{r,t}

Where 'share_richer'/'share_poorer' are computed across *other* regions using either:
    • Unweighted fractions, or
    • Population-weighted shares using others' populations.

Weights are computed period-by-period as:
    w_{r,t} ∝ 1 / MU_{r,t},  normalized so that  ∑_r w_{r,t} = 1 .

Strict pipeline alignment
--------------------------------
- Periods are integers 1..T (no period 0).
- CSV I/O coerces columns to 1..T and the exact region order requested.
- Guarded against zero/invalid populations and empty comparison sets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple, Iterable, Optional
import logging
import math

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ----------------------------- CSV I/O -----------------------------

def load_negishi_weights_from_csv(
    path: Path | str,
    *,
    regions: Iterable[str],
    T: int,
) -> pd.DataFrame:
    """
    Load Negishi weights from CSV and return a DataFrame with:
      - index: exactly the given 'regions', in the same order,
      - columns: 1..T (int),
      - each column sums to 1 (renormalized defensively).

    Accepted input shapes:
      - Regions as rows or columns (auto-detect; we transpose if needed).
      - Period labels may be ints or strings (e.g., "1", "Y2015"); coerced to ints.
      - An optional period 0 column is ignored.

    Validation:
      - All 'regions' must be present (extras are dropped).
      - All periods 1..T must be present after coercion.
      - After per-period renormalization, entries must be finite.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Negishi weights file not found: {path}")

    # Load with automatic delimiter detection
    df = pd.read_csv(path, sep=None, engine="python", index_col=0)

    # Detect whether regions live on index or columns; transpose if needed
    idx_labels = set(map(str, df.index))
    col_labels = set(map(str, df.columns))
    reg_list = list(regions)
    reg_set = set(map(str, reg_list))
    if reg_set.issubset(col_labels) and not reg_set.issubset(idx_labels):
        df = df.T

    # Coerce column labels to ints (allow leading 'Y'/'y')
    def _to_int(label: object) -> int:
        s = str(label).strip()
        if s and s[0] in "Yy":
            s = s[1:]
        return int(s)

    try:
        df.columns = [_to_int(c) for c in df.columns]
    except Exception as e:
        raise ValueError("Period columns must be coercible to ints (e.g., '1' or 'Y2015').") from e

    # Keep only periods 1..T (drop 0 or anything outside) and enforce int dtype
    valid_cols = [t for t in df.columns if isinstance(t, (int, np.integer)) and 1 <= int(t) <= int(T)]
    df = df.loc[:, sorted(set(valid_cols))]
    must_have = list(range(1, int(T) + 1))
    missing_cols = [t for t in must_have if t not in df.columns]
    if missing_cols:
        raise ValueError(f"Negishi file must include periods 1..T; missing: {missing_cols}")

    # Ensure all regions present, in the exact given order
    missing_regions = reg_set - set(map(str, df.index))
    if missing_regions:
        raise ValueError(f"Negishi file missing regions: {sorted(missing_regions)}")
    df = df.loc[reg_list].copy()

    # Convert to float and normalize each period to sum=1
    df = df.astype(float)
    # First pass: handle zero columns by turning 0 to NaN to avoid division by zero
    col_sums = df.sum(axis=0).replace(0.0, np.nan)
    df = df.divide(col_sums, axis=1)

    # Finite-safety: if any column produced NaNs/Infs (e.g., all zeros), set uniform
    for t in df.columns:
        col = df[t]
        if not np.isfinite(col.to_numpy()).all():
            df[t] = 1.0 / len(df)

    # Final re-normalization to eliminate tiny drift
    df = df.divide(df.sum(axis=0), axis=1)

    return df


# ----------------------------- Marginal utility proxies -----------------------------

def _crra_mu(C: float, L: float, eta: float) -> float:
    """
    Negishi marginal-utility proxy consistent with the planner's CRRA objective.

    Planner period utility (CRRA):
        U_{r,t} = L_{r,t} * u(c_pc_th), with c_pc_th = (C_{r,t}/L_{r,t}) * 1000,
        u'(c) = c^(-eta). Constants cancel in per-period normalization.

    Thus dU/dC ∝ (C/L)^(-eta).

    Guard behavior:
    - If L <= 0 or C/L <= 0, return +inf → downstream inversion yields zero weight.
    """
    if L <= 0.0:
        return float("inf")
    c_pc = C / L
    if c_pc <= 0.0:
        return float("inf")
    return c_pc ** (-eta)


def fs_negishi_mu(
    r: str,
    t: int,
    params: Any,
    regions: Iterable[str],
    C: Dict[Tuple[str, int], float],
    L: pd.DataFrame,
    population_weight_envy_guilt: bool,
) -> float:
    """
    Fehr–Schmidt (FS) Negishi marginal-utility proxy at (r,t), matched to the
    model's FS objective (population-weighted variant).

        MU_fs ∝ 1 + α_r * share_richer(r,t) - β_r * share_poorer(r,t)

    'share_*' replicate the weighting used in the model (population-weighted if
    population_weight_envy_guilt=True, else simple fractions). Ties contribute zero.

    Guard behavior:
    - If there are no valid "others" (e.g., single-region runs, or all other L<=0),
      both shares are defined as 0.0 (neutral).
    """
    C_rt = float(C[(r, t)])
    L_rt = float(L.at[r, t])
    if L_rt <= 0.0:
        return float("inf")  # protects downstream inversion

    c_r_pc = C_rt / L_rt

    others: list[tuple[float, float]] = []
    for s in regions:
        if s == r:
            continue
        L_st = float(L.at[s, t])
        if L_st <= 0.0:
            continue
        c_s_pc = C[(s, t)] / L_st
        others.append((c_s_pc, L_st))

    if not others:
        share_richer = 0.0
        share_poorer = 0.0
    elif population_weight_envy_guilt:
        total_pop_others = sum(Ls for (_, Ls) in others)
        if total_pop_others <= 0.0:
            share_richer = 0.0
            share_poorer = 0.0
        else:
            share_richer = sum(Ls for (c_s, Ls) in others if c_s > c_r_pc) / total_pop_others
            share_poorer = sum(Ls for (c_s, Ls) in others if c_r_pc > c_s) / total_pop_others
    else:
        N = len(others)
        share_richer = sum(1 for (c_s, _) in others if c_s > c_r_pc) / N
        share_poorer = sum(1 for (c_s, _) in others if c_r_pc > c_s) / N

    alpha = float(params.FS_alpha[r])
    beta  = float(params.FS_beta[r])

    return 1.0 + alpha * share_richer - beta * share_poorer


# ----------------------------- Inversion & normalization -----------------------------

def _invert_and_normalize(mu_by_region: Dict[str, float]) -> Dict[str, float]:
    """
    Compute Negishi weights ∝ 1/mu, normalized so weights sum to 1.
    Uses a small floor to avoid division blowups if MU≈0 or not finite.
    """
    eps = 1e-15
    inv = {}
    for r, mu in mu_by_region.items():
        val = float(mu)
        if not math.isfinite(val) or val <= eps:
            val = eps
        inv[r] = 1.0 / val
    s = sum(inv.values())
    if s <= 0:
        # Fallback: uniform if something went truly off
        n = max(len(inv), 1)
        return {r: 1.0 / n for r in inv}
    return {r: inv[r] / s for r in inv}


# ----------------------------- Public API -----------------------------

def compute_negishi_weights_from_bau(
    params: Any,
    bau_sol: Dict[str, Dict[Tuple[str, int], float]],
    *,
    utility: str,
    population_weight_envy_guilt: bool,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Compute BAU-based Negishi weights for `utility` in {'crra','fs'}.

    Returns a DataFrame with index=region, columns=period (1..T),
    each column summing to 1 across regions.

    Notes
    -----
    - 'population_weight_envy_guilt' is REQUIRED and used iff utility == 'fs'.
      (For 'crra', it’s ignored but must still be provided to avoid silent defaults.)
    - If 'output_path' is not None, the CSV is written there.
    """
    if "C" not in bau_sol:
        raise ValueError("bau_sol missing 'C' entries.")
    regions = list(params.L.index)
    periods = sorted({t for (_, t) in bau_sol["C"]})

    # Compute marginal utilities
    MU: Dict[Tuple[str, int], float] = {}
    for (r, t), C_rt in bau_sol["C"].items():
        L_rt = float(params.L.at[r, t])
        if utility == "crra":
            eta_r = float(params.crra_eta[r])
            MU[(r, t)] = _crra_mu(C_rt, L_rt, eta_r)
        elif utility == "fs":
            MU[(r, t)] = fs_negishi_mu(
                r=r, t=t, params=params, regions=regions,
                C=bau_sol["C"], L=params.L,
                population_weight_envy_guilt=population_weight_envy_guilt,
            )
        else:
            raise ValueError(f"Unknown utility for Negishi weights: {utility!r}")

    # Invert & normalize period-by-period
    records = []
    for t in periods:
        mu_t = {r: MU[(r, t)] for r in regions}
        w_t = _invert_and_normalize(mu_t)
        # Re-normalize to eliminate drift
        s = sum(w_t.values())
        if s <= 0.0:
            # uniform fallback
            n = max(len(regions), 1)
            w_t = {r: 1.0 / n for r in regions}
        else:
            w_t = {r: w_t[r] / s for r in regions}
        for r in regions:
            records.append({"region": r, "period": t, "weight": w_t[r]})

    df = pd.DataFrame(records).pivot(index="region", columns="period", values="weight")
    # Sort columns (periods) to 1..T if they look like that
    try:
        df = df.loc[regions, sorted(df.columns)]
    except Exception:
        df = df.loc[regions]

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path)
        logger.info("Saved Negishi weights (%s) to %s", utility, str(output_path))

    return df


# ----------------------------- Negishi Weights After FS Discounting -----------------------------
def compute_negishi_weights_from_bau_fs_after_disc(
    params: Any,
    bau_sol: Dict[str, Dict[Tuple[str, int], float]],
    fs_disc: Dict[Tuple[str, int], float],
    *,
    population_weight_envy_guilt: bool,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Compute FS-adjusted Negishi weights, where the marginal utilities (MU) are normalized
    by the FS discounting path (discounted marginal utility).
    
    Arguments:
    - bau_sol: The BAU solution (from which we get consumption `C` and labor `L`).
    - fs_disc: The FS discounting path, which adjusts the MU.
    - population_weight_envy_guilt: Whether to apply population-weighted envy/guilt in FS MU.
    - output_path: Path to save the weights if desired.

    Returns:
    - A DataFrame with the Negishi weights for each region and period, normalized so that
      each column sums to 1.
    """
    regions = list(params.L.index)
    periods = sorted({t for (_, t) in bau_sol["C"]})

    # Compute marginal utilities and normalize by FS discounting
    MU: Dict[Tuple[str, int], float] = {}
    for (r, t), C_rt in bau_sol["C"].items():
        L_rt = float(params.L.at[r, t])
        MU[(r, t)] = fs_negishi_mu(
            r=r, t=t, params=params, regions=regions,
            C=bau_sol["C"], L=params.L,
            population_weight_envy_guilt=population_weight_envy_guilt,
        ) * fs_disc.get((r, t))  # Apply FS discounting here

   # Invert and normalize the marginal utilities to compute Negishi weights
    records = []
    for t in periods:
        mu_t = {r: MU[(r, t)] for r in regions}
        w_t = _invert_and_normalize(mu_t)
        # Re-normalize to eliminate drift
        s = sum(w_t.values())
        if s <= 0.0:
            # uniform fallback
            n = max(len(regions), 1)
            w_t = {r: 1.0 / n for r in regions}
        else:
            w_t = {r: w_t[r] / s for r in regions}
        for r in regions:
            records.append({"region": r, "period": t, "weight": w_t[r]})

    df = pd.DataFrame(records).pivot(index="region", columns="period", values="weight")
    # Sort columns (periods) to 1..T if they look like that
    try:
        df = df.loc[regions, sorted(df.columns)]
    except Exception:
        df = df.loc[regions]

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path)
        logger.info("Saved FS-adjusted Negishi weights to %s", output_path)

    return df