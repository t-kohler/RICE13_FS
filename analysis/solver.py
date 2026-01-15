# RICE13_FS/analysis/solver.py
"""
High-level orchestration for RICE13_FS runs.

Stages
------
  1) BAU baseline (if needed; S fixed to BAU; μ follows a fixed step policy)
  2) Optional Negishi weights (from BAU, BAU+FS-discount, or file) – saved to output_dir
  3) Cooperative planners (CRRA / FS) – GRAND-only (cache-first)
  4) Noncooperative Nash (CRRA / FS) – caches SINGLETON rows (one per region)
  5) Coalition solving – ensure/cache requested coalitions for export

Design choices
--------------
- The CoalitionStore (SQLite) is the single source of truth for coalition exports:
  exports read payoffs/solutions from the cache rather than recomputing.
- Cached entries store *discounted* lifetime payoff rows (computed at write time).
  This requires each stored solution to include discount factors `disc[(r,t)]`.
- FS envy/guilt toggle: `population_weight_envy_guilt` controls how envy/guilt
  shares are computed across other regions; the FS objective itself remains
  population-weighted inside the model.

Notes on time indexing
----------------------
- The economic/climate core is decadal internally (10-year periods). We keep `tstep`
  for backwards compatibility and for calendar-year mapping in plots / μ policies,
  but it should not be treated as a "subdecadal" switch in the core model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime
import re
import pandas as pd

from RICE13_FS.core.data_loader import load_params
from RICE13_FS.common.utils import (
    _read_disc_csv,
    compute_target_vectors,
    fetch_or_solve_coalitions,
    same_exogenous_S,
    build_config_fingerprint,
    payoff_row_discounted,
    build_solution_spec_id,
    _build_fs_discount_series_one_pass,
    _build_fs_discount_series_two_pass,
    normalize_exogenous_S,
    digest_series,
    digest_regional_series,
    stamp_identity,
    is_solution_optimal,
    DIAGNOSTICS_ON,
)
from RICE13_FS.solve.bau import solve_bau
from RICE13_FS.solve.coop import solve_planner
from RICE13_FS.solve.noncoop import solve_nash
from RICE13_FS.solve.coalition import parse_coalition_spec
from RICE13_FS.output.coalition_store import CoalitionStore

from RICE13_FS.analysis.negishi import (
    compute_negishi_weights_from_bau,
    load_negishi_weights_from_csv,
    compute_negishi_weights_from_bau_fs_after_disc
)

logger = logging.getLogger(__name__)


# ----------------------------- Types -----------------------------

@dataclass
class AnalysisResult:
    coop: Optional[dict]
    noncoop: Optional[dict]
    coalitions: Optional[list]
    bau: Optional[dict]
    meta: dict | None = None  # lightweight context so CLI can export


# ----------------------------- Helpers -----------------------------

def _validate_coalition_against_regions(spec: str, regions: list[str], mega_run: bool) -> None:
    s = str(spec).strip()
    if s.lower() in {"all", "*"}:
        if not mega_run:
            raise SystemExit("coalition='all' is only allowed when mega_run=true.")
        return
    if s.upper() == "GRAND":
        return
    # exact-length bitmask
    if re.fullmatch(r"[01]+", s):
        n = len(regions)
        if len(s) != n:
            raise SystemExit(f"Invalid coalition bitmask '{s}': length {len(s)} != number of regions {n}.")
        if "1" not in s:
            raise SystemExit("Invalid coalition bitmask: must contain at least one '1'.")
        return
    # comma-separated region codes
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if parts:
        unknown = [p for p in parts if p not in regions]
        if unknown:
            raise SystemExit(f"Invalid region name(s) in coalition: {unknown}. Valid regions: {regions}")
        return
    raise SystemExit(
        f"Invalid coalition '{s}'. Use 'all' (with mega_run=true), 'GRAND', a 0/1 bitmask of length {len(regions)}, "
        f"or a comma list like 'US,EU'. Valid regions: {regions}"
    )

def _build_solver_opts(config: dict) -> Dict[str, Any]:
    """Normalize IPOPT options; keep None as None (don't stringify)."""
    exec_path = config.get("ipopt_executable", None) or config.get("solver_path", None)
    if isinstance(exec_path, str) and exec_path.strip().lower() in ("", "none", "null"):
        exec_path = None
    return {
        "executable": exec_path,
        "options": {
            "tol": float(config.get("tol_ipopt")),
            "max_iter": int(config.get("max_iter_ipopt")),
        },
    }


def _S_solution_to_df(S_dict: dict, regions, T: int) -> pd.DataFrame:
    """Convert {(r,t)->S} to a regions×period DataFrame with columns 1..T."""
    df = pd.DataFrame(0.0, index=list(regions), columns=list(range(1, int(T) + 1)))
    for (r, t), v in S_dict.items():
        t_int = int(t)
        if 1 <= t_int <= int(T):
            df.at[str(r), t_int] = float(v)
    return df


def _read_exog_S_csv(path: Path) -> pd.DataFrame:
    """
    Robust CSV loader for exogenous S:
      - auto-detect delimiter via sep=None + engine='python'
      - first column is region index
      - strip whitespace on headers/index
    """
    df = pd.read_csv(path, sep=None, engine="python", index_col=0)
    df.index = df.index.astype(str).str.strip()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _export_S_csv(sol: dict, regions, T: int, outdir: Path, tag: str) -> None:
    """Write optimized S to CSV (one file per run tag)."""
    if "S" not in sol:
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    name = f"S_{tag}_{ts}.csv"
    df = _S_solution_to_df(sol["S"], regions, int(T))
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / name)
    logger.info("Saved %s", outdir / name)


def _export_disc_csv(series_map: Dict, T: int, outdir: Path, tag: str) -> None:
    """
    Write discount series to CSV with a timestamped filename.
    - Global: {t -> d_t}  → columns: period,disc
    - Regional: {(r,t) -> d_{r,t}} → columns: region,period,disc
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M_%S")
    name = f"fs_disc_{tag}_{ts}.csv"
    # Detect shape by key type
    if series_map and isinstance(next(iter(series_map.keys())), tuple):
        rows = []
        for (r, t), v in series_map.items():
            rows.append({"region": str(r), "period": int(t), "disc": float(v)})
        df = pd.DataFrame(rows).sort_values(["region", "period"])
    else:
        df = pd.DataFrame(
            {"period": list(range(1, int(T) + 1)),
             "disc": [float(series_map.get(t, 1.0)) for t in range(1, int(T) + 1)]}
        )
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / name, index=False)
    logger.info("Saved %s", outdir / name)


def _resolve_planner_S(config: dict, params, utility: str, coop_crra: Optional[dict]) -> Optional[pd.DataFrame]:
    """Map planner S-mode flags → exogenous S DataFrame or None."""
    if utility == "crra":
        mode = str(config.get("planner_crra_S_mode")).lower()
        if mode == "optimal": return None
        if mode == "bau":     return params.bau_saving_rates.copy()
        if mode == "file":    return _read_exog_S_csv(Path(config.get("planner_crra_S_file", "")))
        raise ValueError(f"Unknown planner_crra_S_mode={mode!r}")
    if utility == "fs":
        mode = str(config.get("planner_fs_S_mode")).lower()
        if mode == "optimal": return None
        if mode == "bau":     return params.bau_saving_rates.copy()
        if mode == "crra":
            if coop_crra is None or "S" not in coop_crra:
                raise RuntimeError("planner_fs_S_mode='crra' but CRRA planner S is not available.")
            return _S_solution_to_df(coop_crra["S"], params.countries, int(config["T"]))
        if mode == "file":    return _read_exog_S_csv(Path(config.get("planner_fs_S_file", "")))
        raise ValueError(f"Unknown planner_fs_S_mode={mode!r}")
    raise ValueError(f"Unknown planner utility={utility!r}")


def _resolve_nash_S(config: dict, params, utility: str,
                    coop_crra: Optional[dict], coop_fs: Optional[dict]) -> Optional[pd.DataFrame]:
    """Map Nash S-mode flags → exogenous S DataFrame or None."""
    if utility == "crra":
        mode = str(config.get("nash_crra_S_mode")).lower()
        if mode == "optimal": return None
        if mode == "bau":     return params.bau_saving_rates.copy()
        if mode == "file":    return _read_exog_S_csv(Path(config.get("nash_crra_S_file", "")))
        if mode == "planner_crra":
            if coop_crra is None or "S" not in coop_crra:
                raise RuntimeError("nash_crra_S_mode='planner_crra' but CRRA planner S is not available.")
            return _S_solution_to_df(coop_crra["S"], params.countries, int(config["T"]))
        if mode == "planner_fs":
            if coop_fs is None or "S" not in coop_fs:
                raise RuntimeError("nash_crra_S_mode='planner_fs' but FS planner S is not available.")
            return _S_solution_to_df(coop_fs["S"], params.countries, int(config["T"]))
        raise ValueError(f"Unknown nash_crra_S_mode={mode!r}")
    if utility == "fs":
        mode = str(config.get("nash_fs_S_mode")).lower()
        if mode == "optimal": return None
        if mode == "bau":     return params.bau_saving_rates.copy()
        if mode == "file":    return _read_exog_S_csv(Path(config.get("nash_fs_S_file", "")))
        if mode == "planner_crra":
            if coop_crra is None or "S" not in coop_crra:
                raise RuntimeError("nash_fs_S_mode='planner_crra' but CRRA planner S is not available.")
            return _S_solution_to_df(coop_crra["S"], params.countries, int(config["T"]))
        if mode == "planner_fs":
            if coop_fs is None or "S" not in coop_fs:
                raise RuntimeError("nash_fs_S_mode='planner_fs' but FS planner S is not available.")
            return _S_solution_to_df(coop_fs["S"], params.countries, int(config["T"]))
        raise ValueError(f"Unknown nash_fs_S_mode={mode!r}")
    raise ValueError(f"Unknown nash utility={utility!r}")


def _resolve_coalition_S(config: dict, params, utility: str,
                         coop_crra: Optional[dict], coop_fs: Optional[dict]) -> Optional[pd.DataFrame]:
    """Map coalition S-mode flags → exogenous S DataFrame or None (strict, per-utility)."""
    if utility == "crra":
        mode = str(config["coalition_crra_S_mode"]).strip().lower()
        if mode == "optimal": return None
        if mode == "bau":     return params.bau_saving_rates.copy()
        if mode == "file":    return _read_exog_S_csv(Path(config["coalition_crra_S_file"]))
        if mode == "planner_crra":
            if coop_crra is None or "S" not in coop_crra:
                raise RuntimeError("coalition_crra_S_mode='planner_crra' but CRRA planner S is not available.")
            return _S_solution_to_df(coop_crra["S"], params.countries, int(config["T"]))
        if mode == "planner_fs":
            if coop_fs is None or "S" not in coop_fs:
                raise RuntimeError("coalition_crra_S_mode='planner_fs' but FS planner S is not available.")
            return _S_solution_to_df(coop_fs["S"], params.countries, int(config["T"]))
        raise ValueError(f"Unknown coalition_crra_S_mode={mode!r}")
    elif utility == "fs":
        mode = str(config["coalition_fs_S_mode"]).strip().lower()
        if mode == "optimal": return None
        if mode == "bau":     return params.bau_saving_rates.copy()
        if mode == "file":    return _read_exog_S_csv(Path(config["coalition_fs_S_file"]))
        if mode == "planner_crra":
            if coop_crra is None or "S" not in coop_crra:
                raise RuntimeError("coalition_fs_S_mode='planner_crra' but CRRA planner S is not available.")
            return _S_solution_to_df(coop_crra["S"], params.countries, int(config["T"]))
        if mode == "planner_fs":
            if coop_fs is None or "S" not in coop_fs:
                raise RuntimeError("coalition_fs_S_mode='planner_fs' but FS planner S is not available.")
            return _S_solution_to_df(coop_fs["S"], params.countries, int(config["T"]))
        raise ValueError(f"Unknown coalition_fs_S_mode={mode!r}")
    else:
        raise ValueError(f"Unknown coalition utility={utility!r}")


def _nash_seed(config, params, T, bau_sol, coop_sol, key):
    """
    Pick Nash seed:
      - 'planner' (default): use planner solution if available, else BAU.
      - 'bau': BAU.
      - 'data': S from params.savings_init and μ from params.mu_init (Data/*.csv).
    """
    mode = str(config.get(key, "planner")).lower()
    if mode == "bau":
        return bau_sol
    if mode == "data":
        S_df = getattr(params, "savings_init", None)
        mu_df = getattr(params, "mu_init", None)
        if S_df is None or mu_df is None:
            logger.warning(
                "%s='data' requested but %s missing; falling back to planner/BAU.",
                key,
                "savings_init" if S_df is None else "mu_init"
            )
            return coop_sol or bau_sol
        regions = list(params.countries)
        periods = list(range(1, int(T) + 1))
        # align to ordering; will raise if rows/cols are missing → fallback above
        S_df = S_df.reindex(index=regions, columns=periods)
        mu_df = mu_df.reindex(index=regions, columns=periods)
        try:
            S0  = {(r, t): float(S_df.at[r, t])  for r in regions for t in periods}
            mu0 = {(r, t): float(mu_df.at[r, t]) for r in regions for t in periods}
        except Exception as e:
            logger.warning("Failed to use 'data' seed (%s); falling back to planner/BAU.", e)
            return coop_sol or bau_sol
        logger.info("Nash seed 'data'.")
        return {"S": S0, "mu": mu0}
    # default: planner
    return coop_sol or bau_sol


def _put_planner_into_cache(store: CoalitionStore, *, solution: dict, regions: List[str], periods: List[int]) -> None:
    """Write GRAND planner to cache (discounted payoffs) — but only if optimal."""
    if store is None or not solution:
        return

    if not is_solution_optimal(solution):
        logger.warning("Not caching planner result because IPOPT did not report optimal termination.")
        return

    vec = tuple(1 for _ in regions)  # GRAND
    label = "GRAND"
    payoff = payoff_row_discounted(solution, regions, periods)
    meta = {
        "converged": True,
        "max_delta": float(solution.get("max_delta", 0.0) or 0.0),
        "S_tag": "planner",
    }
    spec_id = solution.get("spec_id")
    if not spec_id:
        raise ValueError("Planner solution missing 'spec_id' (ensure solvers tag spec_id before caching).")
    store.put(vec=vec, spec_id=spec_id, label=label, payoff=payoff, solution=solution, meta=meta)


def _put_nash_into_cache(store: CoalitionStore, *, solution: dict, regions: List[str], periods: List[int]) -> None:
    """
    Write SINGLETON rows for a Nash solution — only if optimal final eval.
    All entries get the same payoff row (discounted); consumers must pick their columns.
    """
    if not is_solution_optimal(solution):
        logger.warning("Not caching Nash result because IPOPT did not report optimal termination.")
        return
    if solution.get("converged") is False:
        logger.warning("Not caching Nash result because Gauss–Seidel did not converge (hit max iters?).")
        return

    if store is None or not solution:
        return
    payoff = payoff_row_discounted(solution, regions, periods)
    meta = {
        "converged": bool(solution.get("converged", True)),
        "max_delta": (float(solution.get("max_delta", 0.0)) if solution.get("max_delta") is not None else None),
        "S_tag": "nash",
    }
    N = len(regions)
    spec_id = solution.get("spec_id")
    if not spec_id:
        raise ValueError("Nash solution missing 'spec_id' (ensure solvers tag spec_id before caching).")
    # SINGLETONS
    for i in range(N):
        vec = tuple(1 if j == i else 0 for j in range(N))
        label = regions[i]
        store.put(vec=vec, spec_id=spec_id, label=label, payoff=payoff, solution=solution, meta=meta)


def _wrap_cached(vec: List[int], spec_id: str, store: CoalitionStore) -> Optional[Dict[str, Any]]:
    """Small adapter to materialize a cached coalition for the CLI export."""
    if store is None:
        return None
    hit = store.get(tuple(vec), spec_id)
    if not hit or not hit.get("solution"):
        return None
    meta = hit.get("meta") or {}
    return {
        "vector": list(vec),
        "solution": hit["solution"],
        "converged": bool(meta.get("converged", True)),
        "max_delta": float(meta.get("max_delta", 0.0)) if meta.get("max_delta") is not None else 0.0,
        "logfiles": [],
    }

# ----------------------------- Entrypoint -----------------------------

def run_analysis(config: dict, *, diagnostics_dir: Path) -> AnalysisResult:
    """
    Main orchestration used by cli.py. Returns an AnalysisResult with BAU / coop / noncoop / coalitions.
    """
    solver_opts = _build_solver_opts(config)

    # Required basics
    try:
        T = int(config["T"])
        # Legacy compatibility:
        # the model is decadal internally; tstep is used only for year mapping
        # (plots, μ step policy, etc.). Default keeps old behavior (10 years).
        tstep = int(config.get("tstep", 10))
        data_path = Path(config["data_path"])
    except KeyError as e:
        raise KeyError(f"Missing required config key: {e}. Ensure 'T' and 'data_path' are set.")

    # Load parameters
    params = load_params(data_path, T)
    regions = list(params.countries)
    periods = list(range(1, T + 1))

    # ---- Banner (first-line run summary) ------------------------------------
    logger.info(
        "RUN | T=%s, tstep=%s (yrs/period; legacy) | planners(crra=%s, fs=%s) | nash(crra=%s, fs=%s) | "
        "coalitions(crra=%s, fs=%s) | fs_disc(enabled=%s, mode=%s) | negishi(use=%s, source=%s)",
        T, tstep,
        bool(config.get('run_planner_crra')),
        bool(config.get('run_planner_fs')),
        bool(config.get('run_nash_crra')),
        bool(config.get('run_nash_fs')),
        bool(config.get('run_coalition_crra')),
        bool(config.get('run_coalition_fs')),
        bool(config.get('fs_disc_enabled')),
        str(config.get('fs_disc_mode')).lower(),
        bool(config.get('negishi_use')),
        str(config.get('negishi_source', 'off')),
    )

    logger.info("Regions: loaded %d names from Region_names.csv", len(regions))
    # Fail fast now that we know the region set
    _validate_coalition_against_regions(config.get("coalition", ""), regions, bool(config.get("mega_run", False)))

    # Output/side directories
    results_dir = Path(config.get("results_dir", "./Results")).resolve()
    output_dir = Path(config.get("output_dir", results_dir)).resolve()
    failures_path = (results_dir / "failures.txt")
    diagnostics_dir = Path(diagnostics_dir).resolve()

    # ---- Cache (MANDATORY; new & legacy store signatures supported)
    cache_dir = Path(config.get("cache_dir", "./RICE13_FS/Cache")).resolve()
    namespace = str(config.get("cache_namespace", "")).strip() or "default"
    allow_mismatch = bool(config.get("cache_allow_mismatch", False))
    fp = build_config_fingerprint(config, regions)
    store: CoalitionStore
    try:
        # New signature (no cache_scope)
        store = CoalitionStore(
            path=cache_dir,
            namespace=namespace,
            fingerprint=fp,
            allow_mismatch=allow_mismatch,
        )
    except TypeError:
        # Legacy signature: includes cache_scope
        cache_scope = str(config.get("cache_scope", "all")).strip().lower()
        store = CoalitionStore(
            path=cache_dir,
            namespace=namespace,
            fingerprint=fp,
            cache_scope=cache_scope,
            allow_mismatch=allow_mismatch,
        )

    # Flags
    run_bau = bool(config.get("run_bau", True))
    do_negishi = bool(config.get("negishi_use", False))
    # Planner toggles
    run_crra = bool(config.get("run_planner_crra", True))
    run_fs   = bool(config.get("run_planner_fs", False))
    # Nash toggles
    run_nash_crra = bool(config.get("run_nash_crra", False))
    run_nash_fs   = bool(config.get("run_nash_fs", False))
    # Coalition toggles (new config)
    run_coal_crra = bool(config.get("run_coalition_crra", False))
    run_coal_fs   = bool(config.get("run_coalition_fs", False))

    # Negishi needs: compute BAU-based weights whenever any regime of that
    # utility (planner, Nash, or coalitions) actually uses them, not only
    # when the planner is enabled.
    need_crra_negishi = do_negishi and (run_crra or run_nash_crra or run_coal_crra)
    need_fs_negishi   = do_negishi and (run_fs or run_nash_fs or run_coal_fs)

    coalition_spec = config.get("coalition", "none")
    # Predeclare coalitions so that we can safely build AnalysisResult even
    # when no coalition game is requested in the config. When any coalition
    # flavor runs, this will be replaced by a list of materialized entries.
    coalitions: List[Dict[str, Any]] | None = None
    ignore_last_k_periods  = int(config.get("ignore_last_k_periods", 0))
    # Pairing toggle: when true, each CRRA coalition worker will run the matching FS coalition
    # with the requested FS discount regime (off/file/one_pass/two_pass), using the CRRA anchor
    # it just solved.
    pair_in_worker = bool(config.get("run_coalition_fs", False))  # keep simple: pair whenever FS coalitions requested

    pop_w_env = bool(config.get("population_weight_envy_guilt", False))
    # FS discounting knobs (strict; keys validated in CLI)
    fs_disc_enabled = bool(config.get("fs_disc_enabled", False))
    fs_disc_mode = str(config.get("fs_disc_mode", "off")).strip().lower()
    fs_disc_file = config.get("fs_disc_file", None)

    # Read the Negishi source from the config
    negishi_source = str(config.get("negishi_source", "bau")).lower()

    # Anchor spec IDs captured from planner/Nash stages (used strictly later)
    planner_anchor_spec_id_crra: Optional[str] = None
    planner_anchor_spec_id_fs: Optional[str] = None
    nash_anchor_spec_id_crra: Optional[str] = None
    nash_anchor_spec_id_fs: Optional[str] = None

    # ---- BAU
    bau_sol = None
    neg_src = str(config.get("negishi_source", "bau")).lower()
    # BAU is needed if explicitly requested, or if any Negishi mode needs it
    need_bau = run_bau or (do_negishi and neg_src in {"bau", "fs_after_disc"})
    if need_bau:
        logger.info("Running BAU (CRRA objective; μ step policy; S fixed to BAU).")
        bau_sol = solve_bau(params, T, tstep, solver_opts, diagnostics_dir, utility="crra")
        if bau_sol is not None:
            bau_sol["mode"] = "bau"

    # ---- Negishi weights (auto-save to output_dir; BAU/file based only)
    neg_crra_df: Optional[pd.DataFrame] = None
    neg_fs_df:   Optional[pd.DataFrame] = None
    if do_negishi:
        T_int = int(T)
        if negishi_source == "bau":
            if bau_sol is None:
                raise RuntimeError("Negishi requested from BAU but BAU was not run.")
            # CRRA Negishi (BAU-based)
            if need_crra_negishi:
                out = output_dir / "negishi_weights_crra.csv"
                logger.info("Computing Negishi weights from BAU (CRRA).")
                neg_crra_df = compute_negishi_weights_from_bau(
                    params, bau_sol, utility="crra",
                    population_weight_envy_guilt=pop_w_env, output_path=out,
                )
            # FS Negishi (BAU-based) — old mode: no FS discounting in weights
            if need_fs_negishi:
                out_fs = output_dir / "negishi_weights_fs.csv"
                logger.info("Computing Negishi weights from BAU (FS).")
                neg_fs_df = compute_negishi_weights_from_bau(
                    params, bau_sol, utility="fs",
                    population_weight_envy_guilt=pop_w_env, output_path=out_fs,
                )

        elif negishi_source == "fs_after_disc":
            # Hybrid mode: CRRA Negishi weights come from BAU as usual;
            # FS Negishi weights will be computed later, once the FS discount
            # series for the chosen regime (one_pass/two_pass/file) is known.
            if bau_sol is None:
                raise RuntimeError("Negishi requested from BAU+FS-discount but BAU was not run.")
            if need_crra_negishi:
                out = output_dir / "negishi_weights_crra.csv"
                logger.info("Computing Negishi weights from BAU (CRRA) for fs_after_disc mode.")
                neg_crra_df = compute_negishi_weights_from_bau(
                    params, bau_sol, utility="crra",
                    population_weight_envy_guilt=pop_w_env, output_path=out,
                )
            # NOTE: neg_fs_df is intentionally left as None here.
            # It will be built from BAU + FS discount series in the FS planner
            # block once `discount_series_pl` is available.

        elif negishi_source == "file":
            p_crra = Path(config.get("negishi_file_crra_path", ""))
            p_fs   = Path(config.get("negishi_file_fs_path", ""))
            if p_crra and p_crra.exists():
                logger.info("Loading Negishi CRRA weights from %s", p_crra)
                neg_crra_df = load_negishi_weights_from_csv(p_crra, regions=regions, T=T_int)
            if p_fs and p_fs.exists():
                logger.info("Loading Negishi FS weights from %s", p_fs)
                neg_fs_df = load_negishi_weights_from_csv(p_fs, regions=regions, T=T_int)
        else:
            # For clarity: if negishi_use=true but source is 'off' or unknown
            logger.info("Negishi requested but source=%r provides no weights; running without Negishi.", negishi_source)
            neg_crra_df = None
            neg_fs_df = None
 
    # ---- Cooperative planners (CACHE-FIRST GRAND)
    coop_crra = None
    if run_crra:
        exog_S = _resolve_planner_S(config, params, utility="crra", coop_crra=None)
        sid_pl_crra = build_solution_spec_id(
            utility="crra", T=T, countries=params.countries,
            population_weight_envy_guilt=False,
            exogenous_S=normalize_exogenous_S(exog_S, params.countries, T) if exog_S is not None else None,
            negishi_use=do_negishi, negishi_weights=neg_crra_df if do_negishi else None,
        )
        grand_vec = tuple(1 for _ in params.countries)
        hit = store.get(grand_vec, sid_pl_crra)
        if hit is not None and hit.get("solution") is not None and is_solution_optimal(hit["solution"]):
            if exog_S is None or same_exogenous_S(exog_S, hit["solution"].get("S_exogenous")):
                coop_crra = hit["solution"]
                logger.info("Cache hit for planner (CRRA) GRAND — skipping solve.")
        if coop_crra is None:
            logger.info("Running CRRA planner (%s S).", "exogenous" if exog_S is not None else "optimal")
            w_crra = neg_crra_df if do_negishi else None
            if do_negishi and w_crra is None:
                raise RuntimeError("negishi_use=true but CRRA Negishi weights are missing.")
            coop_crra = solve_planner(
                params, T, tstep, utility="crra",
                solver_opts=solver_opts, diagnostics_dir=diagnostics_dir,
                exogenous_S=exog_S, population_weight_envy_guilt=False,
                negishi_weights=w_crra, negishi_use=do_negishi,
            )
            if coop_crra is not None:
                coop_crra = stamp_identity(coop_crra, utility="crra", spec_id=sid_pl_crra)
                coop_crra["mode"] = f"crra_{'exoS' if exog_S is not None else 'optimal'}"
                if neg_crra_df is not None: coop_crra["negishi_weights"] = neg_crra_df
                if exog_S is None: _export_S_csv(coop_crra, params.countries, T, output_dir, "planner_crra")
                
                _put_planner_into_cache(store, solution=coop_crra, regions=regions, periods=periods)
                logger.info("Cached planner (CRRA) GRAND in store.")

    # Record CRRA planner anchor spec-id if available
    if coop_crra is not None and "spec_id" in coop_crra:
        planner_anchor_spec_id_crra = str(coop_crra["spec_id"])

    coop_fs = None
    if run_fs:
        exog_S_fs = _resolve_planner_S(config, params, utility="fs", coop_crra=coop_crra)
        # ----- Resolve discount series / disc_tag for FS planner -----
        discount_series_pl: Optional[Dict[int, float]] = None
        disc_tag_pl = None
        disc_source_pl = "data"
        # Planner two-pass guard toggle
        skip_final_planner_fs = False
        # Guard: ensure CRRA anchor if needed
        need_crra_anchor = fs_disc_enabled and fs_disc_mode in {"one_pass", "two_pass"}
        if need_crra_anchor and (coop_crra is None or "C" not in coop_crra):
            logger.info("FS discount alignment (planner) requires CRRA planner anchor — running it now.")
            exog_S_crra_pl = _resolve_planner_S(config, params, utility="crra", coop_crra=None)
            w_crra = neg_crra_df if do_negishi else None
            # Defensive guard: if Negishi is requested but CRRA weights are missing,
            # fail fast instead of calling the solver with inconsistent flags.
            if do_negishi and w_crra is None:
                raise RuntimeError(
                    "negishi_use=true but CRRA Negishi weights are missing "
                    "(planner anchor for FS discount alignment)."
                )
            # Reconstruct the CRRA planner spec-id here (we may not have built it above if run_planner_crra=false)
            sid_pl_crra = build_solution_spec_id(
                utility="crra", T=T, countries=params.countries,
                population_weight_envy_guilt=False,
                exogenous_S=normalize_exogenous_S(exog_S_crra_pl, params.countries, T) if exog_S_crra_pl is not None else None,
                negishi_use=do_negishi, negishi_weights=neg_crra_df if do_negishi else None,
            )
            coop_crra = solve_planner(
                params, T, tstep, utility="crra",
                solver_opts=solver_opts, diagnostics_dir=diagnostics_dir,
                exogenous_S=exog_S_crra_pl, population_weight_envy_guilt=False,
                negishi_weights=w_crra, negishi_use=do_negishi,
            )
            if coop_crra is not None:
                coop_crra = stamp_identity(coop_crra, utility="crra", spec_id=sid_pl_crra)
                _put_planner_into_cache(store, solution=coop_crra, regions=regions, periods=periods)
        if fs_disc_enabled:
            if fs_disc_mode == "off":
                # Discounting feature enabled but planner is in "off" mode:
                # use baseline data / model-default SDF (no FS-specific alignment).
                disc_source_pl = "data"
                disc_tag_pl = "disc:data"
                logger.info(
                    "FS planner discounting disabled (fs_disc_mode='off'); "
                    "using baseline data discounts."
                )
            elif fs_disc_mode == "file":
                if not fs_disc_file:
                    raise KeyError("fs_disc_mode='file' requires fs_disc_file in config.")
                d = _read_disc_csv(Path(fs_disc_file), regions, int(T))
                discount_series_pl = d
                disc_source_pl = "file"
                logger.info("Loaded FILE discount series: REGIONAL (%s)", fs_disc_file)
                disc_tag_pl = f"disc:file:regional:{digest_regional_series(d, regions, T)}"
            elif fs_disc_mode == "one_pass":
                d = _build_fs_discount_series_one_pass(
                    anchor_crra_sol=coop_crra, params=params, regions=regions, periods=periods,
                    population_weight_envy_guilt=pop_w_env,
                )
                discount_series_pl = d
                disc_source_pl = "one_pass"
                _digest_pl = digest_regional_series(d, regions, T)
                logger.info(
                    "FS planner one-pass: constructed regional discounts (digest=%s).",
                    _digest_pl,
                )
                disc_tag_pl = f"disc:one_pass:planner:regional:{_digest_pl}"
            elif fs_disc_mode == "two_pass":
                # Stage 1: FS one-pass (CRRA g + CRRA ranks), cache-first
                d1 = _build_fs_discount_series_one_pass(
                    anchor_crra_sol=coop_crra, params=params, regions=regions, periods=periods,
                    population_weight_envy_guilt=pop_w_env,
                )
                _digest_d1 = digest_regional_series(d1, regions, T)
                logger.info(
                    "FS planner two-pass (stage 1): constructed one-pass regional discounts (digest=%s).",
                    _digest_d1,
                )
                tag1 = f"disc:one_pass:planner:regional:{_digest_d1}"
                # FS(1) baseline is a technical construct to shape the discount path.
                # In fs_after_disc mode we deliberately run FS(1) *without* Negishi,
                # and only apply Negishi in the final FS run after the discount series
                # is fixed. For BAU/file sources we retain the old behavior.
                use_negishi_fs1 = do_negishi and (negishi_source in {"bau", "file"})
                neg_w_fs1 = neg_fs_df if use_negishi_fs1 else None
                sid_pl_fs1 = build_solution_spec_id(
                    utility="fs", T=T, countries=params.countries,
                    population_weight_envy_guilt=pop_w_env,
                    exogenous_S=normalize_exogenous_S(exog_S_fs, params.countries, T) if exog_S_fs is not None else None,
                    negishi_use=use_negishi_fs1, negishi_weights=neg_w_fs1,
                    disc_tag=tag1,
                )
                grand_vec = tuple(1 for _ in params.countries)
                hit1 = store.get(grand_vec, sid_pl_fs1)
                if hit1 is not None and hit1.get("solution"):
                    fs1 = hit1["solution"]
                    logger.info("Cache hit for FS planner one-pass baseline — reusing for two-pass.")
                else:
                    logger.info("Running FS planner one-pass (two-pass stage 1).")
                    w_fs1 = neg_w_fs1
                    fs1 = solve_planner(
                        params, T, tstep, utility="fs",
                        solver_opts=solver_opts, diagnostics_dir=diagnostics_dir,
                        exogenous_S=exog_S_fs, population_weight_envy_guilt=pop_w_env,
                        negishi_weights=w_fs1, negishi_use=use_negishi_fs1,
                        discount_series=d1, disc_tag=tag1,
                    )
                    if fs1 is not None:
                        fs1["mode"] = f"fs_{str(config.get('planner_fs_S_mode')).lower()}"
                        if neg_w_fs1 is not None:
                            fs1["negishi_weights"] = neg_w_fs1
                        if exog_S_fs is None: _export_S_csv(fs1, params.countries, T, output_dir, "planner_fs")
                        fs1 = stamp_identity(fs1, utility="fs", spec_id=sid_pl_fs1, disc_tag=tag1)
                        _put_planner_into_cache(store, solution=fs1, regions=regions, periods=periods)
                # Stage 2: two-pass series from CRRA growth + FS(one-pass) ranks
                # SAFETY GUARD: require an optimal FS(1) baseline; otherwise skip two-pass entirely.
                if (fs1 is None) or (not is_solution_optimal(fs1)):
                    logger.warning("Skipping FS planner two-pass: one-pass baseline missing or non-optimal.")
                    skip_final_planner_fs = True
                else:
                    d2 = _build_fs_discount_series_two_pass(
                        anchor_crra_sol=coop_crra, fs_baseline_sol=fs1, params=params,
                        regions=regions, periods=periods, population_weight_envy_guilt=pop_w_env,
                    )
                    _digest_pl = digest_regional_series(d2, regions, T)
                    logger.info(
                        "FS planner two-pass (stage 2): constructed regional discounts (digest=%s).",
                        _digest_pl,
                    )
                    discount_series_pl = d2
                    disc_source_pl = "two_pass"
                    disc_tag_pl = f"disc:two_pass:planner:regional:{_digest_pl}"
            else:
                # FS discount enabled but not applicable to planner (e.g., regime='nash')
                disc_source_pl = "data"
                disc_tag_pl = "disc:data"
                logger.info(
                    "FS discounting enabled but not applied to planner regime; "
                    "using baseline data discounts."
                )
        # If Negishi weights should incorporate the FS discounting path,
        # build FS Negishi weights now that the planner discount series is known.
        if do_negishi and negishi_source == "fs_after_disc":
            if bau_sol is None:
                raise RuntimeError("negishi_source='fs_after_disc' requires BAU solution but bau_sol is None.")
            if not discount_series_pl:
                raise RuntimeError(
                    "negishi_source='fs_after_disc' requires a non-empty FS discount series; "
                    "check fs_disc_enabled and fs_disc_mode."
                )
            out_fs = output_dir / "negishi_weights_fs_after_disc.csv"
            logger.info(
                "Computing FS Negishi weights from BAU consumption using FS planner discount series (mode=%s).",
                fs_disc_mode,
            )
            neg_fs_df = compute_negishi_weights_from_bau_fs_after_disc(
                params=params,
                bau_sol=bau_sol,
                fs_disc=discount_series_pl,  # {(r,t)->d^{FS}_{r,t}}
                population_weight_envy_guilt=pop_w_env,
                # Only write the FS-after-disc Negishi weights CSV when diagnostics are enabled.
                # The weights are still computed and used regardless of this flag.
                output_path=(out_fs if DIAGNOSTICS_ON else None),
            )

        # Build spec-id (include disc_tag if known)
        sid_pl_fs = build_solution_spec_id(
            utility="fs", T=T, countries=params.countries,
            population_weight_envy_guilt=pop_w_env,
            exogenous_S=normalize_exogenous_S(exog_S_fs, params.countries, T) if exog_S_fs is not None else None,
            negishi_use=do_negishi, negishi_weights=neg_fs_df if do_negishi else None,
            disc_tag=disc_tag_pl,
        )
        grand_vec = tuple(1 for _ in params.countries)
        hit = store.get(grand_vec, sid_pl_fs)
        if hit is not None and hit.get("solution") is not None and is_solution_optimal(hit["solution"]):
            if exog_S_fs is None or same_exogenous_S(exog_S_fs, hit["solution"].get("S_exogenous")):
                coop_fs = hit["solution"]
                logger.info("Cache hit for planner (FS) GRAND — skipping solve.")
        if (coop_fs is None) and (not skip_final_planner_fs):
            logger.info("Running FS planner (S=%s; envy/guilt pop-weight=%s).",
                        str(config.get("planner_fs_S_mode")).lower(), pop_w_env)
            w_fs = neg_fs_df if do_negishi else None
            if do_negishi and w_fs is None:
                raise RuntimeError("negishi_use=true but FS Negishi weights are missing.")
            coop_fs = solve_planner(
                params, T, tstep, utility="fs",
                solver_opts=solver_opts, diagnostics_dir=diagnostics_dir,
                exogenous_S=exog_S_fs, population_weight_envy_guilt=pop_w_env,
                negishi_weights=w_fs, negishi_use=do_negishi,
                discount_series=discount_series_pl,
                disc_tag=disc_tag_pl,
            )
            if coop_fs is not None:
                coop_fs["mode"] = f"fs_{str(config.get('planner_fs_S_mode')).lower()}"
                if neg_fs_df is not None: coop_fs["negishi_weights"] = neg_fs_df
                if exog_S_fs is None: _export_S_csv(coop_fs, params.countries, T, output_dir, "planner_fs")
                coop_fs = stamp_identity(coop_fs, utility="fs", spec_id=sid_pl_fs, disc_tag=disc_tag_pl)
                # Attach discount provenance
                if fs_disc_enabled:
                    coop_fs["disc_source"] = disc_source_pl
                    coop_fs["disc_regime"] = "planner"
                    # Only export/record digest for CRRA-based modes; handle regional vs global shapes
                    if discount_series_pl and fs_disc_mode in {"one_pass", "two_pass"}:
                        coop_fs["disc_digest"] = (
                            digest_regional_series(discount_series_pl, regions, T)
                            if isinstance(next(iter(discount_series_pl.keys())), tuple)
                            else digest_series(discount_series_pl)
                        )
                        # (2) CSV export
                        _export_disc_csv(discount_series_pl, T, output_dir, tag=f"planner_{fs_disc_mode}")                        
                # Ensure cache uses final spec-id (with disc_tag)
                _put_planner_into_cache(store, solution=coop_fs, regions=regions, periods=periods)
                logger.info("Cached planner (FS) GRAND in store.")
    # Record FS planner anchor spec-id if available
    if coop_fs is not None and "spec_id" in coop_fs:
        planner_anchor_spec_id_fs = str(coop_fs["spec_id"])
        
    # ---- Noncooperative Nash (CACHE-FIRST singletons)
    nash_crra = None
    if run_nash_crra:
        exog_S_crra = _resolve_nash_S(config, params, utility="crra", coop_crra=coop_crra, coop_fs=coop_fs)
        sid_nash_crra = build_solution_spec_id(
            utility="crra", T=T, countries=params.countries,
            population_weight_envy_guilt=pop_w_env,
            exogenous_S=normalize_exogenous_S(exog_S_crra, params.countries, T) if exog_S_crra is not None else None,
            negishi_use=False, negishi_weights=None,
        )
        singletons = [tuple(1 if j == i else 0 for j in range(len(regions))) for i in range(len(regions))]
        hits = []
        for vec in singletons:
            hit = store.get(vec, sid_nash_crra)
            hits.append(bool(hit and hit.get("solution")))
        if all(hits):
            logger.info("Cache hit for Nash (CRRA) singletons — skipping solve.")
            nash_crra = store.get(singletons[0], sid_nash_crra)["solution"]
        if nash_crra is None:
            seed = _nash_seed(config, params, T, bau_sol, coop_crra, "nash_crra_seed")
            seed_mode_crra = str(config.get("nash_crra_seed", "planner")).lower()
            seed_label_crra = "data" if seed_mode_crra == "data" else ("BAU" if seed_mode_crra == "bau" else "planner")
            logger.info("Running Nash (CRRA) with %s seed and S=%s.",
                        seed_label_crra,
                        "exogenous" if exog_S_crra is not None else "optimal")
            nash_crra = solve_nash(
                params, T, tstep,
                utility="crra",
                solver_opts=solver_opts,
                diagnostics_dir=diagnostics_dir / "nash_crra",
                initial_solution=seed,
                exogenous_S=exog_S_crra,
                population_weight_envy_guilt=pop_w_env,
                max_iter=int(config["max_iter_nash"]),
                tol=float(config["tol_mu_nash"]),
                relax=float(config["nash_relax"]),
                ignore_last_k_periods=ignore_last_k_periods,
            )
            if nash_crra is not None and exog_S_crra is None:
                _export_S_csv(nash_crra, params.countries, T, output_dir, "nash_crra")
            if nash_crra is not None:
                nash_crra = stamp_identity(nash_crra, utility="crra", spec_id=sid_nash_crra)
                _put_nash_into_cache(store, solution=nash_crra, regions=regions, periods=periods)
                logger.info("Cached Nash (CRRA) SINGLETONS in store.")

    # Record CRRA Nash anchor spec-id if available
    if nash_crra is not None and "spec_id" in nash_crra:
        nash_anchor_spec_id_crra = str(nash_crra["spec_id"])

    nash_fs = None
    if run_nash_fs:
        exog_S_fs_nash = _resolve_nash_S(config, params, utility="fs", coop_crra=coop_crra, coop_fs=coop_fs)
        # ----- Resolve discount series / disc_tag for FS Nash -----
        discount_series_nash: Optional[Dict[int, float]] = None
        disc_tag_nash = None
        disc_source_nash = "data"
        # Nash two-pass guard toggle
        skip_final_nash_fs = False
        # Guard: ensure CRRA Nash anchor if needed
        need_crra_anchor_n = fs_disc_enabled and fs_disc_mode in {"one_pass", "two_pass"}
        if need_crra_anchor_n and (nash_crra is None or "C" not in nash_crra):
            logger.info("FS discount alignment (Nash) requires CRRA Nash anchor — running it now.")
            exog_S_crra_n = _resolve_nash_S(config, params, utility="crra", coop_crra=coop_crra, coop_fs=coop_fs)
            seed = _nash_seed(config, params, T, bau_sol, coop_crra, "nash_crra_seed")
            nash_crra = solve_nash(
                params, T, tstep,
                utility="crra",
                solver_opts=solver_opts,
                diagnostics_dir=diagnostics_dir / "nash_crra",
                initial_solution=seed,
                exogenous_S=exog_S_crra_n,
                population_weight_envy_guilt=pop_w_env,
                max_iter=int(config["max_iter_nash"]),
                tol=float(config["tol_mu_nash"]),
                relax=float(config["nash_relax"]),
                ignore_last_k_periods=ignore_last_k_periods,
            )
            if nash_crra is not None:
                _put_nash_into_cache(store, solution=nash_crra, regions=regions, periods=periods)
        if fs_disc_enabled:
            if fs_disc_mode == "off":
                disc_source_nash = "data"
                disc_tag_nash = "disc:data"
            elif fs_disc_mode == "file":
                if not fs_disc_file:
                    raise KeyError("fs_disc_mode='file' requires fs_disc_file in config.")
                d = _read_disc_csv(Path(fs_disc_file), regions, int(T))
                discount_series_nash = d
                disc_source_nash = "file"
                logger.info("Loaded FILE discount series: REGIONAL (%s)", fs_disc_file)
                disc_tag_nash = f"disc:file:regional:{digest_regional_series(d, regions, T)}"
            elif fs_disc_mode == "one_pass":
                d = _build_fs_discount_series_one_pass(
                    anchor_crra_sol=nash_crra, params=params, regions=regions, periods=periods,
                    population_weight_envy_guilt=pop_w_env,
                )
                discount_series_nash = d
                disc_source_nash = "one_pass"
                disc_tag_nash = f"disc:one_pass:nash:regional:{digest_regional_series(d, regions, T)}"
            elif fs_disc_mode == "two_pass":
                # Stage 1: FS one-pass (CRRA g + CRRA ranks), cache-first for SINGLETONS
                d1 = _build_fs_discount_series_one_pass(
                    anchor_crra_sol=nash_crra, params=params, regions=regions, periods=periods,
                    population_weight_envy_guilt=pop_w_env,
                )
                tag1 = f"disc:one_pass:nash:regional:{digest_regional_series(d1, regions, T)}"
                sid_fs1 = build_solution_spec_id(
                    utility="fs", T=T, countries=params.countries,
                    population_weight_envy_guilt=pop_w_env,
                    exogenous_S=normalize_exogenous_S(exog_S_fs_nash, params.countries, T) if exog_S_fs_nash is not None else None,
                    negishi_use=False, negishi_weights=None,
                    disc_tag=tag1,
                )
                singletons = [tuple(1 if j == i else 0 for j in range(len(regions))) for i in range(len(regions))]
                hits1 = []
                for vec in singletons:
                    hit1 = store.get(vec, sid_fs1)
                    hits1.append(bool(hit1 and hit1.get("solution")))
                if all(hits1):
                    logger.info("Cache hit for FS Nash one-pass baseline — reusing for two-pass.")
                    fs1 = store.get(singletons[0], sid_fs1)["solution"]
                else:
                    seed1 = _nash_seed(config, params, T, bau_sol, coop_fs, "nash_fs_seed")
                    logger.info("Running Nash (FS) one-pass (two-pass stage 1).")
                    fs1 = solve_nash(
                        params, T, tstep,
                        utility="fs",
                        solver_opts=solver_opts,
                        diagnostics_dir=diagnostics_dir / "nash_fs",
                        initial_solution=seed1,
                        exogenous_S=exog_S_fs_nash,
                        population_weight_envy_guilt=pop_w_env,
                        max_iter=int(config["max_iter_nash"]),
                        tol=float(config["tol_mu_nash"]),
                        relax=float(config["nash_relax"]),
                        ignore_last_k_periods=ignore_last_k_periods,
                        discount_series=d1,
                        disc_tag=tag1,
                    )
                    if fs1 is not None:
                        fs1 = stamp_identity(fs1, utility="fs", spec_id=sid_fs1, disc_tag=tag1)
                        _put_nash_into_cache(store, solution=fs1, regions=regions, periods=periods)
                # SAFETY GUARD: require an optimal FS(1) baseline; otherwise skip two-pass entirely.
                if (fs1 is None) or (not is_solution_optimal(fs1)):
                    logger.warning("Skipping FS Nash two-pass: one-pass baseline missing or non-optimal.")
                    skip_final_nash_fs = True
                else:
                    d2 = _build_fs_discount_series_two_pass(
                        anchor_crra_sol=nash_crra, fs_baseline_sol=fs1, params=params,
                        regions=regions, periods=periods, population_weight_envy_guilt=pop_w_env,
                    )
                    _digest_n = digest_regional_series(d2, regions, T)
                    logger.info("Running FS Nash two-pass (stage 2): constructed regional discounts (digest=%s).", _digest_n)
                    discount_series_nash = d2
                    disc_source_nash = "two_pass"
                    disc_tag_nash = f"disc:two_pass:nash:regional:{_digest_n}"
            else:
                disc_source_nash = "data"
                disc_tag_nash = "disc:data"
        sid_fs = build_solution_spec_id(
            utility="fs", T=T, countries=params.countries,
            population_weight_envy_guilt=pop_w_env,
            exogenous_S=normalize_exogenous_S(exog_S_fs_nash, params.countries, T) if exog_S_fs_nash is not None else None,
            negishi_use=False, negishi_weights=None,
            disc_tag=disc_tag_nash,
        )
        singletons = [tuple(1 if j == i else 0 for j in range(len(regions))) for i in range(len(regions))]
        hits = []
        for vec in singletons:
            hit = store.get(vec, sid_fs)
            hits.append(bool(hit and hit.get("solution")))
        if all(hits):
            logger.info("Cache hit for Nash (FS) singletons — skipping solve.")
            nash_fs = store.get(singletons[0], sid_fs)["solution"]
        if (nash_fs is None) and (not skip_final_nash_fs):
            seed = _nash_seed(config, params, T, bau_sol, coop_fs, "nash_fs_seed")
            seed_mode_fs = str(config.get("nash_fs_seed", "planner")).lower()
            seed_label_fs = "data" if seed_mode_fs == "data" else ("BAU" if seed_mode_fs == "bau" else "planner")
            logger.info("Running Nash (FS) with %s seed and S=%s.",
                        seed_label_fs,
                        "exogenous" if exog_S_fs_nash is not None else "optimal")
            nash_fs = solve_nash(
                params, T, tstep,
                utility="fs",
                solver_opts=solver_opts,
                diagnostics_dir=diagnostics_dir / "nash_fs",
                initial_solution=seed,
                exogenous_S=exog_S_fs_nash,
                population_weight_envy_guilt=pop_w_env,
                max_iter=int(config["max_iter_nash"]),
                tol=float(config["tol_mu_nash"]),
                relax=float(config["nash_relax"]),
                ignore_last_k_periods=ignore_last_k_periods,
                discount_series=discount_series_nash,
                disc_tag=disc_tag_nash, 
            )
            if nash_fs is not None and exog_S_fs_nash is None:
                _export_S_csv(nash_fs, params.countries, T, output_dir, "nash_fs")
            if nash_fs is not None:
                if fs_disc_enabled:
                    nash_fs["disc_source"] = disc_source_nash
                    nash_fs["disc_regime"] = "nash"
                    # Export series only for CRRA-based modes (not 'file' or 'off')
                    if discount_series_nash and fs_disc_mode in {"one_pass", "two_pass"}:
                        nash_fs["disc_digest"] = digest_regional_series(discount_series_nash, regions, T) \
                            if isinstance(next(iter(discount_series_nash.keys())), tuple) else digest_series(discount_series_nash)
                        _export_disc_csv(discount_series_nash, T, output_dir, tag=f"nash_{fs_disc_mode}")
                nash_fs = stamp_identity(nash_fs, utility="fs", spec_id=sid_fs, disc_tag=disc_tag_nash)
                _put_nash_into_cache(store, solution=nash_fs, regions=regions, periods=periods)
                logger.info("Cached Nash (FS) SINGLETONS in store.")

    # Record FS Nash anchor spec-id if available
    if nash_fs is not None and "spec_id" in nash_fs:
        nash_anchor_spec_id_fs = str(nash_fs["spec_id"])

    # ---- Coalition cache materialization (no export here; CLI will export)
    mega_run_flag = bool(config.get("mega_run", False))
    # Predeclare FS coalition context for all code paths so late materialization
    # never hits UnboundLocalError on cache-only re-runs.
    exog_S_coal_fs = None
    neg_w_fs = None

    # Safety: FS discount alignment modes (one_pass/two_pass) for coalitions
    # conceptually require a CRRA anchor. The CLI guards should already enforce
    # this at the config level, but we keep a defensive check here as well.
    if (
        run_coal_fs
        and (not run_coal_crra)
        and bool(fs_disc_enabled)
        and fs_disc_mode in {"one_pass", "two_pass"}
    ):
        raise RuntimeError(
            "FS coalition discount modes 'one_pass'/'two_pass' require CRRA coalitions "
            "(run_coalition_crra=true) so a CRRA anchor is available. "
            "Either enable run_coalition_crra or use fs_disc_mode='off'/'file' for FS-only coalitions."
        )

    # --- Coalitions: CRRA and FS can be run independently per new config
    if run_coal_crra or run_coal_fs:
        # CRRA block (unchanged)
        if run_coal_crra:
            exog_S_coal_crra = _resolve_coalition_S(config, params, utility="crra", coop_crra=coop_crra, coop_fs=coop_fs)
            # Negishi selection for CRRA coalitions
            neg_w_crra = (neg_crra_df if do_negishi else None)
            if do_negishi and neg_w_crra is None:
                raise RuntimeError("negishi_use=true but CRRA Negishi weights are missing.")
            # Targets
            base_vec: Optional[tuple[int, ...]] = None
            if not mega_run_flag:
                base_vec = tuple(parse_coalition_spec(str(coalition_spec), regions=regions))
            vectors = compute_target_vectors(
                regions=regions,
                base_vector=base_vec,
                want_neighbors=bool(config.get("coalition_check_internal")) or bool(config.get("coalition_check_external")),
                mega_run=mega_run_flag,
            )
            # If we intend to pair FS inside workers, pre-resolve FS coalition context here
            exog_S_coal_fs = None
            neg_w_fs = None
            if pair_in_worker:
                exog_S_coal_fs = _resolve_coalition_S(
                    config, params, utility="fs", coop_crra=coop_crra, coop_fs=coop_fs
                )
                neg_w_fs = neg_fs_df if do_negishi else None

            fetch_or_solve_coalitions(
                regions=regions,
                periods=periods,
                utility="crra",
                vectors=vectors,
                store=store,
                params=params,
                tstep=tstep,
                solver_opts=solver_opts,
                exogenous_S=exog_S_coal_crra,
                negishi_use=bool(do_negishi),
                negishi_weights=neg_w_crra,
                population_weight_envy_guilt=False,
                max_iter_nash=int(config["max_iter_nash"]),
                tol_mu_nash=float(config["tol_mu_nash"]),
                relax=float(config["nash_relax"]),
                ignore_last_k_periods=ignore_last_k_periods,
                diagnostics_dir=diagnostics_dir / ("coalition_all" if mega_run_flag else "coalition_partial"),
                workers=int(config["parallel"]),
                planner_anchor_spec_id=planner_anchor_spec_id_crra,
                nash_anchor_spec_id=nash_anchor_spec_id_crra,
                # --- NEW: tell each CRRA worker to immediately run the paired FS for the same coalition ---
                pair_fs_in_worker=pair_in_worker,
                fs_disc_enabled=fs_disc_enabled,
                fs_disc_mode=fs_disc_mode,
                fs_disc_file=fs_disc_file,
                exogenous_S_fs=exog_S_coal_fs,
                negishi_use_fs=bool(do_negishi),
                negishi_weights_fs=neg_w_fs,
                population_weight_envy_guilt_fs=pop_w_env,
                failures_path=failures_path,
                bau_sol=bau_sol,
                fs_negishi_source=negishi_source,
            )

        # FS-only coalition solve: when FS coalitions are requested but CRRA
        # coalitions are *not* run, we must still drive the coalition solver
        # for utility='fs' directly. This is also where FS-only discounting
        # from file (fs_disc_mode='file') is triggered; the per-coalition
        # workers will apply the file-based discount series when utility='fs'.
        if run_coal_fs and (not run_coal_crra):
            # Resolve FS S-policy for coalitions
            exog_S_coal_fs = _resolve_coalition_S(
                config, params, utility="fs", coop_crra=coop_crra, coop_fs=coop_fs
            )
            # Negishi selection for FS coalitions (if enabled)
            neg_w_fs = (neg_fs_df if do_negishi else None)
            if do_negishi and neg_w_fs is None:
                raise RuntimeError("negishi_use=true but FS Negishi weights are missing for FS coalitions.")

            # Target coalition vectors (mega_run/all vs. base+neighbors)
            base_vec_fs: Optional[tuple[int, ...]] = None
            if not mega_run_flag:
                base_vec_fs = tuple(parse_coalition_spec(str(coalition_spec), regions=regions))
            vectors_fs = compute_target_vectors(
                regions=regions,
                base_vector=base_vec_fs,
                want_neighbors=bool(config.get("coalition_check_internal")) or bool(config.get("coalition_check_external")),
                mega_run=mega_run_flag,
            )

            fetch_or_solve_coalitions(
                regions=regions,
                periods=periods,
                utility="fs",
                vectors=vectors_fs,
                store=store,
                params=params,
                tstep=tstep,
                solver_opts=solver_opts,
                exogenous_S=exog_S_coal_fs,
                negishi_use=bool(do_negishi),
                negishi_weights=neg_w_fs,
                population_weight_envy_guilt=pop_w_env,
                max_iter_nash=int(config["max_iter_nash"]),
                tol_mu_nash=float(config["tol_mu_nash"]),
                relax=float(config["nash_relax"]),
                ignore_last_k_periods=ignore_last_k_periods,
                diagnostics_dir=diagnostics_dir / ("coalition_all" if mega_run_flag else "coalition_partial"),
                workers=int(config["parallel"]),
                # Use FS anchors when available (planner/Nash FS spec-ids)
                planner_anchor_spec_id=planner_anchor_spec_id_fs,
                nash_anchor_spec_id=nash_anchor_spec_id_fs,
                # FS-only: no CRRA→FS pairing; workers run FS directly.
                pair_fs_in_worker=False,
                fs_disc_enabled=fs_disc_enabled,
                fs_disc_mode=fs_disc_mode,
                fs_disc_file=fs_disc_file,
                exogenous_S_fs=None,
                negishi_use_fs=False,
                negishi_weights_fs=None,
                population_weight_envy_guilt_fs=False,
                failures_path=failures_path,
                bau_sol=bau_sol,
                fs_negishi_source=negishi_source,
            )

        # FS block
        # Always predeclare these so later export materialization never hits an UnboundLocalError,
        # even when FS coalitions are paired inside CRRA workers (pair_in_worker=True).
        coalitions_aligned_fs: List[Dict[str, Any]] = []
        coalitions_file_fs:    List[Dict[str, Any]] = []

        # Materialize present (base + neighbors) for CLI export
        coalitions = []
        # Materialize whatever was requested (CRRA and/or FS) — both sections above populate the cache.
        # We export both sets if both were requested.
        def _collect(utility: str, exog_df, neg_wgt, pop_env):
            _sid = build_solution_spec_id(
                utility=utility, T=T, countries=params.countries,
                population_weight_envy_guilt=pop_env,
                exogenous_S=normalize_exogenous_S(exog_df, params.countries, T) if exog_df is not None else None,
                negishi_use=bool(do_negishi), negishi_weights=neg_wgt if do_negishi else None,
            )
            _base_vec = None if mega_run_flag else tuple(parse_coalition_spec(str(coalition_spec), regions=regions))
            _vectors = compute_target_vectors(
                regions=regions,
                base_vector=_base_vec,
                want_neighbors=bool(config.get("coalition_check_internal")) or bool(config.get("coalition_check_external")),
                mega_run=mega_run_flag,
            )
            for v in _vectors:
                wrapped = _wrap_cached(list(v), _sid, store)
                if wrapped is not None:
                    coalitions.append(wrapped)
        if run_coal_crra:
            _collect("crra", exog_S_coal_crra, (neg_crra_df if do_negishi else None), False)
        if run_coal_fs:
            if (not fs_disc_enabled) or (fs_disc_mode == "off"):
                # OFF: generic sid (no disc_tag) is correct
                _collect("fs", exog_S_coal_fs, (neg_fs_df if do_negishi else None), pop_w_env)
            elif fs_disc_mode == "file":
                # Per-vector FILE-mode results were appended above
                for w in coalitions_file_fs:
                    if w is not None:
                        coalitions.append(w)
            else:
                # Per-vector aligned FS results were appended during solving
                for w in coalitions_aligned_fs:
                    if w is not None:
                        coalitions.append(w)

    # Bundle meta for export layer
    meta = {
        "countries": regions,
        "periods": periods,
        "L": getattr(params, "L", None),
        "tstep": tstep,
        "base_year": params.base_year,
        "backstop_switch_year": params.backstop_switch_year,
    }

    return AnalysisResult(
        coop={"crra": coop_crra, "fs": coop_fs},
        noncoop={"crra": nash_crra, "fs": nash_fs} if (nash_crra or nash_fs) else None,
        coalitions=coalitions,
        bau=bau_sol,
        meta=meta,
    )