"""Data loader for RICE13_FS

- No `tstep` anymore: the model is decadal by construction (periods are 10-year steps).
- Validate that T <= 59 (data coverage).
- Exact filenames. FS_types rows=countries.
- Rename CRRA parameter alpha to eta.
- All warm starts—including SLR—are loaded from Var_country_independent_init.csv only.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import logging
import math
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------
# Small helpers
# ---------------------------

def _read_csv(path: Path, *, index_col: int | None = 0) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    # sep=None lets pandas sniff delimiters; engine="python" tolerates mixed whitespace
    df = pd.read_csv(path, sep=None, engine="python", index_col=index_col)
    if isinstance(df.index, pd.RangeIndex):
        name_col = str(df.columns[0]).lower()
        if name_col in {"region", "regions", "country", "countries", "name"}:
            df = df.set_index(df.columns[0])
    return df

def _align_regions_as_rows(df: pd.DataFrame, countries: List[str], *, tabname: str) -> pd.DataFrame:
    idx = list(map(str, df.index))
    cols = list(map(str, df.columns))
    if all(r in idx for r in countries):
        return df.loc[countries]
    if all(r in cols for r in countries):
        logger.warning("%s: regions detected on columns; transposing to rows.", tabname)
        return df[countries].T
    raise KeyError(f"{tabname}: regions not found on index nor columns. Expected codes like {countries[:5]}…")

def _maybe_int_columns(df: pd.DataFrame) -> pd.DataFrame:
    # best-effort: convert column labels to int (e.g., "1","2","Y2015"→2015)
    try:
        new_cols = []
        for c in df.columns:
            s = str(c).strip()
            if s and s[0] in "Yy":
                s = s[1:]
            new_cols.append(int(s))
        df = df.copy()
        df.columns = new_cols
        df = df.reindex(sorted(df.columns), axis=1)
    except Exception:
        pass
    return df

def interpolate_series(df: pd.DataFrame, T: int) -> pd.DataFrame:
    """
    Decadal pass-through with optional warm-start.

    Input df may have a warm-start column '0'. We:
      1) Coerce columns to integers when possible (tolerate strings like "0","1","Y2025").
      2) Drop column 0 if present (warm-start).
      3) Require at least T decadal columns; take the first T.
      4) Return columns 1..T (ints), same index, float dtype.
    """
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    df = _maybe_int_columns(df.copy())

    # Separate warm-start (0) if present
    anchors = df.drop(columns=[0]) if 0 in df.columns else df
    D = anchors.shape[1]
    if D < T:
        raise ValueError(f"interpolate_series(decadal): need at least T={T} decadal columns; found {D}.")

    out = anchors.iloc[:, :T].astype(float).copy()
    out.columns = range(1, T + 1)
    return out

def _load_regions(data_path: Path) -> List[str]:
    df = _read_csv(data_path / "Region_names.csv", index_col=None)
    if df.shape[1] == 1:
        countries = df.iloc[:, 0].astype(str).tolist()
    else:
        col = next((c for c in df.columns if str(c).lower() in {"region","regions","country","countries","name"}), df.columns[0])
        countries = df[col].astype(str).tolist()
    logger.info("Regions: loaded %d names from Region_names.csv", len(countries))
    return countries

# ---------------------------
# Params container
# ---------------------------

@dataclass
class Params:
    # Meta
    countries: List[str]
    T: int
    base_year: int
    backstop_switch_year: int

    # Scalar constants
    b11: float; b21: float; b12: float; b22: float; b32: float; b23: float; b33: float
    c1: float; c2: float; c3: float; c4: float
    M_1900: float; eta: float; t2xco2: float
    F_ex_1: float; F_ex_2: float
    T_at_05: float; T_at_15: float; T_lo_0: float
    M_at_0: float; M_up_0: float; M_lo_0: float
    slr_eps: float; omega_exp: float
    eq_therm: float; adj_rate_therm: float; init_slr_therm: float
    exp_gsic: float; tot_ice_gsic: float; melt_rate_gsic: float; eq_temp_gsic: float
    init_vol_gis: float; init_melt_gis: float; min_eq_temp_gis: float
    melt_above_gis: float; exp_on_remain_gis: float
    init_melt_ais: float; intercept_ais: float; melt_low_ais: float; melt_up_ais: float
    infl_pt_ais: float; ratio_Tant_Tglob: float; tot_vol_ais: float; wais_vol_ais: float; other_ais_vol: float

    # Country-dependent static parameters
    gamma: pd.Series; d_k: pd.Series; rho: pd.Series; alpha: pd.Series
    d1: pd.Series; d2: pd.Series; d3: pd.Series
    cat_t: pd.Series; d_3_cat: pd.Series
    pback: pd.Series; d_ab: pd.Series; th_2: pd.Series
    Sig_0: pd.Series; eland_0: pd.Series
    d_el: pd.Series; d_sig: pd.Series; tr_sig: pd.Series; tot_sig: pd.Series
    sig_15_add: pd.Series; sig_9506: pd.Series; add_sig: pd.Series
    catast_coeff: pd.Series
    FS_alpha: pd.Series; FS_beta: pd.Series

    # Per-region CRRA curvature
    crra_eta: pd.Series

    # Initial values for state variables
    E_0: pd.Series; Y_0: pd.Series; rho_0: pd.Series; K_0: pd.Series
    A_0: pd.Series; s_0: pd.Series; L_0: pd.Series

    # Time-varying inputs (raw growth/params, decadal columns)
    g_A: pd.DataFrame; g_L: pd.DataFrame; bau_s_r_t: pd.DataFrame

    # SLR damages
    d1_slr: pd.Series; d2_slr: pd.Series; slr_multiplier: pd.Series

    # Derived time-series (country × period, columns 0..T or 1..T as appropriate)
    L: pd.DataFrame; A: pd.DataFrame
    sigma: pd.DataFrame  # emissions intensity
    eland: pd.DataFrame; f_ex: pd.Series
    backstpr: pd.DataFrame; theta1: pd.DataFrame
    bau_saving_rates: pd.DataFrame

    # Warm-start initializations (country × period)
    U_init: pd.DataFrame; K_init: pd.DataFrame; S_init: pd.DataFrame
    I_init: pd.DataFrame; Q_init: pd.DataFrame; Q_0: pd.Series; Y_init: pd.DataFrame
    savings_init: pd.DataFrame; mu_init: pd.DataFrame; AB_init: pd.DataFrame; D_init: pd.DataFrame
    damage_frac_init: pd.DataFrame; C_init: pd.DataFrame; E_ind_init: pd.DataFrame

    # Warm-start series for global variables (period-indexed 1..T)
    E_tot_init: pd.Series; M_at_init: pd.Series; M_up_init: pd.Series; M_lo_init: pd.Series
    T_at_init: pd.Series; T_lo_init: pd.Series; F_init: pd.Series
    I_0: pd.Series; slr_TE_init: pd.Series
    gsic_remain_init: pd.Series; gsic_melt_init: pd.Series; gsic_cum_init: pd.Series
    gis_remain_init: pd.Series; gis_melt_init: pd.Series; gis_cum_init: pd.Series
    ais_remain_init: pd.Series; ais_melt_init: pd.Series; ais_cum_init: pd.Series
    slr_init: pd.Series

# ---------------------------
# Loader (decadal only)
# ---------------------------

def load_params(data_path: Path, T: int) -> Params:
    """Load all parameters/series for a decadal grid of length T (validate T <= 59)."""
    data_path = Path(data_path)
    if T > 59:
        raise ValueError(f"T={T} exceeds data coverage (max 59).")

    # Regions
    countries = _load_regions(data_path)

    # Core CSVs (original names)
    inv_p     = _read_csv(data_path / "Time_countries_invariant_param.csv").squeeze()
    c_var_p   = _read_csv(data_path / "Countries_variant_param.csv")
    t0_v      = _read_csv(data_path / "Time_0_values.csv")
    tfp_g     = _maybe_int_columns(_read_csv(data_path / "TFP_growth.csv"))
    pop_g     = _maybe_int_columns(_read_csv(data_path / "Pop_growth.csv"))
    bau_s_r_t     = _maybe_int_columns(_read_csv(data_path / "bau_savings_rate.csv"))
    savings_init = _maybe_int_columns(_read_csv(data_path / "saving_rate_t.csv"))
    slr_module_p = _read_csv(data_path / "slr_parameters.csv").squeeze()
    FS_types  = _read_csv(data_path / "FS_types.csv")

    # Country-variant series used below
    Sig_0      = c_var_p.loc['Sig_0'     ].astype(float).reindex(countries)
    sig_15_add = c_var_p.loc['sig_15_add'].astype(float).reindex(countries)
    d_sig      = c_var_p.loc['d_sig'     ].astype(float).reindex(countries)
    tr_sig     = c_var_p.loc['tr_sig'    ].astype(float).reindex(countries)
    tot_sig    = c_var_p.loc['tot_sig'   ].astype(float).reindex(countries)
    d_el       = c_var_p.loc['d_el'      ].astype(float).reindex(countries)
    eland_0    = c_var_p.loc['Eland_0'   ].astype(float).reindex(countries)
    pback      = c_var_p.loc['pback'     ].astype(float).reindex(countries)
    d_ab       = c_var_p.loc['d_ab'      ].astype(float).reindex(countries)
    th_2       = c_var_p.loc['th_2'      ].astype(float).reindex(countries)

    # Per-region CRRA curvature: use alpha row from Countries_variant_param.csv
    crra_eta = c_var_p.loc["alpha"].astype(float).reindex(countries)

    # FS_types: countries as rows
    FS_alpha = FS_types["alpha"].astype(float).reindex(countries)
    FS_beta  = FS_types["beta"].astype(float).reindex(countries)

    # Initial conditions (rows in Time_0_values)
    def _row(name: str) -> pd.Series:
        if name not in t0_v.index:
            raise KeyError(f"Time_0_values.csv missing row '{name}'")
        return t0_v.loc[name].astype(float).reindex(countries)

    E_0 = _row('E_0'); Y_0 = _row('Y_0'); rho_0 = _row('rho_0')
    K_0 = _row('K_0'); A_0 = _row('A_0'); s_0 = _row('s_0'); L_0 = _row('L_0')

    years_0T = list(range(0, T + 1))

    # Population L (decadal)
    L = pd.DataFrame(0.0, index=countries, columns=years_0T)
    for r in countries:
        L.at[r, 0] = L_0[r]
        for j in range(1, T + 1):
            L.at[r, j] = L.at[r, j-1] * math.exp(pop_g.at[r, j] * 10)

    # TFP A (decadal)
    A = pd.DataFrame(0.0, index=countries, columns=years_0T)
    for r in countries:
        A.at[r, 0] = A_0[r]
        for j in range(1, T + 1):
            A.at[r, j] = A.at[r, j-1] * math.exp(tfp_g.at[r, j] * 10)

    # Emissions intensity (sigma)
    g_sig = pd.DataFrame(0.0, index=countries, columns=years_0T)
    for r in countries:
        g_sig.at[r, 0] = 0.0
        for j in range(1, T + 1):
            g_sig.at[r, j] = (tot_sig[r] if j == 1 else (tr_sig[r] + (g_sig.at[r, j-1] - tr_sig[r]) * (1 - d_sig[r])))
    sig = pd.DataFrame(0.0, index=countries, columns=years_0T)
    for r in countries:
        sig.at[r, 0] = Sig_0[r]
        for j in range(1, T + 1):
            if j == 1:
                sig.at[r, j] = sig.at[r, j-1] * math.exp(g_sig.at[r, j] * 10) * sig_15_add[r]
            else:
                sig.at[r, j] = sig.at[r, j-1] * math.exp(g_sig.at[r, j] * 10)
    sigma = sig.copy()
    logger.info("Sigma: built from Sig_0 and g_sig dynamics (decadal)")

    # Land-use emissions (E_land)
    eland = pd.DataFrame(0.0, index=countries, columns=years_0T)
    for r in countries:
        eland.at[r, 0] = eland_0[r]
        for j in range(1, T + 1):
            if j == 1:
                eland.at[r, j] = eland.at[r, j-1] * (1 - d_el[r])
            else:
                eland.at[r, j] = eland.at[r, j-1] * (1 - d_el[r])
    logger.info("E_land: built from Eland_0 and d_el (decadal)")

    # Exogenous forcing f_ex (0..T), standard decadal path
    F_ex_1 = float(inv_p['F_ex_1'])
    F_ex_2 = float(inv_p['F_ex_2'])
    f_ex = pd.Series(index=years_0T, dtype=float)
    f_ex.at[0] = F_ex_1
    for j in range(1, T + 1):
        if j < 11:
            f_ex.at[j] = F_ex_1 + 0.1 * (F_ex_2 - F_ex_1) * j
        else:
            f_ex.at[j] = F_ex_1 + 0.36

    # Backstop / θ1
    
    base_year=2005
    backstop_switch_year=2250
    
    backstpr1 = pd.DataFrame(0.0, index=countries, columns=range(max(T+1, 60)))
    for c in countries:
        p0 = float(pback[c])        # 2005 value in 2005 k$/tC
        d  = float(d_ab[c])         # per-decade decline (e.g., 0.05)
        p_inf = 0.1 * p0
    
        for j in range(max(T+1, 60)):
            year = base_year + 10 * j      # j=0 -> 2005, j=1 -> 2015, ...
            if j == 0:
                val = p0
            else:
                prev = backstpr1.at[c, j - 1]
                if year <= backstop_switch_year: # ≤2245 on decadal grid (2255 is the first >2250)
                    # geometric decline toward 0.1 * p0
                    val = p_inf + (prev - p_inf) * (1.0 - d)
                else:
                    # from 2255 onward: halve each decade (Nordhaus tail)
                    val = prev * 0.5
            backstpr1.at[c, j] = val
    
    # publish periods t=1..T -> 2015, 2025, ...
    backstpr = backstpr1.iloc[:, 1 : T + 1].copy()
    backstpr.columns = range(1, T + 1)

    theta1 = backstpr.multiply(sigma).divide(th_2, axis=0)
    theta1 = theta1.reindex(index=countries, columns=range(1, T + 1))

    # Investment at time 0 and BAU saving rates
    I_0 = Y_0 * bau_s_r_t.iloc[:, 0]
    bau_saving_rates = interpolate_series(bau_s_r_t, T)
    savings_init = interpolate_series(savings_init, T)

    # Warm starts (country × period)
    def _load_ws(name: str) -> pd.DataFrame:
        df = _maybe_int_columns(_read_csv(data_path / name))
        return _align_regions_as_rows(df, countries, tabname=name)

    U_init     = interpolate_series(_load_ws('U_init.csv'),     T)
    K_init     = interpolate_series(_load_ws('K_init.csv'),     T)
    I_init     = interpolate_series(_load_ws('I_init.csv'),     T)
    Q_init_raw = _load_ws('Q_init.csv')
    Y_init     = interpolate_series(_load_ws('Y_init.csv'),     T)
    mu_init    = interpolate_series(_load_ws('mu_init.csv'),    T)
    AB_init    = interpolate_series(_load_ws('AB_init.csv'),    T)
    D_init     = interpolate_series(_load_ws('D_init.csv'),     T)
    C_init     = interpolate_series(_load_ws('C_init.csv'),     T)
    E_ind_init = interpolate_series(_load_ws('E_ind_init.csv'), T)

    oth = _read_csv(data_path / 'Var_country_independent_init.csv')

    Q_0 = Q_init_raw.loc[:, 0]
    Q_init = interpolate_series(Q_init_raw, T)

    oth_interp = interpolate_series(oth, T=T)

    def _oth_row(name: str) -> pd.Series:
        # columns are exactly 1..T; no KeyError
        return oth_interp.loc[name].copy()

    # All SLR warm starts (including slr_TE_init etc) from Var_country_independent_init.csv only
    E_tot_init = _oth_row('E_tot'); M_at_init = _oth_row('M_at'); M_up_init = _oth_row('M_up'); M_lo_init = _oth_row('M_lo')
    T_at_init  = _oth_row('T_at');  T_lo_init  = _oth_row('T_lo'); F_init    = _oth_row('F')
    gsic_remain_init = _oth_row('gsic_remain'); gsic_melt_init = _oth_row('gsic_melt'); gsic_cum_init = _oth_row('gsic_cum')
    gis_remain_init  = _oth_row('gis_remain');  gis_melt_init  = _oth_row('gis_melt');  gis_cum_init  = _oth_row('gis_cum')
    ais_remain_init  = _oth_row('ais_remain');  ais_melt_init  = _oth_row('ais_melt');  ais_cum_init  = _oth_row('ais_cum')
    slr_TE_init      = _oth_row('slr_TE');      slr_init       = _oth_row('slr')

    # Derived pieces for init
    S_init = I_init.divide(Y_init)
    damage_frac_init = D_init.divide(Q_init)

    # Scalars required in Params (load from inv_p)
    b11 = float(inv_p["b11"]); b21 = float(inv_p["b21"]); b12 = float(inv_p["b12"])
    b22 = float(inv_p["b22"]); b32 = float(inv_p["b32"]); b23 = float(inv_p["b23"]); b33 = float(inv_p["b33"])
    c1 = float(inv_p["c1"]); c2 = float(inv_p["c2"]); c3 = float(inv_p["c3"]); c4 = float(inv_p["c4"])
    M_1900 = float(inv_p["M_1900"])
    eta = float(inv_p["eta"])
    t2xco2 = float(inv_p["t2xco2"])
    T_at_05 = float(inv_p["T_at_05"]); T_at_15 = float(inv_p["T_at_15"]); T_lo_0 = float(inv_p["T_lo_0"])
    M_at_0 = float(inv_p["M_at_0"]); M_up_0 = float(inv_p["M_up_0"]); M_lo_0 = float(inv_p["M_lo_0"])
    slr_eps = float(inv_p["slr_eps"]); omega_exp = float(inv_p["omega_exp"])

    # SLR module-specific scalars (from slr_parameters.csv)
    eq_therm = float(slr_module_p["eq_therm"])
    adj_rate_therm = float(slr_module_p["adj_rate_therm"])
    init_slr_therm = float(slr_module_p["init_slr_therm"])
    exp_gsic = float(slr_module_p["exp_gsic"])
    tot_ice_gsic = float(slr_module_p["tot_ice_gsic"])
    melt_rate_gsic = float(slr_module_p["melt_rate_gsic"])
    eq_temp_gsic = float(slr_module_p["eq_temp_gsic"])
    init_vol_gis = float(slr_module_p["init_vol_gis"])
    init_melt_gis = float(slr_module_p["init_melt_gis"])
    min_eq_temp_gis = float(slr_module_p["min_eq_temp_gis"])
    melt_above_gis = float(slr_module_p["melt_above_gis"])
    exp_on_remain_gis = float(slr_module_p["exp_on_remain_gis"])
    init_melt_ais = float(slr_module_p["init_melt_ais"])
    intercept_ais = float(slr_module_p["intercept_ais"])
    melt_low_ais = float(slr_module_p["melt_low_ais"])
    melt_up_ais = float(slr_module_p["melt_up_ais"])
    infl_pt_ais = float(slr_module_p["infl_pt_ais"])
    ratio_Tant_Tglob = float(slr_module_p["ratio_Tant_Tglob"])
    tot_vol_ais = float(slr_module_p["tot_vol_ais"])
    wais_vol_ais = float(slr_module_p["wais_vol_ais"])
    other_ais_vol = float(slr_module_p["other_ais_vol"])

    return Params(
        countries=countries,
        T=T,
        base_year=base_year,
        backstop_switch_year=backstop_switch_year,
        # scalars
        b11=b11, b21=b21, b12=b12, b22=b22, b32=b32, b23=b23, b33=b33,
        c1=c1, c2=c2, c3=c3, c4=c4,
        M_1900=M_1900, eta=eta, t2xco2=t2xco2,
        F_ex_1=F_ex_1, F_ex_2=F_ex_2,
        T_at_05=T_at_05, T_at_15=T_at_15, T_lo_0=T_lo_0,
        M_at_0=M_at_0, M_up_0=M_up_0, M_lo_0=M_lo_0,
        slr_eps=slr_eps, omega_exp=omega_exp,
        eq_therm=eq_therm, adj_rate_therm=adj_rate_therm, init_slr_therm=init_slr_therm,
        exp_gsic=exp_gsic, tot_ice_gsic=tot_ice_gsic, melt_rate_gsic=melt_rate_gsic, eq_temp_gsic=eq_temp_gsic,
        init_vol_gis=init_vol_gis, init_melt_gis=init_melt_gis, min_eq_temp_gis=min_eq_temp_gis,
        melt_above_gis=melt_above_gis, exp_on_remain_gis=exp_on_remain_gis,
        init_melt_ais=init_melt_ais, intercept_ais=intercept_ais, melt_low_ais=melt_low_ais, melt_up_ais=melt_up_ais,
        infl_pt_ais=infl_pt_ais, ratio_Tant_Tglob=ratio_Tant_Tglob, tot_vol_ais=tot_vol_ais, wais_vol_ais=wais_vol_ais, other_ais_vol=other_ais_vol,
        # country-variant
        gamma=c_var_p.loc['gamma'].astype(float).reindex(countries),
        d_k=c_var_p.loc['d_k'].astype(float).reindex(countries),
        rho=c_var_p.loc['rho'].astype(float).reindex(countries),
        alpha=c_var_p.loc['alpha'].astype(float).reindex(countries),
        d1=c_var_p.loc['d_1'].astype(float).reindex(countries),
        d2=c_var_p.loc['d_2'].astype(float).reindex(countries),
        d3=c_var_p.loc['d_3'].astype(float).reindex(countries),
        cat_t=c_var_p.loc['cat_tresh'].astype(float).reindex(countries),
        d_3_cat=c_var_p.loc['d_3_cat'].astype(float).reindex(countries),
        pback=pback, d_ab=d_ab, th_2=th_2,
        Sig_0=Sig_0, eland_0=eland_0, d_el=d_el, d_sig=d_sig, tr_sig=tr_sig, tot_sig=tot_sig,
        sig_15_add=sig_15_add, sig_9506=c_var_p.loc['sig_9506'].astype(float).reindex(countries),
        add_sig=c_var_p.loc['add_sig'].astype(float).reindex(countries),
        catast_coeff=c_var_p.loc['catast_coeff'].astype(float).reindex(countries),
        FS_alpha=FS_alpha, FS_beta=FS_beta,
        # CRRA curvature per region
        crra_eta=crra_eta,
        # initials
        E_0=E_0, Y_0=Y_0, rho_0=rho_0, K_0=K_0, A_0=A_0, s_0=s_0, L_0=L_0,
        # time-varying inputs (decadal tables)
        g_A=tfp_g, g_L=pop_g, bau_s_r_t=bau_s_r_t, savings_init=savings_init,
        # slr damages
        d1_slr=c_var_p.loc['d1_slr'].astype(float).reindex(countries),
        d2_slr=c_var_p.loc['d2_slr'].astype(float).reindex(countries),
        slr_multiplier=c_var_p.loc['slr_multiplier'].astype(float).reindex(countries),
        # derived
        L=L, A=A, sigma=sigma, eland=eland, f_ex=f_ex, backstpr=backstpr, theta1=theta1,
        bau_saving_rates=bau_saving_rates,
        # warm starts (matrices)
        U_init=U_init, K_init=K_init, S_init=S_init, I_init=I_init, Q_init=Q_init, Q_0=Q_0, Y_init=Y_init,
        mu_init=mu_init, AB_init=AB_init, D_init=D_init, damage_frac_init=damage_frac_init,
        C_init=C_init, E_ind_init=E_ind_init,
        # warm starts (series)
        E_tot_init=E_tot_init, M_at_init=M_at_init, M_up_init=M_up_init, M_lo_init=M_lo_init,
        T_at_init=T_at_init, T_lo_init=T_lo_init, F_init=F_init, I_0=I_0,
        gsic_remain_init=gsic_remain_init, gsic_melt_init=gsic_melt_init, gsic_cum_init=gsic_cum_init,
        gis_remain_init=gis_remain_init, gis_melt_init=gis_melt_init, gis_cum_init=gis_cum_init,
        ais_remain_init=ais_remain_init, ais_melt_init=ais_melt_init, ais_cum_init=ais_cum_init,
        slr_TE_init=slr_TE_init, slr_init=slr_init,
    )
