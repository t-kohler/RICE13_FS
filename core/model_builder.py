"""
RICE13_FS — Model builder
-------------------------
Constructs a Pyomo ConcreteModel for the integrated assessment model (IAM),
using preprocessed inputs provided via a `Params` container.

Core design (2026 revision)
---------------------------
• **Time grid** — fixed 10-year periods (decadal). No subdecadal support.
  `_TSTEP=10` is baked into all capital, emissions, and climate recursions.
• **Sea-level rise (SLR)** — always endogenous. The solver warm-starts from
  `params` initial conditions but never accepts an exogenous SLR path.
• **Discounting** — each run can override the geometric discount path by
  passing an optional `discount_series`, global or regional.
• **Utility weighting** —
    – CRRA utility is population-weighted by construction.  
    – FS utility is also population-weighted within each region.  
      The flag `population_weight_envy_guilt` only controls how envy/guilt
      are aggregated *across* regions (population-weighted vs. simple mean).
• **FS smoothing** — the smoothed-positive (`spos_sqrt`) avoids IPOPT
  non-differentiability in envy/guilt terms.
• **Climate module** — thermal expansion, GSIC, GIS, and AIS each follow
  decadal update equations; AIS melt transitions smoothly at ~3 °C via
  a logistic gate (IPOPT-friendly, no kinks).

Return value: a fully initialized, feasible `pe.ConcreteModel`.
"""
from typing import Optional, Mapping, Tuple, Any
import pandas as pd
import pyomo.environ as pe
from RICE13_FS.common.utils import normalize_exogenous_S

# Fixed decadal step (years per period)
_TSTEP = 10


def build_model(
    params,
    T: int,
    utility: str,
    exogenous_S: Optional[pd.DataFrame],
    population_weight_envy_guilt: bool,
    *,
    # Optional discount series. Supports two shapes:
    #   (A) Global per-period: {t -> d_t}
    #   (B) Regional per-period: {(region, t) -> d_{r,t}}
    # When provided, overrides the geometric path from params.rho.
    discount_series: Optional[Mapping[Any, float]] = None,
) -> pe.ConcreteModel:
    """
    Build and return a Pyomo ConcreteModel for the RICE-2013 IAM (decadal grid).

    Parameters
    ----------
    params : Params
        Preprocessed parameter and timeseries container (from data_loader)
    T : int
        Number of modeled periods (1..T) where 0 is the base in params
    utility : {'crra','fs'}
        Utility specification to attach
    exogenous_S : pd.DataFrame or None
        If provided, index=(region,period) and fixes m.S[r,t] to those values
    population_weight_envy_guilt : bool
        If True, envy/guilt are population-weighted shares across other regions;
        else simple averages across other regions.
    discount_series : Mapping or None, optional (keyword-only)
        If given, initializes the discount factor Param `m.disc`.
        Supported shapes:
          - Global: {t → d_t} applied to all regions.
          - Regional: {(region, t) → d_{r,t}} applied per region and period.
        If None (default), `m.disc` is geometric from per-region rho in params.

    Returns
    -------
    m : pe.ConcreteModel
    """
    m = pe.ConcreteModel()

    m.dual = pe.Suffix(direction=pe.Suffix.IMPORT)

    # --- Sets ---
    m.T = pe.RangeSet(1, T)  # Model periods start at 1
    countries = list(params.countries)
    m.REGIONS = pe.Set(initialize=countries, doc="country codes")

    # --- Small helpers for initial values / bounds ---
    def df_init(df):
        return lambda model, r, t: float(df.at[r, t])

    def ser_init(s):
        return lambda model, t: float(s.at[t])

    def savings_bounds(model, region, t):
        # If S is EXOGENOUS, use wide bounds [0,1] everywhere so fixed values
        # never trip “fixed value not within bounds” warnings (e.g., 0.0 in interior).
        if exogenous_S is not None:
            return (0.0, 1.0)
        # Endogenous S: small positive LB in interior periods; final period can be 0.0.
        return (1e-2, 1.0) if t != model.T.last() else (0.0, 1.0)

    # --- Variables (economic core) ---
    m.K     = pe.Var(m.REGIONS, m.T, within=pe.NonNegativeReals, bounds=(1, None), initialize=df_init(params.K_init))
    m.S     = pe.Var(m.REGIONS, m.T, within=pe.NonNegativeReals, bounds=savings_bounds, initialize=df_init(params.S_init))
    m.I     = pe.Var(m.REGIONS, m.T, within=pe.NonNegativeReals, initialize=df_init(params.I_init))
    m.Q     = pe.Var(m.REGIONS, m.T, within=pe.NonNegativeReals, initialize=df_init(params.Q_init))
    m.Y     = pe.Var(m.REGIONS, m.T, within=pe.NonNegativeReals, initialize=df_init(params.Y_init))
    m.mu    = pe.Var(m.REGIONS, m.T, within=pe.NonNegativeReals, bounds=(1e-8, 1.0), initialize=df_init(params.mu_init))
    m.AB    = pe.Var(m.REGIONS, m.T, within=pe.NonNegativeReals, initialize=df_init(params.AB_init))
    m.D     = pe.Var(m.REGIONS, m.T, within=pe.NonNegativeReals, initialize=df_init(params.D_init))
    m.C     = pe.Var(m.REGIONS, m.T, within=pe.NonNegativeReals, bounds=(1e-2, None), initialize=df_init(params.C_init))
    m.E_ind = pe.Var(m.REGIONS, m.T, within=pe.NonNegativeReals, initialize=df_init(params.E_ind_init))

    # Hard-fix last-period savings to 0 if S is endogenous (prevents tiny machine remnants).
    # If S is exogenous, defer to the provided schedule below.
    if exogenous_S is None:
        t_last = m.T.last()
        for r in m.REGIONS:
            m.S[r, t_last].fix(0.0)

    # --- Aggregate emissions and climate cycle ---
    m.E_tot = pe.Var(m.T, within=pe.NonNegativeReals, initialize=ser_init(params.E_tot_init))
    m.M_at  = pe.Var(m.T, within=pe.Reals, bounds=(0.001, None), initialize=ser_init(params.M_at_init))
    m.M_up  = pe.Var(m.T, within=pe.Reals, initialize=ser_init(params.M_up_init))
    m.M_lo  = pe.Var(m.T, within=pe.Reals, initialize=ser_init(params.M_lo_init))
    m.T_at  = pe.Var(m.T, within=pe.Reals, bounds=(0.5, 8), initialize=ser_init(params.T_at_init))
    m.T_lo  = pe.Var(m.T, within=pe.Reals, bounds=(0, 30), initialize=ser_init(params.T_lo_init))
    m.F     = pe.Var(m.T, within=pe.Reals, initialize=ser_init(params.F_init))

    # --- Sea Level Rise (endogenous, warm-started) ---
    # Thermal expansion
    m.slr_TE = pe.Var(m.T, within=pe.NonNegativeReals, initialize=ser_init(params.slr_TE_init))
    # Glaciers & small ice caps (GSIC)
    m.gsic_remain = pe.Var(m.T, initialize=ser_init(params.gsic_remain_init))
    m.gsic_melt   = pe.Var(m.T, initialize=ser_init(params.gsic_melt_init))
    m.gsic_cum    = pe.Var(m.T, within=pe.NonNegativeReals, initialize=ser_init(params.gsic_cum_init), bounds=(0, params.tot_ice_gsic))
    # Greenland (GIS)
    m.gis_remain = pe.Var(m.T, within=pe.NonNegativeReals, initialize=ser_init(params.gis_remain_init))
    m.gis_melt   = pe.Var(m.T, within=pe.NonNegativeReals, initialize=ser_init(params.gis_melt_init))
    m.gis_cum    = pe.Var(m.T, within=pe.NonNegativeReals, initialize=ser_init(params.gis_cum_init), bounds=(0, params.init_vol_gis))
    # Antarctic (AIS)
    m.ais_remain = pe.Var(m.T, within=pe.Reals, initialize=ser_init(params.ais_remain_init))
    m.ais_melt   = pe.Var(m.T, within=pe.Reals, initialize=ser_init(params.ais_melt_init))
    m.ais_cum    = pe.Var(m.T, within=pe.Reals, initialize=ser_init(params.ais_cum_init), bounds=(-1, params.tot_vol_ais))
    # Total SLR
    m.slr = pe.Var(m.T, within=pe.NonNegativeReals, initialize=ser_init(params.slr_init))
    # NOTE: no Param-based exogenous SLR at runtime; endogenous only.

    # --- Population as a Param (country x period) ---
    def pop_init(model, r, t):
        return params.L.loc[r, t]
    m.L = pe.Param(m.REGIONS, m.T, initialize=pop_init, mutable=False)

    # --- Fix S if exogenous schedule provided ---
    if exogenous_S is not None:
        # Normalize to (regions x periods 1..T, integer columns)
        exoS = normalize_exogenous_S(exogenous_S, countries, T)
        for r in m.REGIONS:
            for t in m.T:
                # Clamp defensively into [0,1] before fixing to avoid bound violations.
                # This ensures clean IPOPT initialization even for planner files
                # that contain minor rounding drift.
                v = float(exoS.at[r, t])
                if v < 0.0:
                    v = 0.0
                if v > 1.0:
                    v = 1.0
                m.S[r, t].fix(v)

    # --- Utility specification ---
    # Per-(r,t) discount factor Param so all solvers use m.disc in objectives.
    # The discount path may come from:
    #   (a) discount_series provided by caller (global or regional keys), or
    #   (b) geometric path from params.rho (default).
    def _disc_init(model, r, t):
        # If a discount series is supplied, allow either {(r,t)->d} or {t->d}.
        if discount_series is not None:
            # Try regional key first
            key_rt: Tuple[str, int] = (str(r), int(t))
            v = None
            # Mapping supports .get; guard for both key shapes.
            v = discount_series.get(key_rt) if hasattr(discount_series, "get") else None
            if v is None:
                v = discount_series.get(int(t)) if hasattr(discount_series, "get") else None
            if v is not None:
                return float(v)
        # Fallback: geometric path from per-region rho (decadal step).
        return 1.0 / ((1.0 + float(params.rho[r])) ** (_TSTEP * int(t)))
    m.disc = pe.Param(m.REGIONS, m.T, initialize=_disc_init, mutable=False)

    if utility == 'crra':
        def U_crra_rule(model, r, t):
            # Per-capita consumption in 'thousands'
            c_pc_th = (model.C[r, t] / model.L[r, t]) * 1000.0
            # Use per-region curvature
            eta_r = float(params.crra_eta[r])
            # Population-weighted CRRA: L * u(c_pc)
            return model.L[r, t] * ((c_pc_th ** (1 - eta_r)) / (1 - eta_r) + 1.0)
        m.U = pe.Expression(m.REGIONS, m.T, rule=U_crra_rule)

    elif utility == 'fs':
        # --- Smooth positive part: IPOPT-friendly surrogate for max(x, 0) ---
        # FS block uses per-capita consumption in "thousands", so eps must be
        # expressed in those units (1e-4 ≈ negligible relative to 1 k $ scale).
        def spos_sqrt(x, eps=1e-4):
            return 0.5 * (x + pe.sqrt(x**2 + eps**2))

        # --- Average envy/guilt as Expressions (no aux Vars, no ≥ constraints) ---
        # These use smoothed differences in per-capita consumption between regions.
        # If population_weight_envy_guilt=True, weights ∝ L[s,t]; else simple mean.
        def envy_avg_rule(model, r, t):
            N = len(model.REGIONS)
            def cpc(u): return (model.C[u, t] / model.L[u, t]) * 1000.0
            if population_weight_envy_guilt:
                denom = sum(model.L[s, t] for s in model.REGIONS if s != r)
                num   = sum(model.L[s, t] * spos_sqrt(cpc(s) - cpc(r)) for s in model.REGIONS if s != r)
                return num / denom
            else:
                return sum(spos_sqrt(cpc(s) - cpc(r)) for s in model.REGIONS if s != r) / (N - 1)

        def guilt_avg_rule(model, r, t):
            N = len(model.REGIONS)
            def cpc(u): return (model.C[u, t] / model.L[u, t]) * 1000.0
            if population_weight_envy_guilt:
                denom = sum(model.L[s, t] for s in model.REGIONS if s != r)
                num   = sum(model.L[s, t] * spos_sqrt(cpc(r) - cpc(s)) for s in model.REGIONS if s != r)
                return num / denom
            else:
                return sum(spos_sqrt(cpc(r) - cpc(s)) for s in model.REGIONS if s != r) / (N - 1)

        m.FS_envy_avg  = pe.Expression(m.REGIONS, m.T, rule=envy_avg_rule)
        m.FS_guilt_avg = pe.Expression(m.REGIONS, m.T, rule=guilt_avg_rule)

        # --- FS utility, now using the smoothed averages ---
        def FS_utility(model, r, t):
            alpha = params.FS_alpha[r]
            beta  = params.FS_beta[r]
            payoff_pc = (model.C[r, t] / model.L[r, t]) * 1000.0
            return model.L[r, t] * (payoff_pc - alpha * model.FS_envy_avg[r, t] - beta * model.FS_guilt_avg[r, t])

        m.U = pe.Expression(m.REGIONS, m.T, rule=FS_utility)

    else:
        raise ValueError(f"Unknown utility type '{utility}'")

    # --- Core economic constraints ---
    def K_eq(model, r, t):
        if t == 1:
            return model.K[r, t] == _TSTEP * params.I_0[r] + params.K_0[r] * (1 - params.d_k[r]) ** _TSTEP
        return model.K[r, t] == _TSTEP * model.I[r, t - 1] + model.K[r, t - 1] * (1 - params.d_k[r]) ** _TSTEP
    m.K_eq = pe.Constraint(m.REGIONS, m.T, rule=K_eq)

    m.I_eq = pe.Constraint(m.REGIONS, m.T, rule=lambda model, r, t: model.I[r, t] == model.S[r, t] * model.Y[r, t])

    def Q_eq(model, r, t):
        return model.Q[r, t] == params.A.at[r, t] * model.K[r, t] ** params.gamma[r] * (model.L[r, t] / 1000.0) ** (1 - params.gamma[r])
    m.Q_eq = pe.Constraint(m.REGIONS, m.T, rule=Q_eq)

    def Y_eq(model, r, t):
        return model.Y[r, t] == model.Q[r, t] - model.D[r, t] - model.AB[r, t]
    m.Y_eq = pe.Constraint(m.REGIONS, m.T, rule=Y_eq)

    def AB_eq(model, r, t):
        return model.AB[r, t] == params.theta1.at[r, t] * model.mu[r, t] ** params.th_2[r] * model.Q[r, t]
    m.AB_eq = pe.Constraint(m.REGIONS, m.T, rule=AB_eq)

    def C_eq(model, r, t):
        return model.C[r, t] == model.Y[r, t] - model.I[r, t]
    m.C_eq = pe.Constraint(m.REGIONS, m.T, rule=C_eq)

    def E_ind_eq(model, r, t):
        return model.E_ind[r, t] == params.sigma.at[r, t] * ((1 - model.mu[r, t]) * model.Q[r, t])
    m.E_ind_eq = pe.Constraint(m.REGIONS, m.T, rule=E_ind_eq)

    def E_tot_eq(model, t):
        return model.E_tot[t] == sum(model.E_ind[r, t] + params.eland.at[r, t] for r in model.REGIONS)
    m.E_tot_eq = pe.Constraint(m.T, rule=E_tot_eq)

    # --- Carbon boxes (three-box) ---
    def M_at_eq(model, t):
        if t == 1:
            return model.M_at[t] == sum(params.E_0[r] + params.eland_0[r] for r in model.REGIONS) * _TSTEP \
                                   + params.b11 * params.M_at_0 + params.b21 * params.M_up_0
        return model.M_at[t] == model.E_tot[t - 1] * _TSTEP + params.b11 * model.M_at[t - 1] + params.b21 * model.M_up[t - 1]
    m.M_at_eq = pe.Constraint(m.T, rule=M_at_eq)

    def M_up_eq(model, t):
        if t == 1:
            return model.M_up[t] == params.b12 * params.M_at_0 + params.b22 * params.M_up_0 + params.b32 * params.M_lo_0
        return model.M_up[t] == params.b12 * model.M_at[t - 1] + params.b22 * model.M_up[t - 1] + params.b32 * model.M_lo[t - 1]
    m.M_up_eq = pe.Constraint(m.T, rule=M_up_eq)

    def M_lo_eq(model, t):
        if t == 1:
            return model.M_lo[t] == params.b23 * params.M_up_0 + params.b33 * params.M_lo_0
        return model.M_lo[t] == params.b23 * model.M_up[t - 1] + params.b33 * model.M_lo[t - 1]
    m.M_lo_eq = pe.Constraint(m.T, rule=M_lo_eq)

    # --- Temperature boxes ---
    def T_at_eq(model, t):
        if t == 1:
            return model.T_at[t] == params.T_at_15
        return model.T_at[t] == model.T_at[t - 1] + params.c1 * (model.F[t] - params.c2 * model.T_at[t - 1] - params.c3 * (model.T_at[t - 1] - model.T_lo[t - 1]))
    m.T_at_eq = pe.Constraint(m.T, rule=T_at_eq)

    def T_lo_eq(model, t):
        if t == 1:
            return model.T_lo[t] == params.T_lo_0 + params.c4 * (params.T_at_05 - params.T_lo_0)
        return model.T_lo[t] == model.T_lo[t - 1] + params.c4 * (model.T_at[t - 1] - model.T_lo[t - 1])
    m.T_lo_eq = pe.Constraint(m.T, rule=T_lo_eq)

    def F_eq(model, t):
        # log_2(x) = ln(x)/ln(2)
        return model.F[t] == params.eta * pe.log(model.M_at[t] / params.M_1900) / pe.log(2) + params.f_ex[t]
    m.F_eq = pe.Constraint(m.T, rule=F_eq)

    # --- Sea Level Rise module ---
    # 1) Thermal expansion (Excel logic for 2005 → period 1 warm-start)
    slr_TE_0 = params.init_slr_therm + params.adj_rate_therm * (params.eq_therm * params.T_at_05 - params.init_slr_therm)

    def SLR_TE_rule(m, t):
        if t == m.T.first():
            # 2015: start from slr_TE_0 and update with current T_at
            return m.slr_TE[t] == slr_TE_0 + params.adj_rate_therm * (params.eq_therm * m.T_at[t] - slr_TE_0)
        return m.slr_TE[t] == m.slr_TE[t - 1] + params.adj_rate_therm * (params.eq_therm * m.T_at[t] - m.slr_TE[t - 1])
    m.SLR_TE_eq = pe.Constraint(m.T, rule=SLR_TE_rule)

    # 2) GSIC (scientific formula in all periods; no Excel bug)
    ice_gsic_remain_05 = params.tot_ice_gsic
    gsic_melt_05 = params.melt_rate_gsic * _TSTEP * (ice_gsic_remain_05 / params.tot_ice_gsic) ** params.exp_gsic * (params.T_at_05 - params.eq_temp_gsic)
    gsic_cum_05 = gsic_melt_05

    def GSIC_remain_rule(m, t):
        if t == m.T.first():
            return m.gsic_remain[t] == ice_gsic_remain_05 - gsic_cum_05
        return m.gsic_remain[t] == params.tot_ice_gsic - m.gsic_cum[t - 1]
    m.GSIC_remain = pe.Constraint(m.T, rule=GSIC_remain_rule)

    def GSIC_melt_rule(m, t):
        rem_frac = m.gsic_remain[t] / params.tot_ice_gsic
        temp_term = m.T_at[t] - params.eq_temp_gsic
        return m.gsic_melt[t] == params.melt_rate_gsic * _TSTEP * rem_frac ** params.exp_gsic * temp_term
    m.GSIC_melt = pe.Constraint(m.T, rule=GSIC_melt_rule)

    def GSIC_cum_rule(m, t):
        if t == m.T.first():
            return m.gsic_cum[t] == gsic_cum_05 + m.gsic_melt[t]
        return m.gsic_cum[t] == m.gsic_cum[t - 1] + m.gsic_melt[t]
    m.GSIC_cum = pe.Constraint(m.T, rule=GSIC_cum_rule)

    # 3) Greenland Ice Sheet (Excel-consistent warm-start)
    gis_cum_0 = params.init_melt_gis / 100.0
    gis_remain_0 = params.init_vol_gis - gis_cum_0

    def GIS_remain_rule(m, t):
        if t == m.T.first():
            return m.gis_remain[t] == gis_remain_0
        return m.gis_remain[t] == m.gis_remain[t - 1] - m.gis_melt[t - 1] / 100.0
    m.GIS_remain = pe.Constraint(m.T, rule=GIS_remain_rule)

    def GIS_melt_rule(m, t):
        if t == m.T.first():
            return m.gis_melt[t] == params.init_melt_gis
        therm = params.melt_above_gis * (m.T_at[t] - params.min_eq_temp_gis)
        expo  = 1 - (m.gis_cum[t - 1] / params.init_vol_gis) ** params.exp_on_remain_gis
        return m.gis_melt[t] == (therm + params.init_melt_gis) * expo
    m.GIS_melt = pe.Constraint(m.T, rule=GIS_melt_rule)

    def GIS_cum_rule(m, t):
        if t == m.T.first():
            return m.gis_cum[t] == gis_cum_0 + m.gis_melt[t] / 100.0
        return m.gis_cum[t] == m.gis_cum[t - 1] + m.gis_melt[t] / 100.0
    m.GIS_cum = pe.Constraint(m.T, rule=GIS_cum_rule)

    # 4) Antarctic Ice Sheet — smooth blend at 3°C (no kinks)
    def logistic_gate(x, x0=3.0, k=12.0):
        """
        Smooth step: ~0 for x << x0, ~1 for x >> x0.
        k controls steepness (10–20 is typically good for IPOPT).
        """
        return 1.0 / (1.0 + pe.exp(-k * (x - x0)))

    # Warm-start (Excel logic for 2005 → period 1)
    if params.T_at_05 < 3:
        ais_melt_05 = params.melt_low_ais * params.T_at_05 * params.ratio_Tant_Tglob + params.intercept_ais
    else:
        ais_melt_05 = params.infl_pt_ais * params.melt_low_ais + params.melt_up_ais * (params.T_at_05 - 3.0) + params.intercept_ais
    ais_cum_05 = ais_melt_05 / 100.0

    def AIS_remain_rule(m, t):
        return m.ais_remain[t] == params.tot_vol_ais - m.ais_cum[t]
    m.AIS_remain = pe.Constraint(m.T, rule=AIS_remain_rule)

    def AIS_cum_rule(m, t):
        if t == m.T.first():
            return m.ais_cum[t] == ais_cum_05 + m.ais_melt[t] / 100.0
        return m.ais_cum[t] == m.ais_cum[t - 1] + m.ais_melt[t] / 100.0
    m.AIS_cum = pe.Constraint(m.T, rule=AIS_cum_rule)

    def AIS_melt_rule(m, t):
        """
        low_T(x)  = intercept + melt_low_ais * x * ratio_Tant_Tglob
        high_T(x) = intercept + infl_pt_ais * melt_low_ais + melt_up_ais * (x - 3)
        melt(x)   = (1-g)*low_T + g*high_T, g = logistic_gate(x; x0=3, k=12)
        """
        x = m.T_at[t]
        g = logistic_gate(x, x0=3.0, k=12.0)
        low  = params.intercept_ais + params.melt_low_ais * x * params.ratio_Tant_Tglob
        high = params.intercept_ais + params.infl_pt_ais * params.melt_low_ais + params.melt_up_ais * (x - 3.0)
        return m.ais_melt[t] == (1 - g) * low + g * high
    m.AIS_melt = pe.Constraint(m.T, rule=AIS_melt_rule)

    # 5) Total SLR aggregation
    def SLR_eq(m, t):
        return m.slr[t] == m.slr_TE[t] + m.gsic_cum[t] + m.gis_cum[t] + m.ais_cum[t]
    m.SLR_eq = pe.Constraint(m.T, rule=SLR_eq)

    # 6) Damages (non-SLR + SLR), with elasticity and saturation
    def D_eq(m, mC, t):
        slr_mult   = params.slr_multiplier[mC]
        cat_coeff  = params.catast_coeff[mC]
        cat_thresh = params.cat_t[mC]
        cat_exp    = params.d_3_cat[mC]
        slr_eps    = params.slr_eps
        omega_exp  = params.omega_exp

        temp = m.T_at[t]
        d1, d2, d3 = params.d1[mC], params.d2[mC], params.d3[mC]
        cat_term = cat_coeff * (temp / cat_thresh) ** cat_exp
        non_slr = (d1 * temp + d2 * temp ** d3 + cat_term) * 0.01

        # use SLR level from previous period (except first)
        slr_level = m.slr[t] if t == m.T.first() else m.slr[t - 1]
        slr_lin  = params.d1_slr[mC] * slr_level
        slr_quad = params.d2_slr[mC] * slr_level ** 2

        if t == m.T.first():
            slr_frac = 1.0
        else:
            Q_prev = m.Q[mC, t - 1]
            Q0     = params.Q_0.loc[mC]
            slr_frac = (Q_prev / Q0) ** (1 / slr_eps)

        slr_pct = 100.0 * slr_mult * (slr_lin + slr_quad) * slr_frac
        frac = non_slr + slr_pct / 100.0
        return m.D[mC, t] == m.Q[mC, t] * frac / (1.0 + frac ** omega_exp)
    m.D_eq = pe.Constraint(m.REGIONS, m.T, rule=D_eq)

    return m
