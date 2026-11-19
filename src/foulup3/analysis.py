"""Estimating the effect of fouling up 3 on the leader's chance of winning.

The naive comparison (win % when fouling vs not) is confounded: teams that foul are stronger and tend
to do it with less time left, and less time left by itself makes the lead safer. So we climb a ladder:

    naive  ->  + time left  ->  + situation (timeouts, fouls)  ->  + opponent & team quality, era

and report the leader's win-probability difference (foul - no foul) at every rung, with bootstrap CIs.
Two adjusted estimators are used as a cross-check: outcome regression (g-computation) and
inverse-propensity weighting.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy.special import expit

TIME_BINS = [0, 4, 8, 12, 16, 20, 24]
TIME_LABELS = ["0-4s", "4-8s", "8-12s", "12-16s", "16-20s", "20-24s"]

RHS_TIME = "C(t_bin) + t_entry"
RHS_SITUATION = RHS_TIME + " + to_used_leader + to_used_opp + q4_leader_fouls + penalty_leader"
RHS_FULL = (RHS_SITUATION + " + opp_ft_pct_prior + opp_tp_pct_prior + leader_net_pg_prior"
            " + opp_net_pg_prior + home_leader + season_c + po")
LADDER = [("Naive (no adjustment)", None), ("+ time left", RHS_TIME),
          ("+ timeouts & team fouls", RHS_SITUATION), ("+ opponent, team quality, era", RHS_FULL)]


def prepare(states: pd.DataFrame) -> pd.DataFrame:
    df = states.copy()
    df["t_bin"] = pd.cut(df["t_entry"], TIME_BINS, labels=TIME_LABELS, include_lowest=True)
    df["po"] = (df["season_type"] == "PO").astype(int)
    need = ["opp_ft_pct_prior", "opp_tp_pct_prior", "leader_net_pg_prior", "opp_net_pg_prior",
            "to_used_leader", "to_used_opp"]
    return df.dropna(subset=need).reset_index(drop=True)


# ---------- basic intervals ----------
def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def newcombe_rd(k1: float, n1: float, k0: float, n0: float, z: float = 1.96):
    """Newcombe's hybrid-score interval for p1 - p0."""
    p1, p0 = k1 / n1, k0 / n0
    l1, u1 = wilson(k1, n1, z)
    l0, u0 = wilson(k0, n0, z)
    d = p1 - p0
    lo = d - np.sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2)
    hi = d + np.sqrt((u1 - p1) ** 2 + (p0 - l0) ** 2)
    return d, lo, hi


def naive_table(df: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    rows = []
    groups = [("all", df)] if by is None else list(df.groupby(by, observed=True))
    for name, g in groups:
        t, c = g[g["foul"] == 1], g[g["foul"] == 0]
        row = {"group": name, "n_foul": len(t), "n_nofoul": len(c)}
        for lab, sub in (("foul", t), ("nofoul", c)):
            k, n = sub["leader_win"].sum(), len(sub)
            lo, hi = wilson(k, n)
            row |= {f"win_{lab}": k / n if n else np.nan, f"win_{lab}_lo": lo, f"win_{lab}_hi": hi}
        if len(t) and len(c):
            d, lo, hi = newcombe_rd(t["leader_win"].sum(), len(t), c["leader_win"].sum(), len(c))
            row |= {"rd": d, "rd_lo": lo, "rd_hi": hi}
        rows.append(row)
    return pd.DataFrame(rows)


# ---------- adjusted estimators ----------
def _fit(y, X):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sm.GLM(y, X, family=sm.families.Binomial()).fit(maxiter=200)


def gcomp(df: pd.DataFrame, rhs: str, n_boot: int = 1000, seed: int = 0) -> dict:
    """Outcome-regression (g-computation) estimate of P(win | foul) - P(win | no foul), averaged
    over all states. Bootstrap percentile CI. Variables in `rhs` are confounders; `foul` is added."""
    y, X = patsy.dmatrices(f"leader_win ~ foul + {rhs}", df, return_type="dataframe")
    X1 = np.asarray(patsy.build_design_matrices([X.design_info], df.assign(foul=1))[0])
    X0 = np.asarray(patsy.build_design_matrices([X.design_info], df.assign(foul=0))[0])
    Xa, ya = np.asarray(X), y.to_numpy().ravel()

    def est(idx):
        b = _fit(ya[idx], Xa[idx]).params
        return float((expit(X1[idx] @ b) - expit(X0[idx] @ b)).mean())

    point = est(np.arange(len(df)))
    rng = np.random.default_rng(seed)
    boots = np.array([est(rng.integers(0, len(df), len(df))) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"rd": point, "rd_lo": float(lo), "rd_hi": float(hi), "n": len(df)}


def ipw(df: pd.DataFrame, rhs: str, n_boot: int = 1000, seed: int = 0, clip: float = 0.02) -> dict:
    """Inverse-propensity-weighted (Hajek) estimate; propensity = P(foul | confounders)."""
    Xp = np.asarray(patsy.dmatrix(rhs, df))
    z, y = df["foul"].to_numpy(), df["leader_win"].to_numpy()

    def est(idx):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ps = expit(Xp[idx] @ sm.Logit(z[idx], Xp[idx]).fit(disp=0, maxiter=200).params)
        ps = np.clip(ps, clip, 1 - clip)
        w1, w0 = z[idx] / ps, (1 - z[idx]) / (1 - ps)
        return float((w1 * y[idx]).sum() / w1.sum() - (w0 * y[idx]).sum() / w0.sum())

    point = est(np.arange(len(df)))
    rng = np.random.default_rng(seed)
    boots = np.array([est(rng.integers(0, len(df), len(df))) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"rd": point, "rd_lo": float(lo), "rd_hi": float(hi), "n": len(df)}


def ladder(df: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    """Naive -> fully adjusted, so the reader can see where the bias comes from."""
    n = naive_table(df).iloc[0]
    rows = [{"model": LADDER[0][0], "estimator": "difference in means", "rd": n["rd"],
             "rd_lo": n["rd_lo"], "rd_hi": n["rd_hi"]}]
    for name, rhs in LADDER[1:]:
        r = gcomp(df, rhs, n_boot)
        rows.append({"model": name, "estimator": "regression (g-comp)", **r})
    r = ipw(df, RHS_FULL, n_boot)
    rows.append({"model": "+ opponent, team quality, era", "estimator": "inverse-propensity", **r})
    return pd.DataFrame(rows)


def balance(df: pd.DataFrame, covars: list[str]) -> pd.DataFrame:
    """Standardised mean differences (foul vs no foul), raw and after propensity weighting."""
    Xp = np.asarray(patsy.dmatrix(RHS_FULL, df))
    ps = np.clip(expit(Xp @ sm.Logit(df["foul"].to_numpy(), Xp).fit(disp=0, maxiter=200).params),
                 0.02, 0.98)
    z = df["foul"].to_numpy()
    w = z / ps + (1 - z) / (1 - ps)
    rows = []
    for c in covars:
        x = df[c].to_numpy(float)

        def wm(mask, ww):
            return np.average(x[mask], weights=ww[mask])

        def smd(ww):
            m1, m0 = wm(z == 1, ww), wm(z == 0, ww)
            v1 = np.average((x[z == 1] - m1) ** 2, weights=ww[z == 1])
            v0 = np.average((x[z == 0] - m0) ** 2, weights=ww[z == 0])
            return (m1 - m0) / np.sqrt((v1 + v0) / 2)

        rows.append({"covariate": c, "smd_raw": smd(np.ones_like(w)), "smd_weighted": smd(w)})
    return pd.DataFrame(rows)


# ---------- subgroup and mechanism views ----------
def by_time(df: pd.DataFrame) -> pd.DataFrame:
    """Naive foul vs no-foul comparison inside each window of time left at state entry."""
    return naive_table(df, by="t_bin")


def mechanism(df: pd.DataFrame) -> pd.DataFrame:
    """How the game was settled in regulation, by what the leader did."""
    d = df.copy()
    d["group"] = np.select(
        [d["foul"] == 0, (d["foul"] == 1) & (d["ft_att"] >= 1)],
        ["No foul", "Foul: opponent shot free throws"], "Foul: no free throws (not in bonus)")
    tab = pd.crosstab(d["group"], d["reg_outcome"])
    for c in ("win_reg", "tie_to_ot", "lose_reg"):
        if c not in tab:
            tab[c] = 0
    tab["n"] = tab[["win_reg", "tie_to_ot", "lose_reg"]].sum(axis=1)
    tab["win_overall"] = d.groupby("group")["leader_win"].mean()
    for c in ("win_reg", "tie_to_ot", "lose_reg"):
        tab[c + "_pct"] = tab[c] / tab["n"]
    return tab.reset_index()


def by_opp_ft(df: pd.DataFrame, n_boot: int = 300) -> pd.DataFrame:
    """Does fouling work better against poor free-throw teams? Effect within opponent-FT% terciles."""
    d = df.assign(ft_tercile=pd.qcut(df["opp_ft_pct_prior"], 3, labels=["worst FT", "middle", "best FT"]))
    rows = []
    for name, g in d.groupby("ft_tercile", observed=True):
        g = g.reset_index(drop=True)
        n = naive_table(g).iloc[0]
        a = gcomp(g, RHS_TIME + " + leader_net_pg_prior + opp_net_pg_prior", n_boot)
        rows.append({"opp_ft_tercile": name, "ft_pct_range": f"{g.opp_ft_pct_prior.min():.3f}-"
                     f"{g.opp_ft_pct_prior.max():.3f}", "n_foul": int(n["n_foul"]),
                     "n_nofoul": int(n["n_nofoul"]), "naive_rd": n["rd"],
                     "adj_rd": a["rd"], "adj_lo": a["rd_lo"], "adj_hi": a["rd_hi"]})
    return pd.DataFrame(rows)


def by_era(df: pd.DataFrame, n_boot: int = 300) -> pd.DataFrame:
    rows = []
    for name, g in (("1996-2017", df[df.post_2018 == 0]), ("2018-2024", df[df.post_2018 == 1])):
        g = g.reset_index(drop=True)
        n = naive_table(g).iloc[0]
        a = gcomp(g, RHS_TIME + " + leader_net_pg_prior + opp_net_pg_prior", n_boot)
        rows.append({"era": name, "n_foul": int(n["n_foul"]), "n_nofoul": int(n["n_nofoul"]),
                     "foul_rate": n["n_foul"] / (n["n_foul"] + n["n_nofoul"]), "naive_rd": n["rd"],
                     "adj_rd": a["rd"], "adj_lo": a["rd_lo"], "adj_hi": a["rd_hi"]})
    return pd.DataFrame(rows)


def by_penalty(df: pd.DataFrame, n_boot: int = 300) -> pd.DataFrame:
    """Fouling when it sends the opponent to the line (leader already in the penalty) vs when it only
    buys a sideline inbound. Strata are defined by a pre-decision variable, so this is a fair split."""
    rows = []
    for name, g in (("In the penalty (4+ team fouls): foul = free throws", df[df.penalty_leader == 1]),
                    ("Not in the penalty: foul = sideline inbound", df[df.penalty_leader == 0])):
        g = g.reset_index(drop=True)
        n = naive_table(g).iloc[0]
        a = gcomp(g, RHS_TIME + " + leader_net_pg_prior + opp_net_pg_prior", n_boot)
        rows.append({"situation": name, "n_foul": int(n["n_foul"]), "n_nofoul": int(n["n_nofoul"]),
                     "foul_rate": n["n_foul"] / (n["n_foul"] + n["n_nofoul"]), "naive_rd": n["rd"],
                     "adj_rd": a["rd"], "adj_lo": a["rd_lo"], "adj_hi": a["rd_hi"]})
    return pd.DataFrame(rows)


def adoption(df: pd.DataFrame) -> pd.DataFrame:
    """Share of up-3 states in which the leader fouled, by season."""
    out = df.groupby("season").agg(n=("foul", "size"), fouls=("foul", "sum")).reset_index()
    out["rate"] = out["fouls"] / out["n"]
    out[["lo", "hi"]] = [wilson(k, n) for k, n in zip(out["fouls"], out["n"])]
    return out
