"""Utilitas analisis DESAIN RASIO (artikel 2): gain terhadap baseline, kode rasio, ART ANOVA.

File baru; tidak mengubah modul lain di src/.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

from .utils import ratio_sort_key, strategy_meta


def ratio_numeric(splits_dir, design: str) -> dict:
    """tag rasio -> rasio mayoritas:TB yang SEBENARNYA (orig ≈ 6.03 pada TBX11K)."""
    ss = pd.read_csv(splits_dir / "subsets_summary.csv")
    ss = ss[ss["design"] == design]
    return dict(zip(ss["ratio"].astype(str), ss["actual_ratio_major_to_tb"].astype(float)))


def annotate(df: pd.DataFrame, cfg: dict, rnum: dict) -> pd.DataFrame:
    """Tambah kolom ratio (str), ratio_num, log2r (log2 rasio, 0 = rasio orig), method, dose_pct."""
    df = df.copy()
    df["ratio"] = df["ratio"].astype(str)
    df["ratio_num"] = df["ratio"].map(rnum)
    ref = rnum.get("orig", np.nan)
    df["log2r"] = np.log2(df["ratio_num"] / ref)
    sm = strategy_meta(cfg)
    df["method"] = df["strategy"].map(lambda s: sm.get(s, ("other", np.nan))[0])
    dose = df["strategy"].map(lambda s: sm.get(s, ("other", np.nan))[1])
    df["dose_pct"] = (pd.to_numeric(dose, errors="coerce") * 100).round()
    return df


def ratio_order(tags) -> list[str]:
    """orig (±6:1) dulu, lalu rasio makin parah: orig, 10, 20, 50."""
    return sorted({str(t) for t in tags}, key=lambda t: -1.0 if t == "orig" else ratio_sort_key(t))


def compute_gain(df: pd.DataFrame, metrics: list[str], ref: str) -> pd.DataFrame:
    """Δ = metrik strategi − metrik baseline pada (rasio, model, seed) yang SAMA. Hanya pasangan lengkap."""
    keys = ["ratio", "model", "seed"]
    base = df[df["strategy"] == ref].set_index(keys)[metrics]
    out = df[df["strategy"] != ref].copy()
    idx = pd.MultiIndex.from_frame(out[keys])
    for m in metrics:
        b = base[m].reindex(idx).values
        out[f"base_{m}"] = b
        out[f"gain_{m}"] = out[m].values - b
    have = out[[f"base_{m}" for m in metrics]].notna().all(axis=1)
    return out[have].reset_index(drop=True)


def mean_ci(v, conf: float = 0.95) -> tuple[float, float, float, int]:
    v = np.asarray(pd.Series(v).dropna(), float)
    n = len(v)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    m = v.mean()
    if n < 2:
        return m, np.nan, np.nan, n
    h = stats.t.ppf(0.5 + conf / 2, n - 1) * v.std(ddof=1) / np.sqrt(n)
    return m, m - h, m + h, n


# ----------------------------------------------------------------------------- ART ANOVA
def _effect_estimate(d: pd.DataFrame, y: str, effect: tuple[str, ...]) -> np.ndarray:
    """Estimasi efek (main/interaksi) per baris dengan inklusi-eksklusi rata-rata marginal."""
    est = np.zeros(len(d))
    gm = d[y].mean()
    for k in range(len(effect) + 1):
        for sub in itertools.combinations(effect, k):
            sign = (-1) ** (len(effect) - k)
            if not sub:
                est += sign * gm
            else:
                est += sign * d.groupby(list(sub))[y].transform("mean").values
    return est


def art_anova(df: pd.DataFrame, y: str, factors: list[str], effects: list[tuple[str, ...]]) -> pd.DataFrame:
    """Aligned Rank Transform ANOVA (Wobbrock dkk., CHI 2011) untuk desain faktorial seimbang.

    Untuk tiap efek: respons di-align (residu model sel penuh + estimasi efek itu), dirangking,
    lalu ANOVA faktorial penuh pada rangking; hanya baris efek tersebut yang dibaca.
    """
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm

    d = df[[y] + factors].dropna().copy().reset_index(drop=True)
    for f in factors:
        d[f] = d[f].astype(str)
    resid = d[y] - d.groupby(factors)[y].transform("mean")
    alias = {f: f"F{i}" for i, f in enumerate(factors)}
    terms = [f"C({alias[f]}, Sum)" for f in factors]  # kode efek: Type III sahih walau sel tidak seimbang
    rhs = " * ".join(terms)
    rows = []
    for eff in effects:
        dd = d.rename(columns=alias).copy()
        dd["aligned"] = resid.values + _effect_estimate(d, y, eff)
        dd["r"] = stats.rankdata(dd["aligned"])
        fit = smf.ols(f"r ~ {rhs}", data=dd).fit()
        tab = anova_lm(fit, typ=3)
        key = ":".join(f"C({alias[f]}, Sum)" for f in eff)
        row = tab.loc[key]
        ss_res = tab.loc["Residual", "sum_sq"]
        rows.append({"effect": " × ".join(eff), "F": float(row["F"]), "df": int(row["df"]),
                     "df_resid": int(tab.loc["Residual", "df"]), "p_value": float(row["PR(>F)"]),
                     "partial_eta_sq": float(row["sum_sq"] / (row["sum_sq"] + ss_res))})
    return pd.DataFrame(rows)
