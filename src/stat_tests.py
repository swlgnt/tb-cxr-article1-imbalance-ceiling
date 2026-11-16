"""Uji statistik untuk perbandingan model/strategi.

  * DeLong (Sun & Xu, 2014)        : beda AUC dua model pada test set yang sama
  * McNemar (exact/chi2)           : beda proporsi benar dua model (berpasangan)
  * Friedman + Nemenyi (+ CD)      : >2 perlakuan, blok = kondisi berulang (Demšar, 2006)
  * Wilcoxon signed-rank + Holm    : perbandingan berpasangan terencana + effect size r
  * ANOVA faktorial (Type II)      : efek model × strategi × rasio + partial eta squared
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats


# ----------------------------------------------------------------------------- DeLong
def _midrank(x):
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n)
    out[j] = t
    return out


def _fast_delong(preds_sorted: np.ndarray, m: int):
    """preds_sorted: (k, n) dengan m positif di depan."""
    n = preds_sorted.shape[1] - m
    pos, neg = preds_sorted[:, :m], preds_sorted[:, m:]
    k = preds_sorted.shape[0]
    tx = np.array([_midrank(pos[r]) for r in range(k)])
    ty = np.array([_midrank(neg[r]) for r in range(k)])
    tz = np.array([_midrank(preds_sorted[r]) for r in range(k)])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    return aucs, sx / m + sy / n


def delong_test(y_true_bin, prob_a, prob_b) -> dict:
    """Uji DeLong dua sisi untuk AUC berpasangan (kelas positif = 1)."""
    y = np.asarray(y_true_bin).astype(int)
    order = np.argsort(-y, kind="mergesort")
    m = int(y.sum())
    preds = np.vstack([np.asarray(prob_a)[order], np.asarray(prob_b)[order]])
    aucs, cov = _fast_delong(preds, m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    p = 2 * stats.norm.sf(abs(z))
    se_a, se_b = np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "auc_diff": float(aucs[0] - aucs[1]),
            "auc_a_ci95": [float(aucs[0] - 1.96 * se_a), float(aucs[0] + 1.96 * se_a)],
            "auc_b_ci95": [float(aucs[1] - 1.96 * se_b), float(aucs[1] + 1.96 * se_b)],
            "z": float(z), "p_value": float(p)}


# ----------------------------------------------------------------------------- McNemar
def mcnemar_test(y_true, pred_a, pred_b, exact_threshold: int = 25) -> dict:
    y = np.asarray(y_true)
    ca, cb = np.asarray(pred_a) == y, np.asarray(pred_b) == y
    b = int(np.sum(ca & ~cb))   # A benar, B salah
    c = int(np.sum(~ca & cb))   # A salah, B benar
    if b + c == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0, "method": "none"}
    if b + c < exact_threshold:
        p = stats.binomtest(min(b, c), b + c, 0.5).pvalue
        return {"b": b, "c": c, "statistic": float(min(b, c)), "p_value": float(p), "method": "exact"}
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return {"b": b, "c": c, "statistic": float(chi2), "p_value": float(stats.chi2.sf(chi2, 1)),
            "method": "chi2_cc"}


# ----------------------------------------------------------------------------- koreksi
def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (n - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


# ----------------------------------------------------------------------------- Friedman/Nemenyi
def friedman_nemenyi(df: pd.DataFrame, block_cols: list[str], treat_col: str, value_col: str,
                     higher_is_better: bool = True, alpha: float = 0.05) -> dict:
    """df long-format. Blok = kombinasi block_cols (mis. ratio × strategy × seed)."""
    wide = df.pivot_table(index=block_cols, columns=treat_col, values=value_col, aggfunc="mean").dropna()
    k, n = wide.shape[1], wide.shape[0]
    if k < 2 or n < 2:
        return {"error": f"data kurang (k={k}, n_blok={n})"}
    vals = wide.values if higher_is_better else -wide.values
    ranks = np.apply_along_axis(lambda r: stats.rankdata(-r), 1, vals)  # rank 1 = terbaik
    avg_rank = pd.Series(ranks.mean(0), index=wide.columns).sort_values()
    if k >= 3:
        stat, p = stats.friedmanchisquare(*[wide[c].values for c in wide.columns])
        test = "friedman"
    else:  # hanya 2 perlakuan -> Friedman tidak tepat, gunakan Wilcoxon
        d = wide.iloc[:, 0] - wide.iloc[:, 1]
        stat, p = (stats.wilcoxon(wide.iloc[:, 0], wide.iloc[:, 1]) if (d != 0).any() else (0.0, 1.0))
        test = "wilcoxon"
    kendall_w = stat / (n * (k - 1)) if k >= 3 else np.nan
    se = np.sqrt(k * (k + 1) / (6.0 * n))
    q_alpha = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = q_alpha * se
    rows = []
    for a, b in itertools.combinations(avg_rank.index, 2):
        diff = abs(avg_rank[a] - avg_rank[b])
        p_nem = float(stats.studentized_range.sf(diff / se * np.sqrt(2), k, np.inf))
        rows.append({"a": a, "b": b, "rank_a": avg_rank[a], "rank_b": avg_rank[b], "rank_diff": diff,
                     "p_nemenyi": min(1.0, p_nem), "significant": diff > cd})
    return {"metric": value_col, "treatment": treat_col, "test": test, "k": k, "n_blocks": n,
            "friedman_stat": float(stat), "p_value": float(p), "kendall_w": float(kendall_w),
            "critical_difference": float(cd), "avg_rank": avg_rank.to_dict(),
            "pairwise": pd.DataFrame(rows), "means": wide.mean().to_dict()}


# ----------------------------------------------------------------------------- Wilcoxon
def paired_wilcoxon(df: pd.DataFrame, pair_cols: list[str], treat_col: str, value_col: str,
                    reference: str, group_cols: list[str] | None = None) -> pd.DataFrame:
    """Bandingkan tiap perlakuan vs referensi (mis. tiap strategi vs baseline),
    dipasangkan pada pair_cols, terpisah per group_cols (mis. per model). Koreksi Holm."""
    group_cols = group_cols or []
    out = []
    groups = df.groupby(group_cols) if group_cols else [((), df)]
    for gkey, g in groups:
        wide = g.pivot_table(index=pair_cols, columns=treat_col, values=value_col, aggfunc="mean")
        if reference not in wide:
            continue
        for t in [c for c in wide.columns if c != reference]:
            d = wide[[reference, t]].dropna()
            diff = d[t] - d[reference]
            row = dict(zip(group_cols, gkey if isinstance(gkey, tuple) else (gkey,)))
            row.update({"treatment": t, "reference": reference, "metric": value_col, "n_pairs": len(d),
                        "mean_ref": d[reference].mean(), "mean_treat": d[t].mean(),
                        "mean_diff": diff.mean(), "median_diff": diff.median()})
            if len(d) >= 2 and (diff != 0).any():
                res = stats.wilcoxon(d[t], d[reference], zero_method="wilcox", alternative="two-sided")
                n_nz = int((diff != 0).sum())
                # matched-pairs rank-biserial correlation sebagai effect size
                r_pos = stats.rankdata(np.abs(diff[diff != 0]))
                rb = (r_pos[diff[diff != 0] > 0].sum() - r_pos[diff[diff != 0] < 0].sum()) / r_pos.sum()
                row.update({"W": float(res.statistic), "p_value": float(res.pvalue), "n_nonzero": n_nz,
                            "rank_biserial": float(rb)})
            else:
                row.update({"W": np.nan, "p_value": 1.0, "n_nonzero": 0, "rank_biserial": 0.0})
            out.append(row)
    res = pd.DataFrame(out)
    if len(res):
        res["p_holm"] = holm(res["p_value"].values)
        res["significant_holm"] = res["p_holm"] < 0.05
    return res


# ----------------------------------------------------------------------------- ANOVA
def factorial_anova(df: pd.DataFrame, value_col: str, factors: list[str], max_order: int = 2) -> pd.DataFrame:
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    d = df[[value_col] + factors].dropna().copy()
    d.columns = ["y"] + [f"f_{f}" for f in factors]
    terms = [f"C(f_{f})" for f in factors]
    rhs = []
    for order in range(1, max_order + 1):
        rhs += [":".join(c) for c in itertools.combinations(terms, order)]
    model = smf.ols("y ~ " + " + ".join(rhs), data=d).fit()
    tab = anova_lm(model, typ=2)
    ss_res = tab.loc["Residual", "sum_sq"]
    tab["partial_eta_sq"] = tab["sum_sq"] / (tab["sum_sq"] + ss_res)
    tab.loc["Residual", "partial_eta_sq"] = np.nan
    tab.index = [i.replace("C(f_", "").replace(")", "") for i in tab.index]
    tab.insert(0, "metric", value_col)
    tab.attrs["r_squared"] = model.rsquared
    return tab.reset_index().rename(columns={"index": "term"})
