#!/usr/bin/env python
"""Langkah 12: linear mixed model (robustness check for dependence among runs).

Model per metric (in percentage points):  y ~ condition + architecture, random intercept per architecture-seed family.
Condition effect: likelihood-ratio test between maximum-likelihood fits with and without condition (df = 8).
Condition-vs-baseline coefficients (REML fit) are Holm-corrected within each metric;
GAN - classical averaged over the three doses is reported as a linear contrast.

  python 12_mixed_model.py --runs results/main/tables/all_runs.csv --out results/main/stats/mixed_model.csv
"""
import argparse
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
CONDS = ["baseline", "weighted_ce", "focal", "classic_gap30", "classic_gap50", "classic_gap100",
         "gan_gap30", "gan_gap50", "gan_gap100"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/main/tables/all_runs.csv")
    ap.add_argument("--out", default="results/main/stats/mixed_model.csv")
    a = ap.parse_args()
    df = pd.read_csv(a.runs)
    df = df[df.strategy.isin(CONDS)].copy()
    df["cond"] = pd.Categorical(df.strategy, categories=CONDS)
    df["family"] = df.model + "_" + df.seed.astype(str)
    rows = []
    for y in ("macro_f1", "tb_recall"):
        d = df.copy(); d[y] = d[y] * 100
        fit = lambda f, reml: smf.mixedlm(f, d, groups=d.family).fit(reml=reml, method="powell")
        full, red = fit(f"{y} ~ cond + C(model)", False), fit(f"{y} ~ C(model)", False)
        lr = 2 * (full.llf - red.llf)
        p_lr = stats.chi2.sf(lr, len(CONDS) - 1)
        m = fit(f"{y} ~ cond + C(model)", True)
        names = [f"cond[T.{c}]" for c in CONDS[1:]]
        p_holm = multipletests(m.pvalues[names], method="holm")[1]
        ci = m.conf_int()
        for c, n, ph in zip(CONDS[1:], names, p_holm):
            rows.append(dict(metric=y, term=f"{c} - baseline", estimate_pp=m.params[n], ci_lo=ci.loc[n, 0],
                             ci_hi=ci.loc[n, 1], p=m.pvalues[n], p_holm=ph, lrt_chi2=lr, lrt_df=len(CONDS) - 1,
                             lrt_p=p_lr))
        # GAN - classical averaged over doses
        L = np.zeros(len(m.fe_params))
        for dose in (30, 50, 100):
            L[list(m.fe_params.index).index(f"cond[T.gan_gap{dose}]")] += 1 / 3
            L[list(m.fe_params.index).index(f"cond[T.classic_gap{dose}]")] -= 1 / 3
        t = m.t_test(L[None, :])
        lo, hi = np.asarray(t.conf_int()).ravel()
        rows.append(dict(metric=y, term="GAN - classical (mean over doses)", estimate_pp=float(np.squeeze(t.effect)),
                         ci_lo=lo, ci_hi=hi, p=float(np.squeeze(t.pvalue)), lrt_chi2=lr, lrt_df=len(CONDS) - 1,
                         lrt_p=p_lr))
        print(f"{y}: LRT chi2({len(CONDS) - 1}) = {lr:.2f}, p = {p_lr:.3f}; min Holm p = {p_holm.min():.2f}")
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
