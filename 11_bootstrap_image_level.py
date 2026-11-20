"""Two-level paired bootstrap (model x seed pairs and test images within class).

For each comparison (condition vs baseline; GAN vs classical at equal dose; GAN vs classical pooled over doses)
and each replicate: resample the 20 model x seed pairs with replacement and the test images with replacement
within each class, recompute macro-F1 and TB recall for every run on the resampled test set, and average the
paired differences. Reports mean difference, percentile 95% CI and a two-sided bootstrap p.
Also: same analysis with 'twin' test images (cosine >= 0.97 to a training image) excluded.

  python 11_bootstrap_image_level.py results/main/tables/predictions_all.csv.gz \
         results/main/diagnostics/neighbors.csv results/main/stats/bootstrap_results.csv 2000
(predictions_all.csv.gz is produced by 10_collect_predictions.py)
"""
import sys, json
import numpy as np, pandas as pd

PRED, NEIGH, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
B = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
rng = np.random.default_rng(2026)

df = pd.read_csv(PRED)
df = df[df.source == "main"].copy()
twins = set(pd.read_csv(NEIGH).query("twin == True").image_id)

runs = df[["run_id", "strategy", "model", "seed"]].drop_duplicates().set_index("run_id")
img = np.array(sorted(df.image_id.unique()))
lab = df.drop_duplicates("image_id").set_index("image_id").loc[img, "label"].to_numpy()
P = df.pivot(index="image_id", columns="run_id", values="pred").loc[img]
assert not P.isna().any().any(), "missing predictions"
run_ids = list(P.columns)
pred = P.to_numpy().astype(int)            # n_img x n_runs
col = {r: j for j, r in enumerate(run_ids)}
print("runs", len(run_ids), "images", len(img), "per class", np.bincount(lab))


def metrics_from_weights(W, keep):
    """W: (B, n_img) image weights. Returns macro-F1 and TB recall arrays (B, n_runs)."""
    W = W * keep[None, :]
    C = np.zeros((W.shape[0], pred.shape[1], 3, 3))
    for t in range(3):
        Wt = W[:, lab == t]
        pt = pred[lab == t]
        for p in range(3):
            C[:, :, t, p] = Wt @ (pt == p).astype(float)
    tp = np.stack([C[:, :, k, k] for k in range(3)], -1)
    rowsum = C.sum(3); colsum = C.sum(2)
    rec = tp / np.maximum(rowsum, 1e-12); prec = tp / np.maximum(colsum, 1e-12)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)
    return f1.mean(-1), rec[..., 2]


def weights(B, keep):
    W = np.zeros((B, len(img)))
    for t in range(3):
        idx = np.where((lab == t) & keep)[0]
        draws = rng.integers(0, len(idx), size=(B, len(idx)))
        for b in range(B):
            W[b] += np.bincount(idx[draws[b]], minlength=len(img))
    return W


def pairs(a, b):
    """list of (run_a, run_b) matched by model and seed."""
    ra = runs[runs.strategy == a].reset_index().set_index(["model", "seed"]).run_id
    rb = runs[runs.strategy == b].reset_index().set_index(["model", "seed"]).run_id
    k = ra.index.intersection(rb.index)
    return [(col[ra[i]], col[rb[i]]) for i in k]


conds = ["weighted_ce", "focal", "classic_gap30", "classic_gap50", "classic_gap100",
         "gan_gap30", "gan_gap50", "gan_gap100"]
comps = [(c, "baseline") for c in conds] + [(f"gan_gap{d}", f"classic_gap{d}") for d in (30, 50, 100)]
comps.append(("gan_all", "classic_all"))


def run(keep, label):
    W = weights(B, keep)
    full = np.ones((1, len(img)))
    f_obs, r_obs = metrics_from_weights(full, keep)
    f_bt, r_bt = metrics_from_weights(W, keep)
    out = []
    for a, b in comps:
        if a == "gan_all":
            pr = sum([pairs(f"gan_gap{d}", f"classic_gap{d}") for d in (30, 50, 100)], [])
        else:
            pr = pairs(a, b)
        pr = np.array(pr)
        n = len(pr)
        sel = rng.integers(0, n, size=(B, n))           # resample pairs
        for name, obs, bt in (("macro_f1", f_obs, f_bt), ("tb_recall", r_obs, r_bt)):
            d_obs = 100 * (obs[0, pr[:, 0]] - obs[0, pr[:, 1]]).mean()
            d_rep = 100 * (bt[np.arange(B)[:, None], pr[sel, 0]] - bt[np.arange(B)[:, None], pr[sel, 1]]).mean(1)
            lo, hi = np.percentile(d_rep, [2.5, 97.5])
            p = min(1.0, 2 * min((d_rep <= 0).mean(), (d_rep >= 0).mean()))
            out.append(dict(set=label, comparison=f"{a} - {b}", metric=name, n_pairs=n, diff_pp=d_obs,
                            ci_lo=lo, ci_hi=hi, p_boot=p))
    return pd.DataFrame(out)


allimg = np.ones(len(img), bool)
notwin = np.array([i not in twins for i in img])
print("twins excluded:", (~notwin).sum(), "per class", np.bincount(lab[~notwin], minlength=3))
res = pd.concat([run(allimg, "all"), run(notwin, "no_twins")])
from statsmodels.stats.multitest import multipletests
res["p_holm"] = np.nan
res = res.reset_index(drop=True)
for (s, m), g in res.groupby(["set", "metric"]):
    idx = g.index[g.comparison.str.endswith("baseline")]
    res.loc[idx, "p_holm"] = multipletests(res.loc[idx, "p_boot"], method="holm")[1]
res.to_csv(OUT, index=False)
pd.set_option("display.width", 200)
print(res.round(3).to_string())
