"""Fungsi visualisasi untuk paper (matplotlib, 300 dpi, PNG + PDF)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
                     "figure.dpi": 110, "savefig.dpi": 300, "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "DejaVu Sans"})
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def metric_vs_ratio(summ, metric, models, strategies, ratio_order, ratio_labels, ylabel, path):
    n = len(models)
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.6 * rows), sharey=True, squeeze=False)
    x = np.arange(len(ratio_order))
    for k, m in enumerate(models):
        ax = axes[k // cols, k % cols]
        for j, s in enumerate(strategies):
            g = summ[(summ.model == m) & (summ.strategy == s)].set_index("ratio").reindex(ratio_order)
            if g[f"{metric}_mean"].isna().all():
                continue
            mu, sd = g[f"{metric}_mean"].values, g[f"{metric}_std"].fillna(0).values
            # error bar dipotong pada rentang metrik [0, 1]
            yerr = np.vstack([np.minimum(sd, np.clip(mu, 0, None)), np.minimum(sd, np.clip(1 - mu, 0, None))])
            ax.errorbar(x + (j - (len(strategies) - 1) / 2) * 0.06, mu, yerr=yerr,
                        marker=MARKERS[j % len(MARKERS)], ms=4, capsize=2, lw=1.2, label=s)
        ax.set_title(m)
        ax.set_xticks(x, [ratio_labels.get(r, r) for r in ratio_order], rotation=0)
        ax.set_xlabel("Majority : TB ratio (train)")
        ax.grid(alpha=0.3)
        if k % cols == 0:
            ax.set_ylabel(ylabel)
    for k in range(n, rows * cols):
        axes[k // cols, k % cols].axis("off")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=len(lab), bbox_to_anchor=(0.5, -0.04), frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, path)


def heatmap(mat, row_labels, col_labels, title, path, fmt="{:.3f}", cmap="viridis"):
    fig, ax = plt.subplots(figsize=(1.2 + 1.1 * len(col_labels), 0.8 + 0.45 * len(row_labels)))
    im = ax.imshow(mat, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    lo, hi = np.nanmin(mat), np.nanmax(mat)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7,
                        color="white" if (v - lo) / (hi - lo + 1e-12) < 0.5 else "black")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, path)


def confusion_grid(cms: dict, row_keys, col_keys, class_short, path, normalize=True):
    fig, axes = plt.subplots(len(row_keys), len(col_keys), figsize=(2.3 * len(col_keys), 2.2 * len(row_keys)),
                             squeeze=False)
    for i, r in enumerate(row_keys):
        for j, c in enumerate(col_keys):
            ax = axes[i, j]
            cm = cms.get((r, c))
            if cm is None:
                ax.axis("off")
                continue
            cm = np.asarray(cm, float)
            show = cm / cm.sum(1, keepdims=True) if normalize else cm
            ax.imshow(show, cmap="Blues", vmin=0, vmax=1 if normalize else None)
            for a in range(cm.shape[0]):
                for b in range(cm.shape[1]):
                    ax.text(b, a, f"{show[a, b]:.2f}\n({int(cm[a, b])})", ha="center", va="center", fontsize=6.5,
                            color="white" if show[a, b] > 0.6 else "black")
            ax.set_xticks(range(len(class_short)), class_short, fontsize=7)
            ax.set_yticks(range(len(class_short)), class_short, fontsize=7)
            if i == 0:
                ax.set_title(c, fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{r}\nTrue", fontsize=8)
            if i == len(row_keys) - 1:
                ax.set_xlabel("Predicted", fontsize=8)
    fig.tight_layout()
    save(fig, path)


def roc_pr_curves(curves: dict, title: str, path):
    """curves: label -> (y_true_bin, score)"""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for k, (lab, (y, s)) in enumerate(curves.items()):
        fpr, tpr, _ = roc_curve(y, s)
        pr, rc, _ = precision_recall_curve(y, s)
        a1.plot(fpr, tpr, lw=1.3, label=f"{lab} (AUC={auc(fpr, tpr):.3f})")
        a2.plot(rc, pr, lw=1.3, label=f"{lab} (AP={average_precision_score(y, s):.3f})")
    a1.plot([0, 1], [0, 1], "k--", lw=0.8)
    prev = np.mean(next(iter(curves.values()))[0])
    a2.axhline(prev, color="k", ls="--", lw=0.8)
    a1.set(xlabel="False positive rate", ylabel="True positive rate (sensitivity)", title="ROC — TB vs rest")
    a2.set(xlabel="Recall", ylabel="Precision", title="Precision–Recall — TB vs rest")
    for a in (a1, a2):
        a.legend(loc="lower right" if a is a1 else "lower left", fontsize=7)
        a.grid(alpha=0.3)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    save(fig, path)


def cd_diagram(avg_rank: dict, cd: float, title: str, path):
    """Critical difference diagram (Demšar 2006)."""
    items = sorted(avg_rank.items(), key=lambda kv: kv[1])
    k = len(items)
    lo, hi = 1, max(k, int(np.ceil(max(avg_rank.values()))))
    fig, ax = plt.subplots(figsize=(6, 1.0 + 0.32 * k))
    ax.set_xlim(lo - 0.3, hi + 0.3)
    ax.set_ylim(-(k / 2 + 1.0), 2.1)
    ax.axis("off")
    ax.hlines(1, lo, hi, color="k", lw=1)
    for t in range(lo, hi + 1):
        ax.vlines(t, 1, 1.15, color="k", lw=1)
        ax.text(t, 1.3, str(t), ha="center", fontsize=8)
    ax.hlines(1.65, lo, lo + cd, color="red", lw=2)
    ax.vlines([lo, lo + cd], 1.58, 1.72, color="red", lw=1)
    ax.text(lo + cd + 0.05, 1.65, f"CD = {cd:.2f}", ha="left", va="center", fontsize=7, color="red")
    half = int(np.ceil(k / 2))
    for i, (name, r) in enumerate(items):
        if i < half:
            y = -(i + 0.5) * 0.8
            ax.plot([r, r, lo - 0.1], [1, y, y], color="k", lw=0.8)
            ax.text(lo - 0.15, y, f"{name} ({r:.2f})", ha="right", va="center", fontsize=8)
        else:
            y = -(k - i - 0.5) * 0.8
            ax.plot([r, r, hi + 0.1], [1, y, y], color="k", lw=0.8)
            ax.text(hi + 0.15, y, f"({r:.2f}) {name}", ha="left", va="center", fontsize=8)
    # garis penghubung kelompok yang tidak berbeda signifikan
    ranks = [r for _, r in items]
    cliques, ybar = [], 0.75
    for i in range(k):
        j = i
        while j + 1 < k and ranks[j + 1] - ranks[i] <= cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
    for n_, (i, j) in enumerate(cliques):
        y = ybar - 0.12 * n_
        ax.hlines(y, ranks[i] - 0.03, ranks[j] + 0.03, color="k", lw=2.5)
    ax.set_title(title, fontsize=9, pad=4)
    save(fig, path)


def training_curves(hist_by_label: dict, metric: str, ylabel: str, title: str, path):
    """hist_by_label: label -> list of DataFrame (satu per seed)."""
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for lab, hs in hist_by_label.items():
        L = max(len(h) for h in hs)
        arr = np.full((len(hs), L), np.nan)
        for i, h in enumerate(hs):
            arr[i, : len(h)] = h[metric].values
        mu, sd = np.nanmean(arr, 0), np.nanstd(arr, 0)
        ep = np.arange(L)
        ax.plot(ep, mu, lw=1.3, label=lab)
        ax.fill_between(ep, mu - sd, mu + sd, alpha=0.2)
    ax.set(xlabel="Epoch", ylabel=ylabel, title=title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, path)


def composition_bars(ssum, class_names, ratio_labels, path):
    fig, ax = plt.subplots(figsize=(1.5 + 1.1 * len(ssum), 3.0))
    x = np.arange(len(ssum))
    bottom = np.zeros(len(ssum))
    for c in class_names:
        v = ssum[f"n_{c}"].values
        ax.bar(x, v, bottom=bottom, label=c, width=0.65)
        bottom += v
    ax.set_xticks(x, [ratio_labels.get(r, r) for r in ssum["ratio"]])
    ax.set(xlabel="Training subset (majority : TB)", ylabel="Number of images", title="Training set composition")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, path)


METHOD_STYLE = {"classic": dict(color="#1f77b4", marker="s", label="Classical augmentation"),
                "gan": dict(color="#d62728", marker="o", label="GAN (synthetic)")}


LOSS_STYLE = {"weighted_ce": dict(color="#555555", ls="--", label="Weighted CE (no added images)"),
              "focal": dict(color="#555555", ls=":", label="Focal loss (no added images)"),
              "weighted_focal": dict(color="#555555", ls="-.", label="Weighted focal (no added images)")}


def dose_response(runs, metric, models, ylabel, path, loss_runs=None):
    """Metrik vs dosis (persentase gap ditutup); 0% = original, dipakai kedua metode.
    Panel per model + panel gabungan (mean ± SD atas semua model × seed).
    loss_runs (opsional): run pembanding level loss (weighted_ce, focal) -> garis horizontal rata-rata,
    pita tipis = ± SD, sebagai acuan 'menyeimbangkan lewat loss tanpa menambah citra'."""
    d = runs[runs["dose"].notna()].copy()
    d["dose_pct"] = (d["dose"] * 100).round().astype(int)
    base = d[d["method"] == "original"]
    panels = list(models) + ["All models"]
    cols = min(len(panels), 5)
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.9 * cols, 2.7 * rows), sharey=True, squeeze=False)
    doses = sorted(d["dose_pct"].unique())
    for k, pnl in enumerate(panels):
        ax = axes[k // cols, k % cols]
        for j, meth in enumerate([m for m in ("classic", "gan") if m in set(d["method"])]):
            sub = pd.concat([base, d[d["method"] == meth]])
            if pnl != "All models":
                sub = sub[sub["model"] == pnl]
            g = sub.groupby("dose_pct")[metric].agg(["mean", "std"]).reindex(doses)
            off = (j - 0.5) * 1.2  # geser sedikit agar error bar tidak bertumpuk
            st = METHOD_STYLE[meth]
            mu, sd = g["mean"].values, g["std"].fillna(0).values
            yerr = np.vstack([np.minimum(sd, np.clip(mu, 0, None)), np.minimum(sd, np.clip(1 - mu, 0, None))])
            ax.errorbar(np.array(doses) + off, mu, yerr=yerr, color=st["color"], marker=st["marker"], ms=4,
                        capsize=2, lw=1.3, label=st["label"])
        if loss_runs is not None and len(loss_runs):
            lr = loss_runs if pnl == "All models" else loss_runs[loss_runs["model"] == pnl]
            for s_name, g in lr.groupby("strategy", sort=False):
                v = g[metric].dropna()
                if not len(v):
                    continue
                st = LOSS_STYLE.get(s_name, dict(color="#555555", ls="--", label=s_name))
                ax.axhline(v.mean(), color=st["color"], ls=st["ls"], lw=1.1, label=st["label"], zorder=1)
                if len(v) > 1:
                    ax.axhspan(v.mean() - v.std(ddof=1), v.mean() + v.std(ddof=1), color=st["color"], alpha=0.07,
                               lw=0, zorder=0)
        ax.set_title(pnl, fontweight="bold" if pnl == "All models" else None)
        ax.set_xticks(doses, [f"{x}%" for x in doses])
        ax.set_xlabel("Class gap closed by added TB images")
        ax.grid(alpha=0.3)
        if k % cols == 0:
            ax.set_ylabel(ylabel)
    for k in range(len(panels), rows * cols):
        axes[k // cols, k % cols].axis("off")
    h, lab = axes[0, 0].get_legend_handles_labels()
    order = sorted(range(len(lab)), key=lambda i: "no added images" in lab[i])  # metode dosis dulu, lalu pembanding loss
    h, lab = [h[i] for i in order], [lab[i] for i in order]
    ncol = min(len(lab), 4)
    fig.legend(h, lab, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, -0.03), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.06 * int(np.ceil(len(lab) / ncol)), 1, 1))
    save(fig, path)


def dose_composition(comp, class_names, strategy_order, path):
    """Komposisi data latih per skenario: kelas mayoritas + TB asli + TB tambahan (klasik/GAN)."""
    comp = comp[comp["strategy"].isin(strategy_order)]
    x = np.arange(len(strategy_order))
    fig, ax = plt.subplots(figsize=(3.2 + 1.0 * len(strategy_order), 3.2))
    colors = {"real": None, "classic": "#1f77b4", "gan": "#d62728"}
    bottom = np.zeros(len(strategy_order))
    for ci, c in enumerate(class_names):
        for src in ("real", "classic", "gan"):
            v = np.array([comp[(comp.strategy == s) & (comp.cls == c) & (comp.source == src)]["n"].sum()
                          for s in strategy_order], float)
            if v.sum() == 0:
                continue
            neutral = ["#9e9e9e", "#c7b98a", "#4caf50", "#795548"]  # kelas asli: warna netral
            kw = dict(color=neutral[ci % len(neutral)]) if src == "real" else dict(color=colors[src], hatch="//",
                                                                                   alpha=0.6)
            ax.bar(x, v, bottom=bottom, width=0.65, label=f"{c} ({src})", edgecolor="white", lw=0.5, **kw)
            bottom += v
    ax.set_xticks(x, strategy_order, rotation=30, ha="right")
    ax.set(ylabel="Number of training images", title="Training set composition per scenario")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)  # di luar area batang
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, path)
