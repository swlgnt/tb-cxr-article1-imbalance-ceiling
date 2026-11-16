#!/usr/bin/env python
"""Langkah 4: kumpulkan semua run -> tabel ringkasan, tabel paper (CSV/LaTeX/Excel), ensemble antar-seed.

Keluaran (work/results/tables/):
  all_runs.csv              1 baris per run (meta + metrik test + metrik val + CI bootstrap)
  summary_wide.csv          mean/std/CI95 antar-seed per kondisi
  table_<metrik>.csv/.tex   format paper: baris model×strategi, kolom rasio, "mean ± std"
  per_class_summary.csv     presisi/sensitivitas/spesifisitas/F1/AUC per kelas
  efficiency.csv            parameter, waktu/epoch, throughput inferensi per model
  ensemble_metrics.csv      metrik dari rata-rata probabilitas antar-seed
  table_loss_baselines_<metrik>.csv/.tex   desain dosis + pembanding loss: original vs weighted_ce/focal
                            vs dosis terbesar tiap metode (lihat PANDUAN_PEMBANDING_LOSS.md)
  results.xlsx              semua tabel dalam satu workbook
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.metrics import compute_metrics, flatten_metrics
from src.utils import (get_dirs, get_logger, is_dose_design, load_config, load_json, loss_baselines,
                       ratio_sort_key, strategy_meta)

GROUP = ["design", "ratio", "strategy", "model"]
PRETTY = {"macro_f1": "macro F1-score", "balanced_accuracy": "balanced accuracy", "macro_auc": "macro AUC",
          "tb_recall": "TB sensitivity", "tb_f1": "TB F1-score", "tb_auc": "TB AUC (one-vs-rest)",
          "tb_specificity": "TB specificity", "tb_precision": "TB precision", "mcc": "Matthews correlation coefficient"}


def collect_runs(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for mf in sorted(runs_dir.glob("*/metrics.json")):
        d = load_json(mf)
        meta = {k: v for k, v in d["meta"].items() if not isinstance(v, (dict, list))}
        row = dict(meta)
        row.update(flatten_metrics(d["test"]))
        row.update(flatten_metrics(d["val"], "val_"))
        for k, (lo, hi) in d.get("test_ci95", {}).items():
            row[f"{k}_ci_lo"], row[f"{k}_ci_hi"] = lo, hi
        row["run_dir"] = str(mf.parent)
        rows.append(row)
    return pd.DataFrame(rows)


def order_frame(df: pd.DataFrame, cfg) -> pd.DataFrame:
    df = df.copy()
    df["ratio"] = df["ratio"].astype(str)
    df["_r"] = df["ratio"].map(ratio_sort_key)
    m_order = {m: i for i, m in enumerate(cfg["models"])}
    s_order = {s: i for i, s in enumerate(cfg["strategies"])}
    df["_m"] = df["model"].map(m_order)
    df["_s"] = df["strategy"].map(s_order)
    return df.sort_values(["design", "_r", "_s", "_m"] + (["seed"] if "seed" in df else [])).drop(columns=["_r", "_m", "_s"])


def summarize(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    out = []
    for key, g in df.groupby(GROUP, sort=False):
        row = dict(zip(GROUP, key))
        row["n_seeds"] = len(g)
        for m in metrics:
            v = g[m].dropna().values
            mean = v.mean() if len(v) else np.nan
            sd = v.std(ddof=1) if len(v) > 1 else 0.0
            half = stats.t.ppf(0.975, len(v) - 1) * sd / np.sqrt(len(v)) if len(v) > 1 else np.nan
            row.update({f"{m}_mean": mean, f"{m}_std": sd, f"{m}_ci95_lo": mean - half, f"{m}_ci95_hi": mean + half})
        out.append(row)
    return pd.DataFrame(out)


def paper_table(summ: pd.DataFrame, metric: str, ratio_labels: dict, digits: int = 3) -> pd.DataFrame:
    t = summ.copy()
    t["cell"] = [f"{m:.{digits}f} ± {s:.{digits}f}" for m, s in zip(t[f"{metric}_mean"], t[f"{metric}_std"])]
    tab = t.pivot_table(index=["model", "strategy"], columns="ratio", values="cell", aggfunc="first", sort=False)
    cols = sorted(tab.columns, key=ratio_sort_key)
    tab = tab[cols]
    tab.columns = [ratio_labels.get(c, c) for c in cols]
    return tab


def to_latex(tab: pd.DataFrame, summ: pd.DataFrame, metric: str, caption: str, label: str) -> str:
    """LaTeX booktabs; nilai mean terbaik per kolom dicetak tebal."""
    means = summ.pivot_table(index=["model", "strategy"], columns="ratio", values=f"{metric}_mean", sort=False)
    means = means[sorted(means.columns, key=ratio_sort_key)]
    best = means.max(axis=0).values
    lines = [r"\begin{table}[htbp]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{ll" + "c" * tab.shape[1] + "}", r"\toprule",
             "Model & Strategy & " + " & ".join(tab.columns) + r" \\", r"\midrule"]
    prev = None
    for (model, strat), row in tab.iterrows():
        cells = []
        for j, c in enumerate(row.values):
            mv = means.loc[(model, strat)].values[j]
            cells.append(rf"\textbf{{{c}}}" if np.isclose(mv, best[j]) else str(c))
        mname = model if model != prev else ""
        if prev is not None and model != prev:
            lines.append(r"\addlinespace")
        strat_tex = strat.replace("_", "\\_")
        body = " & ".join(cells).replace("±", "$\\pm$")
        lines.append(f"{mname} & {strat_tex} & {body} " + "\\\\")
        prev = model
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def dose_tables(df: pd.DataFrame, metric: str, digits: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tabel desain dosis: baris = model × metode (klasik/GAN), kolom = dosis (%).
    Baseline (0%) ditampilkan di kedua baris metode. Baris 'All models' = rata-rata gabungan."""
    d = df[df["dose"].notna()].copy()
    if d["ratio"].astype(str).nunique() > 1:  # desain dosis: satu rasio (rasio terbesar, mis. orig)
        last = sorted(d["ratio"].astype(str).unique(), key=ratio_sort_key)[-1]
        d = d[d["ratio"].astype(str) == last]
    base = d[d["method"] == "original"]
    rows = []
    for meth in [m for m in ("classic", "gan") if m in set(d["method"])]:
        sub = pd.concat([base.assign(method=meth), d[d["method"] == meth]])
        sub["dose_pct"] = (sub["dose"] * 100).round().astype(int)
        for model, g in list(sub.groupby("model", sort=False)) + [("All models", sub)]:
            for dp, gg in g.groupby("dose_pct"):
                v = gg[metric].dropna()
                rows.append({"model": model, "method": meth, "dose_pct": dp, "mean": v.mean(),
                             "std": v.std(ddof=1) if len(v) > 1 else 0.0, "n": len(v)})
    long = pd.DataFrame(rows)
    if long.empty:
        return long, long
    long["cell"] = [f"{m:.{digits}f} ± {s:.{digits}f}" for m, s in zip(long["mean"], long["std"])]
    wide = long.pivot_table(index=["model", "method"], columns="dose_pct", values="cell", aggfunc="first", sort=False)
    wide.columns = [f"{c}%" for c in wide.columns]
    return long, wide


def dose_latex(wide: pd.DataFrame, long: pd.DataFrame, metric: str, caption: str) -> str:
    best = long.groupby(["model", "dose_pct"])["mean"].max()
    lines = [r"\begin{table}[htbp]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{tab:dose_{metric}}}",
             r"\begin{tabular}{ll" + "c" * wide.shape[1] + "}", r"\toprule",
             "Model & Augmentation & " + " & ".join(c.replace("%", r"\%") for c in wide.columns) + r" \\", r"\midrule"]
    prev = None
    for (model, meth), row in wide.iterrows():
        cells = []
        for col, c in zip(wide.columns, row.values):
            dp = int(col[:-1])
            m = long[(long.model == model) & (long.method == meth) & (long.dose_pct == dp)]["mean"]
            bold = len(m) and dp > 0 and np.isclose(m.iloc[0], best.get((model, dp), np.nan))
            cells.append((rf"\textbf{{{c}}}" if bold else str(c)).replace("±", r"$\pm$") if isinstance(c, str) else "--")
        if prev is not None and model != prev:
            lines.append(r"\addlinespace")
        mname = model if model != prev else ""
        meth_tex = {"classic": "Classical", "gan": "GAN"}.get(meth, meth)
        lines.append(f"{mname} & {meth_tex} & " + " & ".join(cells) + " " + "\\\\")
        prev = model
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


LOSS_PRETTY = {"weighted_ce": "Weighted CE", "focal": "Focal loss", "weighted_focal": "Weighted focal"}


def loss_tables(df: pd.DataFrame, cfg: dict, metric: str, digits: int, losses: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pembanding level loss vs penambahan citra. Kolom: original (CE), tiap loss baseline, lalu skenario
    dosis TERBESAR tiap metode (mis. classic_gap100, gan_gap100). Dosis terbesar dipilih karena bobot kelas
    inverse setara (dalam ekspektasi) dengan menyeimbangkan kelas 1:1. Baris: tiap model + 'All models'."""
    ref = cfg["analysis"]["reference_strategy"]
    d = df[df["ratio"].astype(str) == sorted(df["ratio"].astype(str).unique(), key=ratio_sort_key)[-1]]
    top = []
    for meth in ("classic", "gan"):
        dm = d[(d["method"] == meth) & d["dose"].notna()]
        if len(dm):
            top.append(dm.loc[dm["dose"].idxmax(), "strategy"])
    cols = [s for s in [ref] + list(losses) + top if s in set(d["strategy"])]
    d = d[d["strategy"].isin(cols)]
    rows = []
    for model, g in list(d.groupby("model", sort=False)) + [("All models", d)]:
        for s in cols:
            v = g.loc[g["strategy"] == s, metric].dropna()
            if len(v):
                rows.append({"model": model, "strategy": s, "mean": v.mean(),
                             "std": v.std(ddof=1) if len(v) > 1 else 0.0, "n": len(v)})
    long = pd.DataFrame(rows)
    if long.empty:
        return long, long
    long["cell"] = [f"{m:.{digits}f} ± {s:.{digits}f}" for m, s in zip(long["mean"], long["std"])]
    wide = long.pivot_table(index="model", columns="strategy", values="cell", aggfunc="first", sort=False)
    return long, wide[[c for c in cols if c in wide.columns]]


def loss_header(s: str, cfg: dict) -> str:
    v = cfg["strategies"].get(s, {})
    if s == cfg["analysis"]["reference_strategy"]:
        return "Original (CE)"
    if s in LOSS_PRETTY:
        return LOSS_PRETTY[s]
    if "dose" in v:
        return f"{'Classical' if v.get('extra') == 'classic' else 'GAN'} {int(round(float(v['dose']) * 100))}%"
    return s


def loss_latex(wide: pd.DataFrame, long: pd.DataFrame, cfg: dict, metric: str, caption: str) -> str:
    best = long.groupby("model")["mean"].max()
    lines = [r"\begin{table}[htbp]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{tab:loss_{metric}}}",
             r"\begin{tabular}{l" + "c" * wide.shape[1] + "}", r"\toprule",
             "Model & " + " & ".join(loss_header(c, cfg).replace("%", r"\%") for c in wide.columns) + r" \\",
             r"\midrule"]
    for model, row in wide.iterrows():
        if model == "All models":
            lines.append(r"\midrule")
        cells = []
        for s, c in zip(wide.columns, row.values):
            m = long[(long.model == model) & (long.strategy == s)]["mean"]
            bold = len(m) and np.isclose(m.iloc[0], best.get(model, np.nan))
            cells.append((rf"\textbf{{{c}}}" if bold else str(c)).replace("±", r"$\pm$") if isinstance(c, str) else "--")
        lines.append(f"{model} & " + " & ".join(cells) + " " + "\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def build_ensembles(df: pd.DataFrame, D: dict, names: list[str], kp: dict) -> pd.DataFrame:
    rows = []
    pcols = [f"prob_{c}" for c in names]
    for key, g in df.groupby(GROUP, sort=False):
        preds = [pd.read_csv(Path(r) / "test_predictions.csv").set_index("image_id") for r in g["run_dir"]]
        base = preds[0][["path", "label"]].copy()
        probs = np.mean([p.loc[base.index, pcols].values for p in preds], axis=0)
        ens = base.copy()
        ens[pcols] = probs
        ens["pred"] = probs.argmax(1)
        tag = "__".join(str(k) for k in key)
        ens.reset_index().to_csv(D["ensembles"] / f"{tag}.csv", index=False)
        m = compute_metrics(ens["label"].values, probs, names, kp)
        row = dict(zip(GROUP, key))
        row["n_seeds"] = len(preds)
        row.update(flatten_metrics(m))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--design", default=None, help="filter desain (default: dari config)")
    ap.add_argument("--digits", type=int, default=3)
    args = ap.parse_args()
    cfg = load_config(args.config)
    D = get_dirs(cfg)
    log = get_logger("agg", str(D["logs"] / "04_aggregate.log"))
    names, kp = cfg["data"]["class_names"], cfg["data"]["short_names"]
    design = args.design or cfg["imbalance"]["design"]

    df = collect_runs(D["runs"])
    if df.empty:
        log.error("Belum ada run yang selesai.")
        return
    df = order_frame(df[df["design"] == design], cfg)
    sm = strategy_meta(cfg)
    df["method"] = df["strategy"].map(lambda s: sm.get(s, ("other", np.nan))[0])
    df["dose"] = df["strategy"].map(lambda s: sm.get(s, ("other", np.nan))[1])
    df.to_csv(D["tables"] / "all_runs.csv", index=False)
    log.info(f"{len(df)} run dikumpulkan ({df.groupby(GROUP).ngroups} kondisi)")
    if "git_commit" in df and df["git_commit"].notna().any():
        commits = df["git_commit"].fillna("tanpa-git").value_counts()
        if len(commits) > 1:
            log.warning("Run berasal dari BEBERAPA versi kode: " + ", ".join(f"{c} ({n} run)" for c, n in commits.items())
                        + ". Pastikan perubahan di antaranya tidak memengaruhi hasil (cek: git log).")
        if df.get("git_dirty", pd.Series(dtype=bool)).fillna(False).astype(bool).any():
            log.warning(f"{int(df['git_dirty'].fillna(False).astype(bool).sum())} run dijalankan dengan perubahan "
                        "yang belum di-commit (kolom git_dirty di all_runs.csv).")

    # label kolom rasio: '1:1', '3:1', ..., 'orig (5.4:1)'
    ssum = pd.read_csv(D["splits"] / "subsets_summary.csv")
    ssum["ratio"] = ssum["ratio"].astype(str)
    ssum = ssum[ssum["design"] == design]
    rl = {r: (f"orig ({a:.1f}:1)" if r == "orig" else f"{r}:1")
          for r, a in zip(ssum["ratio"], ssum["actual_ratio_major_to_tb"])}

    metrics = cfg["analysis"]["key_metrics"]
    all_metric_cols = [c for c in df.columns if c in set(metrics) | {
        "accuracy", "macro_precision", "macro_recall", "weighted_f1", "macro_specificity", "cohen_kappa",
        "log_loss", "brier", "ece", "tb_precision", "tb_ap", "tb_npv"}]
    summ = summarize(df, list(dict.fromkeys(metrics + all_metric_cols)))
    summ.to_csv(D["tables"] / "summary_wide.csv", index=False)

    sheets = {"all_runs": df, "summary": summ}
    for m in metrics:
        tab = paper_table(summ, m, rl, args.digits)
        tab.to_csv(D["tables"] / f"table_{m}.csv")
        (D["tables"] / f"table_{m}.tex").write_text(
            to_latex(tab, summ, m, f"Test {PRETTY.get(m, m.replace('_', ' '))} (mean $\\pm$ SD over seeds).", f"tab:{m}"), encoding="utf-8")
        sheets[f"tab_{m}"[:31]] = tab.reset_index()

    # tabel desain dosis (metrik vs persentase gap yang ditutup)
    if is_dose_design(cfg, list(df["strategy"].unique())):
        for m in metrics:
            long, wide = dose_tables(df, m, args.digits)
            if wide.empty:
                continue
            long.to_csv(D["tables"] / f"dose_long_{m}.csv", index=False)
            wide.to_csv(D["tables"] / f"table_dose_{m}.csv")
            (D["tables"] / f"table_dose_{m}.tex").write_text(dose_latex(
                wide, long, m, f"Test {PRETTY.get(m, m)} by augmentation dose (fraction of the class gap closed; "
                               f"mean $\\pm$ SD over seeds)."), encoding="utf-8")
            sheets[f"dose_{m}"[:31]] = wide.reset_index()

        # pembanding level loss (weighted_ce, focal) vs original dan dosis penyeimbang penuh
        losses = loss_baselines(cfg, list(df["strategy"].unique()))
        if losses:
            for m in metrics:
                long, wide = loss_tables(df, cfg, m, args.digits, losses)
                if wide.empty:
                    continue
                wide.rename(columns=lambda c: loss_header(c, cfg)).to_csv(D["tables"] / f"table_loss_baselines_{m}.csv")
                long.to_csv(D["tables"] / f"loss_baselines_long_{m}.csv", index=False)
                (D["tables"] / f"table_loss_baselines_{m}.tex").write_text(loss_latex(
                    wide, long, cfg, m, f"Test {PRETTY.get(m, m)}: loss-level balancing versus adding TB images "
                                        f"(mean $\\pm$ SD over seeds; best per row in bold)."), encoding="utf-8")
                sheets[f"loss_{m}"[:31]] = wide.rename(columns=lambda c: loss_header(c, cfg)).reset_index()
            log.info(f"Tabel pembanding level loss ({', '.join(losses)}): table_loss_baselines_<metrik>.csv/.tex")

    # per kelas
    pc_rows = []
    for short in kp.values():
        cols = [f"{short}_{k}" for k in ("precision", "recall", "specificity", "f1", "auc", "ap") if f"{short}_{k}" in df]
        s = df.groupby(GROUP, sort=False)[cols].agg(["mean", "std"])
        s.columns = [f"{a}_{b}" for a, b in s.columns]
        pc_rows.append(s)
    pc = pd.concat(pc_rows, axis=1).reset_index()
    pc.to_csv(D["tables"] / "per_class_summary.csv", index=False)
    sheets["per_class"] = pc

    eff = df.groupby("model", sort=False).agg(
        params_millions=("params_millions", "first"), img_size=("img_size", "first"),
        mean_epoch_time_sec=("mean_epoch_time_sec", "mean"), epochs_run=("epochs_run", "mean"),
        train_time_min=("train_time_sec", lambda x: x.mean() / 60),
        inference_img_per_sec=("inference_img_per_sec", "mean"),
        peak_gpu_mem_gb=("peak_gpu_mem_gb", "mean")).reset_index()
    eff.to_csv(D["tables"] / "efficiency.csv", index=False)
    sheets["efficiency"] = eff

    ens = order_frame(build_ensembles(df, D, names, kp), cfg)
    ens.to_csv(D["tables"] / "ensemble_metrics.csv", index=False)
    sheets["ensemble"] = ens

    rank = summ.sort_values("macro_f1_mean", ascending=False)[GROUP + ["n_seeds"] + [f"{m}_mean" for m in metrics]]
    rank.to_csv(D["tables"] / "ranking_by_macro_f1.csv", index=False)
    sheets["ranking"] = rank

    with pd.ExcelWriter(D["tables"] / "results.xlsx", engine="openpyxl") as xw:
        for name, s in sheets.items():
            s.to_excel(xw, sheet_name=name[:31], index=False)
    log.info(f"Tabel tersimpan di {D['tables']}")
    log.info("5 kondisi teratas (macro-F1):\n" + rank.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
