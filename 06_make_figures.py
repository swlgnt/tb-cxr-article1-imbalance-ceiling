#!/usr/bin/env python
"""Langkah 6: gambar untuk paper (work/figures/, PNG 300 dpi + PDF).

  fig01_training_composition       komposisi subset latih per rasio
  fig02_<metrik>_vs_ratio          metrik vs rasio, panel per model, garis per strategi (gambar utama)
  fig02_<metrik>_vs_dose           (desain dosis) metrik vs dosis; pembanding loss = garis horizontal abu-abu
  fig03_heatmap_<metrik>_<strat>   model × rasio
  fig04_confusion_r<rasio>         matriks konfusi ensemble (baris model, kolom strategi)
  fig05_roc_pr_r<rasio>_<strat>    ROC & PR kelas TB per model (ensemble)
  fig06_cd_<treat>_<metrik>        critical difference diagram (Nemenyi)
  fig07_curves_<model>_r<rasio>    kurva pelatihan val macro-F1 per strategi
"""
import argparse
from pathlib import Path

import pandas as pd

from src import plotting as P
from src.metrics import compute_metrics
from src.stat_tests import friedman_nemenyi
from src.utils import get_dirs, get_logger, is_dose_design, load_config, loss_baselines, ratio_sort_key, strategy_meta

LABELS = {"macro_f1": "Macro F1", "tb_recall": "TB sensitivity", "tb_f1": "TB F1", "tb_auc": "TB AUC",
          "balanced_accuracy": "Balanced accuracy", "macro_auc": "Macro AUC", "tb_specificity": "TB specificity",
          "mcc": "MCC"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--cm-ratio", default="orig", help="rasio untuk matriks konfusi & ROC")
    args = ap.parse_args()
    cfg = load_config(args.config)
    D = get_dirs(cfg)
    log = get_logger("fig", str(D["logs"] / "06_figures.log"))
    F = D["figures"]
    names, kp = cfg["data"]["class_names"], cfg["data"]["short_names"]
    tb_idx = names.index(cfg["data"]["minority_class"])
    design = cfg["imbalance"]["design"]

    runs = pd.read_csv(D["tables"] / "all_runs.csv")
    runs["ratio"] = runs["ratio"].astype(str)
    summ = pd.read_csv(D["tables"] / "summary_wide.csv")
    summ["ratio"] = summ["ratio"].astype(str)
    ssum = pd.read_csv(D["splits"] / "subsets_summary.csv")
    ssum["ratio"] = ssum["ratio"].astype(str)
    ssum = ssum[ssum["design"] == design]
    rl = {r: (f"orig\n({a:.1f}:1)" if r == "orig" else f"{r}:1")
          for r, a in zip(ssum["ratio"], ssum["actual_ratio_major_to_tb"])}
    ratio_order = sorted(summ["ratio"].unique(), key=ratio_sort_key)
    models = [m for m in cfg["models"] if m in set(summ["model"])]
    strategies = [s for s in cfg["strategies"] if s in set(summ["strategy"])]
    metrics = [m for m in cfg["analysis"]["key_metrics"] if f"{m}_mean" in summ]

    dose = is_dose_design(cfg, strategies) and "dose" in runs and runs["dose"].notna().any()
    if dose:  # ---- desain dosis: gambar utama = metrik vs dosis
        cf = D["work"] / "results" / "train_composition.csv"
        if cf.exists():
            comp = pd.read_csv(cf)
            comp = comp[comp["ratio"].astype(str) == ratio_order[0]]
            sm = strategy_meta(cfg)  # pembanding loss memakai data latih yang sama dengan original -> tidak digambar
            dose_s = [s for s in strategies if sm.get(s, ("other",))[0] in ("original", "classic", "gan")]
            P.dose_composition(comp, names, dose_s, F / "fig01_training_composition")
        rd = runs[runs["ratio"] == ratio_order[-1]]  # desain dosis: satu rasio
        if len(ratio_order) > 1:
            log.warning(f"Desain dosis dengan beberapa rasio: grafik dosis memakai rasio {ratio_order[-1]}.")
        lb = loss_baselines(cfg, strategies)
        lruns = rd[rd["strategy"].isin(lb)] if lb else None
        for m in metrics:
            if m in rd:
                P.dose_response(rd, m, models, LABELS.get(m, m), F / f"fig02_{m}_vs_dose", loss_runs=lruns)
    else:
        P.composition_bars(ssum.sort_values("ratio", key=lambda s: s.map(ratio_sort_key)), names, rl,
                           F / "fig01_training_composition")

    for m in (metrics if len(ratio_order) > 1 else []):  # grafik vs rasio hanya bila rasio > 1
        P.metric_vs_ratio(summ, m, models, strategies, ratio_order, rl, LABELS.get(m, m), F / f"fig02_{m}_vs_ratio")
        for s in strategies:
            sub = summ[summ.strategy == s].pivot_table(index="model", columns="ratio", values=f"{m}_mean")
            sub = sub.reindex(index=models, columns=ratio_order)
            P.heatmap(sub.values, models, [rl.get(r, r).replace("\n", " ") for r in ratio_order],
                      f"{LABELS.get(m, m)} — {s}", F / f"fig03_heatmap_{m}_{s}")

    # ensemble -> confusion & ROC
    pcols = [f"prob_{c}" for c in names]
    short = [kp[c] for c in names]
    for r in ([args.cm_ratio] if args.cm_ratio in ratio_order else ratio_order[-1:]):
        cms = {}
        for m in models:
            for s in strategies:
                f = D["ensembles"] / f"{design}__{r}__{s}__{m}.csv"
                if f.exists():
                    e = pd.read_csv(f)
                    cms[(m, s)] = compute_metrics(e["label"].values, e[pcols].values, names, kp)["confusion_matrix"]
        if cms:
            P.confusion_grid(cms, models, strategies, short, F / f"fig04_confusion_r{r}")
        for s in strategies:
            curves = {}
            for m in models:
                f = D["ensembles"] / f"{design}__{r}__{s}__{m}.csv"
                if f.exists():
                    e = pd.read_csv(f)
                    curves[m] = ((e["label"].values == tb_idx).astype(int), e[f"prob_{names[tb_idx]}"].values)
            if curves:
                P.roc_pr_curves(curves, f"ratio {rl.get(r, r).replace(chr(10), ' ')} — {s} (seed ensemble)",
                                F / f"fig05_roc_pr_r{r}_{s}")

    # CD diagram
    for treat, blocks in (("model", ["ratio", "strategy", "seed"]), ("strategy", ["ratio", "model", "seed"])):
        for m in ("macro_f1", "tb_recall"):
            if m not in runs:
                continue
            res = friedman_nemenyi(runs, blocks, treat, m, True, cfg["analysis"]["alpha"])
            if "error" not in res:
                P.cd_diagram(res["avg_rank"], res["critical_difference"],
                             f"{LABELS.get(m, m)} — {treat} (Friedman p={res['p_value']:.1e})",
                             F / f"fig06_cd_{treat}_{m}")

    # kurva pelatihan
    for m in models:
        for r in ratio_order:
            hist = {}
            for s in strategies:
                dirs = runs[(runs.model == m) & (runs.ratio == r) & (runs.strategy == s)]["run_dir"]
                hs = [pd.read_csv(Path(d) / "history.csv") for d in dirs if (Path(d) / "history.csv").exists()]
                if hs:
                    hist[s] = hs
            if hist:
                P.training_curves(hist, "val_macro_f1", "Validation macro F1",
                                  f"{m} — ratio {rl.get(r, r).replace(chr(10), ' ')}", F / f"fig07_curves_{m}_r{r}")
    log.info(f"Gambar tersimpan di {F}")


if __name__ == "__main__":
    main()
