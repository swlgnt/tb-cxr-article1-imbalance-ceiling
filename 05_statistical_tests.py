#!/usr/bin/env python
"""Langkah 5: uji statistik.

A. Friedman + Nemenyi antar MODEL     (blok = rasio × strategi × seed)       -> hipotesis 3 proposal
B. Friedman + Nemenyi antar STRATEGI  (blok = rasio × model × seed)
C. Wilcoxon berpasangan tiap strategi vs referensi, per model (Holm)         -> efek augmentasi klasik
D. Tren terhadap rasio: Spearman & regresi metrik ~ log(rasio), per model × strategi
E. ANOVA faktorial model × strategi × rasio (Type II, partial eta^2)
L. Pembanding level loss (desain dosis + weighted_ce/focal; lihat PANDUAN_PEMBANDING_LOSS.md):
     L1 tiap pembanding loss vs original, L2 tiap skenario penambahan citra vs tiap pembanding loss
     (Wilcoxon gabungan lintas model, pasangan model × seed, Holm per metrik)
F. DeLong (AUC kelas TB, one-vs-rest) & McNemar pada prediksi ensemble antar-seed:
     F1 strategi vs referensi untuk tiap model × rasio
     F2 antar-model untuk tiap strategi × rasio
     F3 GAN vs klasik pada dosis sama; F4 skenario penambahan citra vs pembanding loss (per model)
Keluaran: work/results/stats/*.csv + stats_report.txt
"""
import argparse
import itertools
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from src.stat_tests import delong_test, factorial_anova, friedman_nemenyi, holm, mcnemar_test, paired_wilcoxon
from src.utils import get_dirs, get_logger, is_dose_design, load_config, loss_baselines, ratio_sort_key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    D = get_dirs(cfg)
    log = get_logger("stats", str(D["logs"] / "05_stats.log"))
    A = cfg["analysis"]
    alpha, ref = A["alpha"], A["reference_strategy"]
    names = cfg["data"]["class_names"]
    tb_idx = names.index(cfg["data"]["minority_class"])

    warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)
    df = pd.read_csv(D["tables"] / "all_runs.csv").copy()
    df["ratio"] = df["ratio"].astype(str)
    ssum = pd.read_csv(D["splits"] / "subsets_summary.csv")
    actual = dict(zip(ssum["ratio"].astype(str), ssum["actual_ratio_major_to_tb"]))
    df["ratio_num"] = df["ratio"].map(actual)
    report = []
    S = D["stats"]

    def section(t):
        report.append("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)

    lv = {f: sorted(df[f].astype(str).unique(), key=lambda x: (ratio_sort_key(x) if f == "ratio" else 0, x))
          for f in ("ratio", "strategy", "model", "seed")}
    section("STATUS DATA")
    report.append(f"Jumlah run selesai : {len(df)}")
    for f, name in (("ratio", "rasio"), ("strategy", "strategi"), ("model", "model"), ("seed", "seed")):
        report.append(f"{name:<9}({len(lv[f])}): {', '.join(lv[f])}")
    need = {
        "Friedman antar model": len(lv["model"]) >= 2 and len(df.groupby(["ratio", "strategy", "seed"])) >= 2,
        "Friedman antar strategi": len(lv["strategy"]) >= 2,
        f"Wilcoxon vs '{ref}'": ref in lv["strategy"] and len(lv["strategy"]) >= 2,
        **({"Tren terhadap dosis (desain dosis)": True} if is_dose_design(cfg, lv["strategy"])
           else {"Tren terhadap rasio": len(lv["ratio"]) >= 3}),
        "ANOVA faktorial": len(lv["seed"]) >= 2 and any(len(lv[f]) >= 2 for f in ("ratio", "strategy", "model")),
    }
    report.append("Uji yang bisa dihitung dengan data saat ini:")
    for k, ok in need.items():
        report.append(f"  [{'BISA' if ok else 'BELUM'}] {k}")
    if not all(need.values()):
        report.append("Uji berstatus BELUM akan otomatis terisi setelah grid berjalan lebih jauh. Ini NORMAL jika\n"
                      "baru menjalankan uji waktu (--max-runs 1) atau grid belum selesai. Bukan error program.")

    # ---------------- 0: DESAIN DOSIS (analisis utama bila skenario dosis dipakai)
    if is_dose_design(cfg, lv["strategy"]) and "dose" in df and df["dose"].notna().any():
        dd = df[df["dose"].notna()].copy()
        dd["dose_pct"] = (dd["dose"] * 100).round().astype(int)
        prim = A.get("primary_metrics", [])
        section("DESAIN DOSIS (A): tiap skenario vs original - digabung lintas model (pasangan: model × seed)")
        rows = [paired_wilcoxon(dd, ["model", "seed"], "strategy", m, ref) for m in A["key_metrics"]]
        w0 = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        if len(w0):
            pm = w0["metric"].isin(prim)
            if pm.any():
                w0.loc[pm, "p_holm_primary"] = holm(w0.loc[pm, "p_value"].values)
            w0.to_csv(S / "dose_vs_original_pooled.csv", index=False)
            cols = ["metric", "treatment", "n_pairs", "mean_ref", "mean_treat", "mean_diff", "p_value", "p_holm",
                    "rank_biserial"] + (["p_holm_primary"] if "p_holm_primary" in w0 else [])
            report.append(w0[w0["metric"].isin(prim or A["key_metrics"])][cols]
                          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            report.append("(p_holm: dalam satu metrik antar skenario; p_holm_primary: atas metrik primer sekaligus. "
                          "Semua metrik: dose_vs_original_pooled.csv)")
        else:
            report.append("DILEWATI: belum ada pasangan skenario vs original.")

        section("DESAIN DOSIS (B): GAN vs klasik pada dosis yang sama - digabung lintas model (pasangan: model × seed)")
        gc = dd[dd["method"].isin(["gan", "classic"])]
        rows = [paired_wilcoxon(gc, ["model", "seed"], "method", m, "classic", ["dose_pct"]) for m in A["key_metrics"]]
        w1 = pd.concat([r for r in rows if len(r)], ignore_index=True) if any(len(r) for r in rows) else pd.DataFrame()
        if len(w1):
            pm = w1["metric"].isin(prim)
            if pm.any():
                w1.loc[pm, "p_holm_primary"] = holm(w1.loc[pm, "p_value"].values)
            w1.to_csv(S / "dose_gan_vs_classic_pooled.csv", index=False)
            cols = ["metric", "dose_pct", "n_pairs", "mean_ref", "mean_treat", "mean_diff", "p_value", "p_holm",
                    "rank_biserial"] + (["p_holm_primary"] if "p_holm_primary" in w1 else [])
            report.append("mean_ref = klasik, mean_treat = GAN; mean_diff > 0 berarti GAN lebih baik.")
            report.append(w1[w1["metric"].isin(prim or A["key_metrics"])][cols]
                          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        else:
            report.append("DILEWATI: butuh skenario GAN dan klasik pada dosis yang sama.")

        section("DESAIN DOSIS (C): tren metrik terhadap dosis (0% = original), per metode")
        tr_rows = []
        base = dd[dd["method"] == "original"]
        for meth in [x for x in ("classic", "gan") if x in set(dd["method"])]:
            sub = pd.concat([base, dd[dd["method"] == meth]])
            for model, g in list(sub.groupby("model")) + [("ALL", sub)]:
                for m in A["key_metrics"]:
                    gg = g[["dose_pct", m]].dropna()
                    if gg["dose_pct"].nunique() < 3 or gg[m].nunique() < 2:
                        continue
                    rho, p = stats.spearmanr(gg["dose_pct"], gg[m])
                    lr = stats.linregress(gg["dose_pct"] / 100, gg[m])
                    tr_rows.append({"method": meth, "model": model, "metric": m, "spearman_rho": rho, "p_value": p,
                                    "slope_per_100pct": lr.slope, "slope_p": lr.pvalue, "n": len(gg)})
        tdf = pd.DataFrame(tr_rows)
        if len(tdf):
            tdf["p_holm"] = holm(tdf["p_value"].values)
            tdf.to_csv(S / "dose_trend.csv", index=False)
            report.append(tdf[(tdf["model"] == "ALL") & tdf["metric"].isin(prim or A["key_metrics"])]
                          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            report.append("(baris per model: dose_trend.csv)")
        else:
            report.append("DILEWATI: butuh minimal 3 tingkat dosis (termasuk 0%) per metode.")

        section("DESAIN DOSIS (D): ANOVA model × metode × dosis (tanpa original), interaksi orde-2")
        da = dd[dd["method"].isin(["gan", "classic"])].copy()
        da["dose_lvl"] = da["dose_pct"].astype(str)
        fac = [f for f in ("model", "method", "dose_lvl") if da[f].nunique() >= 2]
        lvls = [da[f].nunique() for f in fac]
        npar = 1 + sum(l - 1 for l in lvls) + sum((a - 1) * (b - 1) for a, b in itertools.combinations(lvls, 2))
        if fac and len(da) - npar > 0:
            an0 = []
            for m in (prim or A["key_metrics"]):
                try:
                    t = factorial_anova(da, m, fac, max_order=min(2, len(fac)))
                    an0.append(t)
                    report.append(f"\n[{m}]  R^2={t.attrs.get('r_squared', np.nan):.3f}\n" +
                                  t.drop(columns="metric").to_string(index=False, float_format=lambda x: f"{x:.4g}"))
                except Exception as e:
                    report.append(f"{m}: ANOVA tidak dapat dihitung ({type(e).__name__}: {e})")
            if an0:
                pd.concat(an0).to_csv(S / "dose_anova.csv", index=False)
            report.append("Baris 'method:dose_lvl' = apakah selisih GAN-klasik berubah menurut dosis (interaksi).")
        else:
            report.append("DILEWATI: data belum cukup untuk ANOVA (butuh >= 2 seed dan kedua metode).")

        # ---------------- L: PEMBANDING LEVEL LOSS (weighted_ce, focal; data latih = original)
        losses = loss_baselines(cfg, lv["strategy"])
        if losses:
            section("PEMBANDING LEVEL LOSS (L1): " + ", ".join(losses) + f" vs '{ref}' - digabung lintas model "
                    "(pasangan: model × seed)")
            le = df[df["strategy"].isin([ref] + losses)]
            rows = [paired_wilcoxon(le, ["model", "seed"], "strategy", m, ref) for m in A["key_metrics"]]
            e1 = pd.concat([r for r in rows if len(r)], ignore_index=True) if any(len(r) for r in rows) else pd.DataFrame()
            if len(e1):
                pm = e1["metric"].isin(prim)
                if pm.any():
                    e1.loc[pm, "p_holm_primary"] = holm(e1.loc[pm, "p_value"].values)
                e1.to_csv(S / "loss_vs_original_pooled.csv", index=False)
                cols = ["metric", "treatment", "n_pairs", "mean_ref", "mean_treat", "mean_diff", "p_value", "p_holm",
                        "rank_biserial"] + (["p_holm_primary"] if "p_holm_primary" in e1 else [])
                report.append("Pertanyaan: apakah mengubah fungsi loss saja (tanpa menambah citra) sudah membantu?")
                report.append(e1[e1["metric"].isin(prim or A["key_metrics"])][cols]
                              .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            else:
                report.append("DILEWATI: belum ada pasangan pembanding loss vs original (model × seed yang sama).")

            section("PEMBANDING LEVEL LOSS (L2): tiap skenario penambahan citra vs tiap pembanding loss - digabung "
                    "lintas model (pasangan: model × seed)")
            data_s = sorted(dd.loc[dd["method"].isin(["classic", "gan"]), "strategy"].unique(),
                            key=lambda s: list(cfg["strategies"]).index(s))
            e2 = []
            for L in losses:
                sub = df[df["strategy"].isin([L] + data_s)]
                rr = [paired_wilcoxon(sub, ["model", "seed"], "strategy", m, L) for m in A["key_metrics"]]
                e2 += [r for r in rr if len(r)]
            e2 = pd.concat(e2, ignore_index=True) if e2 else pd.DataFrame()
            if len(e2):
                e2["p_holm"] = np.nan  # koreksi Holm per metrik atas SEMUA perbandingan L2 (semua loss × skenario)
                for m, idx in e2.groupby("metric").groups.items():
                    e2.loc[idx, "p_holm"] = holm(e2.loc[idx, "p_value"].values)
                pm = e2["metric"].isin(prim)
                if pm.any():
                    e2.loc[pm, "p_holm_primary"] = holm(e2.loc[pm, "p_value"].values)
                e2["dose_pct"] = e2["treatment"].map(lambda s: int(round(float(cfg["strategies"][s]["dose"]) * 100)))
                e2["method"] = e2["treatment"].map(lambda s: cfg["strategies"][s].get("extra"))
                e2["_ref"] = e2["reference"].map(losses.index)
                e2["_m"] = e2["metric"].map({m: i for i, m in enumerate(A["key_metrics"])})
                e2 = e2.sort_values(["_m", "_ref", "method", "dose_pct"]).drop(columns=["_ref", "_m"])
                e2.to_csv(S / "data_vs_loss_pooled.csv", index=False)
                cols = ["metric", "reference", "treatment", "n_pairs", "mean_ref", "mean_treat", "mean_diff", "p_value",
                        "p_holm", "rank_biserial"] + (["p_holm_primary"] if "p_holm_primary" in e2 else [])
                report.append("mean_ref = pembanding loss, mean_treat = skenario penambahan citra; mean_diff > 0 berarti "
                              "menambah citra TB lebih baik daripada hanya mengubah loss.")
                report.append(e2[e2["metric"].isin(prim or A["key_metrics"])][cols]
                              .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
                report.append("(semua metrik: data_vs_loss_pooled.csv; perbandingan utama: dosis 100% vs weighted_ce, "
                              "karena bobot kelas inverse setara dengan penyeimbangan 1:1 dalam ekspektasi)")
            else:
                report.append("DILEWATI: butuh pembanding loss dan skenario dosis pada model × seed yang sama.")

    # ---------------- A & B: Friedman
    for treat, blocks in (("model", ["ratio", "strategy", "seed"]), ("strategy", ["ratio", "model", "seed"])):
        section(f"Friedman + Nemenyi antar {treat.upper()} (blok: {' × '.join(blocks)})")
        rows = []
        probe = df.pivot_table(index=blocks, columns=treat, values=A["key_metrics"][0], aggfunc="mean").dropna()
        if probe.shape[1] < 2 or len(probe) < 2:
            report.append(f"DILEWATI: butuh minimal 2 {treat} dan 2 blok lengkap (kombinasi {' × '.join(blocks)} "
                          f"yang memuat semua {treat}).\nSaat ini: {df[treat].nunique()} {treat}, {len(probe)} blok lengkap.")
            continue
        for m in A["key_metrics"]:
            r = friedman_nemenyi(df, blocks, treat, m, A["higher_is_better"], alpha)
            if "error" in r:
                report.append(f"{m}: {r['error']}")
                continue
            r["pairwise"].to_csv(S / f"nemenyi_{treat}_{m}.csv", index=False)
            rows.append({k: v for k, v in r.items() if k not in ("pairwise", "avg_rank", "means")}
                        | {f"rank_{k}": v for k, v in r["avg_rank"].items()})
            ranks = ", ".join(f"{k}={v:.2f}" for k, v in r["avg_rank"].items())
            sname = "chi2" if r["test"] == "friedman" else "Wilcoxon-W"
            report.append(f"{m:<18} [{r['test']}] {sname}={r['friedman_stat']:.2f} p={r['p_value']:.2e} "
                          f"Kendall-W={r['kendall_w']:.3f} "
                          f"CD={r['critical_difference']:.3f} | rank: {ranks}")
        pd.DataFrame(rows).to_csv(S / f"friedman_{treat}.csv", index=False)

    # ---------------- C: Wilcoxon vs referensi
    section(f"Wilcoxon berpasangan: strategi vs '{ref}' per model (pasangan: rasio × seed; koreksi Holm)")
    wl = []
    for m in A["key_metrics"]:
        w = paired_wilcoxon(df, ["ratio", "seed"], "strategy", m, ref, ["model"])
        wl.append(w)
    wil = pd.concat(wl, ignore_index=True) if wl else pd.DataFrame()
    if not len(wil):
        report.append(f"DILEWATI: butuh strategi referensi '{ref}' dan minimal 1 strategi lain pada model, rasio, "
                      f"dan seed yang sama.\nStrategi saat ini: {', '.join(lv['strategy'])}.")
    if len(wil):
        wil["p_holm_global"] = holm(wil["p_value"].values)
        prim = wil["metric"].isin(A.get("primary_metrics", []))
        if prim.any():  # koreksi lintas-metrik hanya di antara metrik primer yang ditetapkan sejak awal
            wil.loc[prim, "p_holm_primary"] = holm(wil.loc[prim, "p_value"].values)
        wil.to_csv(S / "wilcoxon_vs_reference.csv", index=False)
        cols = ["metric", "model", "treatment", "n_pairs", "mean_diff", "p_value", "p_holm", "rank_biserial"]
        cols += ["p_holm_primary"] if "p_holm_primary" in wil else []
        report.append("p_holm = koreksi Holm dalam satu metrik (antar model); p_holm_primary = koreksi Holm "
                      f"atas metrik primer {A.get('primary_metrics', [])} sekaligus (laporkan ini untuk klaim utama).")
        report.append(wil[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # per rasio (efek augmentasi bisa berbeda pada rasio ekstrem) - pasangan: seed
    wr = [paired_wilcoxon(df, ["seed"], "strategy", m, ref, ["model", "ratio"]) for m in A["key_metrics"]]
    wr = pd.concat(wr, ignore_index=True)
    if len(wr):
        wr.to_csv(S / "wilcoxon_vs_reference_per_ratio.csv", index=False)
        report.append("\n(per rasio: lihat wilcoxon_vs_reference_per_ratio.csv; n pasangan = jumlah seed, "
                      "daya uji rendah bila seed < 5)")

    # ---------------- D: tren terhadap rasio
    section("Tren metrik terhadap rasio ketidakseimbangan (Spearman; slope per kenaikan log2 rasio)")
    tr = []
    for (model, strat), g in df.groupby(["model", "strategy"]):
        for m in A["key_metrics"]:
            gg = g[["ratio_num", m]].dropna()
            if gg["ratio_num"].nunique() < 3:
                continue
            if gg[m].nunique() < 2:  # metrik konstan -> korelasi tak terdefinisi
                continue
            rho, p = stats.spearmanr(gg["ratio_num"], gg[m])
            lr = stats.linregress(np.log2(gg["ratio_num"]), gg[m])
            tr.append({"model": model, "strategy": strat, "metric": m, "spearman_rho": rho, "p_value": p,
                       "slope_per_log2_ratio": lr.slope, "slope_p": lr.pvalue, "r2": lr.rvalue ** 2, "n": len(gg)})
    tr = pd.DataFrame(tr)
    if not len(tr):
        if is_dose_design(cfg, lv["strategy"]) and len(lv["ratio"]) == 1:
            report.append("Tidak relevan untuk desain dosis (satu rasio). Tren terhadap dosis ada di DESAIN DOSIS (C).")
        else:
            report.append(f"DILEWATI: butuh minimal 3 rasio per model × strategi. Rasio saat ini: {', '.join(lv['ratio'])}.")
    if len(tr):
        tr["p_holm"] = holm(tr["p_value"].values)
        tr.to_csv(S / "trend_vs_ratio.csv", index=False)
        sel = tr[tr["metric"].isin(["tb_recall", "macro_f1"])]
        report.append(sel.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---------------- E: ANOVA faktorial
    section("ANOVA faktorial (Type II): model × strategi × rasio, interaksi orde-2")
    # ANOVA butuh desain faktorial lengkap: buang strategi/model yang tidak dijalankan di semua sel
    dfa = df.copy()
    for _ in range(3):
        cells = dfa.groupby(["model", "strategy", "ratio"]).size()
        n_r = dfa["ratio"].nunique()
        ok_s = [s for s in dfa["strategy"].unique()
                if len(cells.xs(s, level="strategy")) == dfa["model"].nunique() * n_r]
        dfa = dfa[dfa["strategy"].isin(ok_s)]
        ok_m = [m for m in dfa["model"].unique()
                if len(cells.xs(m, level="model").reset_index().query("strategy in @ok_s")) == len(ok_s) * n_r]
        dfa = dfa[dfa["model"].isin(ok_m)]
    dropped_s = sorted(set(df["strategy"]) - set(dfa["strategy"]))
    dropped_m = sorted(set(df["model"]) - set(dfa["model"]))
    if dropped_s or dropped_m:
        report.append(f"(desain belum lengkap -> dikeluarkan dari ANOVA: strategi {dropped_s}, model {dropped_m})")
    factors = [f for f in ("model", "strategy", "ratio") if dfa[f].nunique() >= 2]
    single = [f for f in ("model", "strategy", "ratio") if f not in factors]
    levels = [dfa[f].nunique() for f in factors]
    n_par = 1 + sum(l - 1 for l in levels) + sum((a - 1) * (b - 1) for a, b in itertools.combinations(levels, 2))
    if not factors or len(dfa) - n_par <= 0:
        report.append("DILEWATI: ANOVA butuh minimal satu faktor dengan >= 2 tingkat DAN ulangan (>= 2 seed per sel)\n"
                      f"agar sisa derajat bebas > 0. Saat ini: {len(dfa)} run, {n_par} parameter model, "
                      f"seed: {', '.join(lv['seed'])}.")
        factors = []
    elif single:
        report.append(f"(faktor dengan 1 tingkat tidak dimasukkan: {', '.join(single)})")
    an = []
    for m in (A["key_metrics"] if factors else []):
        try:
            t = factorial_anova(dfa, m, factors, max_order=min(2, len(factors)))
            an.append(t)
            report.append(f"\n[{m}]  R^2={t.attrs.get('r_squared', np.nan):.3f}\n" +
                          t.drop(columns="metric").to_string(index=False, float_format=lambda x: f"{x:.4g}"))
        except Exception as e:
            report.append(f"{m}: ANOVA tidak dapat dihitung ({type(e).__name__}: {e})")
    if an:
        pd.concat(an).to_csv(S / "anova_factorial.csv", index=False)
    report.append("\nCatatan: periksa asumsi normalitas residu; Friedman/Wilcoxon adalah alternatif non-parametrik.")

    # ---------------- F: DeLong & McNemar pada ensemble
    section("DeLong (AUC TB one-vs-rest) & McNemar pada prediksi ensemble antar-seed")
    pcols = [f"prob_{c}" for c in names]
    design = df["design"].iloc[0]
    ens = {}
    for f in D["ensembles"].glob(f"{design}__*.csv"):
        _, r, s, m = f.stem.split("__")
        ens[(r, s, m)] = pd.read_csv(f).set_index("image_id").sort_index()

    def compare(a, b, label_a, label_b, extra):
        ea, eb = ens[a], ens[b]
        idx = ea.index.intersection(eb.index)
        y = ea.loc[idx, "label"].values
        pa, pb = ea.loc[idx, pcols].values, eb.loc[idx, pcols].values
        dl = delong_test((y == tb_idx).astype(int), pa[:, tb_idx], pb[:, tb_idx])
        mc = mcnemar_test(y, pa.argmax(1), pb.argmax(1))
        # McNemar khusus kasus TB (sensitivitas TB)
        tbm = y == tb_idx
        mc_tb = mcnemar_test(y[tbm], pa[tbm].argmax(1), pb[tbm].argmax(1))
        return {**extra, "a": label_a, "b": label_b, "tb_auc_a": dl["auc_a"], "tb_auc_b": dl["auc_b"],
                "tb_auc_diff": dl["auc_diff"], "delong_z": dl["z"], "delong_p": dl["p_value"],
                "mcnemar_b": mc["b"], "mcnemar_c": mc["c"], "mcnemar_p": mc["p_value"],
                "tb_sens_a": float((pa[tbm].argmax(1) == tb_idx).mean()),
                "tb_sens_b": float((pb[tbm].argmax(1) == tb_idx).mean()), "mcnemar_tb_only_p": mc_tb["p_value"]}

    ratios = sorted({k[0] for k in ens}, key=ratio_sort_key)
    strategies = [s for s in cfg["strategies"] if any(k[1] == s for k in ens)]
    models = [m for m in cfg["models"] if any(k[2] == m for k in ens)]
    f1 = [compare((r, s, m), (r, ref, m), s, ref, {"family": "strategy_vs_ref", "model": m, "ratio": r})
          for m in models for r in ratios for s in strategies
          if s != ref and (r, s, m) in ens and (r, ref, m) in ens]
    f2 = [compare((r, s, a), (r, s, b), a, b, {"family": "model_pairs", "strategy": s, "ratio": r})
          for s in strategies for r in ratios for a, b in itertools.combinations(models, 2)
          if (r, s, a) in ens and (r, s, b) in ens]
    f3 = []
    sm_ = {s: (cfg["strategies"][s].get("extra"), cfg["strategies"][s].get("dose")) for s in strategies}
    for sg, (eg, dg) in sm_.items():
        if eg != "gan":
            continue
        for sc, (ec, dc) in sm_.items():
            if ec == "classic" and dc == dg:
                f3 += [compare((r, sg, m), (r, sc, m), sg, sc, {"family": "gan_vs_classic_same_dose", "model": m,
                                                             "ratio": r, "dose_pct": int(round(dg * 100))})
                       for m in models for r in ratios if (r, sg, m) in ens and (r, sc, m) in ens]
    f4 = []  # skenario penambahan citra vs pembanding level loss, per model
    for L in loss_baselines(cfg, strategies):
        for sd, (ed, dd_) in sm_.items():
            if ed in ("classic", "gan") and dd_ is not None:
                f4 += [compare((r, sd, m), (r, L, m), sd, L, {"family": "data_vs_loss", "model": m, "ratio": r,
                                                            "dose_pct": int(round(float(dd_) * 100))})
                       for m in models for r in ratios if (r, sd, m) in ens and (r, L, m) in ens]
    for name, rows in (("delong_mcnemar_strategy_vs_ref", f1), ("delong_mcnemar_model_pairs", f2),
                       ("delong_mcnemar_gan_vs_classic", f3), ("delong_mcnemar_data_vs_loss", f4)):
        t = pd.DataFrame(rows)
        if len(t):
            for p in ("delong_p", "mcnemar_p", "mcnemar_tb_only_p"):
                t[p + "_holm"] = holm(t[p].values)
            t.to_csv(S / f"{name}.csv", index=False)
            report.append(f"\n[{name}] {len(t)} perbandingan; signifikan DeLong (Holm): "
                          f"{int((t['delong_p_holm'] < alpha).sum())}, McNemar (Holm): "
                          f"{int((t['mcnemar_p_holm'] < alpha).sum())}")

    if not f1 and not f2 and not f3 and not f4:
        report.append("DILEWATI: belum ada pasangan model/strategi pada rasio yang sama (jalankan 04_aggregate_results.py dulu).")
    (S / "stats_report.txt").write_text("\n".join(report), encoding="utf-8")
    log.info(f"Laporan statistik: {S / 'stats_report.txt'}")
    print("\n".join(report))


if __name__ == "__main__":
    main()
