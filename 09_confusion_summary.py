#!/usr/bin/env python
"""Langkah 9: kumpulkan confusion matrix ASLI (test set) dari semua run + pola kesalahan per citra TB.

Membaca  <runs>/<run_id>/metrics.json        -> test.confusion_matrix
         <runs>/<run_id>/test_predictions.csv -> prediksi per citra (bila ada)
Menulis  <out>/confusion_all.csv      satu baris per run (9 sel matriks; baris = label asli, kolom = prediksi)
         <out>/tb_image_errors.csv    per citra TB uji: berapa run memprediksi Normal / non-TB / TB
Urutan kelas: 0 = Normal, 1 = Abnormal_nonTB, 2 = Abnormal_TB.

Contoh (dari folder proyek):
  python 09_confusion_summary.py
  python 09_confusion_summary.py --runs work\\results\\runs work_abl\\results\\runs
"""
import argparse
import glob
import json
import os

import pandas as pd

NAMES = ["normal", "nontb", "tb"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=[os.path.join("work", "results", "runs"),
                                                  os.path.join("work_abl", "results", "runs")])
    ap.add_argument("--out", default=os.path.join("work", "results", "tables"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows, preds = [], []
    for root in args.runs:
        if not os.path.isdir(root):
            print(f"[lewati] folder tidak ada: {root}")
            continue
        source = "ablation" if "work_abl" in os.path.normpath(root) else "main"
        for mpath in sorted(glob.glob(os.path.join(root, "*", "metrics.json"))):
            with open(mpath, encoding="utf-8") as f:
                m = json.load(f)
            meta, cm = m["meta"], m["test"]["confusion_matrix"]
            row = {"source": source, "run_id": meta["run_id"], "strategy": meta["strategy"],
                   "model": meta["model"], "seed": meta["seed"]}
            for i, a in enumerate(NAMES):
                for j, b in enumerate(NAMES):
                    row[f"{a}_to_{b}"] = int(cm[i][j])
            rows.append(row)
            ppath = os.path.join(os.path.dirname(mpath), "test_predictions.csv")
            if os.path.exists(ppath):
                p = pd.read_csv(ppath, usecols=["image_id", "label", "pred"])
                p = p[p["label"] == 2].copy()
                p["source"], p["run_id"] = source, meta["run_id"]
                preds.append(p)

    if not rows:
        raise SystemExit("Tidak ada metrics.json yang ditemukan. Periksa opsi --runs.")
    cm = pd.DataFrame(rows)
    cm.to_csv(os.path.join(args.out, "confusion_all.csv"), index=False)

    for src, g in cm.groupby("source"):
        tb_err = int(g["tb_to_normal"].sum() + g["tb_to_nontb"].sum())
        abn_to_n = int(g["tb_to_normal"].sum() + g["nontb_to_normal"].sum())
        fp_tb = int(g["normal_to_tb"].sum() + g["nontb_to_tb"].sum())
        print(f"\n[{src}] {len(g)} run")
        print(f"  TB salah            : {tb_err}  (ke Normal {int(g['tb_to_normal'].sum())}, ke non-TB {int(g['tb_to_nontb'].sum())})")
        print(f"  Abnormal -> Normal  : {abn_to_n}  (rata-rata {abn_to_n / len(g):.2f} per run dari 886 citra abnormal)")
        print(f"  Prediksi TB palsu   : {fp_tb}  (dari Normal {int(g['normal_to_tb'].sum())}, dari non-TB {int(g['nontb_to_tb'].sum())})")

    if preds:
        p = pd.concat(preds, ignore_index=True)
        t = (p.assign(pred_normal=p["pred"] == 0, pred_nontb=p["pred"] == 1, pred_tb=p["pred"] == 2)
               .groupby(["source", "image_id"])[["pred_normal", "pred_nontb", "pred_tb"]].sum().astype(int).reset_index())
        t["n_runs"] = t[["pred_normal", "pred_nontb", "pred_tb"]].sum(axis=1)
        t["n_errors"] = t["n_runs"] - t["pred_tb"]
        t = t.sort_values(["source", "n_errors"], ascending=[True, False])
        t.to_csv(os.path.join(args.out, "tb_image_errors.csv"), index=False)
        for src, g in t.groupby("source"):
            err = g[g["n_errors"] > 0]
            print(f"  [{src}] citra TB yang pernah salah: {len(err)} dari {len(g)}; "
                  f"10 citra tersulit menyumbang {int(err['n_errors'].head(10).sum())} dari {int(g['n_errors'].sum())} kesalahan")

    print(f"\nTersimpan: {os.path.join(args.out, 'confusion_all.csv')}"
          + (f" dan {os.path.join(args.out, 'tb_image_errors.csv')}" if preds else ""))


if __name__ == "__main__":
    main()
