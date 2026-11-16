#!/usr/bin/env python
"""Langkah 10: kumpulkan prediksi test set per citra dari SEMUA run ke satu file.
"""
import argparse
import glob
import json
import os
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=[os.path.join("work", "results", "runs"), os.path.join("work_abl", "results", "runs")])
    ap.add_argument("--out", default=os.path.join("work", "results", "tables"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    parts, n_missing = [], 0

    for root in args.runs:
        if not os.path.isdir(root):
            print(f"[lewati] folder tidak ada: {root}")
            continue
        
        # FIX 1: Gunakan normpath sebelum pengecekan string 'work_abl'
        norm_root = os.path.normpath(root)
        source = "ablation" if "work_abl" in norm_root else "main"
        
        # FIX 2: Gunakan recursive=True dan '**' agar bisa mencari di folder yang lebih dalam
        search_pattern = os.path.join(root, "**", "metrics.json")
        for mpath in sorted(glob.glob(search_pattern, recursive=True)):
            ppath = os.path.join(os.path.dirname(mpath), "test_predictions.csv")
            
            if not os.path.exists(ppath):
                n_missing += 1
                continue
                
            try:
                with open(mpath, encoding="utf-8") as f:
                    meta = json.load(f)["meta"]
                
                p = pd.read_csv(ppath)
                
                # Memastikan kolom yang dicari benar-benar ada di CSV
                available_cols = ["image_id", "label", "pred"] + [c for c in p.columns if c.startswith("prob_")]
                # Filter hanya kolom yang ada di dalam berkas CSV
                available_cols = [c for c in available_cols if c in p.columns]
                p = p[available_cols]
                
                # Gunakan .get() untuk menghindari KeyError jika key tidak ada di json
                p.insert(0, "seed", meta.get("seed", None))
                p.insert(0, "model", meta.get("model", None))
                p.insert(0, "strategy", meta.get("strategy", None))
                # Menggunakan nama folder sebagai run_id cadangan jika meta["run_id"] kosong
                p.insert(0, "run_id", meta.get("run_id", os.path.basename(os.path.dirname(mpath))))
                p.insert(0, "source", source)
                
                parts.append(p)
            except Exception as e:
                print(f"[ERROR] Gagal memproses {mpath}: {e}")
                continue

    if not parts:
        raise SystemExit(f"Tidak ada test_predictions.csv yang ditemukan. Periksa opsi --runs atau pastikan file ada.")

    df = pd.concat(parts, ignore_index=True)
    
    for c in [c for c in df.columns if c.startswith("prob_")]:
        df[c] = df[c].round(5)
        
    dst = os.path.join(args.out, "predictions_all.csv.gz")
    df.to_csv(dst, index=False, compression="gzip")
    
    print(f"\nOK: {df.run_id.nunique()} run, {len(df)} baris -> {dst}")
    print(df.groupby("source").run_id.nunique().to_string())
    if n_missing:
        print(f"PERINGATAN: {n_missing} run tanpa test_predictions.csv")

if __name__ == "__main__":
    main()
