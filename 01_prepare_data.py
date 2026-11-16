#!/usr/bin/env python
"""Langkah 1: indeks TBX11K -> pemetaan 3 kelas -> cache resize -> deteksi duplikat.

Keluaran (work/data/):
  index_all.csv        semua citra + label sumber + kelas (termasuk yang di-exclude)
  index.csv            citra berlabel yang dipakai (dengan dup_group)
  label_counts.csv     jumlah per label sumber dan kelas penelitian
  duplicates.csv       pasangan hampir-duplikat
  data_report.json     ringkasan untuk ditulis di bagian Metode
"""
import argparse

import pandas as pd

from src.dedup import add_dup_groups, merge_with_embedding_pairs
from src.tbx11k import add_image_meta, apply_class_map, build_index, cache_resized
from src.utils import get_dirs, get_logger, load_config, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config(args.config)
    D = get_dirs(cfg)
    log = get_logger("prep", str(D["logs"] / "01_prepare_data.log"))
    dcfg = cfg["data"]

    df = build_index(cfg["paths"]["data_root"], dcfg["index_mode"], dcfg.get("csv"), log)
    df = apply_class_map(df, dcfg["class_map"], dcfg["class_names"])
    df = add_image_meta(df, args.workers)
    df.to_csv(D["data"] / "index_all.csv", index=False)

    counts = (df.groupby(["source_folder", "source_label", "cls"]).size()
                .rename("n").reset_index().sort_values(["cls", "source_label"]))
    counts.to_csv(D["data"] / "label_counts.csv", index=False)
    log.info("Jumlah per label sumber -> kelas penelitian:\n" + counts.to_string(index=False))

    lab = df[df["label"] >= 0].reset_index(drop=True)
    if dcfg.get("cache_size"):
        log.info(f"Membuat cache resize {dcfg['cache_size']} px ...")
        lab = cache_resized(lab, D["cache"], dcfg["cache_size"], args.workers)

    report = {"n_all_images": len(df), "n_labeled_used": len(lab),
              "per_class_before_dedup": lab["cls"].value_counts().to_dict(),
              "image_size_counts": lab.groupby(["width", "height"]).size().astype(int)
                                      .rename("n").reset_index().to_dict("records")}

    if cfg["dedup"]["enabled"]:
        hcol = "path_cached" if "path_cached" in lab else "path"
        lab, pairs = add_dup_groups(lab, hcol, cfg["dedup"]["max_hamming"], args.workers, log,
                                    cfg["dedup"].get("hash_size", 16), cfg["dedup"].get("max_group_size", 20))
        pairs.to_csv(D["data"] / "duplicates.csv", index=False)
        report["n_duplicate_pairs"] = len(pairs)
        report["n_groups_size_gt1"] = int((lab["dup_group"].value_counts() > 1).sum())
        if cfg["dedup"].get("drop_label_conflicts") and len(pairs) and pairs["label_conflict"].any():
            conf_groups = lab.loc[pd.concat([pairs.loc[pairs.label_conflict, "i"],
                                             pairs.loc[pairs.label_conflict, "j"]]), "dup_group"].unique()
            # konflik dinilai pada kelas penelitian, bukan label sumber
            g = lab[lab["dup_group"].isin(conf_groups)].groupby("dup_group")["label"].nunique()
            bad = g[g > 1].index
            n_drop = int(lab["dup_group"].isin(bad).sum())
            lab = lab[~lab["dup_group"].isin(bad)].reset_index(drop=True)
            report["n_dropped_label_conflict"] = n_drop
            log.warning(f"{n_drop} citra dibuang karena duplikat dengan kelas berbeda.")
    else:
        lab["dup_group"] = range(len(lab))

    # ---- lapis kedua: citra sangat mirip menurut fitur CNN (mis. foto pasien yang sama) -> satu grup
    ecfg = cfg["dedup"].get("embedding", {})
    if cfg["dedup"]["enabled"] and ecfg.get("enabled"):
        import numpy as np
        from src.embeddings import compute_embeddings, mutual_topk_pairs
        hcol = "path_cached" if "path_cached" in lab else "path"
        log.info(f"Menghitung embedding {ecfg.get('model', 'resnet50.tv_in1k')} untuk {len(lab)} citra ...")
        emb = compute_embeddings(lab[hcol].tolist(), ecfg.get("model", "resnet50.tv_in1k"), 224,
                                 ecfg.get("pretrained", True), logger=log)
        np.save(D["data"] / "embeddings.npy", emb)
        lab[["image_id"]].to_csv(D["data"] / "embeddings_ids.csv", index=False)
        ep = mutual_topk_pairs(emb, float(ecfg.get("threshold", 0.97)), int(ecfg.get("mutual_k", 2)))
        pos = {iid: k for k, iid in enumerate(lab["image_id"])}
        ph = pairs if cfg["dedup"]["enabled"] and len(pairs) else pd.DataFrame(columns=["image_i", "image_j"])
        ph = ph[ph["image_i"].isin(pos) & ph["image_j"].isin(pos)]
        ph = pd.DataFrame({"i": ph["image_i"].map(pos), "j": ph["image_j"].map(pos)})
        lab, info = merge_with_embedding_pairs(lab, ep, ph, cfg["dedup"].get("max_group_size", 20), log)
        epd = pd.DataFrame(ep, columns=["i", "j", "cosine"])
        if len(epd):
            epd["image_i"], epd["image_j"] = lab["image_id"].values[epd.i], lab["image_id"].values[epd.j]
            epd["cls_i"], epd["cls_j"] = lab["cls"].values[epd.i], lab["cls"].values[epd.j]
        epd.sort_values("cosine", ascending=False).to_csv(D["data"] / "embedding_pairs.csv", index=False)
        report["embedding_dedup"] = {**info, "threshold": ecfg.get("threshold", 0.97), "model": ecfg.get("model")}
        log.info(f"Embedding: {info['n_embedding_pairs']} pasangan tetangga timbal balik >= {ecfg.get('threshold', 0.97)}; grup >1 "
                 f"(pHash -> gabungan): {info['groups_gt1_phash']} -> {info['groups_gt1_combined']}; "
                 f"{info['n_images_in_groups']} citra berada dalam grup")

    report["per_class_final"] = lab["cls"].value_counts().to_dict()
    lab.to_csv(D["data"] / "index.csv", index=False)
    save_json(report, D["data"] / "data_report.json")
    log.info(f"Selesai. Kelas akhir: {report['per_class_final']}")


if __name__ == "__main__":
    main()
