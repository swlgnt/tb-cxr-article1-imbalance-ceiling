#!/usr/bin/env python
"""Langkah 2: split train/val/test tetap + subset latih per rasio ketidakseimbangan.

Keluaran (work/splits/):
  base_split.csv, train_full.csv, val.csv, test.csv
  train_<design>_r<rasio>.csv      subset latih per rasio
  split_summary.csv                komposisi train/val/test per kelas
  subsets_summary.csv              komposisi subset latih per rasio (Tabel dataset di paper)
"""
import argparse

import pandas as pd

from src.splitting import make_base_split, make_imbalance_subsets
from src.utils import get_dirs, get_logger, load_config, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    D = get_dirs(cfg)
    log = get_logger("split", str(D["logs"] / "02_make_splits.log"))
    names = cfg["data"]["class_names"]

    df = pd.read_csv(D["data"] / "index.csv")
    sc = cfg["split"]
    df = make_base_split(df, sc["test_frac"], sc["val_frac"], sc["seed"],
                         "dup_group" if cfg["dedup"]["enabled"] else None)
    df.to_csv(D["splits"] / "base_split.csv", index=False)
    for s in ("train", "val", "test"):
        df[df["split"] == s].to_csv(D["splits"] / (f"{s}_full.csv" if s == "train" else f"{s}.csv"), index=False)

    summ = pd.crosstab(df["split"], df["cls"]).reindex(index=["train", "val", "test"], columns=names)
    summ["total"] = summ.sum(axis=1)
    summ.to_csv(D["splits"] / "split_summary.csv")
    log.info("Komposisi split:\n" + summ.to_string())

    ic = cfg["imbalance"]
    minority = names.index(cfg["data"]["minority_class"])
    subsets, ssum = make_imbalance_subsets(df[df["split"] == "train"], ic["design"], ic["ratios"], minority,
                                           sc["seed"], ic.get("minority_n"), ic.get("majority_n"), log)
    for tag, sub in subsets.items():
        sub.to_csv(D["splits"] / f"train_{ic['design']}_r{tag}.csv", index=False)
    ssum = ssum.rename(columns={f"n_class{i}": f"n_{c}" for i, c in enumerate(names)})
    ssum.to_csv(D["splits"] / "subsets_summary.csv", index=False)
    log.info("Subset latih per rasio:\n" + ssum.to_string(index=False))
    save_json({"split": sc, "imbalance": ic, "split_summary": summ.to_dict(),
               "subsets": ssum.to_dict("records")}, D["splits"] / "splits_report.json")


if __name__ == "__main__":
    main()
