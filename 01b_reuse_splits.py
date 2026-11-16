#!/usr/bin/env python
"""Langkah 1b (pengganti langkah 1-2 untuk versi resolusi lain, mis. 512 px):
salin indeks data dan split dari folder hasil yang sudah ada (mis. work/ versi 224 px) ke folder
hasil baru, lalu siapkan cache citra pada resolusi baru.

Mengapa disalin, bukan dibuat ulang? Deteksi duplikat (pHash) dihitung dari citra cache; resolusi
cache berbeda bisa menghasilkan grup duplikat sedikit berbeda dan akhirnya split yang berbeda.
Dengan menyalin, citra latih/validasi/uji IDENTIK dengan versi sebelumnya sehingga perbandingan
antar-resolusi adil (hanya resolusi yang berbeda).

Contoh:
  python 01b_reuse_splits.py --config config_512.yaml --from ./work
"""
import argparse
import hashlib
import shutil
from pathlib import Path

import pandas as pd

from src.tbx11k import cache_resized
from src.utils import get_dirs, get_logger, load_config, save_json


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_512.yaml")
    ap.add_argument("--from", dest="src", default="./work", help="folder hasil sumber (berisi data/ dan splits/)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="timpa split yang sudah ada di folder tujuan")
    args = ap.parse_args()
    cfg = load_config(args.config)
    D = get_dirs(cfg)
    log = get_logger("reuse", str(D["logs"] / "01b_reuse_splits.log"))
    src = Path(args.src).resolve()

    if src == D["work"].resolve():
        raise SystemExit("Folder sumber dan tujuan sama. Periksa --from dan paths.work_dir di config.")
    for need in ("data/index.csv", "splits/test.csv", "splits/val.csv", "splits/train_full.csv"):
        if not (src / need).exists():
            raise SystemExit(f"Tidak ditemukan: {src / need}\nJalankan dulu langkah 1-2 untuk versi sumber, "
                             "atau periksa opsi --from.")
    if (D["splits"] / "test.csv").exists() and not args.force:
        log.info(f"Split sudah ada di {D['splits']} - dilewati (pakai --force untuk menimpa).")
        return

    design = cfg["imbalance"]["design"]
    missing = [r for r in cfg["imbalance"]["ratios"]
               if not (src / "splits" / f"train_{design}_r{r}.csv").exists()]
    if missing:
        raise SystemExit(f"Subset rasio {missing} tidak ada di sumber. Rasio di config harus sama dengan versi sumber.")

    idx = pd.read_csv(src / "data" / "index.csv")
    lost = idx[~idx["path"].map(lambda p: Path(p).exists())]
    if len(lost):
        raise SystemExit(f"{len(lost)} citra asli tidak ditemukan (contoh: {lost['path'].iloc[0]}). "
                         "Data harus berada di lokasi yang sama seperti saat langkah 1 versi sumber.")

    size = cfg["data"].get("cache_size")
    if not size:
        new_cached = None
        log.info("cache_size kosong: CNN dan GAN membaca citra asli langsung.")
    else:
        log.info(f"Menyiapkan cache {size} px untuk {len(idx)} citra di {D['cache']} ...")
        new_cached = cache_resized(idx.drop(columns=["path_cached"], errors="ignore"), D["cache"], size, args.workers)
        new_cached = dict(zip(new_cached["image_id"], new_cached["path_cached"]))
        if "width" in idx:
            small = idx[(idx[["width", "height"]].min(axis=1)) < size]
            if len(small):
                log.warning(f"{len(small)} citra asli lebih kecil dari {size} px dan akan diperbesar (upsampling).")

    def remap(df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(columns=["path_cached"], errors="ignore")
        if new_cached is not None:
            df["path_cached"] = df["image_id"].map(new_cached)
        return df

    remap(idx).to_csv(D["data"] / "index.csv", index=False)
    for f in ("index_all.csv", "label_counts.csv", "duplicates.csv", "data_report.json",
              "embeddings.npy", "embeddings_ids.csv", "embedding_pairs.csv"):
        if (src / "data" / f).exists():
            shutil.copy2(src / "data" / f, D["data"] / f)
    n = 0
    for f in sorted((src / "splits").iterdir()):
        if f.suffix == ".csv" and "image_id" in pd.read_csv(f, nrows=0).columns:
            remap(pd.read_csv(f)).to_csv(D["splits"] / f.name, index=False)
            n += 1
        elif f.is_file():
            shutil.copy2(f, D["splits"] / f.name)

    # bukti bahwa split identik (dicatat untuk bagian Metode / audit)
    ident = {}
    for f in ("train_full.csv", "val.csv", "test.csv"):
        a = pd.read_csv(src / "splits" / f)["image_id"].sort_values().tolist()
        b = pd.read_csv(D["splits"] / f)["image_id"].sort_values().tolist()
        ident[f] = a == b
    assert all(ident.values()), f"Split tidak identik: {ident}"
    save_json({"source_work_dir": str(src), "target_work_dir": str(D["work"]), "cache_size": size,
               "identical_image_ids": ident, "n_split_files": n,
               "source_test_csv_md5": md5(src / "splits" / "test.csv")}, D["splits"] / "reused_from.json")
    log.info(f"{n} file split disalin dari {src / 'splits'}; train/val/test IDENTIK dengan sumber.")
    log.info(f"Komposisi:\n{pd.read_csv(D['splits'] / 'split_summary.csv').to_string(index=False)}")
    log.info("Langkah berikutnya: python 02b_train_gan.py --config " + args.config)


if __name__ == "__main__":
    main()
