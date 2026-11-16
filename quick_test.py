#!/usr/bin/env python
"""Uji cepat instalasi (±3-10 menit). TIDAK butuh dataset TBX11K.

Membuat dataset tiruan kecil berstruktur TBX11K di folder _quick_test/, lalu menjalankan
SELURUH pipeline (langkah 1, 2, 2b, 2c, 3-7) dengan pengaturan sangat kecil. Jika semua langkah [LULUS],
instalasi Anda benar dan Anda siap memakai data asli.

Catatan: angka akurasi pada uji ini TIDAK bermakna (data tiruan, 2 epoch, tanpa pretrained).

Cara pakai:
  python quick_test.py            # jalankan uji, hapus folder uji bila lulus
  python quick_test.py --keep     # simpan folder _quick_test/ untuk dilihat hasilnya
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

HERE = Path(__file__).resolve().parent
QT = HERE / "_quick_test"


def make_fake_tbx11k(root: Path, seed: int = 0):
    rng = np.random.default_rng(seed)
    for d in ["imgs/health", "imgs/sick", "imgs/tb", "imgs/test", "annotations/xml"]:
        (root / d).mkdir(parents=True, exist_ok=True)

    def img(kind):
        a = rng.normal(110, 25, (96, 96))
        if kind == "sick":
            a[20:50, 20:50] += 50
        if kind == "tb":
            a[10:30, 55:85] += 80
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))

    for i in range(120):
        img("health").save(root / f"imgs/health/h{i:04d}.png")
        img("sick").save(root / f"imgs/sick/s{i:04d}.png")
    kinds = ["ActiveTuberculosis"] * 45 + ["ObsoletePulmonaryTuberculosis"] * 6 + ["both"] * 3 + \
            ["PulmonaryTuberculosis"] * 3
    for i, k in enumerate(kinds):
        img("tb").save(root / f"imgs/tb/tb{i:04d}.png")
        objs = ["ActiveTuberculosis", "ObsoletePulmonaryTuberculosis"] if k == "both" else [k]
        body = "".join(f"<object><name>{o}</name></object>" for o in objs)
        (root / f"annotations/xml/tb{i:04d}.xml").write_text(
            f"<annotation><filename>tb{i:04d}.png</filename>{body}</annotation>", encoding="utf-8")
    for i in range(10):
        img("health").save(root / f"imgs/test/t{i:04d}.png")
    shutil.copy(root / "imgs/health/h0000.png", root / "imgs/health/h9000.png")  # duplikat sengaja


def make_config(workers: int) -> Path:
    cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    cfg["paths"] = {"data_root": str(QT / "TBX11K"), "work_dir": str(QT / "work")}
    cfg["data"]["cache_size"] = 64
    cfg["imbalance"]["ratios"] = ["orig"]
    cfg["models"] = {"resnet18": {"timm_name": "resnet18", "img_size": 64, "batch_size": 16}}
    cfg["experiment"] = {"models": ["resnet18"], "strategies": ["baseline", "classic_gap50", "gan_gap50", "gan_gap100",
                                                                "weighted_ce", "focal"],
                         "seeds": [1, 2]}
    cfg["gan"].update(img_size=64, batch_size=16, steps=60, log_every=20, sample_every=30, eval_every=30,
                      n_eval_samples=64, ckpt_every=30)
    cfg["train"].update(pretrained=False, epochs=2, min_epochs=1, num_workers=workers, patience=3,
                        lr=1e-3, keep_checkpoint=True)
    cfg["evaluation"]["n_bootstrap"] = 50
    p = QT / "quick_config.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return p


def run(step_name, args, log):
    t0 = time.time()
    print(f"  {step_name:<38}", end="", flush=True)
    r = subprocess.run([sys.executable] + args, cwd=HERE, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    log.write(f"\n\n##### {step_name}\n{r.stdout}\n{r.stderr}")
    ok = r.returncode == 0
    print(f"{'[LULUS]' if ok else '[GAGAL]'}  ({time.time() - t0:.0f} dtk)")
    if not ok:
        tail = (r.stderr or r.stdout).strip().splitlines()[-15:]
        print("    --- pesan error terakhir ---")
        for line in tail:
            print("    " + line)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="jangan hapus folder _quick_test")
    ap.add_argument("--workers", type=int, default=2, help="num_workers DataLoader (uji multiprocessing)")
    args = ap.parse_args()

    print("UJI CEPAT INSTALASI\n")
    if QT.exists():
        shutil.rmtree(QT)
    QT.mkdir()
    print("  Membuat dataset tiruan ...", end="", flush=True)
    make_fake_tbx11k(QT / "TBX11K")
    cfgp = str(make_config(args.workers))
    print(" selesai\n")

    ext_csv = QT / "external.csv"
    steps = [
        ("Langkah 1  prepare_data", ["01_prepare_data.py", "--config", cfgp, "--workers", "2"]),
        ("Langkah 2  make_splits", ["02_make_splits.py", "--config", cfgp]),
        ("Langkah 2b train_gan", ["02b_train_gan.py", "--config", cfgp]),
        ("Langkah 2c generate_synthetic", ["02c_generate_synthetic.py", "--config", cfgp]),
        ("Langkah 3  run_experiments (12 run)", ["03_run_experiments.py", "--config", cfgp]),
        ("Langkah 4  aggregate_results", ["04_aggregate_results.py", "--config", cfgp]),
        ("Langkah 5  statistical_tests", ["05_statistical_tests.py", "--config", cfgp]),
        ("Langkah 6  make_figures", ["06_make_figures.py", "--config", cfgp]),
        ("Langkah 7  evaluate_external", ["07_evaluate_external.py", "--config", cfgp, "--csv", str(ext_csv),
                                          "--name", "uji", "--filter", "*__rorig__*__s1"]),
    ]
    log = open(QT / "quick_test_log.txt", "w", encoding="utf-8")
    t0 = time.time()
    all_ok = True
    for name, a in steps:
        if name.startswith("Langkah 7"):
            import pandas as pd
            t = pd.read_csv(QT / "work/splits/test.csv").sample(20, random_state=0)
            t[["path", "cls"]].to_csv(ext_csv, index=False)
        if not run(name, a, log):
            all_ok = False
            break
    log.close()

    expected = ["work/synthetic/gan/quality_report.json", "work/synthetic/gan/fig_real_vs_synthetic.png",
                "work/results/tables/results.xlsx", "work/results/stats/stats_report.txt",
                "work/figures/fig02_tb_recall_vs_dose.png", "work/results/external/uji/external_all_runs.csv",
                "work/results/tables/table_loss_baselines_macro_f1.csv", "work/results/stats/loss_vs_original_pooled.csv",
                "work/results/stats/data_vs_loss_pooled.csv"]
    if all_ok:
        missing = [e for e in expected if not (QT / e).exists()]
        if missing:
            all_ok = False
            print(f"\n  File keluaran tidak ditemukan: {missing}")

    print(f"\nTotal waktu: {(time.time() - t0) / 60:.1f} menit")
    if all_ok:
        print("\n>>> SEMUA LANGKAH LULUS. Instalasi Anda benar. <<<")
        if args.keep:
            print(f"Folder uji disimpan di: {QT}")
        else:
            shutil.rmtree(QT, ignore_errors=True)
            print("Folder uji sudah dihapus. (Gunakan --keep untuk menyimpannya.)")
        print("Langkah berikutnya: unduh TBX11K (PANDUAN_PC.md Bagian 7).")
    else:
        print(f"\n>>> ADA LANGKAH YANG GAGAL. <<<\nLog lengkap: {QT / 'quick_test_log.txt'}")
        print("Cocokkan pesan error dengan bagian Troubleshooting di PANDUAN_PC.md.")
        print("Jika error menyebut DataLoader/worker/BrokenPipe, coba: python quick_test.py --workers 0")
        sys.exit(1)


if __name__ == "__main__":
    main()
