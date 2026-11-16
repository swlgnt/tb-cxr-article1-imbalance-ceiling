#!/usr/bin/env python
"""Langkah 2c: bangkitkan citra TB sintetis dari generator terbaik + laporan kualitas.

Jumlah citra otomatis = kebutuhan terbesar untuk menyeimbangkan kelas TB di antara semua
subset rasio (mis. pada rasio asli: n_mayoritas - n_TB).

Keluaran (work/synthetic/<nama>/):
  images/*.png              citra sintetis
  synthetic.csv             daftar citra (path, cls, label, generator) -> dibaca strategi 'gan_basic'
  quality_report.json       FID, KID, SSIM, diversitas, indikator memorisasi
  nearest_neighbors.csv     tiap citra sintetis -> citra latih terdekat (jarak, SSIM)
  fig_real_vs_synthetic.png kisi citra asli vs sintetis (untuk paper)
  fig_nearest_pairs.png     8 citra sintetis paling mirip citra latih (cek memorisasi manual)
  fig_nn_distance.png       histogram jarak tetangga terdekat: sintetis vs asli
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.gan import Generator, gen_batch_size  # noqa: E402
from src.gan_eval import full_quality_report, load_uint8, to_uint8  # noqa: E402
from src.tbx11k import restore_cache  # noqa: E402
from src.utils import get_dirs, get_logger, git_info, load_config, load_json, save_json, set_seed  # noqa: E402


def needed_count(D, cfg) -> int:
    """Kebutuhan terbesar di antara semua strategi GAN yang dijalankan:
    mode lama 'balance' = gap penuh; desain dosis = round(dosis_maks × gap)."""
    ss = pd.read_csv(D["splits"] / "subsets_summary.csv")
    ss = ss[ss["design"] == cfg["imbalance"]["design"]]
    ss = ss[ss["ratio"].astype(str).isin([str(r) for r in cfg["imbalance"]["ratios"]])]
    tb = f"n_{cfg['data']['minority_class']}"
    cls_cols = [f"n_{c}" for c in cfg["data"]["class_names"]]
    gap = int((ss[cls_cols].max(axis=1) - ss[tb]).max())
    strat = [cfg["strategies"][s] for s in cfg["experiment"]["strategies"]]
    doses = [float(s["dose"]) for s in strat if s.get("extra") == "gan"]
    old = any(s.get("synthetic_csv") and "dose" not in s for s in strat)
    return max([int(round(d * gap)) for d in doses] + ([gap] if old or not doses else []))


def grid(ax_row, imgs, title):
    for ax, im in zip(ax_row, imgs):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.axis("off")
    ax_row[0].set_title(title, loc="left", fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--name", default=None, help="nama generator (default: gan.name di config)")
    ap.add_argument("--n", type=int, default=None, help="jumlah citra (default: otomatis)")
    ap.add_argument("--n-quality", type=int, default=2000, help="jumlah citra untuk evaluasi kualitas")
    args = ap.parse_args()
    cfg = load_config(args.config)
    D = get_dirs(cfg)
    g = cfg["gan"]
    name = args.name or g["name"]
    gdir = D["work"] / "gan" / name
    out = D["work"] / "synthetic" / name
    (out / "images").mkdir(parents=True, exist_ok=True)
    log = get_logger("syn", str(D["logs"] / f"02c_synthetic_{name}.log"))
    if not (gdir / "best_G.pt").exists():
        raise SystemExit(f"Generator belum ada: {gdir / 'best_G.pt'}. Jalankan 02b_train_gan.py dulu.")

    restore_cache(cfg, D, log)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(gdir / "best_G.pt", map_location=dev, weights_only=False)
    gc = ck["gan_cfg"]
    G = Generator(gc["z_dim"], gc["img_size"], gc["base_channels"], gc.get("max_channels", 512),
                  gc.get("upsample", "nearest")).to(dev)
    G.load_state_dict(ck["G_ema"])
    G.eval()
    n = args.n or needed_count(D, cfg)
    log.info(f"Generator '{name}' (step {ck['step']}, FID {ck['fid']:.2f}) -> membangkitkan {n} citra")

    # ---------------- bangkitkan
    set_seed(g["seed"] + 1)
    for old in (out / "images").glob("*.png"):
        old.unlink()
    tb = cfg["data"]["minority_class"]
    label = cfg["data"]["class_names"].index(tb)
    rows, fakes = [], []
    gbs = gen_batch_size(gc["img_size"], g.get("gen_batch_size"))
    with torch.no_grad():
        for i in range(0, n, gbs):
            z = torch.randn(min(gbs, n - i), gc["z_dim"], device=dev)
            u8 = to_uint8(G(z)).cpu()
            for k in range(len(u8)):
                iid = f"{name}_{i + k:06d}"
                Image.fromarray(u8[k, 0].numpy()).save(out / "images" / f"{iid}.png")
                rows.append({"path": f"images/{iid}.png", "cls": tb, "label": label, "generator": name,
                             "image_id": iid})
            if sum(len(f) for f in fakes) < args.n_quality:
                fakes.append(u8)
    pd.DataFrame(rows).to_csv(out / "synthetic.csv", index=False)
    log.info(f"{n} citra tersimpan di {out / 'images'}")

    # ---------------- kualitas
    gd = load_json(gdir / "gan_data.json")
    base = pd.read_csv(D["data"] / "index.csv").set_index("image_id")
    pcol = "path_cached" if "path_cached" in base else "path"
    tr_paths = base.loc[gd["train_image_ids"], pcol].tolist()
    va = pd.read_csv(D["splits"] / "val.csv").query("cls == @tb")
    S = gc["img_size"]
    real_tr = load_uint8(tr_paths, S)
    real_va = load_uint8(va[pcol if pcol in va else "path"].tolist(), S) if len(va) else None
    fake = torch.cat(fakes)[: args.n_quality]
    log.info(f"Evaluasi kualitas: {len(fake)} sintetis vs {len(real_tr)} TB latih"
             + (f" & {len(real_va)} TB validasi" if real_va is not None else ""))
    rep, ex = full_quality_report(fake, real_tr, real_va, dev, seed=g["seed"])
    rep.update({"generator": name, "best_step": ck["step"], "n_generated": n, "img_size": S, **git_info()})
    save_json(rep, out / "quality_report.json")
    nn_df = pd.DataFrame({"synthetic_id": [r["image_id"] for r in rows[: len(fake)]],
                          "nearest_train_id": np.array(gd["train_image_ids"])[ex["nn_idx_fake"]],
                          "feature_dist": ex["nn_dist_fake"], "ssim": ex["ssim_nn"]}).sort_values("feature_dist")
    nn_df.to_csv(out / "nearest_neighbors.csv", index=False)

    # ---------------- gambar
    rng = np.random.default_rng(0)
    k = max(1, min(8, len(fake), len(real_tr)))  # tangguh bila citra sintetis sangat sedikit
    fig, axes = plt.subplots(2, k, figsize=(1.6 * k, 3.6))
    grid(axes[0], real_tr[rng.choice(len(real_tr), k, replace=False), 0].numpy(), "Real TB (train)")
    grid(axes[1], fake[rng.choice(len(fake), k, replace=False), 0].numpy(), f"Synthetic TB ({name})")
    fig.tight_layout(); fig.savefig(out / "fig_real_vs_synthetic.png", dpi=300); plt.close(fig)

    top = nn_df.head(k)
    fid_idx = {r["image_id"]: j for j, r in enumerate(rows[: len(fake)])}
    tr_idx = {iid: j for j, iid in enumerate(gd["train_image_ids"])}
    fig, axes = plt.subplots(2, k, figsize=(1.6 * k, 3.8))
    grid(axes[0], [fake[fid_idx[s], 0].numpy() for s in top["synthetic_id"]], "Synthetic (closest to train)")
    grid(axes[1], [real_tr[tr_idx[t], 0].numpy() for t in top["nearest_train_id"]], "Nearest real train image")
    for ax, s in zip(axes[1], top["ssim"]):
        ax.text(0.5, -0.08, f"SSIM {s:.2f}", transform=ax.transAxes, ha="center", va="top", fontsize=7)
    fig.tight_layout(); fig.savefig(out / "fig_nearest_pairs.png", dpi=200); plt.close(fig)

    if "nn_dist_real_val" in ex:
        fig, ax = plt.subplots(figsize=(4.8, 3))
        ax.hist(ex["nn_dist_fake"], bins=40, alpha=.6, density=True, label="synthetic → train")
        ax.hist(ex["nn_dist_real_val"], bins=20, alpha=.6, density=True, label="real val → train")
        ax.axvline(rep["memorization_threshold_p5"], color="k", ls="--", lw=1, label="5th pct. real")
        ax.set(xlabel="Nearest-neighbour distance (Inception features)", ylabel="Density")
        ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(out / "fig_nn_distance.png", dpi=200); plt.close(fig)

    # ---------------- ringkasan
    log.info("=" * 60)
    log.info(f"FID vs TB latih     : {rep['fid_vs_train']:.2f}")
    if "fid_vs_val" in rep:
        log.info(f"FID vs TB validasi  : {rep['fid_vs_val']:.2f}  (acuan asli-vs-asli: {rep['fid_train_vs_val_reference']:.2f})")
        log.info(f"KID vs TB validasi  : {rep['kid_vs_val']:.4f} ± {rep['kid_vs_val_std']:.4f}")
        log.info(f"Sintetis lebih dekat ke citra latih daripada 5% citra asli terdekat: "
                 f"{100 * rep['frac_fake_closer_than_real_p5']:.1f}% (idealnya <= ~5%)")
    log.info(f"SSIM ke citra latih terdekat (rata-rata/maks): {rep['ssim_to_nearest_train_mean']:.3f} / "
             f"{rep['ssim_to_nearest_train_max']:.3f}")
    log.info(f"Diversitas (SSIM antarpasangan; makin rendah makin beragam): sintetis "
             f"{rep['diversity_pairwise_ssim_fake']:.3f} vs asli {rep['diversity_pairwise_ssim_real']:.3f}")
    log.info("WAJIB: buka fig_real_vs_synthetic.png dan fig_nearest_pairs.png untuk pemeriksaan visual.")
    log.info("Langkah berikutnya: pastikan 'gan_basic' ada di experiment.strategies, lalu 03_run_experiments.py")


if __name__ == "__main__":
    main()
