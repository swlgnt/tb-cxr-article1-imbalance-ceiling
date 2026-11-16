#!/usr/bin/env python
"""Langkah 2b: latih GAN dasar (DCGAN) HANYA pada citra TB dari data latih.

Pada desain `fix_minority`, himpunan citra TB identik di semua subset rasio, sehingga SATU
generator berlaku untuk semua rasio tanpa kebocoran. Skrip memverifikasi hal ini dan
memastikan tidak ada citra val/test yang dipakai.

Keluaran (work/gan/<nama>/):
  samples/step_XXXXXX.png   contoh citra berkala (pantau kualitas secara visual)
  losses.csv, fid_log.csv   riwayat loss dan FID/KID
  best_G.pt                 generator (EMA) dengan FID terbaik  -> dipakai langkah 2c
  last.pt                   checkpoint lengkap untuk melanjutkan bila terputus
  fig_losses.png, fig_fid.png
Aman diulang: jika terputus, jalankan perintah yang sama untuk melanjutkan.
"""
import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torchvision.utils import save_image  # noqa: E402

from src.gan import EMA, Discriminator, Generator, diff_augment, gen_batch_size, init_dcgan_weights  # noqa: E402
from src.gan_eval import fid_from_features, inception_features, kid_from_features, load_uint8, to_uint8  # noqa: E402
from src.tbx11k import restore_cache  # noqa: E402
from src.utils import get_dirs, get_logger, git_info, keep_awake, load_config, save_json, set_seed  # noqa: E402


def tb_training_set(D, cfg, ratio, log):
    """Ambil citra TB data latih; verifikasi identik antar rasio dan tidak bocor ke val/test."""
    design = cfg["imbalance"]["design"]
    tb_cls = cfg["data"]["minority_class"]
    files = sorted(D["splits"].glob(f"train_{design}_r*.csv"))
    if not files:
        raise FileNotFoundError("Subset latih belum ada. Jalankan 02_make_splits.py dulu.")
    sets = {f.stem.split("_r")[-1]: pd.read_csv(f).query("cls == @tb_cls") for f in files}
    ids = {r: frozenset(d["image_id"]) for r, d in sets.items()}
    identical = len(set(ids.values())) == 1
    if ratio is None:
        if not identical:
            raise SystemExit("Himpunan TB berbeda antar rasio (mis. desain fix_majority). "
                             "Latih per rasio: python 02b_train_gan.py --ratio <rasio>")
        tb = next(iter(sets.values()))
        log.info(f"Himpunan TB identik di semua rasio {sorted(sets)} -> satu generator untuk semua.")
    else:
        tb = sets[str(ratio)]
    held = set(pd.read_csv(D["splits"] / "val.csv")["image_id"]) | set(pd.read_csv(D["splits"] / "test.csv")["image_id"])
    leak = set(tb["image_id"]) & held
    assert not leak, f"KEBOCORAN: {len(leak)} citra TB latih juga ada di val/test!"
    return tb.reset_index(drop=True)


def plot_logs(out, losses, fids):
    if len(losses):
        fig, ax = plt.subplots(1, 2, figsize=(9, 3))
        l = losses.copy()
        vc = ["d_loss", "g_loss", "d_real", "d_fake"]
        l[vc] = l[vc].rolling(10, min_periods=1).mean()  # haluskan nilai saja, bukan sumbu step
        ax[0].plot(l["step"], l["d_loss"], label="D loss")
        ax[0].plot(l["step"], l["g_loss"], label="G loss")
        ax[0].set_xlabel("step"); ax[0].legend(); ax[0].grid(alpha=.3)
        ax[1].plot(l["step"], l["d_real"], label="D(real)")
        ax[1].plot(l["step"], l["d_fake"], label="D(fake)")
        ax[1].set_xlabel("step"); ax[1].set_ylim(0, 1); ax[1].legend(); ax[1].grid(alpha=.3)
        fig.tight_layout(); fig.savefig(out / "fig_losses.png", dpi=150); plt.close(fig)
    if len(fids):
        fig, ax = plt.subplots(figsize=(4.5, 3))
        ax.plot(fids["step"], fids["fid"], marker="o", ms=3)
        b = fids.loc[fids["fid"].idxmin()]
        ax.scatter([b["step"]], [b["fid"]], color="red", zorder=3, label=f"best {b['fid']:.1f} @ {int(b['step'])}")
        ax.set(xlabel="step", ylabel="FID vs TB train (lower = better)"); ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(out / "fig_fid.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--ratio", default=None, help="hanya untuk desain fix_majority")
    ap.add_argument("--restart", action="store_true", help="mulai dari awal (hapus progres sebelumnya)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    D = get_dirs(cfg)
    g = cfg["gan"]
    name = g["name"] + (f"_r{args.ratio}" if args.ratio else "")
    out = D["work"] / "gan" / name
    (out / "samples").mkdir(parents=True, exist_ok=True)
    log = get_logger("gan", str(D["logs"] / f"02b_gan_{name}.log"))
    if args.restart:
        for f in ("last.pt", "best_G.pt", "losses.csv", "fid_log.csv", "done.json"):
            (out / f).unlink(missing_ok=True)
    if (out / "done.json").exists():
        log.info(f"GAN '{name}' sudah selesai dilatih (hapus {out / 'done.json'} atau pakai --restart untuk mengulang).")
        return

    restore_cache(cfg, D, log)
    set_seed(g["seed"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tb = tb_training_set(D, cfg, args.ratio, log)
    pcol = "path_cached" if "path_cached" in tb else "path"
    S = g["img_size"]
    real_u8 = load_uint8(tb[pcol], S)  # semua citra TB dimuat ke RAM (kecil)
    real = (real_u8.float() / 127.5 - 1).to(dev)
    n_real = len(real)
    log.info(f"Citra TB latih untuk GAN: {n_real} | resolusi {S}x{S} | device {dev}")
    val_tb = pd.read_csv(D["splits"] / "val.csv").query("cls == @cfg['data']['minority_class']")
    save_json({"n_train_tb": n_real, "train_image_ids": tb["image_id"].tolist(),
               "n_val_tb": len(val_tb), "gan_cfg": g}, out / "gan_data.json")

    G = Generator(g["z_dim"], S, g["base_channels"], g.get("max_channels", 512), g.get("upsample", "nearest")).to(dev)
    Dn = Discriminator(S, g["base_channels"], g.get("max_channels", 512)).to(dev)
    G.apply(init_dcgan_weights)
    Dn.apply(init_dcgan_weights)
    opt_g = torch.optim.Adam(G.parameters(), lr=g["lr_g"], betas=tuple(g["betas"]))
    opt_d = torch.optim.Adam(Dn.parameters(), lr=g["lr_d"], betas=tuple(g["betas"]))
    ema = EMA(G, g["ema_decay"])
    gbs = gen_batch_size(S, g.get("gen_batch_size"))
    n_grid = 64 if S <= 256 else 16  # kisi contoh: 8x8 (<=256 px) atau 4x4 (512 px)
    z_fixed = torch.randn(n_grid, g["z_dim"], generator=torch.Generator().manual_seed(0)).to(dev)
    start, best_fid = 0, float("inf")
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=dev, weights_only=False)
        G.load_state_dict(ck["G"]); Dn.load_state_dict(ck["D"]); ema.shadow.load_state_dict(ck["EMA"])
        opt_g.load_state_dict(ck["opt_g"]); opt_d.load_state_dict(ck["opt_d"])
        start, best_fid = ck["step"], ck["best_fid"]
        log.info(f"[RESUME] dari step {start} (FID terbaik sejauh ini {best_fid:.2f})")

    feats_real = inception_features(real_u8, dev)
    losses = pd.read_csv(out / "losses.csv") if (out / "losses.csv").exists() else pd.DataFrame()
    fids = pd.read_csv(out / "fid_log.csv") if (out / "fid_log.csv").exists() else pd.DataFrame()
    bs, pol, smooth = g["batch_size"], g.get("diffaugment", ""), g.get("real_label", 1.0)
    params = {"G_params_M": sum(p.numel() for p in G.parameters()) / 1e6,
              "D_params_M": sum(p.numel() for p in Dn.parameters()) / 1e6}
    log.info(f"Parameter G {params['G_params_M']:.2f} M | D {params['D_params_M']:.2f} M | DiffAugment: {pol or '-'}")
    rows, t0 = [], time.time()

    def evaluate(step):
        nonlocal best_fid, fids
        ema.shadow.eval()
        fake = []
        with torch.no_grad():
            for i in range(0, g["n_eval_samples"], gbs):
                z = torch.randn(min(gbs, g["n_eval_samples"] - i), g["z_dim"], device=dev)
                fake.append(to_uint8(ema.shadow(z)).cpu())
        f_fake = inception_features(torch.cat(fake), dev)
        fid = fid_from_features(f_fake, feats_real)
        kid, kid_sd = kid_from_features(f_fake, feats_real)
        is_best = fid < best_fid
        if is_best:
            best_fid = fid
            torch.save({"G_ema": ema.shadow.state_dict(), "step": step, "fid": fid, "gan_cfg": g,
                        "name": name, "n_train_tb": n_real}, out / "best_G.pt")
        fids = pd.concat([fids, pd.DataFrame([{"step": step, "fid": fid, "kid": kid, "kid_std": kid_sd,
                                               "best": is_best}])], ignore_index=True)
        fids.to_csv(out / "fid_log.csv", index=False)
        log.info(f"  [EVAL] step {step} | FID {fid:.2f} | KID {kid:.4f} | {'TERBAIK' if is_best else ''}")

    with keep_awake():
        for step in range(start + 1, g["steps"] + 1):
            G.train(); Dn.train()
            xr = real[torch.randint(0, n_real, (bs,), device=dev)]
            z = torch.randn(bs, g["z_dim"], device=dev)
            xf = G(z)
            # --- discriminator
            lr_ = Dn(diff_augment(xr, pol))
            lf_ = Dn(diff_augment(xf.detach(), pol))
            d_loss = (F.binary_cross_entropy_with_logits(lr_, torch.full_like(lr_, smooth))
                      + F.binary_cross_entropy_with_logits(lf_, torch.zeros_like(lf_)))
            opt_d.zero_grad(set_to_none=True); d_loss.backward(); opt_d.step()
            # --- generator (non-saturating)
            lg = Dn(diff_augment(xf, pol))
            g_loss = F.binary_cross_entropy_with_logits(lg, torch.ones_like(lg))
            opt_g.zero_grad(set_to_none=True); g_loss.backward(); opt_g.step()
            ema.update(G)

            if step % 25 == 0:
                rows.append({"step": step, "d_loss": d_loss.item(), "g_loss": g_loss.item(),
                             "d_real": torch.sigmoid(lr_).mean().item(), "d_fake": torch.sigmoid(lf_).mean().item()})
            if step % g["log_every"] == 0:  # nilai langkah saat ini (bukan baris rekaman terakhir)
                spd = (step - start) / (time.time() - t0)
                eta = (g["steps"] - step) / max(spd, 1e-9) / 60
                log.info(f"step {step}/{g['steps']} | D {d_loss.item():.3f} G {g_loss.item():.3f} | "
                         f"D(real) {torch.sigmoid(lr_).mean().item():.2f} "
                         f"D(fake) {torch.sigmoid(lf_).mean().item():.2f} | "
                         f"{spd:.1f} step/dtk | sisa ~{eta:.0f} menit")
            if step % g["sample_every"] == 0 or step == g["steps"]:
                with torch.no_grad():
                    ema.shadow.eval()
                    grid = torch.cat([ema.shadow(z_fixed[i:i + gbs]) for i in range(0, n_grid, gbs)])
                    save_image(grid * 0.5 + 0.5, out / "samples" / f"step_{step:06d}.png", nrow=int(n_grid ** 0.5))
            if step % g["eval_every"] == 0 or step == g["steps"]:
                evaluate(step)
            if step % g["ckpt_every"] == 0 or step == g["steps"]:
                losses = pd.concat([losses, pd.DataFrame(rows)], ignore_index=True); rows = []
                losses.to_csv(out / "losses.csv", index=False)
                torch.save({"G": G.state_dict(), "D": Dn.state_dict(), "EMA": ema.shadow.state_dict(),
                            "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(), "step": step,
                            "best_fid": best_fid}, out / "last.pt")
                plot_logs(out, losses, fids)

    b = fids.loc[fids["fid"].idxmin()]
    save_json({"name": name, "best_step": int(b["step"]), "best_fid": float(b["fid"]), "best_kid": float(b["kid"]),
               "steps": g["steps"], "train_minutes": round((time.time() - t0) / 60, 1), **params,
               **git_info()}, out / "done.json")
    (out / "last.pt").unlink(missing_ok=True)
    log.info(f"SELESAI. Generator terbaik: step {int(b['step'])}, FID {b['fid']:.2f}. "
             f"Periksa visual: {out / 'samples'}. Langkah berikutnya: python 02c_generate_synthetic.py")


if __name__ == "__main__":
    main()
