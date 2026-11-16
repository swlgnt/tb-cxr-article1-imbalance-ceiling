"""Evaluasi kualitas citra sintetis.

  FID  (Heusel et al., 2017)  : jarak distribusi fitur Inception-v3 (bobot FID standar, torch-fidelity)
  KID  (Binkowski et al., 2018): seperti FID tetapi tak bias untuk sampel kecil -> lebih cocok untuk
                                 ~600 citra TB. Laporkan keduanya.
  SSIM ke tetangga terdekat    : kemiripan struktural citra sintetis dengan citra latih terdekat
  Diversitas (SSIM antarpasangan): makin tinggi dibanding citra asli -> indikasi mode collapse
  Memorisasi                   : apakah citra sintetis "terlalu dekat" dengan citra latih tertentu,
                                 dibandingkan kedekatan alami citra asli (val) ke citra latih.
Catatan: Inception dilatih pada ImageNet, sehingga FID/KID pada citra medis bersifat relatif
(untuk membandingkan generator), bukan ukuran absolut realisme klinis.
"""
from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import linalg

_FE = None


def load_uint8(paths, size: int) -> torch.Tensor:
    arr = []
    for p in paths:
        with Image.open(p) as im:
            im = im.convert("L").resize((size, size), Image.BICUBIC)
            arr.append(np.asarray(im, dtype=np.uint8))
    return torch.from_numpy(np.stack(arr))[:, None]  # (N,1,H,W) uint8


def to_uint8(x: torch.Tensor) -> torch.Tensor:
    """Keluaran generator [-1,1] -> uint8 [0,255]."""
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


@torch.no_grad()
def inception_features(x_uint8: torch.Tensor, device, batch: int = 64) -> np.ndarray:
    global _FE
    if _FE is None:
        from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
        _FE = FeatureExtractorInceptionV3("inception-v3-compat", ["2048"]).eval()
    _FE.to(device)
    out = []
    for i in range(0, len(x_uint8), batch):
        b = x_uint8[i:i + batch].to(device)
        if b.size(1) == 1:
            b = b.expand(-1, 3, -1, -1)
        out.append(_FE(b)[0].double().cpu())
    return torch.cat(out).numpy()


def fid_from_features(f1: np.ndarray, f2: np.ndarray) -> float:
    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    # Tanpa argumen 'disp': argumen itu dihapus di SciPy versi baru (kompatibel versi lama & baru)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        covmean = linalg.sqrtm(s1 @ s2)
        if isinstance(covmean, tuple):  # jaga-jaga untuk versi yang mengembalikan (hasil, galat)
            covmean = covmean[0]
        if not np.isfinite(covmean).all():
            off = np.eye(s1.shape[0]) * 1e-6
            covmean = linalg.sqrtm((s1 + off) @ (s2 + off))
            if isinstance(covmean, tuple):
                covmean = covmean[0]
    covmean = np.real(covmean)
    fid = float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))
    return max(fid, 0.0)  # galat numerik kecil bisa membuatnya sedikit negatif


def kid_from_features(f1: np.ndarray, f2: np.ndarray, n_subsets: int = 50, subset_size: int = 500,
                      seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    m = min(subset_size, len(f1), len(f2))
    d = f1.shape[1]
    vals = []
    for _ in range(n_subsets):
        x = f1[rng.choice(len(f1), m, replace=False)]
        y = f2[rng.choice(len(f2), m, replace=False)]
        kxx = (x @ x.T / d + 1) ** 3
        kyy = (y @ y.T / d + 1) ** 3
        kxy = (x @ y.T / d + 1) ** 3
        mmd = ((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
               - 2 * kxy.mean())
        vals.append(mmd)
    return float(np.mean(vals)), float(np.std(vals))


def _gauss_window(size=11, sigma=1.5, device="cpu"):
    g = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None]


def ssim_pairs(a: torch.Tensor, b: torch.Tensor, size: int = 128) -> torch.Tensor:
    """SSIM per pasangan; a, b uint8 (N,1,H,W). Dihitung pada resolusi `size`."""
    a = F.interpolate(a.float() / 255, size=(size, size), mode="area")
    b = F.interpolate(b.float() / 255, size=(size, size), mode="area")
    w = _gauss_window(device=a.device)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_a, mu_b = F.conv2d(a, w), F.conv2d(b, w)
    saa = F.conv2d(a * a, w) - mu_a ** 2
    sbb = F.conv2d(b * b, w) - mu_b ** 2
    sab = F.conv2d(a * b, w) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2))
    return m.mean(dim=[1, 2, 3])


def nearest_neighbors(query: np.ndarray, ref: np.ndarray, chunk: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Jarak L2 & indeks tetangga terdekat di ruang fitur."""
    rn = (ref ** 2).sum(1)
    dists, idxs = [], []
    for i in range(0, len(query), chunk):
        q = query[i:i + chunk]
        d = (q ** 2).sum(1)[:, None] + rn[None] - 2 * q @ ref.T
        j = d.argmin(1)
        dists.append(np.sqrt(np.maximum(d[np.arange(len(q)), j], 0)))
        idxs.append(j)
    return np.concatenate(dists), np.concatenate(idxs)


def pairwise_ssim_diversity(x: torch.Tensor, n_pairs: int = 500, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(x), n_pairs)
    j = rng.integers(0, len(x), n_pairs)
    keep = i != j
    return float(ssim_pairs(x[i[keep]], x[j[keep]]).mean())


def full_quality_report(fake_u8, real_train_u8, real_val_u8, device, seed: int = 0) -> tuple[dict, dict]:
    f_fake = inception_features(fake_u8, device)
    f_tr = inception_features(real_train_u8, device)
    rep = {"n_fake": len(fake_u8), "n_real_train": len(real_train_u8),
           "fid_vs_train": fid_from_features(f_fake, f_tr)}
    rep["kid_vs_train"], rep["kid_vs_train_std"] = kid_from_features(f_fake, f_tr, seed=seed)
    d_fake, nn_fake = nearest_neighbors(f_fake, f_tr)
    extra = {"nn_dist_fake": d_fake, "nn_idx_fake": nn_fake}
    if real_val_u8 is not None and len(real_val_u8) > 1:
        f_va = inception_features(real_val_u8, device)
        rep["n_real_val"] = len(real_val_u8)
        rep["fid_vs_val"] = fid_from_features(f_fake, f_va)
        rep["kid_vs_val"], rep["kid_vs_val_std"] = kid_from_features(f_fake, f_va, seed=seed)
        rep["fid_train_vs_val_reference"] = fid_from_features(f_tr, f_va)  # batas bawah realistis
        d_real, _ = nearest_neighbors(f_va, f_tr)
        thr1, thr5 = np.percentile(d_real, 1), np.percentile(d_real, 5)
        rep["memorization_threshold_p1"] = float(thr1)
        rep["memorization_threshold_p5"] = float(thr5)
        rep["frac_fake_closer_than_real_p1"] = float((d_fake < thr1).mean())
        rep["frac_fake_closer_than_real_p5"] = float((d_fake < thr5).mean())
        rep["median_nn_dist_real_val"] = float(np.median(d_real))
        extra["nn_dist_real_val"] = d_real
    rep["median_nn_dist_fake"] = float(np.median(d_fake))
    ss = ssim_pairs(fake_u8, real_train_u8[torch.from_numpy(nn_fake)])
    rep["ssim_to_nearest_train_mean"] = float(ss.mean())
    rep["ssim_to_nearest_train_max"] = float(ss.max())
    extra["ssim_nn"] = ss.numpy()
    rep["diversity_pairwise_ssim_fake"] = pairwise_ssim_diversity(fake_u8, seed=seed)
    rep["diversity_pairwise_ssim_real"] = pairwise_ssim_diversity(real_train_u8, seed=seed)
    return rep, extra
