"""Fitur (embedding) citra dari CNN pretrained ImageNet, untuk mendeteksi citra yang sangat mirip
(mis. foto pasien yang sama) yang tidak tertangkap pHash."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image


@torch.no_grad()
def compute_embeddings(paths, model_name: str = "resnet50.tv_in1k", size: int = 224, pretrained: bool = True,
                       batch: int = 64, device=None, logger=None) -> np.ndarray:
    import timm
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=0).eval().to(device)
    dc = timm.data.resolve_model_data_config(model)
    mean = torch.tensor(dc["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(dc["std"], device=device).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(paths), batch):
        arr = []
        for p in paths[i:i + batch]:
            with Image.open(p) as im:
                arr.append(np.asarray(im.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0)
        x = torch.from_numpy(np.stack(arr))[:, None].to(device).expand(-1, 3, -1, -1)
        f = model((x - mean) / std).float()
        out.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy())
        if logger and (i // batch) % 20 == 0:
            logger.info(f"  embedding {min(i + batch, len(paths))}/{len(paths)}")
    return np.concatenate(out).astype(np.float32)


def nearest(query: np.ndarray, ref: np.ndarray, exclude_self: bool = False, chunk: int = 1024):
    """Tetangga terdekat (cosine; embedding sudah ternormalisasi). Kembalikan (indeks, similaritas)."""
    idx, sim = [], []
    for s in range(0, len(query), chunk):
        m = query[s:s + chunk] @ ref.T
        if exclude_self:
            m[np.arange(m.shape[0]), np.arange(s, s + m.shape[0])] = -2
        j = m.argmax(1)
        idx.append(j)
        sim.append(m[np.arange(len(j)), j])
    return np.concatenate(idx), np.concatenate(sim)


def pairs_above(emb: np.ndarray, threshold: float, chunk: int = 1024):
    """Semua pasangan (i<j) dengan similaritas >= threshold."""
    out = []
    for s in range(0, len(emb), chunk):
        m = emb[s:s + chunk] @ emb.T
        ii, jj = np.nonzero(m >= threshold)
        for i, j in zip(ii + s, jj):
            if j > i:
                out.append((int(i), int(j), float(m[i - s, j])))
    return out


def mutual_topk_pairs(emb: np.ndarray, threshold: float, k: int = 2, chunk: int = 1024):
    """Pasangan (i<j) yang saling berada di k tetangga terdekat satu sama lain DAN similaritas >= threshold.
    Mencegah efek berantai: rontgen dada umumnya mirip satu sama lain, sehingga ambang saja
    menghasilkan ribuan pasangan yang menyambung menjadi grup raksasa."""
    n = len(emb)
    topk = np.zeros((n, k), int)
    topv = np.zeros((n, k), np.float32)
    for s in range(0, n, chunk):
        m = emb[s:s + chunk] @ emb.T
        m[np.arange(m.shape[0]), np.arange(s, s + m.shape[0])] = -2  # bukan diri sendiri
        idx = np.argpartition(-m, kth=min(k, n - 1) - 1, axis=1)[:, :k]
        val = np.take_along_axis(m, idx, 1)
        o = np.argsort(-val, 1)
        topk[s:s + m.shape[0]], topv[s:s + m.shape[0]] = np.take_along_axis(idx, o, 1), np.take_along_axis(val, o, 1)
    nbr = [set(r) for r in topk]
    out = []
    for i in range(n):
        for jj, v in zip(topk[i], topv[i]):
            if v >= threshold and i < jj and i in nbr[jj]:
                out.append((int(i), int(jj), float(v)))
    return out
