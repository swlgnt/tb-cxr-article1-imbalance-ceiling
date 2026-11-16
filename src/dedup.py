"""Deteksi citra duplikat / hampir-duplikat dengan perceptual hash (pHash).

TBX11K tidak menyediakan ID pasien, sehingga pembagian berbasis pasien tidak mungkin.
Sebagai pengganti, citra yang hampir identik dikelompokkan (group) dan selalu
ditempatkan pada split yang sama untuk mencegah kebocoran data train -> test.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn

_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def phash_bits(path: str, hash_size: int = 16, highfreq: int = 4) -> np.ndarray:
    """pHash -> bit terkemas (uint8). hash_size=16 -> 256 bit (lebih diskriminatif untuk CXR)."""
    n = hash_size * highfreq
    with Image.open(path) as im:
        a = np.asarray(im.convert("L").resize((n, n), Image.LANCZOS), dtype=np.float64)
    d = dctn(a, norm="ortho")[:hash_size, :hash_size]
    bits = (d > np.median(d.flatten()[1:])).flatten()
    return np.packbits(bits)


def compute_hashes(paths, hash_size: int = 16, workers: int = 8) -> np.ndarray:
    with ThreadPoolExecutor(workers) as ex:
        return np.stack(list(ex.map(lambda p: phash_bits(p, hash_size), paths)))


def _hamming_block(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _POP8[np.bitwise_xor(a[:, None, :], b[None, :, :])].sum(-1, dtype=np.int32)


class _UF:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def _groups_from_pairs(n: int, pairs) -> np.ndarray:
    uf = _UF(n)
    for i, j, _ in pairs:
        uf.union(i, j)
    roots = np.array([uf.find(i) for i in range(n)])
    return np.unique(roots, return_inverse=True)[1]


def find_duplicate_groups(hashes: np.ndarray, max_dist: int, max_group_size: int = 20, chunk: int = 256):
    """Kembalikan (group_id, pasangan, info).

    Union-find bersifat transitif (A~B, B~C -> A,B,C satu grup) sehingga pada citra yang
    strukturnya mirip (CXR) dapat terbentuk grup raksasa. Grup > max_group_size dipecah ulang
    hanya dengan pasangan identik (hamming = 0) dan dilaporkan untuk diperiksa manual."""
    n = len(hashes)
    pairs = []
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        d = _hamming_block(hashes[s:e], hashes)
        ii, jj = np.nonzero(d <= max_dist)
        for i, j in zip(ii + s, jj):
            if j > i:
                pairs.append((int(i), int(j), int(d[i - s, j])))
    gid = _groups_from_pairs(n, pairs)
    sizes = np.bincount(gid)
    big = np.nonzero(sizes > max_group_size)[0]
    info = {"n_big_groups_split": int(len(big)), "big_group_sizes": sizes[big].tolist()}
    if len(big):
        in_big = np.isin(gid, big)
        kept = [p for p in pairs if not (in_big[p[0]] or in_big[p[1]]) or p[2] == 0]
        gid = _groups_from_pairs(n, kept)
    return gid, pairs, info


def add_dup_groups(df: pd.DataFrame, path_col: str, max_dist: int, workers: int, logger,
                   hash_size: int = 16, max_group_size: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info(f"Menghitung pHash {hash_size * hash_size}-bit untuk {len(df)} citra ...")
    h = compute_hashes(df[path_col].tolist(), hash_size, workers)
    gid, pairs, info = find_duplicate_groups(h, max_dist, max_group_size)
    if info["n_big_groups_split"]:
        logger.warning(f"{info['n_big_groups_split']} grup terlalu besar {info['big_group_sizes']} (efek berantai) "
                       f"-> hanya pasangan identik yang dipertahankan. Pertimbangkan menurunkan max_hamming.")
    df = df.copy().reset_index(drop=True)
    df["phash"] = [x.tobytes().hex() for x in h]
    df["dup_group"] = gid
    pr = pd.DataFrame(pairs, columns=["i", "j", "hamming"])
    if len(pr):
        pr["image_i"], pr["image_j"] = df.loc[pr.i, "image_id"].values, df.loc[pr.j, "image_id"].values
        pr["label_i"], pr["label_j"] = df.loc[pr.i, "source_label"].values, df.loc[pr.j, "source_label"].values
        pr["label_conflict"] = pr["label_i"] != pr["label_j"]
    n_grp = (df["dup_group"].value_counts() > 1).sum()
    logger.info(f"Pasangan hampir-duplikat (hamming <= {max_dist}): {len(pr)}; grup berukuran >1: {n_grp}")
    if len(pr) and pr["label_conflict"].any():
        logger.warning(f"{int(pr['label_conflict'].sum())} pasangan duplikat memiliki label BERBEDA "
                       "(cek kemungkinan label noise; lihat duplicates.csv).")
    return df, pr


def merge_with_embedding_pairs(df: pd.DataFrame, emb_pairs, phash_pairs: pd.DataFrame, max_group_size: int,
                               logger) -> tuple[pd.DataFrame, dict]:
    """Gabungkan grup pHash dengan pasangan sangat mirip menurut embedding CNN.
    Citra dalam satu grup selalu masuk split yang sama. Grup > max_group_size (efek berantai)
    dipecah ulang hanya dengan pasangan pHash."""
    n = len(df)
    strong = [(int(i), int(j), 0) for i, j in zip(phash_pairs.get("i", []), phash_pairs.get("j", []))]
    weak = [(i, j, 0) for i, j, _ in emb_pairs]
    gid = _groups_from_pairs(n, strong + weak)
    sizes = np.bincount(gid)
    big = np.nonzero(sizes > max_group_size)[0]
    if len(big):
        in_big = np.isin(gid, big)
        kept = strong + [p for p in weak if not (in_big[p[0]] or in_big[p[1]])]
        gid = _groups_from_pairs(n, kept)
        logger.warning(f"{len(big)} grup embedding terlalu besar {sizes[big].tolist()} -> dipecah ulang. "
                       "Pertimbangkan menaikkan dedup.embedding.threshold.")
    df = df.copy()
    before = (df["dup_group"].value_counts() > 1).sum() if "dup_group" in df else 0
    df["dup_group"] = gid
    after = (df["dup_group"].value_counts() > 1).sum()
    info = {"n_embedding_pairs": len(emb_pairs), "groups_gt1_phash": int(before), "groups_gt1_combined": int(after),
            "n_images_in_groups": int((df["dup_group"].map(df["dup_group"].value_counts()) > 1).sum())}
    return df, info
