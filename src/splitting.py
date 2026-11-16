"""Pembagian data dan pembuatan subset ketidakseimbangan terkontrol.

Prinsip desain:
  * Split train/val/test dibuat SEKALI (split_seed) dan dipakai semua eksperimen,
    sehingga semua model diuji pada test set yang identik (syarat uji berpasangan
    seperti DeLong & McNemar).
  * Val dan test selalu mengikuti distribusi alami (tidak pernah di-resample).
  * Rasio ketidakseimbangan hanya dimanipulasi pada data latih.
  * Subset bersarang (nested): subset rasio kecil adalah bagian dari subset rasio
    besar, sehingga perbedaan antar-rasio tidak tercampur dengan perbedaan sampel acak.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .utils import ratio_tag


def _sgkf_take_fold(df: pd.DataFrame, frac: float, seed: int, group_col: str | None):
    n_splits = max(2, int(round(1.0 / frac)))
    y = df["label"].values
    groups = df[group_col].values if group_col else np.arange(len(df))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest_idx, take_idx = next(sgkf.split(np.zeros(len(df)), y, groups))
    return df.iloc[rest_idx], df.iloc[take_idx]


def make_base_split(df: pd.DataFrame, test_frac: float, val_frac: float, seed: int,
                    group_col: str | None = "dup_group") -> pd.DataFrame:
    """Tambahkan kolom 'split' in {train,val,test}. val_frac dihitung terhadap total data."""
    df = df.reset_index(drop=True).copy()
    rest, test = _sgkf_take_fold(df, test_frac, seed, group_col)
    val_frac_rest = val_frac / (1.0 - test_frac)
    train, val = _sgkf_take_fold(rest, val_frac_rest, seed + 1, group_col)
    df["split"] = ""
    df.loc[train.index, "split"] = "train"
    df.loc[val.index, "split"] = "val"
    df.loc[test.index, "split"] = "test"
    if group_col:  # verifikasi: tidak ada grup duplikat yang menyeberang split
        leak = df.groupby(group_col)["split"].nunique()
        assert (leak <= 1).all(), "Grup duplikat bocor antar split!"
    return df


def _nested_take(idx: np.ndarray, k: int) -> np.ndarray:
    return idx[: max(0, min(k, len(idx)))]


def make_imbalance_subsets(train: pd.DataFrame, design: str, ratios: list, minority_label: int,
                           seed: int, minority_n: int | None = None, majority_n: int | None = None,
                           logger=None) -> tuple[dict, pd.DataFrame]:
    """Bangun subset latih untuk setiap rasio (mayoritas : minoritas).

    design = 'fix_minority' : jumlah kelas minoritas (TB) tetap; tiap kelas mayoritas = r * n_TB.
    design = 'fix_majority' : jumlah tiap kelas mayoritas tetap; n_TB = n_mayoritas / r.
    ratio 'orig'            : seluruh data latih apa adanya.
    """
    rng = np.random.default_rng(seed)
    perm = {c: rng.permutation(train.index[train["label"] == c].values) for c in sorted(train["label"].unique())}
    majors = [c for c in perm if c != minority_label]
    subsets, rows = {}, []

    for r in ratios:
        tag = ratio_tag(r)
        if tag == "orig":
            idx = np.concatenate(list(perm.values()))
        elif design == "fix_minority":
            n_min = minority_n or len(perm[minority_label])
            n_min = min(n_min, len(perm[minority_label]))
            target = int(round(float(r) * n_min))
            parts = [_nested_take(perm[minority_label], n_min)]
            for c in majors:
                if target > len(perm[c]) and logger:
                    logger.warning(f"Rasio {tag}: kelas {c} hanya {len(perm[c])} < target {target}; dibatasi.")
                parts.append(_nested_take(perm[c], target))
            idx = np.concatenate(parts)
        elif design == "fix_majority":
            n_maj = majority_n or min(len(perm[c]) for c in majors)
            n_maj = min(n_maj, *(len(perm[c]) for c in majors))
            n_min = int(round(n_maj / float(r)))
            if n_min > len(perm[minority_label]) and logger:
                logger.warning(f"Rasio {tag}: TB hanya {len(perm[minority_label])} < target {n_min}; dibatasi.")
            parts = [_nested_take(perm[minority_label], n_min)] + [_nested_take(perm[c], n_maj) for c in majors]
            idx = np.concatenate(parts)
        else:
            raise ValueError(f"design tidak dikenal: {design}")

        sub = train.loc[np.sort(idx)].copy()
        sub["ratio"] = tag
        subsets[tag] = sub
        cnt = sub["label"].value_counts().to_dict()
        n_tb = cnt.get(minority_label, 0)
        row = {"design": design, "ratio": tag, "n_total": len(sub)}
        row.update({f"n_class{c}": cnt.get(c, 0) for c in perm})
        row["actual_ratio_major_to_tb"] = round(max(cnt.get(c, 0) for c in majors) / n_tb, 3) if n_tb else np.inf
        rows.append(row)

    # verifikasi sifat bersarang (hanya untuk rasio numerik)
    num = sorted([t for t in subsets if t != "orig"], key=float)
    for a, b in zip(num, num[1:]):
        sa, sb = set(subsets[a].index), set(subsets[b].index)
        if design == "fix_minority" and not sa <= sb and logger:
            logger.warning(f"Subset {a} bukan bagian dari {b} (cek pembatasan jumlah).")
    return subsets, pd.DataFrame(rows)
