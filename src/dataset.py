"""Dataset, transformasi (augmentasi klasik), sampler oversampling, dan data sintetis."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms as T


# ----------------------------------------------------------------------------- transformasi
class ToGray:
    """Pastikan grayscale 1 kanal (kelas, bukan lambda, agar aman untuk multiprocessing)."""

    def __call__(self, img: Image.Image) -> Image.Image:
        return img.convert("L")


class Expand3:
    """1 kanal -> 3 kanal agar cocok dengan bobot pretrained ImageNet."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.expand(3, -1, -1)


class CLAHE:
    """Contrast Limited Adaptive Histogram Equalization (opsional, butuh opencv)."""

    def __init__(self, clip_limit: float = 2.0, tile: int = 8):
        import cv2  # noqa: F401  (gagal lebih awal jika opencv tidak terpasang)
        self.clip, self.tile = clip_limit, tile

    def __call__(self, img: Image.Image) -> Image.Image:
        import cv2
        a = np.asarray(img.convert("L"))
        a = cv2.createCLAHE(clipLimit=self.clip, tileGridSize=(self.tile, self.tile)).apply(a)
        return Image.fromarray(a)


class BorderCrop:
    """Buang tepi citra (fraksi tiap sisi) untuk menghilangkan penanda teks/bingkai yang bisa
    menjadi petunjuk non-patologis (shortcut)."""

    def __init__(self, frac: float):
        self.frac = float(frac)

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        dx, dy = int(round(w * self.frac)), int(round(h * self.frac))
        return img.crop((dx, dy, w - dx, h - dy))


class IntensityNorm:
    """Samakan distribusi intensitas antar citra: 'equalize' (histogram equalization) atau
    'zscore' (standarisasi per citra), untuk mengurangi perbedaan alat/pengolahan antar sumber."""

    def __init__(self, mode: str):
        self.mode = mode

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.mode == "equalize":
            from PIL import ImageOps
            return ImageOps.equalize(img)
        if self.mode == "zscore":
            a = np.asarray(img, dtype=np.float32)
            a = (a - a.mean()) / (a.std() + 1e-6) * 0.2 * 255 + 127.5
            return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        return img


class GaussianNoise:
    """Noise injection (augmentasi klasik pada Bab II proposal)."""

    def __init__(self, std_range=(0.0, 0.03), p: float = 0.5):
        self.lo, self.hi = std_range
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            std = self.lo + (self.hi - self.lo) * torch.rand(1).item()
            x = (x + torch.randn_like(x) * std).clamp(0, 1)
        return x


def build_transforms(img_size: int, train: bool, augment: bool, aug: dict, mean, std, clahe: dict | None,
                     prep: dict | None = None):
    """prep = bagian 'data' dari config (border_crop, intensity_norm). Diterapkan sama pada data latih,
    validasi, uji, salinan klasik, dan citra GAN."""
    prep = prep or {}
    pre = [ToGray()]
    if prep.get("border_crop"):
        pre.append(BorderCrop(prep["border_crop"]))
    if prep.get("intensity_norm") and prep["intensity_norm"] != "none":
        pre.append(IntensityNorm(prep["intensity_norm"]))
    if clahe and clahe.get("enabled"):
        pre.append(CLAHE(clahe.get("clip_limit", 2.0), clahe.get("tile", 8)))
    if train and augment:
        geo = [
            T.RandomResizedCrop(img_size, scale=tuple(aug.get("crop_scale", (0.85, 1.0))),
                                ratio=tuple(aug.get("crop_ratio", (0.95, 1.05)))),
            T.RandomApply([T.RandomAffine(degrees=aug.get("rotation_deg", 10),
                                          translate=tuple(aug.get("translate", (0.08, 0.08))),
                                          scale=tuple(aug.get("scale", (0.95, 1.05))))],
                          p=aug.get("affine_p", 0.7)),
        ]
        if aug.get("hflip_p", 0.5) > 0:
            geo.append(T.RandomHorizontalFlip(aug["hflip_p"]))
        if aug.get("vflip_p", 0.0) > 0:
            geo.append(T.RandomVerticalFlip(aug["vflip_p"]))
        photo = [T.RandomApply([T.ColorJitter(brightness=aug.get("brightness", 0.2),
                                              contrast=aug.get("contrast", 0.2))], p=aug.get("jitter_p", 0.7))]
        if aug.get("blur_p", 0.0) > 0:
            photo.append(T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.0))], p=aug["blur_p"]))
        post = [T.ToTensor()]
        if aug.get("noise_p", 0.3) > 0:
            post.append(GaussianNoise(tuple(aug.get("noise_std", (0.0, 0.03))), aug["noise_p"]))
        tail = [Expand3(), T.Normalize(mean, std)]
        return T.Compose(pre + geo + photo + post + tail)
    return T.Compose(pre + [T.Resize((img_size, img_size)), T.ToTensor(),
                            Expand3(), T.Normalize(mean, std)])


# ----------------------------------------------------------------------------- dataset
class CXRDataset(Dataset):
    """copy_transform: transformasi augmentasi untuk baris salinan klasik (aug_copy=True).
    copy_mode='fixed' -> augmentasi tiap salinan TETAP antar epoch (setara himpunan citra tambahan
    yang dibuat sekali, seperti citra GAN); 'online' -> augmentasi baru setiap epoch."""

    def __init__(self, df: pd.DataFrame, transform, path_col: str = "path", copy_transform=None,
                 copy_mode: str = "fixed", copy_seed: int = 0):
        self.df = df.reset_index(drop=True)
        col = "path_cached" if ("path_cached" in self.df and path_col == "auto") else (
            "path" if path_col == "auto" else path_col)
        self.paths = self.df[col].tolist()
        self.labels = self.df["label"].astype(int).to_numpy()
        self.transform = transform
        self.copy_transform = copy_transform
        self.copy_mode, self.copy_seed = copy_mode, copy_seed
        self.is_copy = (self.df["aug_copy"].fillna(False).astype(bool).to_numpy()
                        if "aug_copy" in self.df else np.zeros(len(self.df), bool))
        self.copy_id = self.df["copy_id"].fillna(-1).astype(int).to_numpy() if "copy_id" in self.df else None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            img = im.convert("L")
        if self.copy_transform is not None and self.is_copy[i]:
            if self.copy_mode == "fixed":
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(self.copy_seed * 1_000_003 + int(self.copy_id[i]))
                    x = self.copy_transform(img)
            else:
                x = self.copy_transform(img)
            return x, int(self.labels[i]), i
        return self.transform(img), int(self.labels[i]), i


class BalancedOversampleSampler(Sampler):
    """Tiap epoch: semua indeks asli + tambahan acak (dengan pengembalian) dari kelas
    yang lebih kecil sampai setiap kelas = ukuran kelas terbesar. Karena augmentasi
    on-the-fly, salinan berulang tidak identik -> 'augmentasi klasik penyeimbang'."""

    def __init__(self, labels: np.ndarray, seed: int = 0):
        self.labels = np.asarray(labels)
        self.cls_idx = {c: np.nonzero(self.labels == c)[0] for c in np.unique(self.labels)}
        self.n_max = max(len(v) for v in self.cls_idx.values())
        self.seed, self.epoch = seed, 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return self.n_max * len(self.cls_idx)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        parts = []
        for idx in self.cls_idx.values():
            extra = rng.choice(idx, self.n_max - len(idx), replace=True) if len(idx) < self.n_max else []
            parts.append(np.concatenate([idx, extra]).astype(int))
        allidx = np.concatenate(parts)
        rng.shuffle(allidx)
        return iter(allidx.tolist())


# ----------------------------------------------------------------------------- data sintetis
def load_synthetic(csv_path: str, class_names: list[str]) -> pd.DataFrame:
    """CSV citra sintetis: kolom 'path' dan 'label' (int) ATAU 'cls' (nama kelas).
    Kolom opsional: 'generator', 'quality_score'."""
    df = pd.read_csv(csv_path)
    if "label" not in df and "cls" in df:
        df["label"] = df["cls"].map({c: i for i, c in enumerate(class_names)})
    if "cls" not in df:
        df["cls"] = df["label"].map(dict(enumerate(class_names)))
    base = Path(csv_path).parent
    df["path"] = [str(p if Path(p).is_absolute() else base / p) for p in df["path"]]
    df["image_id"] = df.get("image_id", pd.Series([f"syn_{i}" for i in range(len(df))]))
    df["is_synthetic"] = True
    return df


def add_synthetic(train: pd.DataFrame, syn: pd.DataFrame, mode: str, seed: int,
                  per_class: dict | None = None) -> pd.DataFrame:
    """Gabungkan citra sintetis HANYA ke data latih.
    mode='balance'  : tambah per kelas sampai setara kelas terbesar
    mode='all'      : tambah semua
    mode='fixed'    : per_class = {label: jumlah}"""
    rng = np.random.default_rng(seed)
    train = train.copy()
    train["is_synthetic"] = False
    counts = train["label"].value_counts()
    n_max = counts.max()
    picks = []
    for c in sorted(syn["label"].unique()):
        pool = syn[syn["label"] == c]
        if mode == "all":
            k = len(pool)
        elif mode == "balance":
            k = int(n_max - counts.get(c, 0))
        elif mode == "fixed":
            k = int((per_class or {}).get(int(c), 0))
        else:
            raise ValueError(mode)
        k = max(0, min(k, len(pool)))
        if k:
            picks.append(pool.iloc[rng.choice(len(pool), k, replace=False)])
    return pd.concat([train] + picks, ignore_index=True) if picks else train


# ----------------------------------------------------------------------------- desain dosis
def gap_counts(train_df: pd.DataFrame, dose: float, target_labels: list[int]) -> dict:
    """Jumlah citra tambahan per kelas target = dosis × (n_kelas_terbesar − n_kelas).
    Mis. 2660 vs 560, dosis 0,30 -> round(0,30 × 2100) = 630."""
    cnt = train_df["label"].value_counts()
    n_max = int(cnt.max())
    return {int(c): int(round(dose * max(0, n_max - int(cnt.get(c, 0))))) for c in target_labels}


def nested_order(n_pool: int, k: int, seed: int) -> np.ndarray:
    """Urutan indeks tetap: gabungan permutasi berulang. Mengambil k pertama memberi himpunan
    BERSARANG antar dosis (630 ⊂ 1050 ⊂ 2100) dan setiap citra asli dipakai merata
    (bila k > n_pool, tiap citra dipakai floor/ceil(k/n_pool) kali)."""
    rng = np.random.default_rng(seed)
    reps = int(np.ceil(k / max(n_pool, 1))) if k else 0
    return np.concatenate([rng.permutation(n_pool) for _ in range(reps)])[:k] if reps else np.array([], int)


def build_train_set(cfg: dict, strategy: str, ratio: str, train_df: pd.DataFrame, work_dir) -> pd.DataFrame:
    """Susun data latih untuk satu strategi. Dipakai IDENTIK oleh pelatihan dan --dry-run.

    Kolom tambahan: source ('real' | 'classic' | 'gan'), aug_copy (True untuk salinan klasik),
    copy_id (nomor salinan, untuk augmentasi tetap per salinan)."""
    from pathlib import Path
    scfg = cfg["strategies"][strategy]
    names = cfg["data"]["class_names"]
    df = train_df.copy()
    df["source"], df["aug_copy"], df["copy_id"] = "real", False, -1
    sel_seed = int(cfg.get("dose_design", {}).get("selection_seed", cfg["split"]["seed"]))

    if "dose" in scfg:  # ---------------- desain dosis (penutupan gap)
        extra = scfg.get("extra", "none")
        targets = [names.index(c) for c in scfg.get("dose_classes", [cfg["data"]["minority_class"]])]
        need = gap_counts(df, float(scfg["dose"]), targets)
        parts = [df]
        if extra == "classic":
            for c, k in need.items():
                pool = df[(df["label"] == c) & (df["source"] == "real")].sort_values("image_id")
                order = nested_order(len(pool), k, sel_seed + c)
                add = pool.iloc[order].copy()
                add["source"], add["aug_copy"] = "classic", True
                add["copy_id"] = np.arange(len(add))
                add["image_id"] = [f"{i}__copy{j}" for i, j in zip(add["image_id"], add["copy_id"])]
                parts.append(add)
        elif extra == "gan":
            p = Path(str(scfg["synthetic_csv"]).format(ratio=ratio))
            p = p if p.is_absolute() else Path(work_dir) / p
            syn = load_synthetic(str(p), names)
            if scfg.get("synthetic_generator"):
                syn = syn[syn["generator"] == scfg["synthetic_generator"]]
            for c, k in need.items():
                pool = syn[syn["label"] == c].sort_values("image_id").reset_index(drop=True)
                if k > len(pool):
                    raise ValueError(f"Strategi '{strategy}' butuh {k} citra sintetis kelas {names[c]}, "
                                     f"tersedia {len(pool)}. Jalankan ulang 02c_generate_synthetic.py.")
                order = np.random.default_rng(sel_seed + c).permutation(len(pool))[:k]  # bersarang antar dosis
                add = pool.iloc[order].copy()
                add["source"], add["aug_copy"], add["copy_id"] = "gan", False, -1
                parts.append(add)
        elif extra != "none":
            raise ValueError(f"extra tidak dikenal: {extra}")
        out = pd.concat(parts, ignore_index=True)
    elif scfg.get("synthetic_csv"):  # ---------------- mode lama (gan_basic: balance/all/fixed)
        p = Path(str(scfg["synthetic_csv"]).format(ratio=ratio))
        p = p if p.is_absolute() else Path(work_dir) / p
        syn = load_synthetic(str(p), names)
        if scfg.get("synthetic_generator"):
            syn = syn[syn["generator"] == scfg["synthetic_generator"]]
        out = add_synthetic(df.drop(columns=["source", "aug_copy", "copy_id"]), syn,
                            scfg.get("synthetic_mode", "balance"), sel_seed, scfg.get("synthetic_per_class"))
        out["source"] = np.where(out["is_synthetic"], "gan", "real")
        out["aug_copy"], out["copy_id"] = False, -1
    else:
        out = df
    if "path_cached" in out:
        out["path_cached"] = out["path_cached"].fillna(out["path"])
    return out.reset_index(drop=True)


def composition_table(train: pd.DataFrame, class_names: list[str]) -> pd.DataFrame:
    t = (train.assign(cls=train["label"].map(dict(enumerate(class_names))))
              .groupby(["cls", "source"]).size().rename("n").reset_index())
    return t
