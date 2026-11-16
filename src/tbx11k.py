"""Membangun indeks citra TBX11K -> label sumber -> kelas penelitian (3 kelas).

Mode yang didukung:
  xml     : struktur asli TBX11K (imgs/{health,sick,tb}/ + annotations/xml/*.xml)
  coco    : struktur asli dengan anotasi COCO JSON (annotations/json/*.json)
  csv     : CSV apa pun (mis. data.csv versi Kaggle) dengan pemetaan kolom di config
  folders : root/<label_sumber>/*.png
  auto    : coba xml -> coco -> csv -> folders
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# nama folder asli TBX11K -> label sumber
FOLDER_TO_SOURCE = {"health": "healthy", "healthy": "healthy", "sick": "sick_non_tb"}


def _list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXT)


def tb_subtype_from_names(names: list[str]) -> str:
    """Tentukan subtipe TB dari nama kategori bounding box."""
    n = [x.strip().lower() for x in names]
    has_active = any("active" in x for x in n)
    has_latent = any(("obsolete" in x) or ("latent" in x) for x in n)
    has_uncertain = any(x == "pulmonarytuberculosis" or "uncertain" in x for x in n)
    if has_active and has_latent:
        return "active_latent_tb"
    if has_active:
        return "active_tb"
    if has_latent:
        return "latent_tb"
    if has_uncertain:
        return "uncertain_tb"
    return "tb_unannotated"


def _parse_voc_xml(xml_path: Path) -> list[str]:
    root = ET.parse(xml_path).getroot()
    return [o.findtext("name", default="") for o in root.iter("object")]


def _find_imgs_dir(root: Path) -> Path | None:
    for cand in (root / "imgs", root / "TBX11K" / "imgs", root):
        if (cand / "tb").is_dir() and ((cand / "health").is_dir() or (cand / "healthy").is_dir()):
            return cand
    return None


def build_index_xml(root: Path, logger) -> pd.DataFrame:
    imgs_dir = _find_imgs_dir(root)
    if imgs_dir is None:
        raise FileNotFoundError("Folder imgs/{health,sick,tb} tidak ditemukan.")
    ann_root = imgs_dir.parent / "annotations"
    xml_dir = ann_root / "xml"
    json_dir = ann_root / "json"

    tb_names: dict[str, list[str]] = {}
    if xml_dir.is_dir() and any(xml_dir.glob("*.xml")):
        for x in xml_dir.glob("*.xml"):
            tb_names[x.stem] = _parse_voc_xml(x)
        logger.info(f"Anotasi XML dibaca: {len(tb_names)} file")
    elif json_dir.is_dir() and any(json_dir.glob("*.json")):
        for jf in json_dir.glob("*.json"):
            d = json.loads(jf.read_text(encoding="utf-8"))
            if not {"images", "annotations", "categories"} <= set(d):
                continue
            cats = {c["id"]: c["name"] for c in d["categories"]}
            id2stem = {im["id"]: Path(im["file_name"]).stem for im in d["images"]}
            for a in d["annotations"]:
                stem = id2stem.get(a["image_id"])
                if stem is not None:
                    tb_names.setdefault(stem, []).append(cats.get(a["category_id"], ""))
        logger.info(f"Anotasi COCO JSON dibaca: {len(tb_names)} citra beranotasi")
    else:
        logger.warning("Anotasi tidak ditemukan: subtipe TB tidak bisa dibedakan (semua -> tb_unannotated).")

    rows = []
    for sub in sorted(p for p in imgs_dir.iterdir() if p.is_dir()):
        name = sub.name.lower()
        if name in FOLDER_TO_SOURCE:
            src = FOLDER_TO_SOURCE[name]
            rows += [{"path": str(p), "source_folder": sub.name, "source_label": src} for p in _list_images(sub)]
        elif name == "tb":
            for p in _list_images(sub):
                st = tb_subtype_from_names(tb_names.get(p.stem, []))
                rows.append({"path": str(p), "source_folder": sub.name, "source_label": st})
        else:
            logger.info(f"Folder '{sub.name}' dilewati (tanpa label publik / bukan bagian studi).")
    return pd.DataFrame(rows)


def build_index_csv(root: Path, csv_cfg: dict, logger) -> pd.DataFrame:
    f = Path(csv_cfg["file"])
    f = f if f.is_absolute() else root / f
    df = pd.read_csv(f)
    path_col, label_col = csv_cfg["path_col"], csv_cfg["label_col"]
    prefix = csv_cfg.get("path_prefix") or ""
    label_map = csv_cfg.get("label_map") or {}

    def resolve(p):
        p = Path(str(p))
        if p.is_absolute():
            return p
        cand = root / prefix / p
        if cand.exists():
            return cand
        hits = list(root.rglob(p.name))  # fallback: cari berdasarkan nama file
        return hits[0] if hits else cand

    out = pd.DataFrame({
        "path": [str(resolve(p)) for p in df[path_col]],
        "source_folder": df[csv_cfg["folder_col"]].astype(str) if csv_cfg.get("folder_col") else "csv",
        "source_label": [label_map.get(str(v), str(v)) for v in df[label_col]],
    })
    # versi Kaggle memiliki 1 baris per bounding box -> gabungkan per citra
    if out["path"].duplicated().any():
        logger.info("Terdapat baris ganda per citra (kemungkinan 1 baris per bbox); digabung per citra.")
        out = (out.groupby("path", as_index=False)
                  .agg(source_folder=("source_folder", "first"),
                       source_label=("source_label", lambda s: _merge_labels(list(s)))))
    missing = ~out["path"].map(lambda p: Path(p).exists())
    if missing.any():
        logger.warning(f"{missing.sum()} path pada CSV tidak ditemukan dan dibuang.")
        out = out[~missing]
    return out


def _merge_labels(labels: list[str]) -> str:
    s = set(labels)
    if len(s) == 1:
        return labels[0]
    if {"active_tb", "latent_tb"} <= s:
        return "active_latent_tb"
    for pref in ("active_tb", "latent_tb", "uncertain_tb"):
        if pref in s:
            return pref
    return "conflict:" + "|".join(sorted(s))


def build_index_folders(root: Path, logger) -> pd.DataFrame:
    rows = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        rows += [{"path": str(p), "source_folder": sub.name, "source_label": sub.name} for p in _list_images(sub)]
    return pd.DataFrame(rows)


def build_index(root: str | Path, mode: str, csv_cfg: dict | None, logger) -> pd.DataFrame:
    root = Path(root)
    if mode == "auto":
        if _find_imgs_dir(root) is not None:
            mode = "xml"
        elif csv_cfg and csv_cfg.get("file"):
            mode = "csv"
        else:
            mode = "folders"
        logger.info(f"Mode terdeteksi otomatis: {mode}")
    if mode in ("xml", "coco"):
        df = build_index_xml(root, logger)
    elif mode == "csv":
        df = build_index_csv(root, csv_cfg or {}, logger)
    elif mode == "folders":
        df = build_index_folders(root, logger)
    else:
        raise ValueError(f"Mode tidak dikenal: {mode}")
    if df.empty:
        raise RuntimeError("Tidak ada citra yang terbaca. Periksa paths.data_root dan data.index_mode.")
    df["image_id"] = df["path"].map(lambda p: Path(p).stem)
    if df["image_id"].duplicated().any():  # nama sama di folder berbeda
        df["image_id"] = df["source_folder"].astype(str) + "/" + df["image_id"]
    return df


def apply_class_map(df: pd.DataFrame, class_map: dict, class_names: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["cls"] = df["source_label"].map(lambda s: class_map.get(s, "exclude"))
    bad = set(df["cls"]) - set(class_names) - {"exclude"}
    if bad:
        raise ValueError(f"class_map menghasilkan kelas tak dikenal: {bad}")
    name2id = {c: i for i, c in enumerate(class_names)}
    df["label"] = df["cls"].map(lambda c: name2id.get(c, -1)).astype(int)
    return df


def _img_meta(path: str) -> tuple[int, int, str]:
    with Image.open(path) as im:
        return im.width, im.height, im.mode


def add_image_meta(df: pd.DataFrame, workers: int = 8) -> pd.DataFrame:
    with ThreadPoolExecutor(workers) as ex:
        meta = list(ex.map(_img_meta, df["path"]))
    df = df.copy()
    df["width"], df["height"], df["mode"] = zip(*meta)
    return df


def _cache_one(args):
    src, dst, size = args
    dst = Path(dst)
    if dst.exists():
        return str(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("L")
        w, h = im.size
        s = size / min(w, h)  # sisi terpendek = size, rasio aspek dipertahankan
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
        im.save(dst, optimize=True)
    return str(dst)


def cache_resized(df: pd.DataFrame, cache_dir: str | Path, size: int, workers: int = 8) -> pd.DataFrame:
    """Simpan salinan grayscale ber-resize agar loading saat training jauh lebih cepat."""
    cache_dir = Path(cache_dir)
    jobs = [(p, cache_dir / f"{iid.replace('/', '__')}.png", size) for p, iid in zip(df["path"], df["image_id"])]
    with ThreadPoolExecutor(workers) as ex:
        out = list(ex.map(_cache_one, jobs))
    df = df.copy()
    df["path_cached"] = out
    return df


def restore_cache(cfg: dict, D: dict, logger) -> None:
    """Pastikan cache citra tersedia. Di Colab, disk lokal kosong setiap sesi baru, sehingga
    cache dibuat ulang dari data asli (path tercatat di index.csv) secara otomatis."""
    idx_file = D["data"] / "index.csv"
    if not idx_file.exists():
        return
    df = pd.read_csv(idx_file)
    if "path_cached" not in df or not cfg["data"].get("cache_size"):
        return
    missing = df[~df["path_cached"].map(lambda p: Path(p).exists())]
    if missing.empty:
        return
    no_src = missing[~missing["path"].map(lambda p: Path(p).exists())]
    if len(no_src):
        raise SystemExit(
            f"{len(no_src)} citra asli tidak ditemukan (contoh: {no_src['path'].iloc[0]}).\n"
            f"Data belum tersedia di sesi ini. Di Colab: jalankan sel 'Unduh data' terlebih dahulu.\n"
            f"Di PC: periksa paths.data_root di config.")
    logger.info(f"Memulihkan cache: {len(missing)} citra dibuat ulang di {Path(missing['path_cached'].iloc[0]).parent} ...")
    size = cfg["data"]["cache_size"]
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(_cache_one, [(p, c, size) for p, c in zip(missing["path"], missing["path_cached"])]))
    logger.info("Cache siap.")
