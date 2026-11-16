"""Utilitas umum: konfigurasi, seed, I/O, logging."""
from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

CLASS_NAMES_DEFAULT = ["Normal", "Abnormal_nonTB", "Abnormal_TB"]

# Konsol Windows kadang memakai encoding lama (cp1252) -> karakter seperti "×" membuat error.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | os.PathLike) -> dict:
    """Baca config YAML. Kunci `_base_` = config induk yang diwarisi (mis. config_colab.yaml
    mewarisi config.yaml dan hanya mengganti path/pengaturan perangkat)."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base_dir = Path(path).resolve().parent
    if cfg.get("_base_"):
        parent = load_config(base_dir / cfg.pop("_base_"))
        cfg = _deep_merge(parent, cfg)
    # path relatif dihitung dari folder config
    base = base_dir
    for key in ("data_root", "work_dir", "cache_dir"):
        if key in cfg.get("paths", {}) and cfg["paths"][key] is not None:
            p = Path(cfg["paths"][key])
            cfg["paths"][key] = str(p if p.is_absolute() else (base / p).resolve())
    return cfg


def get_logger(name: str = "tb", log_file: str | None = None, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(Path(log_file).resolve())
                   for h in logger.handlers):
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            logger.addHandler(fh)
    return logger


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        else:
            torch.backends.cudnn.benchmark = True
    except ImportError:
        pass


def seed_worker(worker_id: int) -> None:
    import torch
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


class NumpyEncoder(json.JSONEncoder):
    def default(self, o: Any):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def save_json(obj: Any, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)
    os.replace(tmp, path)  # tulis atomik: aman jika sesi Colab terputus


def load_json(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ratio_tag(ratio) -> str:
    """1 -> '1', 2.5 -> '2.5', 'orig' -> 'orig' (dipakai di nama file/run)."""
    if isinstance(ratio, str):
        return ratio
    return str(int(ratio)) if float(ratio).is_integer() else str(ratio)


def make_run_id(design: str, ratio, strategy: str, model: str, seed: int) -> str:
    return f"{design}__r{ratio_tag(ratio)}__{strategy}__{model}__s{seed}"


def parse_run_id(run_id: str) -> dict:
    design, r, strategy, model, s = run_id.split("__")
    return {"design": design, "ratio": r[1:], "strategy": strategy, "model": model, "seed": int(s[1:])}


def get_dirs(cfg: dict) -> dict:
    w = Path(cfg["paths"]["work_dir"])
    d = {
        "work": w, "data": w / "data", "splits": w / "splits",
        "cache": Path(cfg["paths"]["cache_dir"]) if cfg["paths"].get("cache_dir") else w / "data" / "cache",
        "runs": w / "results" / "runs", "tables": w / "results" / "tables", "stats": w / "results" / "stats",
        "ensembles": w / "results" / "ensembles", "external": w / "results" / "external",
        "figures": w / "figures", "logs": w / "logs",
    }
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


def ratio_sort_key(tag: str) -> float:
    return float("inf") if tag == "orig" else float(tag)


class keep_awake:
    """Cegah Windows masuk mode sleep selama eksperimen berjalan (layar boleh mati).
    Di Linux/macOS tidak melakukan apa-apa."""

    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001

    def __enter__(self):
        self.active = False
        if sys.platform.startswith("win"):
            try:
                import ctypes
                ctypes.windll.kernel32.SetThreadExecutionState(self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED)
                self.active = True
            except Exception:
                pass
        return self

    def __exit__(self, *exc):
        if self.active:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(self.ES_CONTINUOUS)
        return False


def git_info(repo_dir: str | os.PathLike | None = None) -> dict:
    """Versi kode saat ini (commit, cabang, ada perubahan belum di-commit?).
    Disimpan di setiap hasil agar angka di paper bisa dilacak ke kode yang menghasilkannya."""
    import subprocess
    repo_dir = str(repo_dir or Path(__file__).resolve().parents[1])

    def run(*a, strip=True):
        r = subprocess.run(["git", *a], cwd=repo_dir, capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip())
        return r.stdout.strip() if strip else r.stdout

    try:
        commit = run("rev-parse", "HEAD")
        dirty_files = [ln[3:] for ln in run("status", "--porcelain", "--untracked-files=no", strip=False).splitlines()
                       if ln.strip()]
        try:
            tag = run("describe", "--tags", "--exact-match")
        except RuntimeError:
            tag = None
        return {"git_commit": commit[:10], "git_branch": run("rev-parse", "--abbrev-ref", "HEAD"),
                "git_tag": tag, "git_dirty": bool(dirty_files), "git_dirty_files": dirty_files[:20]}
    except Exception:
        return {"git_commit": None, "git_branch": None, "git_tag": None, "git_dirty": None, "git_dirty_files": []}


def strategy_meta(cfg: dict) -> dict:
    """strategi -> (metode, dosis). Desain dosis: baseline = ('original', 0.0)."""
    strat = cfg["strategies"]
    dose_design = any("dose" in v for v in strat.values())
    out = {}
    for name, v in strat.items():
        if "dose" in v:
            out[name] = (v.get("extra", "none") if v.get("extra", "none") != "none" else "original", float(v["dose"]))
        elif dose_design and not v.get("synthetic_csv") and not v.get("augment") and v.get("balance", "none") == "none" \
                and v.get("loss", "ce") == "ce":
            out[name] = ("original", 0.0)
        elif dose_design and _is_loss_baseline(v):
            out[name] = ("loss", float("nan"))  # data latih = original, hanya fungsi loss yang berbeda
        else:
            out[name] = ("other", float("nan"))
    return out


LOSS_LEVEL = ("weighted_ce", "focal", "weighted_focal")


def _is_loss_baseline(v: dict) -> bool:
    """Pembanding level loss: data latih IDENTIK dengan original (tanpa augmentasi, tanpa oversampling,
    tanpa citra tambahan); yang berbeda hanya fungsi loss."""
    return (v.get("loss", "ce") in LOSS_LEVEL and not v.get("augment") and v.get("balance", "none") == "none"
            and not v.get("synthetic_csv") and "dose" not in v and v.get("extra", "none") == "none")


def loss_baselines(cfg: dict, strategies=None) -> list[str]:
    """Strategi pembanding level loss (mis. weighted_ce, focal) di antara `strategies`, urut sesuai config."""
    strategies = set(strategies if strategies is not None else cfg["experiment"]["strategies"])
    return [s for s, v in cfg["strategies"].items() if s in strategies and _is_loss_baseline(v)]


def is_dose_design(cfg: dict, strategies=None) -> bool:
    strategies = strategies or cfg["experiment"]["strategies"]
    return any("dose" in cfg["strategies"][s] for s in strategies)
