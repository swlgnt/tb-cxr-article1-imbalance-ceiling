#!/usr/bin/env python
"""Langkah 0: periksa apakah PC siap menjalankan penelitian.

Yang diperiksa:
  1. Versi Python dan paket yang dibutuhkan
  2. GPU NVIDIA, CUDA, dan VRAM
  3. RAM dan ruang disk kosong
  4. Lokasi dataset TBX11K
  5. Mengunduh bobot pretrained keempat model (sekali saja, lalu tersimpan di cache)
  6. (GPU) Uji batch size: apakah batch_size di config muat di memori GPU

Cara pakai:
  python 00_check_environment.py
  python 00_check_environment.py --skip-download     # lewati unduhan bobot
  python 00_check_environment.py --skip-batch-test   # lewati uji batch size
"""
import argparse
import importlib
import platform
import shutil
import sys
from pathlib import Path

OK, WARN, FAIL = "[ OK ]", "[PERINGATAN]", "[GAGAL]"
problems = []


def say(tag, msg):
    print(f"{tag} {msg}")
    if tag == FAIL:
        problems.append(msg)


def header(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def check_python():
    header("1. Python dan paket")
    v = sys.version_info
    if v < (3, 9):
        say(FAIL, f"Python {v.major}.{v.minor} terlalu lama. Butuh Python 3.10-3.12.")
    elif v >= (3, 13):
        say(WARN, f"Python {v.major}.{v.minor}: sebagian paket mungkin belum mendukung. Disarankan 3.11.")
    else:
        say(OK, f"Python {v.major}.{v.minor}.{v.micro} ({platform.system()} {platform.release()})")
    import os
    env = os.environ.get("CONDA_DEFAULT_ENV", "")
    venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if (env and env != "base") or venv:
        say(OK, f"Environment aktif: {env or sys.prefix}")
    else:
        say(WARN, "Tampaknya Anda tidak berada di environment khusus (mis. 'tbstudy'). "
                  "Jalankan: conda activate tbstudy")
    pkgs = {"torch": "torch", "torchvision": "torchvision", "timm": "timm", "numpy": "numpy",
            "pandas": "pandas", "sklearn": "scikit-learn", "scipy": "scipy", "statsmodels": "statsmodels",
            "matplotlib": "matplotlib", "yaml": "pyyaml", "PIL": "pillow", "openpyxl": "openpyxl",
            "torch_fidelity": "torch-fidelity"}
    missing = []
    for mod, pipname in pkgs.items():
        try:
            m = importlib.import_module(mod)
            say(OK, f"{pipname:<13} {getattr(m, '__version__', '')}")
        except ImportError:
            missing.append(pipname)
            say(FAIL, f"{pipname} belum terpasang")
    if missing:
        if "torch" in missing or "torchvision" in missing:
            print("      -> PyTorch dipasang terpisah, lihat PANDUAN_PC.md Bagian 4.")
        rest = [m for m in missing if m not in ("torch", "torchvision")]
        if rest:
            print("      -> Jalankan: pip install -r requirements.txt")
        return False
    import torch
    if tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) < (2, 3):
        say(FAIL, f"PyTorch {torch.__version__} terlalu lama; butuh >= 2.3")
    return True


def check_gpu():
    header("2. GPU dan CUDA")
    import torch
    if "+cpu" in torch.__version__ or not torch.cuda.is_available():
        say(WARN, f"GPU tidak terdeteksi oleh PyTorch (torch {torch.__version__}).")
        print("      Pelatihan di CPU SANGAT lambat (bisa puluhan kali lebih lambat): tidak realistis untuk grid penuh.")
        print("      Penyebab umum: (a) PC tidak punya GPU NVIDIA, (b) driver NVIDIA belum terpasang/terlalu lama,")
        print("      (c) PyTorch versi CPU yang terpasang. Lihat PANDUAN_PC.md Bagian 3-4 dan Troubleshooting.")
        return None
    dev = torch.cuda.get_device_properties(0)
    vram = dev.total_memory / 1024**3
    say(OK, f"GPU: {dev.name} | VRAM {vram:.1f} GB | CUDA (PyTorch) {torch.version.cuda} | "
            f"compute capability {dev.major}.{dev.minor}")
    if torch.cuda.device_count() > 1:
        say(OK, f"Terdeteksi {torch.cuda.device_count()} GPU (yang dipakai: GPU 0)")
    if vram < 4:
        say(WARN, "VRAM < 4 GB: kurangi batch_size dan pertimbangkan img_size lebih kecil.")
    elif vram < 8:
        say(WARN, "VRAM < 8 GB: kemungkinan batch_size DenseNet201/Xception perlu dikurangi (lihat uji batch).")
    try:
        x = torch.randn(2048, 2048, device="cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            y = x @ x
        torch.cuda.synchronize()
        say(OK, "Uji komputasi GPU + mixed precision (AMP) berhasil")
        del x, y
    except Exception as e:
        say(FAIL, f"Uji komputasi GPU gagal: {e}")
    return vram


def check_system(cfg):
    header("3. RAM dan disk")
    try:
        import psutil
        ram = psutil.virtual_memory().total / 1024**3
        tag = OK if ram >= 16 else WARN
        say(tag, f"RAM: {ram:.1f} GB" + ("" if ram >= 16 else " (disarankan >= 16 GB; set train.num_workers: 2)"))
        cpu = psutil.cpu_count(logical=False) or psutil.cpu_count()
        say(OK, f"CPU: {cpu} core fisik")
    except ImportError:
        say(WARN, "psutil tidak terpasang; RAM tidak diperiksa (pip install psutil)")
    work = Path(cfg["paths"]["work_dir"])
    probe = work if work.exists() else Path.cwd()
    free = shutil.disk_usage(probe.anchor or probe).free / 1024**3
    tag = OK if free >= 50 else (WARN if free >= 25 else FAIL)
    say(tag, f"Disk kosong di drive kerja ({probe.anchor or probe}): {free:.0f} GB "
             "(disarankan >= 50 GB: dataset ~ 5 GB, cache, ~15 GB checkpoint)")
    p = str(Path.cwd())
    if "onedrive" in p.lower():
        say(WARN, "Folder proyek ada di dalam OneDrive. Ribuan file hasil akan ikut disinkron dan bisa "
                  "mengganggu. Pindahkan ke mis. C:\\penelitian\\tb_imbalance_study")
    if " " in p:
        say(WARN, f"Path folder mengandung spasi: '{p}'. Biasanya aman, tetapi lebih baik tanpa spasi.")


def check_data(cfg):
    header("4. Dataset TBX11K")
    root = Path(cfg["paths"]["data_root"])
    if not root.exists():
        say(WARN, f"Folder data belum ada: {root}")
        print("      Unduh & ekstrak TBX11K ke folder tersebut (PANDUAN_PC.md Bagian 7),")
        print("      atau ubah paths.data_root di config.yaml.")
        return
    from src.tbx11k import _find_imgs_dir
    imgs = _find_imgs_dir(root)
    if imgs is None:
        say(WARN, f"Folder ditemukan ({root}) tetapi struktur imgs/health, imgs/sick, imgs/tb tidak ditemukan.")
        sub = [p.name for p in root.iterdir()][:15]
        print(f"      Isi folder: {sub}")
        print("      Periksa apakah ada folder bertingkat (mis. TBX11K/TBX11K/imgs) lalu sesuaikan data_root.")
        return
    counts = {d.name: sum(1 for _ in d.glob("*.png")) for d in imgs.iterdir() if d.is_dir()}
    say(OK, f"Struktur TBX11K ditemukan di {imgs}: {counts}")
    ann = imgs.parent / "annotations"
    if (ann / "xml").is_dir() or (ann / "json").is_dir():
        say(OK, "Folder anotasi ditemukan (dibutuhkan untuk membedakan TB aktif vs laten)")
    else:
        say(WARN, "Folder annotations/xml atau annotations/json tidak ditemukan: subtipe TB tidak bisa dibedakan.")


def download_weights(cfg):
    header("5. Unduh bobot pretrained ImageNet (sekali saja)")
    import timm
    for key, m in cfg["models"].items():
        try:
            model = timm.create_model(m["timm_name"], pretrained=True, num_classes=3)
            n = sum(p.numel() for p in model.parameters()) / 1e6
            say(OK, f"{key:<12} {m['timm_name']:<26} {n:.1f} juta parameter")
            del model
        except Exception as e:
            say(FAIL, f"{key}: gagal mengunduh bobot ({type(e).__name__}: {str(e)[:150]})")
            print("      Periksa koneksi internet / proxy kampus. Bobot diunduh dari huggingface.co.")
    try:
        from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
        FeatureExtractorInceptionV3("inception-v3-compat", ["2048"])
        say(OK, "Inception-v3 untuk FID/KID (evaluasi GAN) siap")
    except Exception as e:
        say(FAIL, f"Gagal mengunduh bobot Inception untuk FID ({type(e).__name__}: {str(e)[:120]}). "
                  "Bobot diunduh dari github.com.")


def batch_test(cfg):
    header("6. Uji batch size di GPU (forward + backward, AMP)")
    import timm
    import torch
    tcfg = cfg["train"]
    suggestions = {}
    for key, m in cfg["models"].items():
        bs = m.get("batch_size", tcfg["batch_size"])
        size = m.get("img_size", 224)
        model = timm.create_model(m["timm_name"], pretrained=False, num_classes=3).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda")
        fit = None
        for trial in sorted({bs, bs // 2, bs // 4, 8, 4}, reverse=True):
            if trial < 2 or (fit is not None):
                continue
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                x = torch.randn(trial, 3, size, size, device="cuda")
                y = torch.randint(0, 3, (trial,), device="cuda")
                for _ in range(2):
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.float16):
                        loss = torch.nn.functional.cross_entropy(model(x), y)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated() / 1024**3
                fit = (trial, peak)
            except torch.cuda.OutOfMemoryError:
                x = y = loss = None  # lepaskan memori
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
        del model, opt
        torch.cuda.empty_cache()
        if fit is None:
            say(FAIL, f"{key}: bahkan batch 4 tidak muat di GPU")
        elif fit[0] == bs:
            say(OK, f"{key:<12} batch_size {bs} muat (puncak memori {fit[1]:.1f} GB)")
        else:
            say(WARN, f"{key:<12} batch_size {bs} TIDAK muat -> gunakan {fit[0]} (puncak {fit[1]:.1f} GB)")
            suggestions[key] = fit[0]
    if suggestions:
        print("\n      Ubah di config.yaml bagian 'models', contoh:")
        for k, v in suggestions.items():
            print(f"        {k}: {{..., batch_size: {v}}}")
        print("      PENTING: tetapkan batch size SEBELUM grid dimulai dan jangan diubah di tengah jalan,")
        print("      agar semua kondisi untuk satu model dilatih dengan pengaturan yang sama.")


def gan_batch_test(cfg):
    header("7. Uji batch size GAN di GPU")
    import torch
    import torch.nn.functional as F
    from src.gan import Discriminator, Generator
    g = cfg["gan"]
    S, bs = g["img_size"], g["batch_size"]
    G = Generator(g["z_dim"], S, g["base_channels"], g.get("max_channels", 512), g.get("upsample", "nearest")).cuda()
    Dn = Discriminator(S, g["base_channels"], g.get("max_channels", 512)).cuda()
    og, od = torch.optim.Adam(G.parameters()), torch.optim.Adam(Dn.parameters())
    fit = None
    for trial in sorted({bs, bs // 2, 4, 2}, reverse=True):
        if trial < 2 or fit is not None:
            continue
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            z = torch.randn(trial, g["z_dim"], device="cuda")
            real = torch.randn(trial, 1, S, S, device="cuda")
            fake = G(z)
            ld = F.softplus(-Dn(real)).mean() + F.softplus(Dn(fake.detach())).mean()
            od.zero_grad(); ld.backward(); od.step()
            lg = F.softplus(-Dn(fake)).mean()
            og.zero_grad(); lg.backward(); og.step()
            torch.cuda.synchronize()
            fit = (trial, torch.cuda.max_memory_allocated() / 1024**3)
        except torch.cuda.OutOfMemoryError:
            z = real = fake = ld = lg = None
            torch.cuda.empty_cache()
    del G, Dn
    torch.cuda.empty_cache()
    if fit is None:
        say(FAIL, f"GAN {S}px: bahkan batch 2 tidak muat. Turunkan gan.img_size atau gan.base_channels.")
    elif fit[0] == bs:
        say(OK, f"GAN {S}px batch_size {bs} muat (puncak memori {fit[1]:.1f} GB)")
    else:
        say(WARN, f"GAN {S}px batch_size {bs} TIDAK muat -> gunakan gan.batch_size: {fit[0]} (puncak {fit[1]:.1f} GB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-batch-test", action="store_true")
    args = ap.parse_args()

    print("PEMERIKSAAN LINGKUNGAN - studi ketidakseimbangan kelas TBX11K")
    if not check_python():
        print("\nPerbaiki paket yang belum terpasang, lalu jalankan skrip ini lagi.")
        sys.exit(1)
    from src.utils import load_config
    cfg = load_config(args.config)
    vram = check_gpu()
    check_system(cfg)
    check_data(cfg)
    if not args.skip_download:
        download_weights(cfg)
    if vram and not args.skip_batch_test:
        batch_test(cfg)
        if "gan_basic" in cfg["experiment"]["strategies"] or any(
                v.get("synthetic_csv") for v in cfg["strategies"].values()):
            gan_batch_test(cfg)

    header("RINGKASAN")
    if problems:
        print(f"Ada {len(problems)} masalah yang HARUS diperbaiki:")
        for p in problems:
            print(f"  - {p}")
        print("Lihat bagian Troubleshooting di PANDUAN_PC.md.")
        sys.exit(1)
    print("Tidak ada masalah fatal. Baca juga baris [PERINGATAN] di atas bila ada.")
    print("Langkah berikutnya: python quick_test.py")


if __name__ == "__main__":
    main()
