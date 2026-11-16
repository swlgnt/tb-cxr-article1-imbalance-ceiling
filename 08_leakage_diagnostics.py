#!/usr/bin/env python
"""Langkah 8: diagnosis kebocoran data dan shortcut. Jalankan SETELAH langkah 1-2 (uji 'neighbors'
paling informatif bila run baseline langkah 3 sudah ada).

Uji:
  lowlevel  : model sederhana dari statistik global citra (histogram, rata-rata, kontras, tepi)
              + metadata (ukuran asli, ukuran file). Akurasi tinggi -> kelas dapat dibedakan dari ciri
              teknis, bukan penyakit (SHORTCUT).
  thumbnail : model sederhana dari citra 16x16 dan 32x32 piksel (lesi TB tidak terlihat). Akurasi
              tinggi -> petunjuk non-patologis global (SHORTCUT).
  border    : model sederhana HANYA dari tepi citra (bagian tengah/paru ditutup). Akurasi tinggi ->
              penanda teks/bingkai/tepi membawa informasi label (SHORTCUT).
  neighbors : tiap citra uji dicari "kembarannya" di data latih (fitur CNN). Banyak kembaran berlabel
              sama + akurasi CNN jauh lebih tinggi pada citra berkembaran -> KEBOCORAN (pasien sama).
  shuffle   : CNN dilatih dengan label latih DIACAK. Akurasi uji harus turun ke tingkat tebakan.
              Bila tetap tinggi -> BUG PIPELINE (label bocor ke prediksi).

Pra-pemrosesan di config (data.border_crop, data.intensity_norm) DITERAPKAN pada uji lowlevel,
thumbnail, dan border, sehingga Anda bisa memeriksa apakah opsi pelemah shortcut menurunkan sinyalnya.

Contoh:
  python 08_leakage_diagnostics.py                       # semua uji
  python 08_leakage_diagnostics.py --tests lowlevel,thumbnail,border,neighbors   # tanpa GPU berat
  python 08_leakage_diagnostics.py --config config_512.yaml
Keluaran: <work_dir>/diagnostics/ (diagnostics_report.txt, .json, CSV, gambar)
"""
import argparse
import copy
import os
from concurrent.futures import ThreadPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, recall_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.dataset import BorderCrop, IntensityNorm  # noqa: E402
from src.utils import get_dirs, get_logger, load_config, load_json, save_json  # noqa: E402

ALL_TESTS = ["lowlevel", "thumbnail", "border", "neighbors", "shuffle"]


# ----------------------------------------------------------------------------- ekstraksi ciri
def _features(path: str, prep: dict, size: int = 256) -> dict:
    with Image.open(path) as im:
        img = im.convert("L")
    if prep.get("border_crop"):
        img = BorderCrop(prep["border_crop"])(img)
    if prep.get("intensity_norm") and prep["intensity_norm"] != "none":
        img = IntensityNorm(prep["intensity_norm"])(img)
    a = np.asarray(img.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    pct = np.percentile(a, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    hist = np.histogram(a, bins=32, range=(0, 1), density=True)[0]
    gx, gy = np.abs(np.diff(a, axis=1)).mean(), np.abs(np.diff(a, axis=0)).mean()
    low = np.r_[a.mean(), a.std(), pct, hist, (a < 0.04).mean(), (a > 0.96).mean(), gx, gy]
    b = int(size * 0.10)  # pita tepi 10%
    band = np.ones_like(a, bool)
    band[b:-b, b:-b] = False
    masked = np.where(band, a, 0.0)
    bthumb = np.asarray(Image.fromarray((masked * 255).astype(np.uint8)).resize((16, 16), Image.BILINEAR),
                        dtype=np.float32).ravel() / 255.0
    bv = a[band]
    border = np.r_[bthumb, bv.mean(), bv.std(), np.percentile(bv, [5, 25, 50, 75, 95])]
    t16 = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((16, 16), Image.BILINEAR), np.float32).ravel()
    t32 = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((32, 32), Image.BILINEAR), np.float32).ravel()
    return {"low": low, "border": border, "t16": t16 / 255.0, "t32": t32 / 255.0}


def extract(df: pd.DataFrame, pcol: str, prep: dict, workers: int = 8) -> dict:
    with ThreadPoolExecutor(workers) as ex:
        feats = list(ex.map(lambda p: _features(p, prep), df[pcol]))
    out = {k: np.stack([f[k] for f in feats]) for k in feats[0]}
    meta = []
    for _, r in df.iterrows():
        meta.append([r.get("width", np.nan), r.get("height", np.nan),
                     os.path.getsize(r["path"]) if os.path.exists(r["path"]) else np.nan])
    out["meta"] = np.asarray(meta, np.float32)
    return out


def fit_eval(name, Xtr, ytr, Xva, yva, Xte, yte, tb_idx) -> dict:
    """Pilih model (regresi logistik vs gradient boosting) berdasarkan akurasi VALIDASI, laporkan di uji."""
    cands = {"logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0)),
             "gboost": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=0)}
    best = None
    for k, m in cands.items():
        m.fit(np.nan_to_num(Xtr), ytr)
        va = accuracy_score(yva, m.predict(np.nan_to_num(Xva)))
        if best is None or va > best[1]:
            best = (k, va, m)
    pred = best[2].predict(np.nan_to_num(Xte))
    rec = recall_score(yte, pred, average=None, labels=sorted(set(yte)), zero_division=0)
    return {"features": name, "model": best[0], "val_acc": float(best[1]), "test_acc": float(accuracy_score(yte, pred)),
            "test_macro_f1": float(f1_score(yte, pred, average="macro")),
            "test_tb_recall": float(rec[tb_idx]) if tb_idx < len(rec) else float("nan"),
            "test_recall_per_class": [float(x) for x in rec]}


# ----------------------------------------------------------------------------- uji tetangga
def neighbors_test(cfg, D, tr, te, pcol, out, log, thr, names):
    from src.embeddings import compute_embeddings, nearest
    ecfg = cfg["dedup"].get("embedding", {})
    ef, idf = D["data"] / "embeddings.npy", D["data"] / "embeddings_ids.csv"
    if ef.exists() and idf.exists():
        emb = np.load(ef)
        pos = {iid: k for k, iid in enumerate(pd.read_csv(idf)["image_id"])}
        if all(i in pos for i in tr["image_id"]) and all(i in pos for i in te["image_id"]):
            etr, ete = emb[[pos[i] for i in tr["image_id"]]], emb[[pos[i] for i in te["image_id"]]]
            log.info("Memakai embedding dari langkah 1.")
        else:
            ef = None
    else:
        ef = None
    if ef is None:
        log.info("Menghitung embedding untuk uji tetangga ...")
        m, pre = ecfg.get("model", "resnet50.tv_in1k"), ecfg.get("pretrained", True)
        etr = compute_embeddings(tr[pcol].tolist(), m, 224, pre, logger=log)
        ete = compute_embeddings(te[pcol].tolist(), m, 224, pre, logger=log)
    j, sim = nearest(ete, etr)
    res = te[["image_id", "label"]].copy()
    res["nn_train_id"], res["nn_label"], res["nn_cosine"] = tr["image_id"].values[j], tr["label"].values[j], sim
    res["twin"] = res["nn_cosine"] >= thr
    # akurasi CNN per citra uji dari run baseline (bila ada)
    runs = sorted(D["runs"].glob(f"*__{cfg['experiment']['strategies'][0]}__*/test_predictions.csv"))
    if runs:
        acc = pd.concat([pd.read_csv(f)[["image_id", "correct"]] for f in runs]).groupby("image_id")["correct"].mean()
        res["cnn_acc"] = res["image_id"].map(acc)
    res.to_csv(out / "neighbors.csv", index=False)
    r = {"threshold": thr, "n_test": len(res), "frac_twins": float(res["twin"].mean()),
         "one_nn_accuracy": float((res["nn_label"] == res["label"]).mean()),
         "median_nn_cosine": float(res["nn_cosine"].median()),
         "twin_label_agreement": float((res.loc[res.twin, "nn_label"] == res.loc[res.twin, "label"]).mean())
         if res["twin"].any() else float("nan")}
    if "cnn_acc" in res and res["cnn_acc"].notna().any():
        r["cnn_acc_on_twins"] = float(res.loc[res.twin, "cnn_acc"].mean()) if res["twin"].any() else float("nan")
        r["cnn_acc_on_clean"] = float(res.loc[~res.twin, "cnn_acc"].mean())
        r["cnn_runs_used"] = len(runs)
    # gambar: histogram & pasangan paling mirip
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(res["nn_cosine"], bins=50, color="#555")
    ax.axvline(thr, color="red", ls="--", label=f"threshold {thr}")
    ax.set(xlabel="Cosine similarity: test image -> nearest training image", ylabel="Count")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(out / "fig_neighbors_hist.png", dpi=200); plt.close(fig)
    top = res.sort_values("nn_cosine", ascending=False).head(8)
    trp = tr.set_index("image_id")[pcol]
    tep = te.set_index("image_id")[pcol]
    fig, axes = plt.subplots(2, len(top), figsize=(1.7 * len(top), 3.9), squeeze=False)
    for k, (_, row) in enumerate(top.iterrows()):
        for rr, (p, lab) in enumerate(((tep[row.image_id], row.label), (trp[row.nn_train_id], row.nn_label))):
            with Image.open(p) as im:
                axes[rr, k].imshow(im.convert("L"), cmap="gray")
            axes[rr, k].set_title(f"{'test' if rr == 0 else 'train'}: {names[lab][:10]}", fontsize=6)
            axes[rr, k].axis("off")
        axes[1, k].text(0.5, -0.08, f"cos {row.nn_cosine:.3f}", transform=axes[1, k].transAxes, ha="center",
                        va="top", fontsize=6)
    fig.tight_layout(); fig.savefig(out / "fig_neighbors_top_pairs.png", dpi=200); plt.close(fig)
    return r


# ----------------------------------------------------------------------------- uji label acak
def shuffle_test(cfg, D, tr, va, te, out, log):
    from src.trainer import run_one
    dc = cfg.get("diagnostics", {})
    c = copy.deepcopy(cfg)
    c["strategies"]["diag_shuffled_labels"] = {"augment": False, "balance": "none", "loss": "ce"}
    ep = int(dc.get("shuffle_epochs", 5))
    c["train"].update(epochs=ep, min_epochs=ep, patience=ep + 1, keep_checkpoint=False)
    rng = np.random.default_rng(12345)

    def shuffled(df):
        """Label acak yang INDEPENDEN SECARA EKSAK dari label asli: tabel silang (asli × acak) dibuat
        sama dengan harapan proporsional (n_asli × n_acak / N). Permutasi biasa pada data kecil
        menyisakan keselarasan kebetulan yang bisa dimanfaatkan model."""
        df = df.copy()
        y = df["label"].to_numpy()
        cls_of = dict(zip(df["label"], df["cls"]))
        labs = np.unique(y)
        rows = np.array([(y == a).sum() for a in labs])
        cols = rows.copy()  # jumlah per kelas tetap sama
        E = np.outer(rows, cols) / rows.sum()
        T = np.floor(E).astype(int)
        frac = E - T
        for i, j in sorted(np.ndindex(T.shape), key=lambda ij: -frac[ij]):  # pembulatan matriks
            if T[i].sum() < rows[i] and T[:, j].sum() < cols[j]:
                T[i, j] += 1
        assert (T.sum(1) == rows).all() and (T.sum(0) == cols).all(), "pembulatan gagal"
        new = np.empty_like(y)
        for i, a in enumerate(labs):
            idx = rng.permutation(np.nonzero(y == a)[0])
            new[idx] = np.repeat(labs, T[i])
        df["label"] = new
        df["cls"] = [cls_of[v] for v in new]
        return df
    # label VALIDASI juga diacak: validasi dipakai memilih epoch terbaik; bila labelnya asli, pemilihan
    # itu sendiri membocorkan informasi label ke model. Label UJI tetap asli (yang dinilai).
    sh, va_sh = shuffled(tr), shuffled(va)
    model = dc.get("shuffle_model", "resnet18")
    log.info(f"Uji label acak: {model}, {ep} epoch, label latih diacak ...")
    res = run_one(c, cfg["imbalance"]["design"], str(cfg["imbalance"]["ratios"][-1]), "diag_shuffled_labels", model, 0,
                  sh, va_sh, te, out / "runs", log, overwrite=True)
    mj = load_json(out / "runs" / f"{res['run_id']}" / "metrics.json")
    m, ci = mj["test"], mj.get("test_ci95", {})
    maj = te["label"].value_counts(normalize=True).max()
    return {"model": model, "epochs": ep, "test_acc": m["accuracy"], "test_macro_f1": m["macro_f1"],
            "test_macro_auc": m["macro_auc"], "macro_auc_ci95": ci.get("macro_auc", [float("nan")] * 2),
            "majority_acc": float(maj)}


# ----------------------------------------------------------------------------- utama
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--tests", default=",".join(ALL_TESTS))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config(args.config)
    D = get_dirs(cfg)
    out = D["work"] / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    log = get_logger("diag", str(D["logs"] / "08_diagnostics.log"))
    tests = [t.strip() for t in args.tests.split(",") if t.strip()]
    names = cfg["data"]["class_names"]
    tb_idx = names.index(cfg["data"]["minority_class"])
    dc = cfg.get("diagnostics", {})
    strong = float(dc.get("strong_shortcut_acc", 0.90))

    for f in ("train_full.csv", "val.csv", "test.csv"):
        if not (D["splits"] / f).exists():
            raise SystemExit("Split belum ada. Jalankan langkah 1-2 (atau 01b) terlebih dahulu.")
    tr, va, te = (pd.read_csv(D["splits"] / f) for f in ("train_full.csv", "val.csv", "test.csv"))
    from src.tbx11k import restore_cache
    restore_cache(cfg, D, log)
    pcol = "path_cached" if "path_cached" in tr else "path"
    maj = float(te["label"].value_counts(normalize=True).max())
    prep = {k: cfg["data"].get(k) for k in ("border_crop", "intensity_norm")}
    R = {"majority_acc": maj, "n_train": len(tr), "n_val": len(va), "n_test": len(te), "prep": prep, "tests": {}}
    lines = [f"DIAGNOSIS KEBOCORAN & SHORTCUT  (config: {args.config})",
             f"Data: latih {len(tr)}, validasi {len(va)}, uji {len(te)} | akurasi tebakan kelas terbanyak: {maj:.3f}",
             f"Pra-pemrosesan: border_crop={prep['border_crop']}, intensity_norm={prep['intensity_norm']}", ""]

    if any(t in tests for t in ("lowlevel", "thumbnail", "border")):
        log.info("Mengekstrak ciri sederhana (latih/validasi/uji) ...")
        F = {s: extract(d, pcol, prep, args.workers) for s, d in (("tr", tr), ("va", va), ("te", te))}
        y = {"tr": tr["label"].values, "va": va["label"].values, "te": te["label"].values}
        groups = {"lowlevel": [("statistik intensitas global", "low"), ("metadata (ukuran asli, ukuran file)", "meta")],
                  "thumbnail": [("citra 16x16", "t16"), ("citra 32x32", "t32")],
                  "border": [("tepi citra saja (tengah ditutup)", "border")]}
        for t in [t for t in ("lowlevel", "thumbnail", "border") if t in tests]:
            rows = []
            for label, key in groups[t]:
                if key == "meta" and np.isnan(F["tr"][key]).all():
                    continue
                r = fit_eval(label, F["tr"][key], y["tr"], F["va"][key], y["va"], F["te"][key], y["te"], tb_idx)
                r["verdict"] = ("KUAT" if r["test_acc"] >= strong else
                                "SEDANG" if r["test_acc"] >= maj + 0.5 * (strong - maj) else "LEMAH")
                rows.append(r)
                log.info(f"[{t}] {label}: akurasi uji {r['test_acc']:.3f} (sinyal shortcut {r['verdict']})")
            R["tests"][t] = rows
            pd.DataFrame(rows).to_csv(out / f"{t}.csv", index=False)

    if "neighbors" in tests:
        thr = float(dc.get("neighbor_threshold", 0.97))
        R["tests"]["neighbors"] = neighbors_test(cfg, D, tr, te, pcol, out, log, thr, names)

    if "shuffle" in tests:
        R["tests"]["shuffle"] = shuffle_test(cfg, D, tr, va, te, out, log)

    # ---------------- laporan
    lines.append("=" * 78)
    lines.append("A. SHORTCUT: seberapa jauh label bisa ditebak TANPA melihat lesi?")
    lines.append(f"   (tebakan kelas terbanyak = {maj:.3f}; sinyal KUAT bila akurasi >= {strong:.2f})")
    lines.append("=" * 78)
    flag_shortcut = False
    for t in ("lowlevel", "thumbnail", "border"):
        for r in R["tests"].get(t, []):
            lines.append(f"  {r['features']:<40} akurasi {r['test_acc']:.3f} | macro-F1 {r['test_macro_f1']:.3f} | "
                         f"sens. TB {r['test_tb_recall']:.3f} -> {r['verdict']}")
            flag_shortcut |= r["verdict"] == "KUAT"
    if not any(t in R["tests"] for t in ("lowlevel", "thumbnail", "border")):
        lines.append("  (tidak dijalankan)")

    lines += ["", "=" * 78, "B. KEBOCORAN: apakah citra uji punya 'kembaran' di data latih?", "=" * 78]
    flag_leak = False
    nb = R["tests"].get("neighbors")
    if nb:
        lines.append(f"  Citra uji dengan kembaran (cosine >= {nb['threshold']}): {100 * nb['frac_twins']:.1f}% "
                     f"| kesamaan label kembaran: {nb['twin_label_agreement']:.3f}")
        lines.append(f"  Akurasi 1-tetangga-terdekat (tanpa pelatihan): {nb['one_nn_accuracy']:.3f}")
        if "cnn_acc_on_clean" in nb:
            gap = nb.get("cnn_acc_on_twins", np.nan) - nb["cnn_acc_on_clean"]
            lines.append(f"  Akurasi CNN baseline ({nb['cnn_runs_used']} run): citra berkembaran "
                         f"{nb.get('cnn_acc_on_twins', float('nan')):.3f} vs tanpa kembaran {nb['cnn_acc_on_clean']:.3f}")
            flag_leak = nb["frac_twins"] > 0.05 and gap > 0.02
        else:
            lines.append("  (akurasi CNN per kelompok tersedia setelah run baseline langkah 3 ada)")
        flag_leak |= nb["frac_twins"] > 0.10
        lines.append("  Periksa visual: fig_neighbors_top_pairs.png (apakah pasangan teratas pasien yang sama?)")
    else:
        lines.append("  (tidak dijalankan)")

    lines += ["", "=" * 78, "C. BUG PIPELINE: uji label acak", "=" * 78]
    flag_bug = False
    sh = R["tests"].get("shuffle")
    if sh:
        lo, hi = sh["macro_auc_ci95"]
        auc_bad = (lo > 0.5) if np.isfinite(lo) else sh["test_macro_auc"] > 0.60  # CI 95% tidak memuat 0,5
        flag_bug = sh["test_acc"] > sh["majority_acc"] + 0.05 or auc_bad
        lines.append(f"  Akurasi uji dengan label latih+validasi diacak: {sh['test_acc']:.3f} (harus ~{sh['majority_acc']:.3f}) | "
                     f"macro-AUC {sh['test_macro_auc']:.3f}, CI95 [{lo:.3f}, {hi:.3f}] (harus memuat 0,5) -> "
                     f"{'MASALAH' if flag_bug else 'LULUS'}")
    else:
        lines.append("  (tidak dijalankan)")

    lines += ["", "=" * 78, "KESIMPULAN & SARAN", "=" * 78]
    if flag_bug:
        lines.append("  [!] Uji label acak GAGAL: ada informasi label yang bocor di pipeline. Hentikan analisis, "
                     "laporkan ke pengembang kode.")
    if flag_leak:
        lines.append("  [!] INDIKASI kebocoran lintas split (perlu dikonfirmasi visual). Buka fig_neighbors_top_pairs.png:")
        lines.append("      - bila pasangan teratas benar-benar citra/pasien yang sama -> turunkan dedup.embedding.threshold "
                     "(mis. 0.95), ulangi langkah 1-2 dan semua run;")
        lines.append("      - bila pasangan tampak berbeda -> kemiripan hanya mencerminkan kemiripan umum rontgen dada "
                     "(fitur ImageNet tidak andal mengenali pasien yang sama), bukan kebocoran.")
    if flag_shortcut:
        lines.append("  [!] Sinyal shortcut KUAT: kelas dapat dibedakan dari ciri teknis tanpa melihat lesi. Akurasi "
                     "CNN yang sangat tinggi mungkin sebagian berasal dari ciri ini.")
        lines.append("      Saran: (1) coba data.border_crop: 0.06 dan/atau data.intensity_norm: equalize, jalankan "
                     "ulang uji ini untuk melihat apakah sinyal turun; (2) tulis sebagai keterbatasan di paper; "
                     "(3) validasi eksternal (RSUD Haji) menjadi bukti utama generalisasi.")
    if not (flag_bug or flag_leak or flag_shortcut):
        lines.append("  Tidak ada indikasi kuat kebocoran, bug, maupun shortcut pada uji yang dijalankan.")
    lines.append("  Catatan: tanpa ID pasien, kebocoran hanya dapat DIMINIMALKAN, tidak dijamin nol.")

    R["flags"] = {"shortcut": flag_shortcut, "leakage": flag_leak, "pipeline_bug": flag_bug}
    save_json(R, out / "diagnostics_report.json")
    (out / "diagnostics_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    log.info(f"Laporan: {out / 'diagnostics_report.txt'}")


if __name__ == "__main__":
    main()
