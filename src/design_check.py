"""Pemeriksa kecukupan desain eksperimen untuk uji statistik di 05_statistical_tests.py.

Dijalankan SEBELUM pelatihan (03_run_experiments.py, termasuk --dry-run), sehingga kekurangan
desain (mis. hanya 1 seed atau 2 rasio) diketahui sebelum menghabiskan waktu GPU.
"""
from __future__ import annotations


def wilcoxon_min_p(n_pairs: int) -> float:
    """p dua sisi terkecil yang MUNGKIN dicapai Wilcoxon signed-rank eksak dengan n pasangan
    (terjadi bila semua selisih bertanda sama)."""
    return 2.0 / 2 ** n_pairs if n_pairs > 0 else 1.0


def check_design(ratios, strategies, models, seeds, reference: str, alpha: float = 0.05,
                 n_metrics: int = 8, n_primary: int = 0) -> tuple[list[tuple[str, bool, str]], dict]:
    r, s, m, k = len(ratios), len(strategies), len(models), len(seeds)
    n_runs = r * s * m * k
    checks = []

    # Friedman antar model: blok = rasio × strategi × seed
    blocks_m = r * s * k
    checks.append(("Perbandingan antar model (Friedman + Nemenyi)", m >= 3 and blocks_m >= 2,
                   f"{m} model, {blocks_m} blok" + ("" if m >= 3 else " - butuh >= 3 model")))

    # Antar strategi
    blocks_s = r * m * k
    if s >= 3:
        checks.append(("Perbandingan antar strategi (Friedman + Nemenyi)", blocks_s >= 2, f"{s} strategi, {blocks_s} blok"))
    else:
        checks.append(("Perbandingan antar strategi (Wilcoxon untuk 2 strategi)", s == 2 and blocks_s >= 6,
                       f"{s} strategi, {blocks_s} pasangan" + ("" if s == 2 else " - butuh >= 2 strategi")))

    # Wilcoxon strategi vs referensi, per model: pasangan = rasio × seed
    n_pairs = r * k
    fam = m * max(s - 1, 1)  # keluarga koreksi Holm per metrik
    pmin = wilcoxon_min_p(n_pairs)
    ok_w = reference in strategies and s >= 2 and pmin * fam < alpha
    checks.append((f"Wilcoxon tiap strategi vs '{reference}' per model", ok_w,
                   f"{n_pairs} pasangan/model; p terkecil mungkin {pmin:.4f}, x{fam} uji (Holm) = {pmin * fam:.3f} "
                   + ("< " if pmin * fam < alpha else ">= ") + f"{alpha}"
                   + ("" if reference in strategies else f" - strategi '{reference}' tidak ada")))

    # Tren terhadap rasio
    checks.append(("Tren metrik terhadap rasio (Spearman/regresi)", r >= 3, f"{r} rasio" + ("" if r >= 3 else " - butuh >= 3")))

    # ANOVA faktorial: sisa derajat bebas > 0 dan ulangan
    lv = [x for x in (m, s, r) if x >= 2]
    n_par = 1 + sum(x - 1 for x in lv) + sum((a - 1) * (b - 1) for i, a in enumerate(lv) for b in lv[i + 1:])
    df_res = n_runs - n_par
    checks.append(("ANOVA faktorial (model × strategi × rasio)", k >= 2 and df_res >= 10,
                   f"{n_runs} run, {n_par} parameter, sisa df {df_res}" + ("" if k >= 2 else " - butuh >= 2 seed")))

    # Ulangan untuk mean ± SD
    checks.append(("Mean ± SD dan CI antar seed per kondisi", k >= 3, f"{k} seed" + ("" if k >= 3 else " - disarankan >= 3")))

    info = {"n_runs": n_runs, "wilcoxon_pairs": n_pairs, "wilcoxon_min_p": pmin, "holm_family": fam,
            "all_metrics_family": fam * n_metrics,
            "min_p_all_metrics": pmin * fam * n_metrics,
            "primary_family": fam * n_primary, "min_p_primary": pmin * fam * n_primary}
    return checks, info


def check_dose_design(cfg, ratios, strategies, models, seeds) -> tuple[list, dict]:
    A = cfg["analysis"]
    alpha, ref = A["alpha"], A["reference_strategy"]
    sc = {s: cfg["strategies"][s] for s in strategies}
    r, s, m, k = len(ratios), len(strategies), len(models), len(seeds)
    meth = {}
    for name, v in sc.items():
        if "dose" in v and v.get("extra") in ("classic", "gan"):
            meth.setdefault(v["extra"], set()).add(float(v["dose"]))
    checks = []
    blocks_m = r * s * k
    checks.append(("Perbandingan antar model (Friedman + Nemenyi)", m >= 3 and blocks_m >= 2, f"{m} model, {blocks_m} blok"))
    checks.append(("Perbandingan antar skenario (Friedman + Nemenyi)", s >= 3 and r * m * k >= 2,
                   f"{s} skenario, {r * m * k} blok (model × seed)"))
    pairs = r * m * k
    pmin = wilcoxon_min_p(pairs)
    fam = max(s - 1, 1)
    checks.append((f"Tiap skenario vs '{ref}' (Wilcoxon gabungan lintas model)", ref in strategies and pmin * fam < alpha,
                   f"{pairs} pasangan; p terkecil mungkin x{fam} uji = {pmin * fam:.2g}"
                   + ("" if ref in strategies else f" - strategi '{ref}' tidak ada")))
    common = sorted(meth.get("gan", set()) & meth.get("classic", set()))
    checks.append(("GAN vs klasik pada dosis yang sama (Wilcoxon gabungan)", len(common) >= 1 and pmin * max(len(common), 1) < alpha,
                   f"{len(common)} dosis sepadan {[int(d * 100) for d in common]}, {pairs} pasangan per dosis"))
    lv = {x: len(v | ({0.0} if ref in strategies else set())) for x, v in meth.items()}
    checks.append(("Tren metrik terhadap dosis (0% = original)", all(v >= 3 for v in lv.values()) and bool(lv),
                   ", ".join(f"{x}: {v} tingkat" for x, v in lv.items()) or "tidak ada skenario dosis"))
    n_doses = len(set().union(*meth.values())) if meth else 0
    n_an = r * m * k * sum(len(v) for v in meth.values())
    lvls = [x for x in (m, len(meth), n_doses) if x >= 2]
    n_par = 1 + sum(x - 1 for x in lvls) + sum((a - 1) * (b - 1) for i, a in enumerate(lvls) for b in lvls[i + 1:])
    checks.append(("ANOVA model × metode × dosis", k >= 2 and n_an - n_par >= 10,
                   f"{n_an} run, {n_par} parameter, sisa df {n_an - n_par}"))
    from .utils import loss_baselines
    lb = loss_baselines(cfg, strategies)
    if lb:  # pembanding level loss: vs original (L1) dan vs tiap skenario penambahan citra (L2)
        n_data = sum(len(v) for v in meth.values())
        fam_l = max(len(lb) * n_data, 1)
        checks.append((f"Pembanding level loss {lb}: vs '{ref}' dan vs skenario penambahan citra (Wilcoxon gabungan)",
                       ref in strategies and pmin * fam_l < alpha,
                       f"{pairs} pasangan; L2: {fam_l} uji per metrik, p terkecil mungkin x jumlah uji = {pmin * fam_l:.2g}"))
    checks.append(("Per model: DeLong & McNemar (prediksi ensemble per citra uji)", True, "selalu tersedia"))
    checks.append(("Mean ± SD dan CI antar seed per kondisi", k >= 3, f"{k} seed"))
    prim = A.get("primary_metrics", [])
    if prim:
        f = fam * len(prim)
        checks.append((f"Koreksi Holm atas metrik primer {prim}", pmin * f < alpha, f"{f} uji; p terkecil mungkin x jumlah uji = {pmin * f:.2g}"))
    return checks, {"n_runs": r * s * m * k}


def design_report(cfg: dict, ratios, strategies, models, seeds) -> tuple[str, bool]:
    A = cfg["analysis"]
    if any("dose" in cfg["strategies"][x] for x in strategies):
        checks, info = check_dose_design(cfg, ratios, strategies, models, seeds)
        lines = ["KECUKUPAN DESAIN STATISTIK - desain dosis (setelah grid ini SELESAI):",
                 f"  {len(ratios)} rasio × {len(strategies)} skenario × {len(models)} model × {len(seeds)} seed "
                 f"= {info['n_runs']} run"]
        lines += [f"  [{'OK' if ok else 'KURANG'}] {n}: {d}" for n, ok, d in checks]
        if len(ratios) > 1:
            lines.append(f"  PERINGATAN: desain dosis sebaiknya memakai satu rasio (mis. [orig]); saat ini {list(ratios)}.")
        lines.append("  Catatan: uji utama digabung lintas model (pasangan model × seed). Wilcoxon per model hanya punya "
                     f"{len(seeds)} pasangan sehingga bersifat deskriptif; perbandingan per model memakai DeLong.")
        return "\n".join(lines), all(ok for _, ok, _ in checks)
    prim = A.get("primary_metrics", [])
    checks, info = check_design(ratios, strategies, models, seeds, A["reference_strategy"], A["alpha"],
                                len(A["key_metrics"]), len(prim))
    lines = ["KECUKUPAN DESAIN STATISTIK (setelah grid ini SELESAI):",
             f"  {len(ratios)} rasio × {len(strategies)} strategi × {len(models)} model × {len(seeds)} seed "
             f"= {info['n_runs']} run"]
    for name, ok, detail in checks:
        lines.append(f"  [{'OK' if ok else 'KURANG'}] {name}: {detail}")
    if prim:
        okp = info["min_p_primary"] < A["alpha"]
        checks.append((f"Koreksi Holm atas metrik primer {prim}", okp,
                       f"{info['primary_family']} uji; p terkecil mungkin x jumlah uji = {info['min_p_primary']:.3f}"))
        lines.append(f"  [{'OK' if okp else 'KURANG'}] {checks[-1][0]}: {checks[-1][2]}")
    if info["min_p_all_metrics"] >= A["alpha"]:
        lines.append(f"  Catatan: koreksi atas SEMUA {len(A['key_metrics'])} metrik sekaligus ({info['all_metrics_family']} uji) "
                     "tidak akan pernah signifikan; karena itu klaim utama memakai metrik primer, metrik lain sekunder.")
    all_ok = all(ok for _, ok, _ in checks)
    return "\n".join(lines), all_ok
