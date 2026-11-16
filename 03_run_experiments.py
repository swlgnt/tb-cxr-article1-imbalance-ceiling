#!/usr/bin/env python
"""Langkah 3: jalankan grid eksperimen (rasio × strategi × model × seed).

Aman dijalankan berulang: run yang sudah selesai dilewati, run yang terputus dilanjutkan.
Contoh:
  python 03_run_experiments.py                                  # seluruh grid di config
  python 03_run_experiments.py --models resnet18 --seeds 42     # sebagian
  python 03_run_experiments.py --shard 0/2                      # bagi ke 2 mesin/akun Colab
  python 03_run_experiments.py --dry-run                        # tampilkan daftar run saja
"""
import argparse
import itertools
import time
import traceback

import pandas as pd

from src.design_check import design_report
from src.tbx11k import restore_cache
from src.dataset import build_train_set, composition_table
from src.trainer import run_one
from src.utils import get_dirs, get_logger, git_info, keep_awake, load_config, make_run_id, ratio_tag


def csv_list(s, typ=str):
    return [typ(x) for x in s.split(",")] if s else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--models")
    ap.add_argument("--strategies")
    ap.add_argument("--seeds")
    ap.add_argument("--ratios")
    ap.add_argument("--shard", default="0/1", help="i/n: jalankan bagian ke-i dari n")
    ap.add_argument("--max-runs", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument("--require-clean", action="store_true",
                    help="tolak berjalan bila ada perubahan kode/config yang belum di-commit")
    args = ap.parse_args()

    cfg = load_config(args.config)
    D = get_dirs(cfg)
    log = get_logger("exp", str(D["logs"] / f"03_run_{time.strftime('%Y%m%d')}.log"))
    ex, ic = cfg["experiment"], cfg["imbalance"]
    design = ic["design"]
    models = csv_list(args.models) or ex["models"]
    strategies = csv_list(args.strategies) or ex["strategies"]
    seeds = csv_list(args.seeds, int) or ex["seeds"]
    ratios = csv_list(args.ratios) or [ratio_tag(r) for r in ic["ratios"]]
    for s in strategies:
        assert s in cfg["strategies"], f"strategi '{s}' tidak ada di config"
    for m in models:
        assert m in cfg["models"], f"model '{m}' tidak ada di config"

    # seed paling luar -> hasil parsial tetap seimbang antar kondisi
    grid = list(itertools.product(seeds, ratios, strategies, models))
    si, sn = map(int, args.shard.split("/"))
    grid = [g for k, g in enumerate(grid) if k % sn == si]
    todo = [g for g in grid if args.overwrite or not
            (D["runs"] / make_run_id(design, g[1], g[2], g[3], g[0]) / "metrics.json").exists()]
    # ---- komposisi data latih per skenario (fungsi yang SAMA dengan pelatihan)
    # Selalu untuk SELURUH strategi di config (+ yang dipilih), agar --strategies sebagian (mis. hanya
    # weighted_ce,focal) tidak menimpa train_composition.csv dan laporan desain dengan daftar yang terpotong.
    all_strats = list(dict.fromkeys(list(ex["strategies"]) + strategies))
    comp_rows = []
    for r in ratios:
        trp = D["splits"] / f"train_{design}_r{r}.csv"
        if not trp.exists():
            continue
        tr = pd.read_csv(trp)
        for s in all_strats:
            scfg = cfg["strategies"][s]
            if scfg.get("extra") == "gan" or (scfg.get("synthetic_csv") and "dose" not in scfg):
                from src.dataset import gap_counts
                base = tr.assign(source="real")
                add = []
                if "dose" in scfg:
                    tl = [cfg["data"]["class_names"].index(c) for c in scfg.get("dose_classes", [cfg["data"]["minority_class"]])]
                    for c, k in gap_counts(tr, float(scfg["dose"]), tl).items():
                        add.append(pd.DataFrame({"label": [c] * k, "source": "gan"}))
                built = pd.concat([base] + add, ignore_index=True)
            else:
                built = build_train_set(cfg, s, r, tr, cfg["paths"]["work_dir"])
            t = composition_table(built, cfg["data"]["class_names"])
            t.insert(0, "strategy", s)
            t.insert(0, "ratio", r)
            comp_rows.append(t)
    if comp_rows:
        comp = pd.concat(comp_rows, ignore_index=True)
        comp.to_csv(D["work"] / "results" / "train_composition.csv", index=False)
        lines = ["KOMPOSISI DATA LATIH PER SKENARIO:"]
        for (r, s), g in comp.groupby(["ratio", "strategy"], sort=False):
            n_add = int(g.loc[g["source"] != "real", "n"].sum())
            lines.append(f"  SCENARIO: {s} (rasio {r}) | tambahan = {n_add}")
            for _, row in g.iterrows():
                lines.append(f"      {row['cls']:<16} {row['source']:<8} {row['n']:>6}")
        log.info("\n" + "\n".join(lines))

    rep, ok = design_report(cfg, ratios, all_strats, models, seeds)
    if len(all_strats) > len(strategies):
        rep += (f"\n  (laporan di atas untuk seluruh strategi di config; yang dijalankan sekarang: {', '.join(strategies)})")
    (log.info if ok else log.warning)("\n" + rep)
    log.info(f"Grid: {len(grid)} run (shard {args.shard}); belum selesai: {len(todo)}")
    if args.dry_run:
        for g in todo:
            print(make_run_id(design, g[1], g[2], g[3], g[0]))
        return
    if args.max_runs:
        todo = todo[: args.max_runs]
    from pathlib import Path
    for s in sorted({g[2] for g in todo}):  # hanya strategi yang benar-benar akan dijalankan
        sc = cfg["strategies"][s].get("synthetic_csv")
        if sc:
            for r in ratios:
                p = Path(str(sc).format(ratio=r))
                p = p if p.is_absolute() else Path(cfg["paths"]["work_dir"]) / p
                if not p.exists():
                    raise SystemExit(f"Strategi '{s}' butuh citra sintetis, tetapi file belum ada:\n  {p}\n"
                                     "Jalankan dulu: python 02b_train_gan.py  lalu  python 02c_generate_synthetic.py\n"
                                     f"(atau keluarkan '{s}' dari experiment.strategies / pakai --strategies).")

    gi = git_info()
    if gi["git_commit"] is None:
        log.warning("Folder ini belum menjadi repositori Git: versi kode tidak tercatat di hasil "
                    "(lihat PANDUAN_GITHUB.md).")
    else:
        log.info(f"Versi kode: commit {gi['git_commit']} ({gi['git_branch']}"
                 + (f", tag {gi['git_tag']}" if gi["git_tag"] else "") + ")")
        if gi["git_dirty"]:
            msg = ("Ada perubahan yang BELUM di-commit: " + ", ".join(gi["git_dirty_files"]) +
                   ". Hasil run akan ditandai git_dirty=True. Disarankan: git add . && git commit -m \"...\"")
            if args.require_clean:
                raise SystemExit(msg)
            log.warning(msg)
    snap = D["work"] / "config_snapshots" / f"{time.strftime('%Y%m%d_%H%M%S')}_{gi['git_commit'] or 'nogit'}.yaml"
    snap.parent.mkdir(parents=True, exist_ok=True)
    import yaml  # simpan config GABUNGAN (termasuk warisan _base_), bukan hanya file yang dipanggil
    snap.write_text(f"# config efektif dari: {args.config}\n" + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                    encoding="utf-8")

    if not (D["splits"] / "val.csv").exists():
        raise SystemExit(f"Split data belum ada di {D['splits']}.\n"
                         "Jalankan dulu: python 01_prepare_data.py  lalu  python 02_make_splits.py")
    restore_cache(cfg, D, log)
    val = pd.read_csv(D["splits"] / "val.csv")
    test = pd.read_csv(D["splits"] / "test.csv")
    train_cache = {}
    status_file = D["runs"].parent / "runs_status.csv"
    with keep_awake():
        _loop(todo, cfg, D, design, val, test, train_cache, status_file, log, args)
    log.info("Semua run dalam daftar ini sudah diproses.")


def _loop(todo, cfg, D, design, val, test, train_cache, status_file, log, args):
    for n, (seed, ratio, strat, model) in enumerate(todo, 1):
        log.info(f"===== [{n}/{len(todo)}] =====")
        if ratio not in train_cache:
            train_cache[ratio] = pd.read_csv(D["splits"] / f"train_{design}_r{ratio}.csv")
        t0 = time.time()
        try:
            res = run_one(cfg, design, ratio, strat, model, seed, train_cache[ratio], val, test,
                          D["runs"], log, args.overwrite)
        except Exception as e:  # satu run gagal tidak menghentikan seluruh grid
            log.error(f"GAGAL {make_run_id(design, ratio, strat, model, seed)}: {e}\n{traceback.format_exc()}")
            res = {"run_id": make_run_id(design, ratio, strat, model, seed), "status": f"error: {e}"}
            if args.stop_on_error:
                raise
        res["minutes"] = round((time.time() - t0) / 60, 2)
        res["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        pd.DataFrame([res]).to_csv(status_file, mode="a", header=not status_file.exists(), index=False)


if __name__ == "__main__":
    main()
