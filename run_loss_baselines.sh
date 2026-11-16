#!/usr/bin/env bash
# Pembanding level loss: weighted_ce dan focal (4 model x 5 seed x 2 strategi = 40 run per resolusi).
# Linux / Google Colab. Lihat PANDUAN_PEMBANDING_LOSS.md
#   bash run_loss_baselines.sh          -> versi 224 px (config.yaml, folder work)
#   bash run_loss_baselines.sh 512      -> versi 512 px (config_512.yaml, folder work_512)
set -e
cd "$(dirname "$0")"
CFG=config.yaml; WD=work
if [ "${1:-}" = "512" ]; then CFG=config_512.yaml; WD=work_512; fi
if [ ! -f "$WD/splits/test.csv" ]; then
  echo "[GAGAL] Split belum ada di $WD/splits. Jalankan run_all.sh (atau run_all_512) minimal sampai langkah 2."
  exit 1
fi
python 03_run_experiments.py --config "$CFG" --strategies weighted_ce,focal --dry-run
python 03_run_experiments.py --config "$CFG" --strategies weighted_ce,focal
python 04_aggregate_results.py --config "$CFG"
python 05_statistical_tests.py --config "$CFG"
python 06_make_figures.py --config "$CFG"
echo "SELESAI. Lihat $WD/results/tables/table_loss_baselines_*.csv dan $WD/results/stats/stats_report.txt"
