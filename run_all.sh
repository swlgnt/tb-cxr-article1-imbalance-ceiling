#!/usr/bin/env bash
# Menjalankan seluruh pipeline (Linux/macOS). Jalankan setelah: conda activate tbstudy
# Pemakaian: bash run_all.sh            (seluruh pipeline)
#            bash run_all.sh analysis   (hanya langkah 4-6)
set -e
cd "$(dirname "$0")"
if [ "$1" != "analysis" ]; then
  if [ -f work/splits/test.csv ]; then
    echo "[INFO] Langkah 1-2 sudah pernah dijalankan - dilewati."
  else
    python 01_prepare_data.py
    python 02_make_splits.py
  fi
  if python -c "import sys; from src.utils import load_config; c=load_config('config.yaml'); sys.exit(0 if any(c['strategies'][s].get('synthetic_csv') for s in c['experiment']['strategies']) else 1)"; then
    if [ ! -f work/synthetic/gan/synthetic.csv ]; then
      python 02b_train_gan.py
      python 02c_generate_synthetic.py
      echo "[PENTING] Periksa work/synthetic/gan/fig_real_vs_synthetic.png dulu. Enter untuk lanjut, Ctrl+C untuk berhenti."
      read -r _
    fi
  fi
  python 03_run_experiments.py
fi
python 04_aggregate_results.py
python 05_statistical_tests.py
python 06_make_figures.py
echo "SELESAI. Hasil di work/results dan work/figures"
