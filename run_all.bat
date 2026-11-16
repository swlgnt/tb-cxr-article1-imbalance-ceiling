@echo off
REM ======================================================================
REM  Menjalankan seluruh pipeline (Windows).
REM  WAJIB dijalankan dari "Anaconda Prompt" SETELAH:  conda activate tbstudy
REM  Cara: ketik  run_all.bat  lalu Enter (di dalam folder proyek).
REM  Aman diulang: langkah 1-2 dilewati bila sudah ada, run yang selesai dilewati,
REM  run yang terputus dilanjutkan.
REM ======================================================================
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"

python -c "import torch, timm, yaml" >nul 2>nul
if errorlevel 1 (
    echo [GAGAL] Environment belum siap. Buka Anaconda Prompt, jalankan: conda activate tbstudy
    echo         lalu jalankan file ini dari jendela tersebut.
    pause
    exit /b 1
)

if exist "work\splits\test.csv" (
    echo [INFO] Langkah 1-2 sudah pernah dijalankan - dilewati.
) else (
    echo ===== Langkah 1: menyiapkan data =====
    python 01_prepare_data.py || goto :error
    echo ===== Langkah 2: membuat split =====
    python 02_make_splits.py || goto :error
)

python -c "import sys; from src.utils import load_config; c=load_config('config.yaml'); sys.exit(0 if any(c['strategies'][s].get('synthetic_csv') for s in c['experiment']['strategies']) else 1)"
if errorlevel 1 goto :skipgan
if exist "work\synthetic\gan\synthetic.csv" (
    echo [INFO] Citra sintetis GAN sudah ada - langkah 2b-2c dilewati.
    goto :skipgan
)
echo ===== Langkah 2b: melatih GAN dasar =====
python 02b_train_gan.py || goto :error
echo ===== Langkah 2c: membangkitkan citra sintetis =====
python 02c_generate_synthetic.py || goto :error
echo.
echo [PENTING] Periksa dulu work\synthetic\gan\fig_real_vs_synthetic.png dan quality_report.json
echo           Tutup jendela ini bila kualitas buruk. Tekan tombol apa saja untuk lanjut ke pelatihan CNN.
pause
:skipgan
echo ===== Langkah 3: pelatihan (bisa berjam-jam/berhari-hari) =====
python 03_run_experiments.py || goto :error
echo ===== Langkah 4: rekap hasil =====
python 04_aggregate_results.py || goto :error
echo ===== Langkah 5: uji statistik =====
python 05_statistical_tests.py || goto :error
echo ===== Langkah 6: gambar =====
python 06_make_figures.py || goto :error

echo.
echo ======================================================
echo  SELESAI. Hasil ada di folder work\results dan work\figures
echo ======================================================
pause
exit /b 0

:error
echo.
echo [GAGAL] Ada langkah yang error. Baca pesan di atas dan file di folder work\logs
echo Lihat bagian Troubleshooting di PANDUAN_PC.md
pause
exit /b 1
