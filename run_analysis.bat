@echo off
REM Hanya langkah 4-6 (rekap, statistik, gambar). Bisa dijalankan kapan saja,
REM juga saat pelatihan baru sebagian selesai (di jendela Anaconda Prompt lain).
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
python -c "import torch, timm, yaml" >nul 2>nul
if errorlevel 1 (
    echo [GAGAL] Environment belum siap. Jalankan dulu: conda activate tbstudy
    pause
    exit /b 1
)
python 04_aggregate_results.py || goto :error
python 05_statistical_tests.py || goto :error
python 06_make_figures.py || goto :error
echo SELESAI. Lihat work\results dan work\figures
pause
exit /b 0
:error
echo [GAGAL] Lihat pesan di atas.
pause
exit /b 1
