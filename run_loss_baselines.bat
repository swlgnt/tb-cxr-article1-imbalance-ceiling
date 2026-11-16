@echo off
REM ======================================================================
REM  Pembanding level loss: weighted_ce dan focal
REM  4 model x 5 seed x 2 strategi = 40 run per resolusi. Lihat PANDUAN_PEMBANDING_LOSS.md
REM  WAJIB dijalankan dari "Anaconda Prompt" SETELAH:  conda activate tbstudy
REM    run_loss_baselines.bat        -> versi 224 px (config.yaml, folder work)
REM    run_loss_baselines.bat 512    -> versi 512 px (config_512.yaml, folder work_512)
REM  Aman diulang: run yang sudah selesai dilewati, run yang terputus dilanjutkan.
REM  JANGAN dijalankan bersamaan dengan run_all.bat pada GPU yang sama.
REM ======================================================================
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
set CFG=config.yaml
set WD=work
if /I "%~1"=="512" set CFG=config_512.yaml
if /I "%~1"=="512" set WD=work_512

python -c "import torch, timm, yaml" >nul 2>nul
if errorlevel 1 (
    echo [GAGAL] Environment belum siap. Buka Anaconda Prompt, jalankan: conda activate tbstudy
    pause
    exit /b 1
)
if not exist "%WD%\splits\test.csv" (
    echo [GAGAL] Split belum ada di %WD%\splits.
    echo         Jalankan dulu run_all.bat ^(atau run_all_512.bat^) minimal sampai langkah 2.
    pause
    exit /b 1
)

echo ===== Pemeriksaan: daftar run pembanding loss yang belum selesai =====
python 03_run_experiments.py --config %CFG% --strategies weighted_ce,focal --dry-run || goto :error
echo.
echo [INFO] Periksa daftar di atas: seharusnya hanya run ...__weighted_ce__... dan ...__focal__...
echo        Tekan tombol apa saja untuk mulai melatih, atau tutup jendela ini untuk batal.
pause

echo ===== Langkah 3: pelatihan pembanding loss =====
python 03_run_experiments.py --config %CFG% --strategies weighted_ce,focal || goto :error
echo ===== Langkah 4: rekap SEMUA run (desain dosis + pembanding loss) =====
python 04_aggregate_results.py --config %CFG% || goto :error
echo ===== Langkah 5: uji statistik =====
python 05_statistical_tests.py --config %CFG% || goto :error
echo ===== Langkah 6: gambar =====
python 06_make_figures.py --config %CFG% || goto :error

echo.
echo ======================================================
echo  SELESAI. Lihat:
echo    %WD%\results\tables\table_loss_baselines_macro_f1.csv  ^(dan _tb_recall^)
echo    %WD%\results\stats\stats_report.txt  bagian PEMBANDING LEVEL LOSS
echo    %WD%\figures\fig02_macro_f1_vs_dose.png  ^(garis abu-abu = pembanding loss^)
echo ======================================================
pause
exit /b 0

:error
echo.
echo [GAGAL] Ada langkah yang error. Baca pesan di atas dan file di folder %WD%\logs
pause
exit /b 1
