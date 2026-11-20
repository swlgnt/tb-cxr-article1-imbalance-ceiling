# DCGAN Versus Classical Augmentation at Matched Doses for Class-Imbalanced Tuberculosis Chest X-Ray Classification

Code, data splits and per-run results for the conference paper of the same title
(Swalaganata, Astuti & Hendradi; accepted at the 2027 9th International Seminar on Research of Information
Technology and Intelligent Systems, ISRITI; in press).

The study trains four ImageNet-pretrained CNNs (ResNet-18, ResNet-50, DenseNet201, Xception) with five seeds
for three-class chest radiograph classification (normal, non-TB abnormality, TB) on TBX11K. TB training images
are supplemented with 30%, 50% or 100% of the class gap using DCGAN images or augmented copies of real images;
weighted cross-entropy and focal loss serve as loss-level comparators (180 runs). A shortcut and leakage audit
and a preprocessing ablation (30 runs) are included.

## Contents

| Folder / file | What it holds |
|---|---|
| `*.py`, `src/` | Pipeline: data indexing and de-duplication, splits, DCGAN, classifier training, analysis, figures, audit |
| `config.yaml` | Settings of the main grid (180 runs) |
| `config_abl.yaml` | Preprocessing ablation (30 runs) |
| `splits/` | TBX11K image IDs with class, split and near-duplicate group |
| `results/main/` | Main grid: `tables/all_runs.csv` (one row per run), summary tables, statistical tests, ensembles, GAN logs and image quality, shortcut and leakage audit |
| `results/ablation/` | The same outputs for the preprocessing ablation |
| `figures/` | Figures as produced by the scripts |

Code comments and console messages are partly in Indonesian.

## Data

TBX11K is publicly available from its authors (Liu et al., CVPR 2020) and is not redistributed here;
`splits/` lists the image IDs used. Trained model weights are not included because of their size.

## Reproducing

1. Python 3 with PyTorch (CUDA), then `pip install -r requirements.txt`.
2. Place TBX11K under `data/TBX11K` (see `paths` in `config.yaml`).
3. `run_all.bat` (or the numbered scripts 01-06 in order) for the main grid; `run_loss_baselines.bat` for
   the loss-level comparators; `08_leakage_diagnostics.py` for the audit; `config_abl.yaml` with
   `03_run_experiments.py` for the ablation; `09_confusion_summary.py` for per-image error patterns.
4. Robustness analyses reported in the paper:
   - `10_collect_predictions.py` collects the test-set predictions of all runs into
     `results/main/tables/predictions_all.csv.gz` (already included);
   - `11_bootstrap_image_level.py results/main/tables/predictions_all.csv.gz results/main/diagnostics/neighbors.csv results/main/stats/bootstrap_results.csv 2000`
     runs the two-level paired bootstrap (model x seed pairs and test images within class), with and without
     twin test images;
   - `12_mixed_model.py --runs results/main/tables/all_runs.csv --out results/main/stats/mixed_model.csv` fits the
     linear mixed model (likelihood-ratio test for condition, Holm-corrected coefficients).

## Citation

See `CITATION.cff`. Please also cite TBX11K (Liu et al., 2020) when using the splits.

## License

Code: MIT (see `LICENSE`). Tables in `results/`: CC BY 4.0.
