# ENGE707 Assessment 3 — Cat/Dog Facial Feature Classification

8-class image classification (CatEye, CatFace, CatMouth, CatNose, DogEye, DogFace, DogMouth, DogNose) for the
ENGE707 Kaggle competition. Metric: accuracy.

## Layout

```
01_eda.ipynb                        Task 1   data checks, geometry, colour, duplicates, examples
02_preprocessing_and_splits.ipynb   Task 2   folds, ShiftSim calibration, adversarial validation, benchmarks
03_baseline_cnn.ipynb               Task 3/4 CNN from scratch, learning curves, first submission
04_transfer_learning.ipynb          Task 3/4 pretrained fine-tuning + ablations (fold 0)
05_cv_ensemble_training.ipynb                5-fold training of a chosen preset, OOF predictions
06_evaluation_error_analysis.ipynb  Task 5/6 metrics, confusion, size/aspect buckets, hardest images, val vs LB
07_inference_submission.ipynb                TTA + ensemble -> final submission
src/        config.py (paths, presets)  data.py  transforms.py (incl. ShiftSim)  models.py  engine.py  utils.py  plots.py
data/       folds.csv, shiftsim_params.json      written once by notebook 02, never regenerated
results/    experiments.csv (one row per run/fold), submissions.csv (one row per submission file)
submissions/                                     submission CSVs
outputs/    checkpoints/ preds/ logs/ figures/ cache/   (git-ignored)
```

Every training run is a named **preset** in `src/config.py` (`PRESETS`). The preset name is also the
checkpoint folder, so a finished run is never retrained by accident and an interrupted run resumes from its last epoch.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name enge707 --display-name "ENGE707 (.venv)"
```

The Kaggle files go in the repo root: `train/train/*.jpg`, `test/test/*.jpg`, `train_labels.csv`,
`sample_submission.csv`. On Kaggle/Colab set `ENGE707_DATA_ROOT` to the folder that contains them.

## Running

Short notebooks (01, 02, 06) can be run interactively in Jupyter / VS Code. Long training notebooks run best in the
background from a terminal in the repo root, so the laptop can be left alone (`caffeinate` keeps the Mac awake;
the Mac must stay on mains power with the lid open, or the training stops):

```bash
source .venv/bin/activate
caffeinate -is nohup papermill 03_baseline_cnn.ipynb outputs/logs/03_run.ipynb --log-output > outputs/logs/03_run.log 2>&1 &
tail -f outputs/logs/03_run.log        # watch progress (Ctrl+C stops watching, not training)
```

Parameters are changed with `-p NAME value`, e.g.

```bash
caffeinate -is nohup papermill 04_transfer_learning.ipynb outputs/logs/04_abl.ipynb \
  -p RUNS "abl_base,abl_plain,abl_letterbox,abl_scale04,abl_scale08,abl_nolowres,abl_nosmooth,abl_96,abl_160" \
  --log-output > outputs/logs/04_abl.log 2>&1 &
caffeinate -is nohup papermill 05_cv_ensemble_training.ipynb outputs/logs/05_convnext.ipynb \
  -p RUN convnext_tiny_shift --log-output > outputs/logs/05_convnext.log 2>&1 &
```

The executed copy in `outputs/logs/*.ipynb` contains all outputs. To get the figures into the notebook in the repo
root, open it afterwards and *Run All*: finished runs are loaded from disk instead of retrained, so this takes minutes.

Only one training job should run at a time (they share the GPU).

## Measured speed (M1 Pro, MPS)

| preset | model | px | min / epoch | default epochs |
|---|---|---|---|---|
| cnn_* | SimpleCNN | 96 | ~1.5 | 20 |
| abl_* | EfficientNet-B0 | 128 | ~5 | 6 |
| convnext_tiny_* | ConvNeXt-Tiny | 128 | ~11 | 10 |
| vit_small_shift | ViT-S/16 | 128 | ~5 | 10 |

The 224 px presets (`*_224`) are ~4x slower on MPS and are intended for a CUDA GPU.

## Submissions

Each submission is validated against `sample_submission.csv` before it is written and logged in
`results/submissions.csv`. After uploading it to Kaggle, type the public score into the `lb_public` column;
notebook 06 plots validation accuracy against it.
