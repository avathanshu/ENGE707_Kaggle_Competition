"""Paths, constants, run presets and the device helper. Every notebook imports from here."""
import json
import os
from dataclasses import dataclass, asdict, field, replace
from pathlib import Path

# Must be set before torch is imported: lets unsupported MPS ops fall back to the CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch  # noqa: E402
import resource  # noqa: E402

# macOS terminals default to 256 open files per process; DataLoader workers pass every batch through file
# descriptors, so a long run with several loaders can hit "Too many open files". Raise the soft limit
# (inherited by the worker processes) as far as the hard limit allows.
_soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
_target = 65536 if _hard == resource.RLIM_INFINITY else min(65536, _hard)
if _soft < _target:
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (_target, _hard))
    except (ValueError, OSError):
        pass

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
# Raw Kaggle files. On a Kaggle / Colab GPU set the environment variable ENGE707_DATA_ROOT
# to the folder that holds train/, test/ and the two CSVs (e.g. /kaggle/input/<competition>).
DATA_ROOT = Path(os.environ.get("ENGE707_DATA_ROOT", ROOT))
TRAIN_DIR = DATA_ROOT / "train" / "train"
TEST_DIR = DATA_ROOT / "test" / "test"
LABELS_CSV = DATA_ROOT / "train_labels.csv"
SAMPLE_SUB_CSV = DATA_ROOT / "sample_submission.csv"

DATA_DIR = ROOT / "data"                 # small derived tables (tracked in git)
RESULTS_DIR = ROOT / "results"           # experiment / submission logs (tracked)
SUB_DIR = ROOT / "submissions"           # submission CSVs (tracked)
OUT_DIR = ROOT / "outputs"               # heavy files (git-ignored)
CKPT_DIR = OUT_DIR / "checkpoints"
PRED_DIR = OUT_DIR / "preds"
LOG_DIR = OUT_DIR / "logs"
FIG_DIR = OUT_DIR / "figures"
CACHE_DIR = OUT_DIR / "cache"

FOLDS_CSV = DATA_DIR / "folds.csv"
SHIFTSIM_JSON = DATA_DIR / "shiftsim_params.json"
EXPERIMENTS_CSV = RESULTS_DIR / "experiments.csv"
SUBMISSIONS_CSV = RESULTS_DIR / "submissions.csv"

# ---------------------------------------------------------------- constants
CLASSES = ["CatEye", "CatFace", "CatMouth", "CatNose", "DogEye", "DogFace", "DogMouth", "DogNose"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
SEED = 42
N_FOLDS = 5

# Default ShiftSim parameters; notebook 02 calibrates them against the test geometry
# and stores the result in data/shiftsim_params.json, which then takes precedence.
DEFAULT_SHIFTSIM = dict(p_keep=0.5, scale_min=0.6, aspect_max=1.33, p_round=0.25, shrink_min=0.5, seed=SEED)


def make_dirs():
    for d in [DATA_DIR, RESULTS_DIR, SUB_DIR, CKPT_DIR, PRED_DIR, LOG_DIR, FIG_DIR, CACHE_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def get_device():
    """cuda -> mps -> cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_shiftsim_params():
    if SHIFTSIM_JSON.exists():
        return json.loads(SHIFTSIM_JSON.read_text())
    return dict(DEFAULT_SHIFTSIM)


# ---------------------------------------------------------------- training config
@dataclass
class TrainConfig:
    run_name: str
    model_name: str = "convnext_tiny.fb_in22k_ft_in1k"   # or "simple_cnn"
    pretrained: bool = True
    img_size: int = 224
    resize_mode: str = "squash"        # "squash" (default) or "letterbox" (pad to square first)
    aug: str = "shift"                 # "none", "plain", "shift" or "sim" (see src/transforms.py)
    crop_scale_min: float = 0.6        # RandomResizedCrop area floor
    crop_ratio: tuple = (0.6, 1.67)    # RandomResizedCrop aspect range
    lowres_p: float = 0.3              # probability of the low-resolution degradation
    epochs: int = 12
    batch_size: int = 64
    lr: float = 2e-4                   # backbone learning rate
    head_lr: float = 1e-3              # classifier-head learning rate
    weight_decay: float = 0.05
    warmup_epochs: int = 1
    label_smoothing: float = 0.1
    ema_decay: float = 0.999           # 0 switches EMA off
    drop_rate: float = 0.0             # dropout before the classifier
    drop_path_rate: float = 0.1        # stochastic depth (timm models only)
    cnn_width: int = 32                # channels of the first block (simple_cnn only)
    patience: int = 4                  # early stopping on val_shift accuracy (0 = off)
    amp: bool = True                   # float16 autocast on cuda / mps
    drop_degenerate: bool = True       # drop training images with short side < 8 px or aspect > 3
    num_workers: int = 6
    seed: int = SEED
    train_limit: int = 0               # >0: use only this many training images (smoke tests)
    notes: str = ""

    def to_dict(self):
        return asdict(self)


_CNN = dict(model_name="simple_cnn", pretrained=False, img_size=96, batch_size=128, lr=2e-3, head_lr=2e-3,
            weight_decay=1e-4, ema_decay=0.0, drop_rate=0.3, drop_path_rate=0.0, epochs=20, patience=0)
# Measured on the M1 Pro (MPS, training images/s): convnext_tiny 224px ~22, 128px ~78;
# efficientnet_b0 128px ~180; vit_small 128px ~200. Hence 128 px locally and 224 px only on a cloud GPU.
_FAST = dict(model_name="efficientnet_b0.ra_in1k", img_size=128, epochs=6, batch_size=128, lr=3e-4, head_lr=1e-3)

# Named runs. A notebook picks one by name; the name is also the folder of its checkpoints.
PRESETS = {
    # baseline CNN from scratch (notebook 03), ~1.5 min/epoch locally
    "cnn_noaug":  dict(_CNN, aug="none", label_smoothing=0.0),
    "cnn_shift":  dict(_CNN, aug="shift"),
    # main pretrained model, the key augmentation A/B (notebook 04), ~11 min/epoch locally
    "convnext_tiny_plain": dict(img_size=128, epochs=10, aug="plain"),
    "convnext_tiny_shift": dict(img_size=128, epochs=10, aug="shift"),
    "convnext_tiny_sim":   dict(img_size=128, epochs=10, aug="sim"),
    # fast ablations on fold 0 (notebook 04), ~5 min/epoch; each differs from abl_base in one setting
    "abl_base":        dict(_FAST),
    "abl_plain":       dict(_FAST, aug="plain"),
    "abl_sim":         dict(_FAST, aug="sim"),
    # resolution A/B against abl_sim (~1.5 h locally); fp32 because fp16 autocast diverged to NaN at 224 px on MPS
    "abl_sim_224":     dict(_FAST, aug="sim", img_size=224, amp=False),
    "abl_letterbox":   dict(_FAST, resize_mode="letterbox"),
    "abl_scale04":     dict(_FAST, crop_scale_min=0.4),
    "abl_scale08":     dict(_FAST, crop_scale_min=0.8),
    "abl_nolowres":    dict(_FAST, lowres_p=0.0),
    "abl_nosmooth":    dict(_FAST, label_smoothing=0.0),
    "abl_96":          dict(_FAST, img_size=96),
    "abl_160":         dict(_FAST, img_size=160),
    # extra architectures for the ensemble (notebooks 04/05)
    "vit_small_shift":  dict(model_name="vit_small_patch16_224.augreg_in21k_ft_in1k", img_size=128, epochs=10,
                             lr=1e-4, head_lr=5e-4),
    "effnet_b0_shift":  dict(_FAST, epochs=12),
    # 224 px versions - only practical on a CUDA GPU (Kaggle / Colab)
    "convnext_tiny_shift_224": dict(img_size=224, epochs=12, batch_size=96),
    "vit_small_shift_224":     dict(model_name="vit_small_patch16_224.augreg_in21k_ft_in1k", img_size=224,
                                    epochs=12, batch_size=96, lr=1e-4, head_lr=5e-4),
    # quick end-to-end check of the pipeline (a few minutes)
    "smoke_test":       dict(_FAST, epochs=1, train_limit=1500, img_size=64),
    "smoke_cnn":        dict(_CNN, epochs=2, train_limit=1500, img_size=64),
}


def make_config(run_name, **overrides):
    """Build a TrainConfig from a preset name plus optional overrides."""
    if run_name not in PRESETS:
        raise KeyError(f"Unknown run '{run_name}'. Available: {list(PRESETS)}")
    cfg = TrainConfig(run_name=run_name, **PRESETS[run_name])
    cfg = replace(cfg, **overrides)
    cfg.crop_ratio = tuple(cfg.crop_ratio)
    return cfg


def load_run_config(run_name, fold):
    """Read back the config that was saved next to a checkpoint."""
    d = json.loads((CKPT_DIR / run_name / f"fold{fold}" / "config.json").read_text())
    d["crop_ratio"] = tuple(d["crop_ratio"])
    return TrainConfig(**d)
