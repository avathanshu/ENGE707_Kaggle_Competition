"""Loading the CSVs, scanning image files, the PyTorch Dataset and DataLoader helper."""
import hashlib
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .config import (CACHE_DIR, CLASS_TO_IDX, FOLDS_CSV, LABELS_CSV, SAMPLE_SUB_CSV, TEST_DIR, TRAIN_DIR)
from .transforms import apply_framing


# ---------------------------------------------------------------- tables
def load_train_df():
    """train_labels.csv with zero-padded string ids, file path and integer label."""
    df = pd.read_csv(LABELS_CSV, dtype={"image_id": str})
    df["path"] = [str(TRAIN_DIR / f"{i}.jpg") for i in df["image_id"]]
    df["label"] = df["class"].map(CLASS_TO_IDX)
    return df


def load_test_df():
    """Test ids in sample_submission order. `sub_id` is the id exactly as Kaggle expects it."""
    sub = pd.read_csv(SAMPLE_SUB_CSV)
    df = pd.DataFrame({"sub_id": sub["image_id"]})
    df["image_id"] = df["sub_id"].astype(int).map(lambda i: f"{i:05d}")
    df["path"] = [str(TEST_DIR / f"{i}.jpg") for i in df["image_id"]]
    df["label"] = -1
    return df


def load_folds_df():
    """Training table merged with data/folds.csv (fold, group, w, h, degenerate). Created in notebook 02."""
    if not FOLDS_CSV.exists():
        raise FileNotFoundError("data/folds.csv not found - run notebook 02 first.")
    folds = pd.read_csv(FOLDS_CSV, dtype={"image_id": str})
    return load_train_df().merge(folds.drop(columns=["class"]), on="image_id", how="left")


# ---------------------------------------------------------------- file scan (EDA)
def scan_image(path):
    """Header facts, hashes and simple colour / sharpness statistics of one image file."""
    raw = open(path, "rb").read()
    out = {"path": path, "bytes": len(raw), "md5": hashlib.md5(raw).hexdigest()}
    try:
        img = Image.open(path)
        out.update(format=img.format, mode=img.mode, w=img.width, h=img.height,
                   exif="exif" in img.info, progressive=bool(img.info.get("progressive", False)),
                   qtables=hashlib.md5(repr(getattr(img, "quantization", None)).encode()).hexdigest()[:8])
        rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
        out["ok"] = True
    except Exception as e:  # unreadable file
        out.update(ok=False, error=str(e))
        return out
    gray = rgb.mean(axis=2)
    hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
    out.update(mean_r=rgb[..., 0].mean(), mean_g=rgb[..., 1].mean(), mean_b=rgb[..., 2].mean(),
               brightness=gray.mean(), contrast=gray.std(), saturation=hsv[..., 1].mean(),
               sharpness=cv2.Laplacian(gray, cv2.CV_32F).var(),
               greyscale=bool(np.abs(rgb[..., 0] - rgb[..., 1]).mean() + np.abs(rgb[..., 1] - rgb[..., 2]).mean() < 2))
    # 64-bit difference hash: near-duplicate detection
    small = np.asarray(Image.fromarray(gray.astype(np.uint8)).resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    out["dhash"] = "%016x" % int("".join("1" if b else "0" for b in bits), 2)
    return out


def get_image_meta(split, n_jobs=8):
    """Scan every image of 'train' or 'test' once (multiprocessing) and cache the table."""
    cache = CACHE_DIR / f"{split}_meta.csv"
    if cache.exists():
        return pd.read_csv(cache, dtype={"image_id": str, "dhash": str})
    df = load_train_df() if split == "train" else load_test_df()
    with Pool(n_jobs) as pool:
        rows = list(tqdm(pool.imap(scan_image, df["path"], chunksize=256), total=len(df), desc=f"scan {split}", mininterval=10))
    meta = df.merge(pd.DataFrame(rows), on="path")
    cache.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(cache, index=False)
    return meta


# ---------------------------------------------------------------- Dataset / DataLoader
class FaceDataset(Dataset):
    """Returns (image tensor, label). Optional steps before the transform, in order:
    ShiftSim (deterministic per image id), a TTA framing, and a horizontal flip."""

    def __init__(self, df, transform, shift_sim=None, framing="full", hflip=False):
        self.paths = df["path"].tolist()
        self.ids = df["image_id"].astype(int).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.transform, self.shift_sim, self.framing, self.hflip = transform, shift_sim, framing, hflip

    def __len__(self):
        return len(self.paths)

    def load(self, i):
        """The PIL image exactly as it enters the transform (useful for plotting)."""
        img = Image.open(self.paths[i]).convert("RGB")
        if self.shift_sim is not None:
            img = self.shift_sim(img, key=self.ids[i])
        img = apply_framing(img, self.framing)
        if self.hflip:
            img = ImageOps.mirror(img)
        return img

    def __getitem__(self, i):
        return self.transform(self.load(i)), self.labels[i]


def make_loader(dataset, batch_size, shuffle=False, num_workers=6, drop_last=False, persistent=False):
    """persistent=True keeps the worker processes alive between epochs (used for the training loop)."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                      num_workers=num_workers, persistent_workers=persistent and num_workers > 0,
                      pin_memory=torch.cuda.is_available(), prefetch_factor=4 if num_workers > 0 else None)
