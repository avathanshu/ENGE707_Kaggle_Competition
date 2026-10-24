"""Preprocessing, training augmentation, the ShiftSim test-framing simulator and TTA framings."""
import math
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import v2

from .config import load_shiftsim_params

BICUBIC = v2.InterpolationMode.BICUBIC


# ---------------------------------------------------------------- geometry statistics
def geometry_stats(w, h):
    """Summary of image sizes in the form of the train-vs-test geometry table."""
    w, h = np.asarray(w), np.asarray(h)
    ar, short = w / h, np.minimum(w, h)
    pct = lambda m: 100 * float(np.mean(m))
    return pd.Series({
        "square %": pct(w == h),
        "odd dimension %": pct((w % 2 == 1) | (h % 2 == 1)),
        "both multiples of 10 %": pct((w % 10 == 0) & (h % 10 == 0)),
        "aspect < 0.80 %": pct(ar < 0.80),
        "aspect 0.80-0.995 %": pct((ar >= 0.80) & (ar < 0.995)),
        "aspect 0.995-1.03 %": pct((ar >= 0.995) & (ar <= 1.03)),
        "aspect 1.03-1.30 %": pct((ar > 1.03) & (ar <= 1.30)),
        "aspect 1.30-1.50 %": pct((ar > 1.30) & (ar <= 1.50)),
        "aspect > 1.50 %": pct(ar > 1.50),
        "short side p5": float(np.percentile(short, 5)),
        "short side p50": float(np.percentile(short, 50)),
        "short side p95": float(np.percentile(short, 95)),
    })


def geometry_distance(stats, target, size_weight=20):
    """L1 distance between two geometry_stats results: sum of the absolute differences of the percentage rows
    plus size_weight x the relative error of the shorter-side quantiles (a 10% size error costs 2 points)."""
    pct_rows = [r for r in target.index if r.endswith("%")]
    size_rows = [r for r in target.index if r.startswith("short side")]
    d = (stats[pct_rows] - target[pct_rows]).abs().sum()
    d += size_weight * ((stats[size_rows] - target[size_rows]).abs() / target[size_rows]).sum()
    return float(d)


# ---------------------------------------------------------------- ShiftSim
class ShiftSim:
    """Simulates the framing perturbation seen in the test set.

    With probability p_keep the image is returned unchanged. Otherwise a sub-box is cut out:
    area fraction uniform in [scale_min, 1], aspect ratio multiplied by a log-uniform factor in
    [1/aspect_max, aspect_max], random position. With probability p_round the crop is then
    resized to a 'round' size (each side shrunk by a factor in [shrink_min, 1] and rounded to a
    multiple of 10). Passing `key` (the image id) makes the result deterministic for that image,
    which is how the fixed val_shift set is built.
    """

    def __init__(self, p_keep=0.5, scale_min=0.6, aspect_max=1.33, p_round=0.25, shrink_min=0.5, seed=42):
        self.p_keep, self.scale_min, self.aspect_max = p_keep, scale_min, aspect_max
        self.p_round, self.shrink_min, self.seed = p_round, shrink_min, seed

    def plan(self, w, h, rng):
        """Return (box, output_size) for an image of size (w, h). box=None means unchanged."""
        if rng.random() < self.p_keep:
            return None, (w, h)
        area = rng.uniform(self.scale_min, 1.0)
        factor = math.exp(rng.uniform(-math.log(self.aspect_max), math.log(self.aspect_max)))
        cw = min(w, max(1, round(w * math.sqrt(area * factor))))
        ch = min(h, max(1, round(h * math.sqrt(area / factor))))
        x0 = int(rng.integers(0, w - cw + 1))
        y0 = int(rng.integers(0, h - ch + 1))
        out = (cw, ch)
        if rng.random() < self.p_round:
            shrink = rng.uniform(self.shrink_min, 1.0)
            out = (max(10, round(cw * shrink / 10) * 10), max(10, round(ch * shrink / 10) * 10))
        return (x0, y0, x0 + cw, y0 + ch), out

    def rng_for(self, key):
        return np.random.default_rng([self.seed, int(key)]) if key is not None else np.random.default_rng()

    def __call__(self, img, key=None):
        box, out = self.plan(*img.size, self.rng_for(key))
        if box is None:
            return img
        img = img.crop(box)
        if out != img.size:
            img = img.resize(out, Image.BICUBIC)
        return img

    def simulate_sizes(self, w, h, keys=None, rng=None):
        """Output sizes ShiftSim would produce, without touching any pixels (used for calibration)."""
        out_w, out_h = np.empty(len(w), int), np.empty(len(w), int)
        for i, (wi, hi) in enumerate(zip(w, h)):
            r = self.rng_for(keys[i]) if keys is not None else rng
            _, (out_w[i], out_h[i]) = self.plan(int(wi), int(hi), r)
        return out_w, out_h


def calibrate_shiftsim(w, h, target, n_trials=200, seed=42):
    """Random search for the ShiftSim parameters whose output geometry is closest to `target`."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_trials):
        params = dict(p_keep=rng.uniform(0.0, 0.7), scale_min=rng.uniform(0.4, 1.0),
                      aspect_max=rng.uniform(1.0, 1.6), p_round=rng.uniform(0.0, 0.7),
                      shrink_min=rng.uniform(0.3, 1.0))
        sim = ShiftSim(**params, seed=seed)
        sw, sh = sim.simulate_sizes(w, h, rng=np.random.default_rng(t))
        rows.append(dict(params, distance=geometry_distance(geometry_stats(sw, sh), target)))
    return pd.DataFrame(rows).sort_values("distance").reset_index(drop=True)


# ---------------------------------------------------------------- augmentation pieces
class RandomLowRes:
    """With probability p, shrink the image to a random small size (independent width/height,
    sometimes rounded to multiples of 10) and enlarge it back to its original size."""

    def __init__(self, p=0.3, min_side=12, max_side=64, p_round=0.3):
        self.p, self.min_side, self.max_side, self.p_round = p, min_side, max_side, p_round

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        short = random.randint(self.min_side, self.max_side)
        w, h = short, max(4, round(short * random.uniform(0.8, 1.25)))
        if random.random() < self.p_round:
            w, h = max(10, round(w / 10) * 10), max(10, round(h / 10) * 10)
        return img.resize((w, h), Image.BICUBIC).resize(img.size, Image.BICUBIC)


class PadToSquare:
    """Letterbox: paste the image in the centre of a square canvas filled with the ImageNet mean colour."""

    def __call__(self, img):
        w, h = img.size
        s = max(w, h)
        canvas = Image.new("RGB", (s, s), (124, 116, 104))
        canvas.paste(img, ((s - w) // 2, (s - h) // 2))
        return canvas


# ---------------------------------------------------------------- full pipelines
def _finish(mean, std):
    return [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)]


def build_eval_transform(cfg, mean, std):
    """Identical for validation and test: (optional letterbox) -> squash-resize to a square -> normalise."""
    s = cfg.img_size
    steps = [PadToSquare()] if cfg.resize_mode == "letterbox" else []
    steps += [v2.Resize((s, s), interpolation=BICUBIC, antialias=True)]
    return v2.Compose(steps + _finish(mean, std))


def build_train_transform(cfg, mean, std):
    """aug='none': eval transform only. 'plain': resize + flip + colour jitter.
    'shift': aggressive crop / aspect augmentation (RandomResizedCrop area 0.6-1, aspect 0.6-1.67) + degradations.
    'sim': random ShiftSim (calibrated test framing) + the same degradations, no RandomResizedCrop."""
    s = cfg.img_size
    steps = [PadToSquare()] if cfg.resize_mode == "letterbox" else []
    if cfg.aug == "none":
        steps += [v2.Resize((s, s), interpolation=BICUBIC, antialias=True)]
    elif cfg.aug == "plain":
        steps += [v2.Resize((s, s), interpolation=BICUBIC, antialias=True),
                  v2.RandomHorizontalFlip(),
                  v2.ColorJitter(0.2, 0.2, 0.2, 0.02)]
    elif cfg.aug == "shift":
        steps += [v2.RandomResizedCrop(s, scale=(cfg.crop_scale_min, 1.0), ratio=cfg.crop_ratio,
                                       interpolation=BICUBIC, antialias=True),
                  RandomLowRes(p=cfg.lowres_p),
                  v2.RandomHorizontalFlip(),
                  v2.RandomApply([v2.RandomRotation(15)], p=0.3),
                  v2.ColorJitter(0.2, 0.2, 0.2, 0.02),
                  v2.RandomGrayscale(p=0.05),
                  v2.RandomApply([v2.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.2),
                  v2.ToImage(),
                  v2.RandomApply([v2.JPEG((40, 95))], p=0.3)]
    elif cfg.aug == "sim":
        # Random ShiftSim at the strength calibrated to the test geometry (notebook 02), then squash-resize.
        # Unlike 'shift' it never shrinks a crop below ~65% area or changes the aspect by more than ~14%,
        # so a Mouth crop is not turned into a Nose-like image with a Mouth label.
        steps += [ShiftSim(**load_shiftsim_params()),
                  v2.Resize((s, s), interpolation=BICUBIC, antialias=True),
                  RandomLowRes(p=cfg.lowres_p),
                  v2.RandomHorizontalFlip(),
                  v2.ColorJitter(0.2, 0.2, 0.2, 0.02),
                  v2.RandomGrayscale(p=0.05),
                  v2.RandomApply([v2.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.2),
                  v2.ToImage(),
                  v2.RandomApply([v2.JPEG((40, 95))], p=0.3)]
    else:
        raise ValueError(f"unknown aug '{cfg.aug}'")
    return v2.Compose(steps + _finish(mean, std))


# ---------------------------------------------------------------- test-time augmentation framings
TTA_FRAMINGS = ["full", "center90", "narrow", "wide"]


def apply_framing(img, framing):
    """Re-frame an image before the eval transform: full image, 90% centre crop,
    aspect x0.85 (crop width) or aspect x1.15 (crop height)."""
    w, h = img.size
    if framing == "full":
        return img
    cw, ch = {"center90": (0.9 * w, 0.9 * h), "narrow": (0.85 * w, h), "wide": (w, h / 1.15)}[framing]
    cw, ch = max(1, round(cw)), max(1, round(ch))
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    return img.crop((x0, y0, x0 + cw, y0 + ch))
