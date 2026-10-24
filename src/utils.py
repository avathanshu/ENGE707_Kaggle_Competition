"""Small helpers: seeding, CSV logging, prediction files, submissions and ensembling."""
import itertools
import json
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from .config import CLASSES, PRED_DIR, SAMPLE_SUB_CSV, SUB_DIR, SUBMISSIONS_CSV


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def upsert_csv(path, row, keys):
    """Append `row` to a CSV, replacing an existing row with the same values in `keys`."""
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path, dtype=str)
        same = np.ones(len(old), bool)
        for k in keys:
            same &= old[k].astype(str) == str(row[k])
        new = pd.concat([old[~same], new.astype(str)], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)


def connected_groups(df, cols):
    """Group id per row: rows sharing a value in any of `cols` end up in the same group (union-find)."""
    parent = list(range(len(df)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for col in cols:
        first = {}
        for i, v in enumerate(df[col].values):
            if v in first:
                parent[find(i)] = find(first[v])
            else:
                first[v] = i
    roots = [find(i) for i in range(len(df))]
    return pd.factorize(pd.Series(roots))[0]


# ---------------------------------------------------------------- prediction files
def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def pred_path(run_name, fold, split, tta=False):
    """split: 'val_clean', 'val_shift' or 'test'."""
    return PRED_DIR / run_name / f"fold{fold}_{split}{'_tta' if tta else ''}.npz"


def save_preds(path, ids, logits, labels=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=np.asarray(ids, dtype=str), logits=logits.astype(np.float32),
             labels=np.asarray(labels if labels is not None else [-1] * len(ids)))


def load_preds(path):
    d = np.load(path)
    return d["ids"], d["logits"], d["labels"]


def accuracy(logits_or_probs, labels):
    return float((np.asarray(logits_or_probs).argmax(1) == np.asarray(labels)).mean())


def log_loss(probs, labels):
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-7, 1)).mean())


# ---------------------------------------------------------------- submissions
def validate_submission(sub):
    """Raise if the submission does not match sample_submission.csv exactly in ids and label set."""
    sample = pd.read_csv(SAMPLE_SUB_CSV)
    assert list(sub.columns) == ["image_id", "class"], sub.columns
    assert len(sub) == len(sample) == 29443, len(sub)
    assert (sub["image_id"].values == sample["image_id"].values).all(), "ids differ from sample_submission"
    assert sub["class"].isin(CLASSES).all(), "unknown class name"
    assert sub["class"].notna().all()
    return True


def write_submission(test_df, probs, name, runs, val_clean=None, val_shift=None, notes=""):
    """Write submissions/<name>.csv from test probabilities (rows in test_df order) and log it."""
    sub = pd.DataFrame({"image_id": test_df["sub_id"].values, "class": np.array(CLASSES)[probs.argmax(1)]})
    validate_submission(sub)
    path = SUB_DIR / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)
    upsert_csv(SUBMISSIONS_CSV, dict(file=path.name, created=now(), runs=runs,
                                     val_clean=val_clean, val_shift=val_shift,
                                     lb_public="", lb_private="", notes=notes), keys=["file"])
    print(f"wrote {path}  ({len(sub)} rows, validated)")
    print(sub["class"].value_counts(normalize=True).round(3).to_dict())
    return path


# ---------------------------------------------------------------- ensembling
def fit_ensemble_weights(prob_list, labels, step=0.1):
    """Grid search over weights that sum to 1; maximise accuracy, ties broken by log-loss."""
    n = len(prob_list)
    if n == 1:
        return np.array([1.0]), accuracy(prob_list[0], labels)
    grid = np.round(np.arange(0, 1 + 1e-9, step), 6)
    best = None
    for w in itertools.product(grid, repeat=n - 1):
        if sum(w) > 1 + 1e-9:
            continue
        weights = np.array(list(w) + [1 - sum(w)])
        p = sum(wi * pi for wi, pi in zip(weights, prob_list))
        score = (accuracy(p, labels), -log_loss(p, labels))
        if best is None or score > best[0]:
            best = (score, weights)
    return best[1], best[0][0]
