"""Second-stage 'correction' model: CNN class probabilities (one or more runs) + image geometry -> class.

Trained on out-of-fold validation predictions only (each training image predicted by the fold model that did
not train on it), on both the clean and the ShiftSim view of every image, so it sees the original rigid
framing and the simulated test framing. Applied to each fold's test probabilities, then averaged.
No test labels are used.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .config import CACHE_DIR, load_shiftsim_params
from .data import load_folds_df, load_test_df
from .transforms import ShiftSim
from .utils import load_preds, pred_path, softmax


def geometry_features(w, h):
    w, h = np.asarray(w, float), np.asarray(h, float)
    return np.c_[np.log(w), np.log(h), np.log(w / h)]


def features(prob_list, w, h, use_geometry=True):
    """Log-probabilities of every run side by side, optionally followed by log width / height / aspect."""
    parts = [np.log(np.clip(p, 1e-6, 1)) for p in prob_list]
    if use_geometry:
        parts.append(geometry_features(w, h))
    return np.concatenate(parts, axis=1)


def oof_table(runs, folds, use_geometry=True):
    """One row per (image, view): the runs' OOF probabilities and the geometry the image had when fed."""
    meta = load_folds_df().set_index("image_id")
    sim = ShiftSim(**load_shiftsim_params())
    X, y, fold, split, probs = [], [], [], [], []
    for s in ["val_clean", "val_shift"]:
        for f in folds:
            loaded = [load_preds(pred_path(r, f, s)) for r in runs]
            ids, labels = loaded[0][0], loaded[0][2]
            assert all((l[0] == ids).all() for l in loaded)
            w, h = meta.loc[ids, "w"].values, meta.loc[ids, "h"].values
            if s == "val_shift":
                w, h = sim.simulate_sizes(w, h, keys=ids.astype(int))
            p = [softmax(l[1]) for l in loaded]
            X.append(features(p, w, h, use_geometry))
            probs.append(np.mean(p, axis=0))
            y.append(labels)
            fold.append(np.full(len(ids), f))
            split.append(np.full(len(ids), s))
    return tuple(np.concatenate(a) for a in (X, y, fold, split, probs))


def make_model(seed=42):
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=seed)


def cross_validate(X, y, fold, split, base_probs):
    """Leave-one-fold-out accuracy of the corrector vs the plain averaged CNN probabilities, per view."""
    pred = np.empty(len(y), int)
    for f in np.unique(fold):
        m = make_model().fit(X[fold != f], y[fold != f])
        pred[fold == f] = m.predict(X[fold == f])
    base = base_probs.argmax(1)
    out = {}
    for s in ["val_clean", "val_shift"]:
        k = split == s
        out[s] = {"CNN average": (base[k] == y[k]).mean(), "corrected": (pred[k] == y[k]).mean()}
    return pd.DataFrame(out), pred


def test_inputs(runs, folds):
    """Per fold: list of each run's test probabilities; plus the test table and the test image sizes."""
    test = load_test_df()
    meta = pd.read_csv(CACHE_DIR / "test_meta.csv", dtype={"image_id": str}).set_index("image_id")
    w, h = meta.loc[test["image_id"], "w"].values, meta.loc[test["image_id"], "h"].values
    per_fold = []
    for f in folds:
        loaded = [load_preds(pred_path(r, f, "test")) for r in runs]
        assert all((l[0] == test["image_id"].values).all() for l in loaded)
        per_fold.append([softmax(l[1]) for l in loaded])
    return test, w, h, per_fold


def predict_test(model, runs, folds, use_geometry=True):
    test, w, h, per_fold = test_inputs(runs, folds)
    return test, np.mean([model.predict_proba(features(p, w, h, use_geometry)) for p in per_fold], axis=0)
