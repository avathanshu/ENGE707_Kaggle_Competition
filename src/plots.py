"""Plotting helpers shared by the notebooks. Every figure is also saved to outputs/figures/."""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image

from .config import FIG_DIR


def savefig(name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_DIR / f"{name}.png", dpi=130, bbox_inches="tight")


def show_grid(images, titles=None, ncols=12, size=1.4, suptitle=None, name=None, fontsize=7):
    """Show PIL images or file paths in a grid. Each image keeps its own aspect ratio inside its cell,
    so the framing (square / portrait / landscape) stays visible."""
    n = len(images)
    nrows = max(1, int(np.ceil(n / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * size, nrows * (size + 0.25)), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, im in enumerate(images):
        if not isinstance(im, Image.Image):
            im = Image.open(im).convert("RGB")
        ax = axes.flat[i]
        ax.imshow(im)
        ax.set_title(titles[i] if titles is not None else f"{im.width}x{im.height}", fontsize=fontsize)
    if suptitle:
        fig.suptitle(suptitle, y=1.0)
    plt.tight_layout()
    if name:
        savefig(name)
    plt.show()


def denormalize(x, mean, std):
    """Normalised CHW tensor -> HWC numpy image in [0, 1] for display."""
    img = x.permute(1, 2, 0).numpy() * np.array(std) + np.array(mean)
    return np.clip(img, 0, 1)


def plot_history(hist, title="", name=None):
    """Loss and accuracy per epoch for train, val_clean and val_shift."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    for split, style in [("train", "-"), ("val_clean", "--"), ("val_shift", "-.")]:
        axes[0].plot(hist["epoch"], hist[f"{split}_loss"], style, marker="o", label=split)
        axes[1].plot(hist["epoch"], hist[f"{split}_acc"], style, marker="o", label=split)
    axes[2].plot(hist["epoch"], hist["lr"], marker="o")
    for ax, t in zip(axes, ["loss (train includes label smoothing)", "accuracy", "learning rate (end of epoch)"]):
        ax.set_title(t)
        ax.set_xlabel("epoch")
    axes[0].legend()
    axes[1].legend()
    fig.suptitle(title)
    plt.tight_layout()
    if name:
        savefig(name)
    plt.show()


def plot_confusion(cm, classes, title="", normalize=True, ax=None):
    """Confusion matrix heatmap (rows = true class). normalize=True shows row percentages."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    data = cm / cm.sum(axis=1, keepdims=True) * 100 if normalize else cm
    sns.heatmap(data, annot=True, fmt=".1f" if normalize else "d", cmap="Blues", cbar=False,
                xticklabels=classes, yticklabels=classes, ax=ax, annot_kws={"fontsize": 8})
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title + (" (row %)" if normalize else " (counts)"))
    return ax
