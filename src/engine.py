"""Training loop (resumable, logs every epoch), prediction with TTA, and speed benchmarks."""
import math
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.utils import ModelEmaV3
from tqdm.auto import tqdm

from .config import CKPT_DIR, EXPERIMENTS_CSV, LOG_DIR, get_device, load_run_config, load_shiftsim_params
from .data import FaceDataset, load_folds_df, load_test_df, make_loader
from .models import create_model, get_mean_std
from .transforms import ShiftSim, build_eval_transform, build_train_transform
from .utils import load_preds, now, pred_path, save_json, save_preds, seed_everything, upsert_csv

# The fp16 GradScaler skips the optimiser step when it sees inf/NaN gradients (normal in the first steps);
# the scheduler still steps, which triggers a harmless PyTorch warning.
warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step()` before `optimizer.step()`")


def autocast(device, enabled):
    enabled = enabled and device.type in ("cuda", "mps")
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)


# ---------------------------------------------------------------- optimiser and schedule
def build_optimizer(model, cfg):
    """AdamW with a higher learning rate for the classifier head and no weight decay on biases / norms."""
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    groups = {}
    for p in model.parameters():
        if p.requires_grad:
            groups.setdefault((id(p) in head_ids, p.ndim > 1), []).append(p)
    param_groups = [dict(params=ps, lr=cfg.head_lr if is_head else cfg.lr,
                         weight_decay=cfg.weight_decay if decay else 0.0)
                    for (is_head, decay), ps in groups.items()]
    return torch.optim.AdamW(param_groups)


def build_scheduler(optimizer, cfg, steps_per_epoch):
    """Linear warm-up for cfg.warmup_epochs, then cosine decay to 1% of the base rate (stepped per batch)."""
    warm, total = cfg.warmup_epochs * steps_per_epoch, cfg.epochs * steps_per_epoch

    def factor(step):
        if step < warm:
            return (step + 1) / warm
        progress = min(1.0, (step - warm) / max(1, total - warm))
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


# ---------------------------------------------------------------- one epoch / prediction
def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, scaler, ema, amp):
    model.train()
    loss_sum, correct, seen = torch.zeros((), device=device), torch.zeros((), device=device), 0
    for x, y in tqdm(loader, desc="train", leave=False, mininterval=30):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast(device, amp):
            logits = model(x)
            loss = criterion(logits, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if ema is not None:
            ema.update(model)
        loss_sum += loss.detach().float() * len(y)
        correct += (logits.argmax(1) == y).sum()
        seen += len(y)
    return loss_sum.item() / seen, correct.item() / seen


@torch.no_grad()
def predict_logits(model, loader, device, amp=True, desc="predict"):
    model.eval()
    out = []
    for x, _ in tqdm(loader, desc=desc, leave=False, mininterval=30):
        with autocast(device, amp):
            out.append(model(x.to(device)).float().cpu())
    return torch.cat(out).numpy()


def score(logits, labels):
    """Plain cross-entropy (no label smoothing) and accuracy."""
    loss = F.cross_entropy(torch.from_numpy(logits), torch.tensor(np.array(labels))).item()
    return loss, float((logits.argmax(1) == np.asarray(labels)).mean())


# ---------------------------------------------------------------- full training run
def fit(cfg, fold=0, force=False):
    """Train cfg on all folds except `fold`, validate on `fold` (clean and ShiftSim), save the best
    checkpoint, its predictions (val_clean, val_shift, test) and one row in results/experiments.csv.
    Resumes from the last checkpoint if interrupted; returns immediately if the run already finished."""
    run_dir = CKPT_DIR / cfg.run_name / f"fold{fold}"
    log_path = LOG_DIR / cfg.run_name / f"fold{fold}_history.csv"
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    if force and last_path.exists():
        last_path.unlink()
    if last_path.exists() and torch.load(last_path, map_location="cpu", weights_only=False)["finished"]:
        print(f"[{cfg.run_name} fold {fold}] already finished - loading saved history (force=True retrains)")
        return pd.read_csv(log_path)

    seed_everything(cfg.seed + fold)
    device = get_device()
    save_json(cfg.to_dict(), run_dir / "config.json")

    # ---- data
    df = load_folds_df()
    train_df, val_df = df[df["fold"] != fold], df[df["fold"] == fold].reset_index(drop=True)
    if cfg.drop_degenerate:
        train_df = train_df[~train_df["degenerate"]]
    if cfg.train_limit:
        train_df = train_df.sample(cfg.train_limit, random_state=cfg.seed)
        val_df = val_df.sample(min(len(val_df), cfg.train_limit // 2), random_state=cfg.seed).reset_index(drop=True)

    model = create_model(cfg).to(device)
    mean, std = get_mean_std(model)
    eval_tf = build_eval_transform(cfg, mean, std)
    shift_sim = ShiftSim(**load_shiftsim_params())
    train_loader = make_loader(FaceDataset(train_df, build_train_transform(cfg, mean, std)), cfg.batch_size,
                               shuffle=True, num_workers=cfg.num_workers, drop_last=True, persistent=True)
    val_loaders = {"val_clean": make_loader(FaceDataset(val_df, eval_tf), cfg.batch_size * 2,
                                            num_workers=cfg.num_workers, persistent=True),
                   "val_shift": make_loader(FaceDataset(val_df, eval_tf, shift_sim=shift_sim), cfg.batch_size * 2,
                                            num_workers=cfg.num_workers, persistent=True)}
    print(f"[{cfg.run_name} fold {fold}] device={device}  train={len(train_df)}  val={len(val_df)}  "
          f"steps/epoch={len(train_loader)}")

    # ---- optimisation
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler(device.type, enabled=cfg.amp and device.type in ("cuda", "mps"))
    ema = ModelEmaV3(model, decay=cfg.ema_decay, use_warmup=True) if cfg.ema_decay > 0 else None
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    history, best_acc, bad_epochs, start_epoch = [], -1.0, 0, 0
    if last_path.exists():  # resume
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        if ema is not None:
            ema.load_state_dict(ck["ema"])
        history, best_acc, bad_epochs, start_epoch = ck["history"], ck["best_acc"], ck["bad_epochs"], ck["epoch"] + 1
        print(f"resumed after epoch {ck['epoch'] + 1}")

    # ---- epochs
    finished = False
    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, criterion,
                                                device, scaler, ema, cfg.amp)
        eval_model = ema.module if ema is not None else model
        row = dict(epoch=epoch + 1, lr=optimizer.param_groups[0]["lr"], train_loss=train_loss, train_acc=train_acc)
        for name, loader in val_loaders.items():
            row[f"{name}_loss"], row[f"{name}_acc"] = score(predict_logits(eval_model, loader, device, cfg.amp, name),
                                                            val_df["label"].values)
        row["seconds"] = time.time() - t0
        if not math.isfinite(train_loss):
            raise RuntimeError(f"training loss is {train_loss} in epoch {epoch + 1} - run diverged, stopping")
        history.append(row)
        print(f"epoch {epoch + 1:2d}/{cfg.epochs}  train {train_loss:.4f}/{train_acc:.4f}  "
              f"clean {row['val_clean_loss']:.4f}/{row['val_clean_acc']:.4f}  "
              f"shift {row['val_shift_loss']:.4f}/{row['val_shift_acc']:.4f}  {row['seconds']:.0f}s")
        if epoch == start_epoch:
            print(f"  ETA for the remaining {cfg.epochs - epoch - 1} epochs: "
                  f"{row['seconds'] * (cfg.epochs - epoch - 1) / 60:.0f} min")

        if row["val_shift_acc"] > best_acc:  # model selection on val_shift
            best_acc, bad_epochs = row["val_shift_acc"], 0
            torch.save(eval_model.state_dict(), best_path)
        else:
            bad_epochs += 1
        finished = epoch + 1 == cfg.epochs or (cfg.patience and bad_epochs >= cfg.patience)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(log_path, index=False)
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                        ema=ema.state_dict() if ema is not None else None, history=history, epoch=epoch,
                        best_acc=best_acc, bad_epochs=bad_epochs, finished=False), last_path)
        if finished:
            if epoch + 1 < cfg.epochs:
                print(f"early stopping: no val_shift improvement for {cfg.patience} epochs")
            break

    # ---- predictions of the best checkpoint
    model.load_state_dict(torch.load(best_path, map_location="cpu"))
    for name, loader in val_loaders.items():
        save_preds(pred_path(cfg.run_name, fold, name), val_df["image_id"],
                   predict_logits(model, loader, device, cfg.amp, name), val_df["label"].values)
    test_df = load_test_df()
    test_loader = make_loader(FaceDataset(test_df, eval_tf), cfg.batch_size * 2, num_workers=cfg.num_workers)
    save_preds(pred_path(cfg.run_name, fold, "test"), test_df["image_id"],
               predict_logits(model, test_loader, device, cfg.amp, "test"))

    hist = pd.DataFrame(history)
    best = hist.loc[hist["val_shift_acc"].idxmax()]
    upsert_csv(EXPERIMENTS_CSV, dict(
        run=cfg.run_name, fold=fold, finished_at=now(), model=cfg.model_name, img_size=cfg.img_size, aug=cfg.aug,
        resize_mode=cfg.resize_mode, crop_scale_min=cfg.crop_scale_min, lowres_p=cfg.lowres_p,
        label_smoothing=cfg.label_smoothing, epochs_run=len(hist), best_epoch=int(best["epoch"]),
        train_acc=round(best["train_acc"], 4), val_clean_acc=round(best["val_clean_acc"], 4),
        val_shift_acc=round(best["val_shift_acc"], 4), val_clean_loss=round(best["val_clean_loss"], 4),
        val_shift_loss=round(best["val_shift_loss"], 4), minutes=round(hist["seconds"].sum() / 60, 1),
        n_train=len(train_df), n_val=len(val_df), notes=cfg.notes), keys=["run", "fold"])

    ck = torch.load(last_path, map_location="cpu", weights_only=False)
    ck["finished"] = True
    torch.save(ck, last_path)
    return hist


# ---------------------------------------------------------------- inference with saved checkpoints
def load_trained_model(run_name, fold, device=None):
    cfg = load_run_config(run_name, fold)
    model = create_model(cfg, pretrained=False)
    model.load_state_dict(torch.load(CKPT_DIR / run_name / f"fold{fold}" / "best.pt", map_location="cpu"))
    return model.to(device or get_device()).eval(), cfg


def predict_view(model, cfg, df, framing="full", hflip=False, shift_sim=None, device=None, num_workers=6):
    """Logits for one view of the images: a framing, optionally mirrored, optionally after ShiftSim."""
    device = device or get_device()
    ds = FaceDataset(df, build_eval_transform(cfg, *get_mean_std(model)), shift_sim=shift_sim,
                     framing=framing, hflip=hflip)
    return predict_logits(model, make_loader(ds, 256, num_workers=num_workers), device, cfg.amp,
                          f"{framing}{' mirrored' if hflip else ''}")


def predict_tta(model, cfg, df, framings, hflip=True, shift_sim=None, device=None):
    """Average logits over framings x (original, mirrored)."""
    views = [(f, flip) for f in framings for flip in ([False, True] if hflip else [False])]
    return np.mean([predict_view(model, cfg, df, f, flip, shift_sim, device) for f, flip in views], axis=0)


def get_preds(run_name, fold, split, tta=False, framings=None):
    """Load cached predictions; with tta=True they are computed (and cached) on first use."""
    path = pred_path(run_name, fold, split, tta)
    if path.exists():
        return load_preds(path)
    if not tta:
        raise FileNotFoundError(path)
    model, cfg = load_trained_model(run_name, fold)
    if split == "test":
        df = load_test_df()
    else:
        df = load_folds_df()
        df = df[df["fold"] == fold].reset_index(drop=True)
    shift_sim = ShiftSim(**load_shiftsim_params()) if split == "val_shift" else None
    logits = predict_tta(model, cfg, df, framings, shift_sim=shift_sim)
    save_preds(path, df["image_id"], logits, df["label"].values)
    return load_preds(path)


# ---------------------------------------------------------------- benchmarks
def benchmark_model(cfg, batch_size=64, steps=8, device=None):
    """Training and inference images/second on random input (excludes data loading)."""
    device = device or get_device()
    model = create_model(cfg, pretrained=False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=cfg.amp and device.type in ("cuda", "mps"))
    x = torch.randn(batch_size, 3, cfg.img_size, cfg.img_size, device=device)
    y = torch.randint(0, 8, (batch_size,), device=device)

    def sync():
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

    model.train()
    for i in range(steps + 2):  # first 2 steps are warm-up
        if i == 2:
            sync()
            t0 = time.time()
        with autocast(device, cfg.amp):
            loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    sync()
    train_ips = steps * batch_size / (time.time() - t0)
    model.eval()
    with torch.no_grad():
        for i in range(steps + 2):
            if i == 2:
                sync()
                t0 = time.time()
            with autocast(device, cfg.amp):
                model(x)
        sync()
    eval_ips = steps * batch_size / (time.time() - t0)
    return train_ips, eval_ips


def benchmark_loader(loader, n_batches=30):
    """Images/second the DataLoader can deliver (decode + augmentation only)."""
    it = iter(loader)
    next(it)  # start-up of the worker processes is not counted
    t0, n = time.time(), 0
    for _ in range(n_batches):
        x, _ = next(it)
        n += len(x)
    return n / (time.time() - t0)
