#!/usr/bin/env python3
"""
DDP Training Script (single file)
- Loads your NPZ dataset: hackathon_dataset_npz_final/{train,val}/*.npz
- Input: STEM (4ch) + metadata maps (default: thickness + sin/cos rotations)
- Target: potential (1ch)
- Model: SwinUNETR (if available) or fallback UNet
- Mixed precision + grad clip
- Saves: best_model.pt, last_model.pt, loss_curve.png, metrics.csv, sample_pred_epoch_X.png
- Multi-GPU: use torchrun (recommended) or run single GPU normally

Example (single GPU):
  python train_ddp_microscopy.py --data_dir hackathon_dataset_npz_final --epochs 50

Example (2 GPUs on one node):
  torchrun --standalone --nproc_per_node=2 train_ddp_microscopy.py --batch_size 16

If SLURM:
  srun --gres=gpu:2 torchrun --standalone --nproc_per_node=2 train_ddp_microscopy.py ...
"""

import os
import glob
import math
import csv
import time
import argparse
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler

import matplotlib.pyplot as plt

# ----------------------------
# Optional MONAI imports
# ----------------------------
MONAI_OK = True
try:
    from monai.networks.nets import SwinUNETR, UNet
    from monai.transforms import Compose, RandGaussianNoise, RandGaussianSmooth, RandScaleIntensity
except Exception:
    MONAI_OK = False


# =============================================================================
# Utils: DDP / rank helpers
# =============================================================================
def ddp_is_enabled() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if ddp_is_enabled() else 0

def get_world_size() -> int:
    return dist.get_world_size() if ddp_is_enabled() else 1

def is_main_process() -> bool:
    return get_rank() == 0

def setup_ddp():
    """Initialize process group if launched with torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

def cleanup_ddp():
    if ddp_is_enabled():
        dist.destroy_process_group()

def all_reduce_mean(t: torch.Tensor) -> torch.Tensor:
    """All-reduce and average a scalar tensor across ranks."""
    if not ddp_is_enabled():
        return t
    t = t.clone()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= get_world_size()
    return t


# =============================================================================
# Config
# =============================================================================
@dataclass
class TrainCfg:
    data_dir: str
    out_dir: str
    epochs: int
    batch_size: int              # per-process batch size (DDP). total batch = batch_size * world_size
    lr: float
    weight_decay: float
    num_workers: int
    img_size: Tuple[int, int]
    use_amp: bool
    grad_clip: float
    val_interval: int
    save_samples: bool
    model: str                   # "swin" or "unet"
    use_checkpoint: bool         # for SwinUNETR only (slower but memory saving)
    metadata_mode: str           # "raw4" or "sincos7"
    aug: bool


# =============================================================================
# Loss: Pixel + FFT + Gradient (stable FFT)
# =============================================================================
class PhysicsLoss(nn.Module):
    def __init__(self, w_pixel=1.0, w_fft=0.1, w_grad=0.1, eps=1e-8):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.w_pixel = w_pixel
        self.w_fft = w_fft
        self.w_grad = w_grad
        self.eps = eps

    @staticmethod
    def gradients(img: torch.Tensor):
        # img: (B, C, H, W)
        dy = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
        dx = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
        return dy, dx

    def forward(self, pred, target, return_parts=False):
        # 1. Pixel Loss
        loss_pixel = self.l1(pred, target)

        # 2. FFT Loss (Log-space for stability)
        fft_pred = torch.fft.rfft2(pred + 1e-8)
        fft_targ = torch.fft.rfft2(target + 1e-8)
        loss_fft = self.l1(torch.abs(fft_pred), torch.abs(fft_targ))

        # 3. Gradient Loss
        dy_pred, dx_pred = self.gradients(pred)
        dy_tgt, dx_tgt = self.gradients(target)
        loss_grad = self.l1(dy_pred, dy_tgt) + self.l1(dx_pred, dx_tgt)

        # Total Weighted Sum
        total = loss_pixel + 0.1 * loss_fft + 0.1 * loss_grad
        
        # Optional: Return breakdown for debugging
        if return_parts:
            return total, loss_pixel.detach(), loss_fft.detach(), loss_grad.detach()
            
        return total


# =============================================================================
# Dataset
# =============================================================================
class MicroscopyDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str,
        img_size: Tuple[int, int],
        augment: bool,
        metadata_mode: str = "sincos7",
    ):
        self.files = sorted(glob.glob(os.path.join(root_dir, split, "*.npz")))
        self.augment = augment
        self.img_size = img_size
        self.metadata_mode = metadata_mode

        self.transforms = None
        if self.augment:
            if MONAI_OK:
                self.transforms = Compose([
                    RandGaussianNoise(prob=0.5, mean=0.0, std=0.035),
                    RandGaussianSmooth(prob=0.2, sigma_x=(0.2, 0.6), sigma_y=(0.2, 0.6)),
                    RandScaleIntensity(factors=0.1, prob=0.5),
                ])
            else:
                # If MONAI not installed, we just skip augmentation.
                self.transforms = None

    def __len__(self):
        return len(self.files)

    @staticmethod
    def _safe_minmax(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        mn = float(x.min())
        mx = float(x.max())
        return (x - mn) / (mx - mn + 1e-8)

    def _make_metadata_vec(self, data) -> np.ndarray:
        thick = np.array(data["thickness_nm"], dtype=np.float32).reshape(1)  # nm
        thick = thick / 100.0  # scale 0..~1

        rot = np.array(data["rotation"], dtype=np.float32).reshape(-1)  # could be len 3
        # If saved as [rz, rx, ry] or [rz, rx, ry] etc, we accept as-is.
        # Your generator used rotation=[rz, rx, ry]. We'll treat them as degrees.
        if self.metadata_mode == "raw4":
            rot01 = rot / 360.0
            vec = np.concatenate([thick, rot01], axis=0)  # 1 + 3 = 4
            return vec.astype(np.float32)

        # sin/cos encoding (periodic-safe)
        # theta in radians:
        th = rot * (math.pi / 180.0)
        sincos = np.concatenate([np.sin(th), np.cos(th)], axis=0)  # 3 + 3 = 6
        vec = np.concatenate([thick, sincos], axis=0)              # 1 + 6 = 7
        return vec.astype(np.float32)

    def __getitem__(self, idx):
        fpath = self.files[idx]
        try:
            with np.load(fpath, allow_pickle=False) as data:
                img = data["input"].astype(np.float32)  # (4, H, W)
                tgt = data["potential"].astype(np.float32)

                SCALING_FACTOR = 600000.0 

                # Scale down: 3000 Volts becomes 1.0, 1500 Volts becomes 0.5
                tgt = tgt / SCALING_FACTOR

                tgt = np.nan_to_num(tgt, nan=0.0, posinf=1.0, neginf=0.0)

                # Add channel dimension
                if tgt.ndim == 2:
                    tgt = tgt[np.newaxis, ...]

                # Basic sanity (no resizing here; assumes your generator is fixed 256x256)
                if img.shape[1:] != self.img_size or tgt.shape[1:] != self.img_size:
                    # If mismatch, you can add resizing, but you said you don't want changing data.
                    raise ValueError(f"Size mismatch img={img.shape}, tgt={tgt.shape}, expected={self.img_size}")

                # Normalize input only (keep target in physical-ish units)
                img = self._safe_minmax(img)

                # Metadata -> spatial maps
                meta_vec = self._make_metadata_vec(data)  # (K,)
                k = meta_vec.shape[0]
                meta_maps = np.ones((k, img.shape[1], img.shape[2]), dtype=np.float32)
                for i in range(k):
                    meta_maps[i, :, :] = meta_vec[i]

                x = np.concatenate([img, meta_maps], axis=0)  # (4+K, H, W)

            x_t = torch.from_numpy(x)        # float32
            y_t = torch.from_numpy(tgt)      # float32

            # Physics-safe aug: only on the first 4 visual channels
            if self.augment and self.transforms is not None:
                vis = x_t[:4, ...]
                meta = x_t[4:, ...]
                vis = self.transforms(vis)
                x_t = torch.cat([vis, meta], dim=0)

            return x_t, y_t

        except Exception as e:
            # Return zeros rather than crashing
            # NOTE: This can hide dataset issues; use logs to detect frequent corruption.
            k = 7 if self.metadata_mode == "sincos7" else 4
            x = torch.zeros((4 + k, *self.img_size), dtype=torch.float32)
            y = torch.zeros((1, *self.img_size), dtype=torch.float32)
            if is_main_process():
                print(f"⚠️ Error loading {os.path.basename(fpath)}: {e}")
            return x, y


# =============================================================================
# Model builders
# =============================================================================
def build_model(model_name: str, in_ch: int, out_ch: int, img_size: Tuple[int, int], use_checkpoint: bool):
    if not MONAI_OK:
        raise RuntimeError("MONAI is not available in this environment. Install monai to use this script.")

    model_name = model_name.lower().strip()
    if model_name == "swin":
        # SwinUNETR in 2D depends on MONAI version; if it errors, switch to unet.
        return SwinUNETR(
            spatial_dims=2,
            in_channels=in_ch,
            out_channels=out_ch,
            feature_size=48,
            use_checkpoint=use_checkpoint
        )
    elif model_name == "unet":
        # Fast & reliable baseline
        return UNet(
            spatial_dims=2,
            in_channels=in_ch,
            out_channels=out_ch,
            channels=(32, 64, 128, 256),
            strides=(2, 2, 2),
            num_res_units=2,
        )
    else:
        raise ValueError(f"Unknown --model {model_name} (use 'swin' or 'unet').")


# =============================================================================
# Monitoring: plot learning curve
# =============================================================================
def plot_learning_curve(out_path: str, epochs, train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Physics Loss")
    plt.title("Convergence / Learning Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_val_sample(model, loader, device, out_path: str):
    model.eval()
    with torch.no_grad():
        x, y = next(iter(loader))
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with autocast(enabled=True):
            p = model(x)

        # show first sample
        x0 = x[0, 0].detach().float().cpu().numpy()
        y0 = y[0, 0].detach().float().cpu().numpy()
        p0 = p[0, 0].detach().float().cpu().numpy()

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1); plt.imshow(x0, cmap="gray");    plt.title("Input (STEM ch0)"); plt.axis("off")
        plt.subplot(1, 3, 2); plt.imshow(p0, cmap="inferno"); plt.title("Prediction");      plt.axis("off")
        plt.subplot(1, 3, 3); plt.imshow(y0, cmap="inferno"); plt.title("Ground Truth");    plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()


# =============================================================================
# Training / Validation loops
# =============================================================================
def run_one_epoch_train(model, loader, criterion, optimizer, scaler, device, use_amp, grad_clip):
    model.train()
    total = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            p = model(x)
            loss = criterion(p, y)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total += float(loss.detach().item())
        n += 1

    avg = torch.tensor(total / max(n, 1), device=device, dtype=torch.float32)
    avg = all_reduce_mean(avg)
    return float(avg.item())


@torch.no_grad()
def run_one_epoch_val(model, loader, criterion, device, use_amp):
    model.eval()
    total = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            p = model(x)
            loss = criterion(p, y)
        total += float(loss.detach().item())
        n += 1

    avg = torch.tensor(total / max(n, 1), device=device, dtype=torch.float32)
    avg = all_reduce_mean(avg)
    return float(avg.item())


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="hackathon_dataset_npz_final")
    parser.add_argument("--out_dir", type=str, default="checkpoints_ddp")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=16)      # per process
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--img_size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--save_samples", action="store_true")
    parser.add_argument("--model", type=str, default="swin", choices=["swin", "unet"])
    parser.add_argument("--use_checkpoint", action="store_true", help="Swin checkpointing (slower, less VRAM).")
    parser.add_argument("--metadata_mode", type=str, default="sincos7", choices=["raw4", "sincos7"])
    parser.add_argument("--no_aug", action="store_true")
    args = parser.parse_args()

    setup_ddp()

    cfg = TrainCfg(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        img_size=(args.img_size[0], args.img_size[1]),
        use_amp=(not args.no_amp),
        grad_clip=args.grad_clip,
        val_interval=args.val_interval,
        save_samples=args.save_samples,
        model=args.model,
        use_checkpoint=args.use_checkpoint,
        metadata_mode=args.metadata_mode,
        aug=(not args.no_aug),
    )

    if is_main_process():
        os.makedirs(cfg.out_dir, exist_ok=True)
        print(f"✅ Output dir: {cfg.out_dir}")
        print(f"✅ World size: {get_world_size()} | Batch per rank: {cfg.batch_size} | Total batch: {cfg.batch_size * get_world_size()}")
        print(f"✅ Model: {cfg.model} | Metadata: {cfg.metadata_mode} | AMP: {cfg.use_amp} | Aug: {cfg.aug}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. This script expects GPU training.")

    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))

    # Small speed boost on A100/H100
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # Dataset + samplers
    train_ds = MicroscopyDataset(cfg.data_dir, "train", cfg.img_size, augment=cfg.aug, metadata_mode=cfg.metadata_mode)
    val_ds   = MicroscopyDataset(cfg.data_dir, "val",   cfg.img_size, augment=False, metadata_mode=cfg.metadata_mode)

    if ddp_is_enabled():
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=False)
        val_sampler   = DistributedSampler(val_ds, shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler = None

    # persistent_workers speeds up loader after first epoch
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=4 if cfg.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=max(1, cfg.num_workers // 2),
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    # Model
    k = 7 if cfg.metadata_mode == "sincos7" else 4
    in_ch = 4 + k
    out_ch = 1

    model = build_model(cfg.model, in_ch=in_ch, out_ch=out_ch, img_size=cfg.img_size, use_checkpoint=cfg.use_checkpoint).to(device)

    if ddp_is_enabled():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index],
            output_device=device.index,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    criterion = PhysicsLoss(w_pixel=1.0, w_fft=0.1, w_grad=0.1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = GradScaler(enabled=cfg.use_amp)

    # Tracking
    best_val = float("inf")
    train_hist = []
    val_hist = []
    epoch_hist = []

    # Metrics CSV
    metrics_csv = os.path.join(cfg.out_dir, "metrics.csv")
    if is_main_process():
        with open(metrics_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "val_loss", "epoch_time_sec"])

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        if ddp_is_enabled():
            train_sampler.set_epoch(epoch)

        train_loss = run_one_epoch_train(model, train_loader, criterion, optimizer, scaler, device, cfg.use_amp, cfg.grad_clip)

        val_loss = None
        if (epoch % cfg.val_interval) == 0:
            val_loss = run_one_epoch_val(model, val_loader, criterion, device, cfg.use_amp)

        dt = time.time() - t0

        # Only main writes files/plots
        if is_main_process():
            epoch_hist.append(epoch)
            train_hist.append(train_loss)
            if val_loss is None:
                # keep same length for plotting; just mirror train or append NaN
                val_hist.append(float("nan"))
            else:
                val_hist.append(val_loss)

            # Print
            if val_loss is None:
                print(f"Ep {epoch:03d}/{cfg.epochs} | Train {train_loss:.5f} | time {dt:.1f}s")
            else:
                print(f"Ep {epoch:03d}/{cfg.epochs} | Train {train_loss:.5f} | Val {val_loss:.5f} | time {dt:.1f}s")

            # Save metrics
            with open(metrics_csv, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([epoch, train_loss, val_loss if val_loss is not None else "", f"{dt:.3f}"])

            # Plot learning curve
            plot_learning_curve(
                out_path=os.path.join(cfg.out_dir, "loss_curve.png"),
                epochs=epoch_hist,
                train_losses=train_hist,
                val_losses=val_hist,
            )

            # Save sample predictions
            if cfg.save_samples and (val_loss is not None):
                sample_path = os.path.join(cfg.out_dir, f"sample_pred_epoch_{epoch:03d}.png")
                # unwrap for DDP
                m = model.module if hasattr(model, "module") else model
                save_val_sample(m, val_loader, device, sample_path)

            # Save checkpoints
            m = model.module if hasattr(model, "module") else model
            torch.save(m.state_dict(), os.path.join(cfg.out_dir, "last_model.pt"))

            if val_loss is not None and val_loss < best_val:
                best_val = val_loss
                torch.save(m.state_dict(), os.path.join(cfg.out_dir, "best_model.pt"))
                print("   💾 New best_model.pt saved")

    if is_main_process():
        print(f"🏁 Done. Best Val Loss = {best_val:.6f}")
        print(f"Artifacts in: {cfg.out_dir}")

    cleanup_ddp()


if __name__ == "__main__":
    main()
