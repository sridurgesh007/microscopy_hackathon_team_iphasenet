#!/usr/bin/env python3
import os, glob, math, csv, argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ----------------------------
#  MONAI imports
# ----------------------------
MONAI_OK = True
try:
    from monai.networks.nets import SwinUNETR, UNet
except Exception:
    MONAI_OK = False


# ----------------------------
# Helpers
# ----------------------------
def safe_minmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn = float(x.min())
    mx = float(x.max())
    return (x - mn) / (mx - mn + 1e-8)

def make_metadata_vec(data, mode: str):
    # thickness_nm scalar
    thick = np.array(data["thickness_nm"], dtype=np.float32).reshape(1) / 100.0

    rot = np.array(data["rotation"], dtype=np.float32).reshape(-1)  # expected len=3, degrees

    if mode == "raw4":
        rot01 = rot / 360.0
        return np.concatenate([thick, rot01], axis=0).astype(np.float32)  # (4,)

    # sin/cos encoding (periodic-safe)
    th = rot * (math.pi / 180.0)
    sincos = np.concatenate([np.sin(th), np.cos(th)], axis=0)  # (6,)
    return np.concatenate([thick, sincos], axis=0).astype(np.float32)    # (7,)

def make_metadata_maps(meta_vec: np.ndarray, H: int, W: int) -> np.ndarray:
    k = meta_vec.shape[0]
    maps = np.ones((k, H, W), dtype=np.float32)
    for i in range(k):
        maps[i, :, :] = meta_vec[i]
    return maps

def build_model(model_name: str, in_ch: int, out_ch: int, use_checkpoint: bool):
    if not MONAI_OK:
        raise RuntimeError("MONAI not found. Install monai to use SwinUNETR/UNet here.")

    model_name = model_name.lower().strip()
    if model_name == "swin":
        return SwinUNETR(
            spatial_dims=2,
            in_channels=in_ch,
            out_channels=out_ch,
            feature_size=48,
            use_checkpoint=use_checkpoint
        )
    elif model_name == "unet":
        return UNet(
            spatial_dims=2,
            in_channels=in_ch,
            out_channels=out_ch,
            channels=(32, 64, 128, 256),
            strides=(2, 2, 2),
            num_res_units=2,
        )
    else:
        raise ValueError("model_name must be 'swin' or 'unet'")

def load_weights(model, ckpt_path: str, device):
    sd = torch.load(ckpt_path, map_location="cpu")
    # support both raw state_dict and wrapped dict
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    # if keys have "module." prefix from DataParallel/DDP checkpoint
    new_sd = {}
    for k, v in sd.items():
        nk = k.replace("module.", "")
        new_sd[nk] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"✅ Loaded weights: {ckpt_path}")
    if missing:
        print(f"⚠️ Missing keys (ok if minor): {missing[:10]}{'...' if len(missing)>10 else ''}")
    if unexpected:
        print(f"⚠️ Unexpected keys (ok if minor): {unexpected[:10]}{'...' if len(unexpected)>10 else ''}")
    model.to(device).eval()
    return model

def mae(a, b): return float(np.mean(np.abs(a - b)))
def mse(a, b): return float(np.mean((a - b) ** 2))
def rmse(a, b): return float(np.sqrt(mse(a, b)))

def psnr(a, b, data_range=1.0):
    m = mse(a, b)
    if m < 1e-12:
        return 99.0
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(m)

def pearson(a, b):
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b) / denom)

def pct_err_on_peaks(pred, gt, q=99.5):
    # compare high-intensity region only (peaks proxy)
    thr = np.percentile(gt, q)
    mask = gt >= thr
    if mask.sum() < 10:
        return float("nan")
    pe = np.mean(np.abs(pred[mask] - gt[mask]) / (np.abs(gt[mask]) + 1e-8))
    return float(pe)

def save_triptych(inp_ch0, pred, gt, out_png, title=""):
    plt.figure(figsize=(15, 5))
    plt.suptitle(title, fontsize=12)

    plt.subplot(1, 3, 1)
    plt.imshow(inp_ch0, cmap="gray")
    plt.title("Input (STEM ch0)")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(pred, cmap="inferno")
    plt.title("Prediction")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(gt, cmap="inferno")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=170)
    plt.close()


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="hackathon_dataset_npz_final")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val"])
    ap.add_argument("--ckpt", type=str, default="checkpoints_ddp/best_model.pt")
    ap.add_argument("--out_dir", type=str, default="inference_outputs")
    ap.add_argument("--model", type=str, default="swin", choices=["swin", "unet"])
    ap.add_argument("--use_checkpoint", action="store_true", help="Only for swin build (not for loading).")
    ap.add_argument("--metadata_mode", type=str, default="sincos7", choices=["raw4", "sincos7"])
    ap.add_argument("--scaling_factor", type=float, default=600000.0, help="Your global max scale (e.g. 600000).")
    ap.add_argument("--clip01", action="store_true", help="Optional: clip scaled target/pred to [0,1].")
    ap.add_argument("--max_files", type=int, default=0, help="0 = all files")
    ap.add_argument("--save_examples", type=int, default=25, help="how many images to save")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    img_out = os.path.join(args.out_dir, "examples")
    os.makedirs(img_out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.data_dir, args.split, "*.npz")))
    if args.max_files and args.max_files > 0:
        files = files[:args.max_files]

    if len(files) == 0:
        raise RuntimeError(f"No .npz found in {args.data_dir}/{args.split}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device} | Files: {len(files)}")

    k = 7 if args.metadata_mode == "sincos7" else 4
    in_ch = 4 + k
    out_ch = 1

    model = build_model(args.model, in_ch=in_ch, out_ch=out_ch, use_checkpoint=args.use_checkpoint)
    model = load_weights(model, args.ckpt, device)

    # metrics accumulators
    rows = []
    agg = {
        "mae_norm": [],
        "rmse_norm": [],
        "psnr_norm": [],
        "pearson_norm": [],
        "peak_pcterr_norm": [],
        "mae_abs": [],
        "rmse_abs": [],
        "peak_pcterr_abs": [],
        "max_gt_abs": [],
        "max_pred_abs": [],
        "mean_gt_abs": [],
        "mean_pred_abs": [],
    }

    saved = 0

    for i, fpath in enumerate(files):
        with np.load(fpath, allow_pickle=False) as data:
            img = data["input"].astype(np.float32)      # (4,H,W)
            tgt = data["potential"].astype(np.float32)  # (H,W) or (1,H,W)
            if tgt.ndim == 3 and tgt.shape[0] == 1:
                tgt = tgt[0]
            elif tgt.ndim == 2:
                pass
            else:
                # if weird shape, flatten to 2D best-effort
                tgt = np.squeeze(tgt)

            H, W = img.shape[1], img.shape[2]

            # Build model input
            img_norm = safe_minmax(img)  # normalize input only
            meta_vec = make_metadata_vec(data, args.metadata_mode)
            meta_maps = make_metadata_maps(meta_vec, H, W)
            x = np.concatenate([img_norm, meta_maps], axis=0)  # (4+k,H,W)

            # Target scaling for stable training/inference
            tgt_norm = tgt / float(args.scaling_factor)  # scaled 0..~1
            if args.clip01:
                tgt_norm = np.clip(tgt_norm, 0.0, 1.0)

            # Torch inference
            x_t = torch.from_numpy(x).unsqueeze(0).to(device)  # (1,C,H,W)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                pred_norm = model(x_t).float().cpu().numpy()[0, 0]  # (H,W)

            if args.clip01:
                pred_norm = np.clip(pred_norm, 0.0, 1.0)

            # ----- normalized metrics (0..1 space) -----
            mae_n = mae(pred_norm, tgt_norm)
            rmse_n = rmse(pred_norm, tgt_norm)
            psnr_n = psnr(pred_norm, tgt_norm, data_range=1.0)
            pr_n = pearson(pred_norm, tgt_norm)
            pk_n = pct_err_on_peaks(pred_norm, tgt_norm, q=99.5)

            # ----- absolute metrics (back to volts) -----
            pred_abs = pred_norm * float(args.scaling_factor)
            tgt_abs = tgt_norm * float(args.scaling_factor)  # equals original unless clip enabled

            mae_a = mae(pred_abs, tgt_abs)
            rmse_a = rmse(pred_abs, tgt_abs)
            pk_a = pct_err_on_peaks(pred_abs, tgt_abs, q=99.5)

            # global value comparisons
            max_gt = float(np.max(tgt_abs))
            max_pr = float(np.max(pred_abs))
            mean_gt = float(np.mean(tgt_abs))
            mean_pr = float(np.mean(pred_abs))

            # store
            agg["mae_norm"].append(mae_n)
            agg["rmse_norm"].append(rmse_n)
            agg["psnr_norm"].append(psnr_n)
            agg["pearson_norm"].append(pr_n)
            agg["peak_pcterr_norm"].append(pk_n)

            agg["mae_abs"].append(mae_a)
            agg["rmse_abs"].append(rmse_a)
            agg["peak_pcterr_abs"].append(pk_a)

            agg["max_gt_abs"].append(max_gt)
            agg["max_pred_abs"].append(max_pr)
            agg["mean_gt_abs"].append(mean_gt)
            agg["mean_pred_abs"].append(mean_pr)

            rows.append([
                os.path.basename(fpath),
                mae_n, rmse_n, psnr_n, pr_n, pk_n,
                mae_a, rmse_a, pk_a,
                max_gt, max_pr, mean_gt, mean_pr
            ])

            # save a few example triptychs
            if saved < args.save_examples:
                title = f"{os.path.basename(fpath)} | MAE(norm)={mae_n:.4f} | MAE(abs)={mae_a:.1f}V"
                out_png = os.path.join(img_out, f"ex_{saved:04d}.png")
                save_triptych(inp_ch0=img_norm[0], pred=pred_abs, gt=tgt_abs, out_png=out_png, title=title)
                saved += 1

        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{len(files)}")

    # write per-file metrics
    metrics_csv = os.path.join(args.out_dir, "metrics.csv")
    with open(metrics_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "file",
            "mae_norm", "rmse_norm", "psnr_norm", "pearson_norm", "peak_pcterr_norm@99.5",
            "mae_abs_V", "rmse_abs_V", "peak_pcterr_abs@99.5",
            "max_gt_V", "max_pred_V", "mean_gt_V", "mean_pred_V"
        ])
        w.writerows(rows)

    def _mean(x):
        x = np.array(x, dtype=np.float64)
        return float(np.nanmean(x))

    summary_txt = os.path.join(args.out_dir, "summary.txt")
    with open(summary_txt, "w") as f:
        f.write("=== SUMMARY (mean over files) ===\n")
        f.write(f"MAE(norm): { _mean(agg['mae_norm']):.6f}\n")
        f.write(f"RMSE(norm): { _mean(agg['rmse_norm']):.6f}\n")
        f.write(f"PSNR(norm): { _mean(agg['psnr_norm']):.3f}\n")
        f.write(f"Pearson(norm): { _mean(agg['pearson_norm']):.4f}\n")
        f.write(f"Peak %Err(norm @99.5): { _mean(agg['peak_pcterr_norm'])*100:.2f}%\n\n")
        f.write(f"MAE(abs V): { _mean(agg['mae_abs']):.2f}\n")
        f.write(f"RMSE(abs V): { _mean(agg['rmse_abs']):.2f}\n")
        f.write(f"Peak %Err(abs @99.5): { _mean(agg['peak_pcterr_abs'])*100:.2f}%\n\n")
        f.write(f"GT max (V): { _mean(agg['max_gt_abs']):.2f}\n")
        f.write(f"Pred max (V): { _mean(agg['max_pred_abs']):.2f}\n")
        f.write(f"GT mean (V): { _mean(agg['mean_gt_abs']):.2f}\n")
        f.write(f"Pred mean (V): { _mean(agg['mean_pred_abs']):.2f}\n")

    print(f"✅ Saved: {metrics_csv}")
    print(f"✅ Saved: {summary_txt}")
    print(f"🖼️ Examples: {img_out}")


if __name__ == "__main__":
    main()