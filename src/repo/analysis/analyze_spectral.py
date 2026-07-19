"""Post-hoc spectral mechanism analysis for the E207 E2E Capacity checkpoint.

Outputs three complementary diagnostics:
1) analytical PDE frequency response for every learned WPO layer;
2) empirical finite-difference spectral mixing matrix of the full backbone;
3) CASSI measurement-domain leave-one-band-out recovery.

The diagnostics do not alter or train the model. Band removal is an explanatory
intervention because a real snapshot measurement does not expose bandwise terms.
"""

import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--only", choices=("all", "frequency", "impulse", "occlusion", "equal_occlusion", "evidence"), default="all"
    )
    parser.add_argument("--device", default=None, help="Override device, e.g. cpu or cuda")
    return parser.parse_args()


ARGS = parse_args()
with open(ARGS.config, "r", encoding="utf-8") as handle:
    CFG = yaml.safe_load(handle)

if ARGS.device is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(CFG["gpu_id"])

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(ROOT))
os.environ["SMILE_CONFIG"] = str(Path(ARGS.config).resolve())

from dataset import load_mask, load_test
from loss import torch_psnr, torch_sam, torch_ssim
from model.e2e import E2ESMILE, cassi_measure, phi_phi_t, shift_back, shift_cube
from model.wpo3d import WPO3D


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_style():
    style = CFG.get("style", {})
    plt.rcParams.update({
        "font.family": style.get("font_family", "DejaVu Sans"),
        "font.size": style.get("font_size", 10),
        "axes.titlesize": style.get("title_size", 11),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": style.get("dpi", 220),
        "savefig.dpi": style.get("dpi", 220),
        "savefig.bbox": "tight",
    })


def build_model(device):
    model = E2ESMILE(
        dim=CFG["dim"], unet_stage=CFG["unet_stage"], num_blocks=CFG["num_blocks"],
        use_sicmb=CFG["use_sicmb"], use_perchannel=CFG["use_perchannel"],
        post_block=CFG["post_block"], ffn_mult=CFG["ffn_mult"],
        input_mode=CFG["input_mode"], output_dc=CFG["output_dc"],
        dc_gamma_init=CFG.get("dc_gamma_init", 0.30),
        wpo_variant=CFG.get("wpo_variant", "full"), gradient_checkpointing=False,
        step=CFG.get("shift_step", 2), bands=CFG["num_bands"],
    ).to(device)
    checkpoint = torch.load(CFG["checkpoint"], map_location=device)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, checkpoint


def save_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, out_dir, stem):
    fig.savefig(out_dir / f"{stem}.png")
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


def layer_frequency_transfer(module, device):
    """Return the actual per-layer zero-spatial-frequency transfer.

    Only layers with ``module.dim == num_bands`` have a direct wavelength-axis
    interpretation. Wider layers operate on latent feature channels.
    """
    alpha, vs, vl, t = module._get_effective_params()
    alpha = alpha.reshape(-1)
    vs = vs.reshape(-1)
    channels = int(module.dim)
    fft_channels = 1 << (channels - 1).bit_length()
    if alpha.numel() == 1:
        alpha = alpha.repeat(fft_channels)
        vs = vs.repeat(fft_channels)
    elif alpha.numel() < fft_channels:
        alpha = F.pad(alpha, (0, fft_channels - alpha.numel()))
        vs = F.pad(vs, (0, fft_channels - vs.numel()))
    elif alpha.numel() > fft_channels:
        raise ValueError(f"layer dim/parameter mismatch: dim={channels}, alpha={alpha.numel()}")
    fc = torch.fft.fftfreq(fft_channels, device=device)
    omega_sq = (2 * math.pi) ** 2 * vl.square() * fc.square()
    ones = torch.ones_like(fc, dtype=torch.complex64)
    zeros = torch.zeros_like(ones)
    response, _, _ = module._solve_damped(ones, zeros, omega_sq, alpha, t)
    kernel = torch.fft.ifft(response.to(torch.complex64), n=fft_channels).real
    return response.real, kernel, fc, alpha, vs, vl, t, channels, fft_channels


def analytical_frequency_response(model, out_dir, device):
    """Initial-displacement transfer with zero initial velocity at fx=fy=0."""
    layers = [(name, module) for name, module in model.named_modules() if isinstance(module, WPO3D)]
    bands = int(CFG["num_bands"])
    rows, spectral_responses, spectral_kernels, arrays = [], [], [], {}
    with torch.no_grad():
        for layer_idx, (name, module) in enumerate(layers):
            response, kernel, fc, alpha, vs_flat, vl, t, channels, fft_channels = \
                layer_frequency_transfer(module, device)
            response_np = response.cpu().numpy()
            kernel_np = torch.fft.fftshift(kernel).cpu().numpy()
            arrays[f"layer_{layer_idx:02d}_response"] = response_np
            arrays[f"layer_{layer_idx:02d}_kernel"] = kernel_np
            if channels == bands:
                spectral_responses.append(response_np)
                spectral_kernels.append(kernel_np)
            for mode in range(fft_channels):
                rows.append({
                    "layer": layer_idx, "module": name, "mode": mode,
                    "axis_semantics": "wavelength" if channels == bands else "latent_feature",
                    "feature_channels": channels, "fft_channels": fft_channels,
                    "frequency_cycles_per_band": float(fc[mode].cpu()),
                    "response": float(response[mode].cpu()),
                    "kernel_shifted": float(torch.fft.fftshift(kernel)[mode].cpu()),
                    "alpha_mode": float(alpha[mode].cpu()),
                    "vs_mode": float(vs_flat[mode].cpu()),
                    "vl": float(vl.cpu()), "t": float(t.cpu()),
                })
    if not spectral_responses:
        raise RuntimeError("no WPO layer has a direct 28-band wavelength axis")
    np.save(out_dir / "frequency_response.npy", np.stack(spectral_responses))
    np.save(out_dir / "spectral_pde_kernel.npy", np.stack(spectral_kernels))
    np.savez(out_dir / "all_layer_frequency_response.npz", **arrays)
    save_csv(out_dir / "frequency_response.csv", list(rows[0]), rows)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7))
    mean_response = np.mean(np.abs(spectral_responses), axis=0)
    fft_bands = mean_response.shape[0]
    freq = np.fft.fftfreq(fft_bands)
    order = np.argsort(freq)
    axes[0].plot(freq[order], mean_response[order], color=CFG["style"]["color_full"], lw=2)
    axes[0].set(xlabel="Spectral frequency (cycles/band)", ylabel="Mean |transfer|",
                title=f"PDE spectral transfer ({len(spectral_responses)} wavelength-axis layers)")
    axes[0].grid(alpha=.22)
    mean_kernel = np.mean(np.abs(spectral_kernels), axis=0)
    offsets = np.arange(fft_bands) - fft_bands // 2
    axes[1].stem(offsets, mean_kernel, linefmt=CFG["style"]["color_near"], markerfmt="o", basefmt=" ")
    axes[1].set(xlabel="Spectral offset", ylabel="Mean |impulse coefficient|",
                title="Equivalent spectral propagation kernel")
    axes[1].grid(alpha=.22)
    fig.tight_layout()
    save_figure(fig, out_dir, "frequency_response")


def prepare_data(device):
    test = load_test(CFG["test_path"], nC=CFG["num_bands"]).float()
    mask = load_mask(CFG["mask_path"], nC=CFG["num_bands"]).float()
    return test, mask


def empirical_impulse_response(model, test, mask, out_dir, device):
    """Finite-difference Jacobian of backbone output wrt its H-field input."""
    n_scene = min(int(CFG["impulse_num_scenes"]), len(test))
    bands, eps = int(CFG["num_bands"]), float(CFG["impulse_epsilon"])
    patch = int(CFG["impulse_patch_size"])
    batch_size = int(CFG["analysis_batch_size"])
    signed = torch.zeros(bands, bands, device=device)
    absolute = torch.zeros_like(signed)
    mask1 = mask.unsqueeze(0).to(device)
    shifted1 = shift_cube(mask1, CFG.get("shift_step", 2))
    h, w = test.shape[-2:]
    y0, x0 = (h - patch) // 2, (w - patch) // 2

    with torch.no_grad():
        for scene in range(n_scene):
            gt = test[scene:scene + 1].to(device)
            y = cassi_measure(gt, mask1, CFG.get("shift_step", 2))
            h_field = shift_back(y / bands * 2.0, bands, CFG.get("shift_step", 2))
            base = model.backbone(h_field, shifted1)
            for start in range(0, bands, batch_size):
                ids = list(range(start, min(start + batch_size, bands)))
                perturbed = h_field.expand(len(ids), -1, -1, -1).clone()
                for row, band in enumerate(ids):
                    perturbed[row, band, y0:y0 + patch, x0:x0 + patch] += eps
                shifted = shifted1.expand(len(ids), -1, -1, -1)
                delta = (model.backbone(perturbed, shifted) - base) / eps
                local = delta[:, :, y0:y0 + patch, x0:x0 + patch]
                for row, band in enumerate(ids):
                    signed[:, band] += local[row].mean(dim=(-2, -1))
                    absolute[:, band] += local[row].abs().mean(dim=(-2, -1))
            print(f"[impulse] scene {scene + 1}/{n_scene}", flush=True)
    signed /= n_scene
    absolute /= n_scene
    signed_np, absolute_np = signed.cpu().numpy(), absolute.cpu().numpy()
    column_norm = absolute_np / (absolute_np.sum(axis=0, keepdims=True) + 1e-12)
    np.save(out_dir / "empirical_mixing_signed.npy", signed_np)
    np.save(out_dir / "empirical_mixing_abs.npy", absolute_np)
    np.save(out_dir / "empirical_mixing_column_normalized.npy", column_norm)
    rows = []
    for out_band in range(bands):
        for in_band in range(bands):
            rows.append({"output_band": out_band + 1, "input_band": in_band + 1,
                         "signed_response": signed_np[out_band, in_band],
                         "absolute_response": absolute_np[out_band, in_band],
                         "column_normalized_response": column_norm[out_band, in_band]})
    save_csv(out_dir / "empirical_mixing.csv", list(rows[0]), rows)

    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    image = ax.imshow(column_norm, cmap=CFG["style"]["cmap_sequential"], aspect="auto")
    ax.set(xlabel="Perturbed input band", ylabel="Responding output band",
           title="Empirical cross-band mixing (finite difference)")
    fig.colorbar(image, ax=ax, label="Column-normalized |response|")
    fig.tight_layout()
    save_figure(fig, out_dir, "empirical_mixing_map")


def metric_triplet(pred, gt):
    pred = pred.clamp(0, 1)
    return float(torch_psnr(pred, gt)), float(torch_ssim(pred, gt)), float(torch_sam(pred, gt))


def target_band_psnr(pred, gt, band):
    pred_q = (pred[band].clamp(0, 1) * 256).round()
    gt_q = (gt[band] * 256).round()
    mse = (pred_q - gt_q).square().mean().clamp_min(1e-12)
    return float(10 * torch.log10(pred_q.new_tensor(255.0 ** 2) / mse))


def occlusion_recovery(model, test, mask, out_dir, device):
    """Remove known band contributions from simulated measurements."""
    n_scene = min(int(CFG["occlusion_num_scenes"]), len(test))
    bands, step = int(CFG["num_bands"]), int(CFG.get("shift_step", 2))
    radius = int(CFG["occlusion_neighbor_radius"])
    mask1 = mask.unsqueeze(0).to(device)
    shifted = shift_cube(mask1, step)
    ppt = phi_phi_t(mask1, step)
    rows, cube_rows = [], []
    conditions = ("full", "target_missing", "near_only", "far_only")

    with torch.no_grad():
        for scene in range(n_scene):
            gt = test[scene:scene + 1].to(device)
            contributions = shift_cube(gt * mask1, step)
            full_y = contributions.sum(dim=1)
            full_pred = model(full_y, mask1, shifted, ppt)[0]
            full_metrics = metric_triplet(full_pred, gt[0])
            for band in range(bands):
                near = [j for j in range(bands) if j != band and abs(j - band) <= radius]
                far = [j for j in range(bands) if abs(j - band) > radius]
                ys = {
                    "full": full_y,
                    "target_missing": full_y - contributions[:, band],
                    "near_only": contributions[:, near].sum(dim=1) if near else torch.zeros_like(full_y),
                    "far_only": contributions[:, far].sum(dim=1) if far else torch.zeros_like(full_y),
                }
                preds = {}
                for condition in conditions:
                    preds[condition] = full_pred if condition == "full" else model(
                        ys[condition], mask1, shifted, ppt
                    )[0]
                    rows.append({
                        "scene": scene + 1, "target_band": band + 1, "condition": condition,
                        "target_psnr": target_band_psnr(preds[condition], gt[0], band),
                        "target_mae": float((preds[condition][band].clamp(0, 1) - gt[0, band]).abs().mean()),
                    })
                for condition in ("target_missing", "near_only", "far_only"):
                    psnr, ssim, sam = metric_triplet(preds[condition], gt[0])
                    cube_rows.append({"scene": scene + 1, "removed_target_band": band + 1,
                                      "condition": condition, "cube_psnr": psnr,
                                      "cube_ssim": ssim, "cube_sam_rad": sam,
                                      "full_cube_psnr": full_metrics[0],
                                      "full_cube_ssim": full_metrics[1],
                                      "full_cube_sam_rad": full_metrics[2]})
            print(f"[occlusion] scene {scene + 1}/{n_scene}", flush=True)
    save_csv(out_dir / "occlusion_target_band.csv", list(rows[0]), rows)
    save_csv(out_dir / "occlusion_full_cube.csv", list(cube_rows[0]), cube_rows)

    mean = {condition: np.zeros(bands) for condition in conditions}
    for condition in conditions:
        for band in range(bands):
            values = [r["target_psnr"] for r in rows
                      if r["condition"] == condition and r["target_band"] == band + 1]
            mean[condition][band] = np.mean(values)
    summary_rows = []
    for band in range(bands):
        summary_rows.append({"target_band": band + 1, **{
            f"{condition}_target_psnr": mean[condition][band] for condition in conditions
        }})
    save_csv(out_dir / "occlusion_summary_by_band.csv", list(summary_rows[0]), summary_rows)

    style = CFG["style"]
    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    x = np.arange(1, bands + 1)
    ax.plot(x, mean["full"], color=style["color_full"], label="Full measurement", lw=2)
    ax.plot(x, mean["target_missing"], color=style["color_missing"], label="Target contribution removed", lw=2)
    ax.plot(x, mean["near_only"], color=style["color_near"], label=f"Adjacent ±{radius} only", lw=1.6)
    ax.plot(x, mean["far_only"], color=style["color_far"], label=f"Distant bands only", lw=1.6)
    ax.set(xlabel="Target spectral band", ylabel="Target-band PSNR (dB)",
           title="Cross-band recovery under measurement decomposition")
    ax.grid(alpha=.22)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    save_figure(fig, out_dir, "occlusion_recovery")


def equal_count_occlusion(model, test, mask, out_dir, device):
    """Paired context removal with equal near/far band counts.

    Starting from a target-missing measurement, remove K adjacent bands or K
    distant bands. Multiple distant subsets are averaged. This controls band
    count; removed measurement energy is recorded rather than normalized away.
    """
    n_scene = min(int(CFG["occlusion_num_scenes"]), len(test))
    bands, step = int(CFG["num_bands"]), int(CFG.get("shift_step", 2))
    radius = int(CFG["occlusion_neighbor_radius"])
    trials = int(CFG.get("equal_count_far_trials", 8))
    batch_size = int(CFG.get("analysis_batch_size", 4))
    mask1 = mask.unsqueeze(0).to(device)

    # Use the same deterministic distant subsets for every scene.
    far_subsets = {}
    for band in range(bands):
        near = [j for j in range(bands) if j != band and abs(j - band) <= radius]
        far = [j for j in range(bands) if abs(j - band) > radius]
        rng = np.random.RandomState(int(CFG["seed"]) + band)
        far_subsets[band] = [sorted(rng.choice(far, size=len(near), replace=False).tolist())
                             for _ in range(trials)]

    rows = []
    with torch.no_grad():
        for scene in range(n_scene):
            gt = test[scene:scene + 1].to(device)
            contributions = shift_cube(gt * mask1, step)
            full_y = contributions.sum(dim=1)
            for band in range(bands):
                near = [j for j in range(bands) if j != band and abs(j - band) <= radius]
                target_missing = full_y - contributions[:, band]
                specs = [("target_missing", -1, [], target_missing)]
                specs.append(("near_removed", 0, near,
                              target_missing - contributions[:, near].sum(dim=1)))
                for trial, subset in enumerate(far_subsets[band]):
                    specs.append(("matched_far_removed", trial, subset,
                                  target_missing - contributions[:, subset].sum(dim=1)))

                for start in range(0, len(specs), batch_size):
                    chunk = specs[start:start + batch_size]
                    y_batch = torch.cat([item[3] for item in chunk], dim=0)
                    n = len(chunk)
                    mask_b = mask1.expand(n, -1, -1, -1)
                    shifted_b = shift_cube(mask_b, step)
                    ppt_b = phi_phi_t(mask_b, step)
                    pred_batch = model(y_batch, mask_b, shifted_b, ppt_b).clamp(0, 1)
                    for idx, (condition, trial, removed, _) in enumerate(chunk):
                        pred = pred_batch[idx]
                        psnr, ssim, sam = metric_triplet(pred, gt[0])
                        removed_term = contributions[:, removed].sum(dim=1) if removed else torch.zeros_like(full_y)
                        rows.append({
                            "scene": scene + 1, "target_band": band + 1,
                            "condition": condition, "trial": trial,
                            "k_context_removed": len(removed),
                            "removed_bands_1based": " ".join(str(v + 1) for v in removed),
                            "removed_measurement_rms": float(removed_term.square().mean().sqrt()),
                            "target_psnr": target_band_psnr(pred, gt[0], band),
                            "target_mae": float((pred[band] - gt[0, band]).abs().mean()),
                            "cube_psnr": psnr, "cube_ssim": ssim, "cube_sam_rad": sam,
                        })
            print(f"[equal-occlusion] scene {scene + 1}/{n_scene}", flush=True)

    save_csv(out_dir / "equal_count_occlusion_trials.csv", list(rows[0]), rows)
    summary_rows, paired_rows = [], []
    for band in range(bands):
        band_rows = [r for r in rows if r["target_band"] == band + 1]
        for condition in ("target_missing", "near_removed", "matched_far_removed"):
            selected = [r for r in band_rows if r["condition"] == condition]
            summary_rows.append({
                "target_band": band + 1, "condition": condition,
                "target_psnr_mean": float(np.mean([r["target_psnr"] for r in selected])),
                "target_psnr_std": float(np.std([r["target_psnr"] for r in selected])),
                "cube_sam_mean": float(np.mean([r["cube_sam_rad"] for r in selected])),
                "removed_measurement_rms_mean": float(np.mean([r["removed_measurement_rms"] for r in selected])),
                "n": len(selected),
            })
        for scene in range(1, n_scene + 1):
            sr = [r for r in band_rows if r["scene"] == scene]
            base = [r for r in sr if r["condition"] == "target_missing"][0]["target_psnr"]
            near_psnr = [r for r in sr if r["condition"] == "near_removed"][0]["target_psnr"]
            far_psnr = np.mean([r["target_psnr"] for r in sr if r["condition"] == "matched_far_removed"])
            paired_rows.append({
                "scene": scene, "target_band": band + 1,
                "target_missing_psnr": base, "near_removed_psnr": near_psnr,
                "matched_far_removed_psnr": float(far_psnr),
                "drop_remove_near": base - near_psnr,
                "drop_remove_far": base - far_psnr,
                "near_minus_far_importance": (base - near_psnr) - (base - far_psnr),
            })
    save_csv(out_dir / "equal_count_occlusion_by_band.csv", list(summary_rows[0]), summary_rows)
    save_csv(out_dir / "equal_count_occlusion_paired.csv", list(paired_rows[0]), paired_rows)

    def band_curve(condition, field="target_psnr_mean"):
        return np.array([next(r[field] for r in summary_rows
                              if r["target_band"] == band + 1 and r["condition"] == condition)
                         for band in range(bands)])

    base_curve = band_curve("target_missing")
    near_curve = band_curve("near_removed")
    far_curve = band_curve("matched_far_removed")
    near_drop, far_drop = base_curve - near_curve, base_curve - far_curve
    paired_importance = np.array([r["near_minus_far_importance"] for r in paired_rows])
    aggregate = {
        "target_missing_psnr": float(base_curve.mean()),
        "near_removed_psnr": float(near_curve.mean()),
        "matched_far_removed_psnr": float(far_curve.mean()),
        "drop_remove_near": float(near_drop.mean()),
        "drop_remove_far": float(far_drop.mean()),
        "near_minus_far_importance_mean": float(paired_importance.mean()),
        "near_more_important_fraction": float((paired_importance > 0).mean()),
        "num_scene_band_pairs": len(paired_rows),
        "far_trials_per_pair": trials,
    }
    with open(out_dir / "equal_count_occlusion_summary.json", "w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)

    style = CFG["style"]
    x = np.arange(1, bands + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))
    axes[0].plot(x, base_curve, color=style["color_full"], lw=2, label="Target missing")
    axes[0].plot(x, near_curve, color=style["color_near"], lw=1.8, label="Also remove K adjacent")
    axes[0].plot(x, far_curve, color=style["color_far"], lw=1.8, label="Also remove K distant (8 trials)")
    axes[0].set(xlabel="Target spectral band", ylabel="Target-band PSNR (dB)",
                title="Equal-count context removal")
    axes[0].grid(alpha=.22); axes[0].legend(frameon=False)
    axes[1].plot(x, near_drop, color=style["color_near"], lw=1.8, label="PSNR drop: adjacent")
    axes[1].plot(x, far_drop, color=style["color_far"], lw=1.8, label="PSNR drop: distant")
    axes[1].axhline(0, color="black", lw=.8, alpha=.5)
    axes[1].set(xlabel="Target spectral band", ylabel="Drop from target-missing baseline (dB)",
                title="Matched importance of adjacent vs. distant bands")
    axes[1].grid(alpha=.22); axes[1].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, out_dir, "equal_count_occlusion")
    print("[equal-occlusion] " + json.dumps(aggregate), flush=True)


def complete_frequency_evidence(model, test, mask, out_dir, device):
    """Connect learned PDE modes to real test spectra and audit symmetry.

    This is descriptive evidence, not proof that a particular absorption peak is
    generated by one Fourier mode. It also creates the missing SAM occlusion plot.
    """
    bands, step = int(CFG["num_bands"]), int(CFG.get("shift_step", 2))
    fft_bands = 1 << (bands - 1).bit_length()
    mask1 = mask.unsqueeze(0).to(device)
    shifted = shift_cube(mask1, step)
    ppt = phi_phi_t(mask1, step)
    gt_power = torch.zeros(fft_bands, device=device)
    pred_power = torch.zeros_like(gt_power)
    error_power = torch.zeros_like(gt_power)
    curvature = torch.zeros(bands, device=device)
    mae = torch.zeros(bands, device=device)
    band_psnr = torch.zeros(bands, device=device)

    with torch.no_grad():
        for scene in range(len(test)):
            gt = test[scene:scene + 1].to(device)
            y = cassi_measure(gt, mask1, step)
            pred = model(y, mask1, shifted, ppt).clamp(0, 1)
            gt_fft = torch.fft.fft(gt, n=fft_bands, dim=1)
            pred_fft = torch.fft.fft(pred, n=fft_bands, dim=1)
            err_fft = pred_fft - gt_fft
            gt_power += gt_fft.abs().square().mean(dim=(0, 2, 3))
            pred_power += pred_fft.abs().square().mean(dim=(0, 2, 3))
            error_power += err_fft.abs().square().mean(dim=(0, 2, 3))
            mae += (pred - gt).abs().mean(dim=(0, 2, 3))
            mse_band = (pred - gt).square().mean(dim=(0, 2, 3)).clamp_min(1e-12)
            band_psnr += 10 * torch.log10(1.0 / mse_band)
            second = (gt[:, 2:] - 2 * gt[:, 1:-1] + gt[:, :-2]).abs().mean(dim=(0, 2, 3))
            curvature[1:-1] += second
            curvature[0] += (gt[:, 1] - gt[:, 0]).abs().mean()
            curvature[-1] += (gt[:, -1] - gt[:, -2]).abs().mean()
            print(f"[frequency-evidence] scene {scene + 1}/{len(test)}", flush=True)
    n_scene = float(len(test))
    gt_power /= n_scene; pred_power /= n_scene; error_power /= n_scene
    curvature /= n_scene; mae /= n_scene; band_psnr /= n_scene

    # Recompute with each layer's true width. Only 28-channel layers are
    # compared with the physical HSI spectrum; 56/112-channel layers are latent.
    layers = [(name, module) for name, module in model.named_modules() if isinstance(module, WPO3D)]
    layer_records, spectral_responses = [], []
    with torch.no_grad():
        for idx, (name, module) in enumerate(layers):
            response, _, _, alpha, vs, _, _, channels, fft_channels = \
                layer_frequency_transfer(module, device)
            response_np = response.cpu().numpy()
            alpha_np = alpha.cpu().numpy()
            vs_np = vs.cpu().numpy()
            if channels == bands:
                spectral_responses.append(response_np)
            layer_records.append((idx, name, channels, fft_channels, response_np, alpha_np, vs_np))
    if not spectral_responses:
        raise RuntimeError("no 28-channel WPO layer available for wavelength evidence")
    mean_response = np.mean(np.abs(spectral_responses), axis=0)
    if len(mean_response) != fft_bands:
        raise RuntimeError(f"expected {fft_bands}-point spectral response, got {len(mean_response)}")
    freq = np.fft.fftfreq(fft_bands)
    data_rows = []
    for mode in range(fft_bands):
        data_rows.append({
            "mode": mode, "frequency_cycles_per_band": freq[mode],
            "gt_power": float(gt_power[mode].cpu()),
            "pred_power": float(pred_power[mode].cpu()),
            "error_power": float(error_power[mode].cpu()),
            "relative_error_power": float((error_power[mode] / gt_power[mode].clamp_min(1e-12)).cpu()),
            "mean_abs_pde_response": float(mean_response[mode]),
        })
    save_csv(out_dir / "data_frequency_evidence.csv", list(data_rows[0]), data_rows)
    band_rows = [{
        "band": band + 1, "mean_abs_second_difference": float(curvature[band].cpu()),
        "reconstruction_mae": float(mae[band].cpu()), "reconstruction_psnr": float(band_psnr[band].cpu()),
    } for band in range(bands)]
    save_csv(out_dir / "absorption_detail_evidence.csv", list(band_rows[0]), band_rows)

    # A real-valued spectral operator should have mirrored mode coefficients.
    audit_rows = []
    for idx, name, channels, layer_fft, response, alpha, vs in layer_records:
        mirror = np.array([(-k) % layer_fft for k in range(layer_fft)])
        def asymmetry(values):
            return float(np.mean(np.abs(values - values[mirror])) / (np.mean(np.abs(values)) + 1e-12))
        audit_rows.append({
            "layer": idx, "module": name,
            "axis_semantics": "wavelength" if channels == bands else "latent_feature",
            "feature_channels": channels, "fft_channels": layer_fft,
            "response_mirror_asymmetry": asymmetry(response),
            "alpha_mirror_asymmetry": asymmetry(alpha),
            "vs_mirror_asymmetry": asymmetry(vs),
        })
    save_csv(out_dir / "hermitian_symmetry_audit.csv", list(audit_rows[0]), audit_rows)

    order = np.argsort(freq)
    gt_np = gt_power.cpu().numpy(); pred_np = pred_power.cpu().numpy()
    err_np = error_power.cpu().numpy(); curve_np = curvature.cpu().numpy(); mae_np = mae.cpu().numpy()
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2))
    axes[0, 0].plot(freq[order], gt_np[order] / gt_np.max(), lw=2, label="GT power")
    axes[0, 0].plot(freq[order], pred_np[order] / pred_np.max(), lw=1.7, label="Reconstruction power")
    axes[0, 0].plot(freq[order], mean_response[order] / (mean_response.max() + 1e-12), lw=1.5,
                    label="Mean PDE response")
    axes[0, 0].set(title="Data spectrum and learned PDE modes", xlabel="Spectral frequency",
                   ylabel="Normalized magnitude"); axes[0, 0].legend(frameon=False); axes[0, 0].grid(alpha=.2)
    axes[0, 1].plot(freq[order], (err_np / np.maximum(gt_np, 1e-12))[order], color=CFG["style"]["color_missing"])
    axes[0, 1].set(title="Frequency-resolved reconstruction error", xlabel="Spectral frequency",
                   ylabel="Error power / GT power"); axes[0, 1].grid(alpha=.2)
    x = np.arange(1, bands + 1)
    axes[1, 0].plot(x, curve_np / (curve_np.max() + 1e-12), lw=2, label="GT spectral curvature")
    axes[1, 0].plot(x, mae_np / (mae_np.max() + 1e-12), lw=1.8, label="Reconstruction MAE")
    axes[1, 0].set(title="Absorption/detail proxy vs. reconstruction error", xlabel="Spectral band",
                   ylabel="Normalized score"); axes[1, 0].legend(frameon=False); axes[1, 0].grid(alpha=.2)
    axes[1, 1].plot([r["layer"] for r in audit_rows],
                    [r["response_mirror_asymmetry"] for r in audit_rows], marker="o", label="PDE response")
    axes[1, 1].plot([r["layer"] for r in audit_rows],
                    [r["alpha_mirror_asymmetry"] for r in audit_rows], marker="s", label="alpha")
    axes[1, 1].plot([r["layer"] for r in audit_rows],
                    [r["vs_mirror_asymmetry"] for r in audit_rows], marker="^", label="vs")
    axes[1, 1].set(title="Spectral-mode mirror-symmetry audit", xlabel="WPO layer", ylabel="Relative asymmetry")
    axes[1, 1].legend(frameon=False); axes[1, 1].grid(alpha=.2)
    fig.tight_layout(); save_figure(fig, out_dir, "frequency_evidence_complete")

    equal_csv = out_dir / "equal_count_occlusion_trials.csv"
    if equal_csv.exists():
        with open(equal_csv, "r", encoding="utf-8") as handle:
            equal_rows = list(csv.DictReader(handle))
        conditions = ("target_missing", "near_removed", "matched_far_removed")
        curves = {}
        for condition in conditions:
            curves[condition] = np.array([
                np.mean([float(r["cube_sam_rad"]) for r in equal_rows
                         if r["condition"] == condition and int(r["target_band"]) == band + 1])
                for band in range(bands)
            ])
        fig, ax = plt.subplots(figsize=(9.2, 4.0))
        ax.plot(x, curves["target_missing"], lw=2, label="Target missing")
        ax.plot(x, curves["near_removed"], lw=1.8, label="Also remove K adjacent")
        ax.plot(x, curves["matched_far_removed"], lw=1.8, label="Also remove K distant")
        ax.set(title="Equal-count occlusion: full-cube spectral angle", xlabel="Target spectral band",
               ylabel="SAM (radians)"); ax.legend(frameon=False); ax.grid(alpha=.2)
        fig.tight_layout(); save_figure(fig, out_dir, "equal_count_occlusion_sam")

    summary = {
        "mean_response_mirror_asymmetry": float(np.mean([r["response_mirror_asymmetry"] for r in audit_rows])),
        "mean_alpha_mirror_asymmetry": float(np.mean([r["alpha_mirror_asymmetry"] for r in audit_rows])),
        "mean_vs_mirror_asymmetry": float(np.mean([r["vs_mirror_asymmetry"] for r in audit_rows])),
        "curvature_mae_correlation": float(np.corrcoef(curve_np, mae_np)[0, 1]),
    }
    with open(out_dir / "frequency_evidence_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print("[frequency-evidence] " + json.dumps(summary), flush=True)


def write_manifest(out_dir, checkpoint, model):
    info = {
        "experiment": CFG["experiment_name"], "checkpoint": CFG["checkpoint"],
        "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "checkpoint_best_psnr": checkpoint.get("best_psnr") if isinstance(checkpoint, dict) else None,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "interpretation_limits": [
            "Frequency response isolates the learned PDE transfer with zero initial velocity.",
            "Impulse response is a finite-difference response of the nonlinear backbone, not a fixed attention matrix.",
            "Band removal uses simulated bandwise measurement terms and is diagnostic, not a deployable missing-band protocol.",
        ],
        "config": CFG,
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(info, handle, ensure_ascii=False, indent=2)


def main():
    seed_everything(int(CFG["seed"]))
    configure_style()
    if ARGS.device:
        device = torch.device(ARGS.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(CFG["output_root"]) / CFG["experiment_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint = build_model(device)
    write_manifest(out_dir, checkpoint, model)
    print(f"device={device} output={out_dir}", flush=True)

    if ARGS.only in ("all", "frequency"):
        analytical_frequency_response(model, out_dir, device)
        print("[done] analytical frequency response", flush=True)
    if ARGS.only in ("all", "impulse", "occlusion", "equal_occlusion", "evidence"):
        test, mask = prepare_data(device)
        print(f"test={tuple(test.shape)} mask={tuple(mask.shape)}", flush=True)
        if ARGS.only in ("all", "impulse"):
            empirical_impulse_response(model, test, mask, out_dir, device)
            print("[done] empirical impulse response", flush=True)
        if ARGS.only in ("all", "occlusion"):
            occlusion_recovery(model, test, mask, out_dir, device)
            print("[done] occlusion recovery", flush=True)
        if ARGS.only in ("all", "equal_occlusion"):
            equal_count_occlusion(model, test, mask, out_dir, device)
            print("[done] equal-count occlusion", flush=True)
        if ARGS.only in ("all", "evidence"):
            complete_frequency_evidence(model, test, mask, out_dir, device)
            print("[done] complete frequency evidence", flush=True)


if __name__ == "__main__":
    main()


