"""Run SMILE² E2E checkpoints on TSA_real_data measurements.

This follows the real-data protocol used by MST:
  - load `Measurements/scene{i}.mat` with key `meas_real`
  - normalize each measurement by `meas / meas.max() * 0.8`
  - load the 2D coded aperture `mask.mat` with key `mask`
  - save reconstructed HSI as MATLAB key `res`, shape [H, W, C]

The model itself performs the same CASSI prep used in training:
shift-back H0, PhiPhiT, residual adjoint cues for estimate-evolve variants.
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import yaml

from model.smile import SMILE2


def load_config(path):
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)
    return cfg


def build_model(cfg):
    return SMILE2(
        dim=cfg["dim"],
        unet_stage=cfg["unet_stage"],
        num_blocks=cfg["num_blocks"],
        use_spatial_content_modulation=cfg["use_spatial_content_modulation"],
        use_perchannel=cfg["use_perchannel"],
        use_spectral_wave=cfg.get("use_spectral_wave", True),
        post_block=cfg["post_block"],
        ffn_mult=cfg["ffn_mult"],
        input_mode=cfg["input_mode"],
        output_dc=cfg["output_dc"],
        dc_gamma_init=cfg.get("dc_gamma_init", 0.30),
        swp_variant=cfg.get("swp_variant", "full"),
        gradient_checkpointing=False,
        bands=cfg["num_bands"],
        input_adapter=cfg.get("input_adapter", "none"),
        wavelength_cutoff_init=cfg.get("wavelength_cutoff_init", 0.28),
        wave_param_mode=cfg.get("wave_param_mode", "free"),
        wave_basis_count=cfg.get("wave_basis_count", 3),
        num_field_outputs=cfg.get("num_field_outputs", 1),
        share_estimator_evolver_weights=cfg.get("share_estimator_evolver_weights", True),
        return_intermediate_fields=cfg.get("return_intermediate_fields", False),
        field_process_mode=cfg.get("field_process_mode", "plain"),
    )


def strip_module_prefix(state):
    if not any(k.startswith("module.") for k in state.keys()):
        return state
    return {k.replace("module.", "", 1): v for k, v in state.items()}


def load_checkpoint(model, ckpt_path, device):
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except TypeError:
        # PyTorch < 2.6 does not have the weights_only keyword.
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(ckpt, dict):
        state = ckpt.get("model") or ckpt.get("state_dict") or ckpt.get("model_state_dict")
        meta = {k: v for k, v in ckpt.items() if k != "model"}
    else:
        state = ckpt
        meta = {}
    if state is None:
        raise KeyError(f"No model/state_dict found in checkpoint: {ckpt_path}")
    missing, unexpected = model.load_state_dict(strip_module_prefix(state), strict=False)
    if missing or unexpected:
        print(f"[ckpt] missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("  missing first:", missing[:8])
        if unexpected:
            print("  unexpected first:", unexpected[:8])
    return meta


def resolve_real_paths(real_root):
    root = Path(real_root)
    meas_dir = root / "Measurements"
    mask_path = root / "mask.mat"
    if not meas_dir.exists():
        raise FileNotFoundError(f"Measurements dir not found: {meas_dir}")
    if not mask_path.exists():
        raise FileNotFoundError(f"mask.mat not found: {mask_path}")
    return meas_dir, mask_path


def load_mask(mask_path, bands, device):
    mat = sio.loadmat(str(mask_path))
    if "mask" not in mat:
        raise KeyError(f"`mask` key not found in {mask_path}; keys={list(mat.keys())}")
    mask = mat["mask"].astype(np.float32)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = torch.from_numpy(mask).float().to(device)
    mask = mask.unsqueeze(0).unsqueeze(0).repeat(1, bands, 1, 1)
    return mask


def load_measurement(path, device):
    mat = sio.loadmat(str(path))
    if "meas_real" not in mat:
        raise KeyError(f"`meas_real` key not found in {path}; keys={list(mat.keys())}")
    meas = mat["meas_real"].astype(np.float32)
    if meas.ndim == 3:
        meas = np.squeeze(meas)
    meas_max = float(np.max(meas))
    if meas_max <= 0:
        raise ValueError(f"Non-positive measurement max in {path}: {meas_max}")
    meas = meas / meas_max * 0.8
    return torch.from_numpy(meas).float().unsqueeze(0).to(device), meas_max


def final_output(output):
    if isinstance(output, (list, tuple)):
        return output[-1]
    return output


def save_preview_png(res_hwc, png_path):
    try:
        import imageio.v2 as imageio
    except Exception as exc:
        print(f"[preview skip] imageio unavailable: {exc}")
        return
    rgb_idx = [10, 16, 24]
    rgb_idx = [min(i, res_hwc.shape[2] - 1) for i in rgb_idx]
    rgb = res_hwc[:, :, rgb_idx].astype(np.float32)
    lo, hi = np.percentile(rgb, [1, 99])
    rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)
    imageio.imwrite(str(png_path), (rgb * 255.0 + 0.5).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--real-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--num-scenes", type=int, default=5)
    parser.add_argument("--save-preview", action="store_true")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_config(args.config)
    bands = int(cfg.get("num_bands", 28))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[real] config={args.config}")
    print(f"[real] ckpt={args.ckpt}")
    print(f"[real] out={out_dir}")
    print(f"[real] device={device} visible_gpu={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    model = build_model(cfg).to(device).eval()
    meta = load_checkpoint(model, args.ckpt, device)
    print(f"[real] ckpt_meta epoch={meta.get('epoch')} metrics={meta.get('metrics')}")

    meas_dir, mask_path = resolve_real_paths(args.real_root)
    mask = load_mask(mask_path, bands, device)
    print(f"[real] mask={tuple(mask.shape)}")

    rows = []
    with torch.no_grad():
        for idx in range(1, args.num_scenes + 1):
            scene_path = meas_dir / f"scene{idx}.mat"
            if not scene_path.exists():
                print(f"[real] skip missing {scene_path}")
                continue
            meas, meas_max = load_measurement(scene_path, device)
            pred = final_output(model(meas, mask)).clamp(0.0, 1.0)
            res = pred.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
            save_path = out_dir / f"{idx - 1}.mat"
            sio.savemat(str(save_path), {"res": res})
            if args.save_preview:
                save_preview_png(res, out_dir / f"{idx - 1}_rgb.png")
            row = {
                "scene": idx,
                "measurement": str(scene_path),
                "meas_max_before_norm": meas_max,
                "meas_shape": list(meas.shape),
                "res_shape": list(res.shape),
                "output": str(save_path),
            }
            rows.append(row)
            print(f"[real] scene{idx}: meas={tuple(meas.shape)} res={res.shape} -> {save_path}")

    with open(out_dir / "run_meta.json", "w", encoding="utf-8") as fp:
        json.dump({
            "config": str(args.config),
            "ckpt": str(args.ckpt),
            "real_root": str(args.real_root),
            "ckpt_epoch": meta.get("epoch"),
            "ckpt_metrics": meta.get("metrics"),
            "rows": rows,
        }, fp, ensure_ascii=False, indent=2)
    with open(out_dir / "scenes.csv", "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=[
            "scene", "measurement", "meas_max_before_norm", "meas_shape", "res_shape", "output"
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[real] done: {len(rows)} scenes")


if __name__ == "__main__":
    main()


