#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Export HDNet and SMILE-L predictions for TSA real scene4 256 crop.

This complements the existing real4 E2E cache by replacing BIRNAT with HDNet
and adding SMILE-L, without regenerating every baseline.
"""
import os
import sys
import json
import importlib.util
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

OUT = Path("/tmp/smile2_real4_hdnet_smilel_256")
OUT.mkdir(parents=True, exist_ok=True)

SAM = Path("/tmp/sam_e2e_eval")
E2E = Path(os.environ.get("SMILE_E2E_ROOT", "./src/repo")).resolve()
REAL = Path(os.environ.get("SMILE_REAL_DATA", "./datasets/TSA_real_data")).resolve()

SMILE_L_CFG = E2E / "configs/runtime_friend/progressive_244_step2_noshare_sfe.yaml"
SMILE_L_CKPT = E2E / "result/model/2026_07_14_19_31_08_progressive_244_step2_noshare_sfe/best_psnr.pth"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def save_pred(out_dir, name, pred, h0=256, w0=256, meta=None):
    arr = pred[0].detach().float().cpu().numpy().clip(0, 1).astype("float32")
    if arr.shape[-2] >= h0 and arr.shape[-1] >= w0:
        arr = arr[:, :h0, :w0]
    else:
        raise RuntimeError(f"{name} bad shape {arr.shape}, expected at least {(h0, w0)}")
    np.save(out_dir / f"{name}.npy", arr)
    if meta is not None:
        meta["models"].append(name)
    print("[real-export] saved", name, arr.shape, float(arr.min()), float(arr.max()), flush=True)


def main():
    os.chdir(str(SAM))
    sys.path.insert(0, str(SAM))
    ts = load_module("sam_test_sota_real_hdnet", SAM / "test_sota.py")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("[real-export] device", device, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", flush=True)

    nC, STEP = ts.nC, ts.STEP
    meas = sio.loadmat(str(REAL / "Measurements/scene4.mat"))["meas_real"].astype("float32")
    if meas.ndim == 3:
        meas = np.squeeze(meas)
    meas = meas / max(float(meas.max()), 1e-8) * 0.8

    mask2d = sio.loadmat(str(REAL / "mask.mat"))["mask"].astype("float32")
    if mask2d.ndim == 3:
        mask2d = mask2d[:, :, 0]

    crop = 256
    y0 = 200
    x0 = 120
    wp0 = crop + (nC - 1) * STEP
    meas_crop = meas[y0:y0 + crop, x0:x0 + wp0]
    mask_crop = mask2d[y0:y0 + crop, x0:x0 + crop]

    mask3d_np = np.tile(mask_crop[:, :, None], (1, 1, nC)).transpose(2, 0, 1).astype("float32")
    meas_raw = torch.from_numpy(meas_crop).unsqueeze(0).to(device).float()
    mask3d = torch.from_numpy(mask3d_np).to(device).float()
    mask3d_batch = mask3d.unsqueeze(0)

    shift_mask = torch.zeros(1, nC, crop, wp0, device=device)
    for c in range(nC):
        shift_mask[:, c, :, c * STEP:c * STEP + crop] = mask3d_batch[:, c]
    PhiPhiT = shift_mask.sum(1).clamp(1.0)
    meas_norm = meas_raw / nC * 2
    meas_H = torch.zeros(1, nC, crop, crop, device=device)
    for c in range(nC):
        meas_H[:, c] = meas_norm[:, :, c * STEP:c * STEP + crop]

    # keep the shared input artefacts together with the new predictions
    np.save(OUT / "measurement.npy", meas_crop.astype("float32"))
    np.save(OUT / "measurement_full.npy", meas.astype("float32"))
    np.save(OUT / "mask.npy", mask3d_np)
    meta = {
        "scene": "real_scene4",
        "crop": {"x0": int(x0), "y0": int(y0), "size": int(crop)},
        "models": [],
        "smile_l_ckpt": str(SMILE_L_CKPT),
        "smile_l_config": str(SMILE_L_CFG),
    }
    full_inputs = dict(
        meas_raw=meas_raw,
        shift_mask=shift_mask,
        PhiPhiT=PhiPhiT,
        meas_H=meas_H,
        mask3d_batch=mask3d_batch,
    )

    registry = {name: (build, kw, fwd) for name, build, kw, fwd in ts.ALL_MODELS}
    try:
        build, kw, fwd = registry["hdnet"]
        print("[real-export] building hdnet", flush=True)
        model = build(**kw)
        ts._core_model(model).eval()
        with torch.no_grad():
            pred = fwd(model, **full_inputs)
        save_pred(OUT, "hdnet", pred, meta=meta)
        del model, pred
        torch.cuda.empty_cache()
    except Exception as e:
        print("[real-export][skip] hdnet", repr(e), flush=True)

    try:
        print("[real-export] building smile_l", flush=True)
        old = os.getcwd()
        os.chdir(str(E2E))
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["SMILE_CONFIG"] = str(SMILE_L_CFG.relative_to(E2E))
        sys.path.insert(0, str(E2E))
        from train import build_model as smile_build_model

        model = smile_build_model().to(device).eval()
        ckpt = torch.load(str(SMILE_L_CKPT), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=True)
        mask_b = torch.from_numpy(mask3d_np).unsqueeze(0).to(device).float()
        with torch.no_grad():
            pred = model(meas_raw, mask_b)
            if isinstance(pred, (list, tuple)):
                pred = pred[-1]
        save_pred(OUT, "smile_l", pred.clamp(0, 1), meta=meta)
        os.chdir(old)
    except Exception as e:
        print("[real-export][skip] smile_l", repr(e), flush=True)

    with open(OUT / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("[real-export] done", OUT, meta, flush=True)


if __name__ == "__main__":
    main()



