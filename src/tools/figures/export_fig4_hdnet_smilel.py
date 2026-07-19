#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Export only HDNet and SMILE-L predictions for Scene05/Scene07 figure refresh.

This intentionally does not overwrite the existing figure caches. Outputs:
  /tmp/smile2_fig4_hdnet_smilel/scene05/*.npy
  /tmp/smile2_fig4_hdnet_smilel/scene07/*.npy
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT_OUT = Path("/tmp/smile2_fig4_hdnet_smilel")
SAM_CANDIDATES = [Path("/tmp/sam_e2e_eval_runtime"), Path("/tmp/sam_e2e_eval")]
SAM = next(p for p in SAM_CANDIDATES if (p / "test_sota.py").exists())
E2E = Path(os.environ.get("SMILE_E2E_ROOT", "./src/repo")).resolve()
SMILE_L_CFG = E2E / "configs/runtime_friend/progressive_244_step2_noshare_sfe.yaml"
SMILE_L_CKPT = E2E / "result/model/2026_07_14_19_31_08_progressive_244_step2_noshare_sfe/best_psnr.pth"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


os.chdir(str(SAM))
sys.path.insert(0, str(SAM))
ts = load_module("sam_test_sota_fig4", SAM / "test_sota.py")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("[export] SAM", SAM, "device", device, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", flush=True)

mask3d = ts.load_mask(ts.cfg["mask_path"], nC=ts.nC)


def load_scene(scene_no: int):
    scene_name = f"scene{scene_no:02d}"
    scene_path = Path(ts.cfg["test_path"]) / f"{scene_name}.npy"
    scene_np = np.load(str(scene_path)).astype("float32")
    if scene_np.ndim == 3 and scene_np.shape[-1] == ts.nC:
        scene_np = np.transpose(scene_np, (2, 0, 1))
    test_data = torch.from_numpy(scene_np).unsqueeze(0)
    meas_raw, shift_mask, PhiPhiT, meas_H, mask3d_batch = ts.prepare_inputs(test_data, mask3d, step=ts.STEP)
    full_inputs = dict(
        meas_raw=meas_raw[:1],
        shift_mask=shift_mask[:1],
        PhiPhiT=PhiPhiT[:1],
        meas_H=meas_H[:1],
        mask3d_batch=mask3d_batch[:1],
        gt_cuda=test_data[:1].cuda(),
    )
    return scene_name, test_data, full_inputs


def save_pred(out_dir: Path, name: str, pred):
    arr = pred[0].detach().float().cpu().numpy().clip(0, 1).astype("float32")
    np.save(out_dir / f"{name}.npy", arr)
    print("[export] saved", out_dir.name, name, arr.shape, float(arr.min()), float(arr.max()), flush=True)


def build_hdnet():
    registry = {name: (build, kw, fwd) for name, build, kw, fwd in ts.ALL_MODELS}
    build, kw, fwd = registry["hdnet"]
    print("[export] building hdnet", flush=True)
    model = build(**kw)
    ts._core_model(model).eval()
    return model, fwd


def build_smile_l():
    print("[export] building smile_l", flush=True)
    old_cwd = os.getcwd()
    os.chdir(str(E2E))
    os.environ["SMILE_CONFIG"] = str(SMILE_L_CFG)
    if str(E2E) not in sys.path:
        sys.path.insert(0, str(E2E))
    from train import build_model as smile_build_model
    from model.e2e import cassi_measure, phi_phi_t, shift_cube

    model = smile_build_model().to(device).eval()
    ckpt = torch.load(str(SMILE_L_CKPT), map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=True)
    print("[export] smile_l missing", len(missing), "unexpected", len(unexpected), flush=True)
    os.chdir(old_cwd)
    return model, cassi_measure, phi_phi_t, shift_cube


def main():
    ROOT_OUT.mkdir(parents=True, exist_ok=True)
    hdnet, hdnet_fwd = build_hdnet()
    smile_l, cassi_measure, phi_phi_t, shift_cube = build_smile_l()

    meta = {"models": ["hdnet", "smile_l"], "smile_l_ckpt": str(SMILE_L_CKPT), "smile_l_config": str(SMILE_L_CFG)}
    for scene_no in [5, 7]:
        scene_name, test_data, full_inputs = load_scene(scene_no)
        out_dir = ROOT_OUT / scene_name
        out_dir.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            pred = hdnet_fwd(hdnet, **full_inputs)
        save_pred(out_dir, "hdnet", pred)

        gt = test_data[:1].to(device).float()
        mask_b = mask3d.unsqueeze(0).to(device).float()
        y = cassi_measure(gt, mask_b)
        shifted = shift_cube(mask_b)
        ppt = phi_phi_t(mask_b)
        with torch.no_grad():
            pred = smile_l(y, mask_b, shifted, ppt)
            if isinstance(pred, (list, tuple)):
                pred = pred[-1]
        save_pred(out_dir, "smile_l", pred)
        meta[scene_name] = {"output": str(out_dir)}

    (ROOT_OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[export] done", ROOT_OUT, flush=True)


if __name__ == "__main__":
    main()



