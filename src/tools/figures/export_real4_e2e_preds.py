#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys, json, importlib.util
from pathlib import Path
import numpy as np
import torch
import scipy.io as sio

OUT = Path('/tmp/smile2_real4_e2e_preds')
OUT.mkdir(parents=True, exist_ok=True)
SAM = Path('/tmp/sam_e2e_eval')
E2E = Path('.')
REAL = Path('./datasets/TSA_real_data')

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

os.chdir(str(SAM))
sys.path.insert(0, str(SAM))
ts = load_module('sam_test_sota_real', SAM / 'test_sota.py')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('[real-export] device', device, torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu', flush=True)

nC, STEP = ts.nC, ts.STEP
meas = sio.loadmat(str(REAL / 'Measurements/scene4.mat'))['meas_real'].astype('float32')
if meas.ndim == 3:
    meas = np.squeeze(meas)
meas = meas / max(float(meas.max()), 1e-8) * 0.8
mask2d = sio.loadmat(str(REAL / 'mask.mat'))['mask'].astype('float32')
if mask2d.ndim == 3:
    mask2d = mask2d[:, :, 0]
H_full, Wp_full = meas.shape
W_full = Wp_full - (nC - 1) * STEP
# Use a common 512x512 crop for real visual comparison; most E2E baselines
# are trained/evaluated on window-friendly sizes and fail on raw 660x660.
crop = 512
y0 = (H_full - crop) // 2
x0 = (W_full - crop) // 2
H0 = crop
W0 = crop
Wp0 = crop + (nC - 1) * STEP
meas_crop = meas[y0:y0+crop, x0:x0+Wp0]
mask_crop = mask2d[y0:y0+crop, x0:x0+crop]
Hp, W, Wp = crop, crop, Wp0
meas_pad = meas_crop.astype('float32')
mask_pad = mask_crop.astype('float32')
mask3d_np = np.tile(mask_pad[:, :, None], (1, 1, nC)).transpose(2,0,1).astype('float32')
meas_raw = torch.from_numpy(meas_pad).unsqueeze(0).to(device).float()
mask3d = torch.from_numpy(mask3d_np).to(device).float()
mask3d_batch = mask3d.unsqueeze(0)
shift_mask = torch.zeros(1, nC, Hp, Wp, device=device)
for c in range(nC):
    shift_mask[:, c, :, c*STEP:c*STEP+W] = mask3d_batch[:, c]
PhiPhiT = shift_mask.sum(1).clamp(1.0)
meas_norm = meas_raw / nC * 2
meas_H = torch.zeros(1, nC, Hp, W, device=device)
for c in range(nC):
    meas_H[:, c] = meas_norm[:, :, c*STEP:c*STEP+W]

np.save(OUT / 'measurement.npy', meas_crop.astype('float32'))
np.save(OUT / 'measurement_full.npy', meas.astype('float32'))
np.save(OUT / 'mask.npy', mask3d_np)
meta = {'scene': 'real_scene4', 'crop': {'x0': int(x0), 'y0': int(y0), 'size': int(crop)}, 'models': []}
full_inputs = dict(meas_raw=meas_raw, shift_mask=shift_mask, PhiPhiT=PhiPhiT, meas_H=meas_H, mask3d_batch=mask3d_batch)

def save_pred(name, pred):
    arr = pred[0].detach().float().cpu().numpy().clip(0,1).astype('float32')
    if arr.shape[-2] >= H0 and arr.shape[-1] >= W0:
        arr = arr[:, :H0, :W0]
    else:
        print('[real-export][skip]', name, 'bad_shape', arr.shape, 'expected at least', (H0, W0), flush=True)
        return
    np.save(OUT / f'{name}.npy', arr)
    meta['models'].append(name)
    print('[real-export] saved', name, arr.shape, float(arr.min()), float(arr.max()), flush=True)

allowed = ['lambda_net', 'tsa_net', 'mst_l', 'cst_l', 'birnat']
registry = {name: (build, kw, fwd) for name, build, kw, fwd in ts.ALL_MODELS}
for name in allowed:
    try:
        build, kw, fwd = registry[name]
        print('[real-export] building', name, flush=True)
        model = build(**kw)
        ts._core_model(model).eval()
        with torch.no_grad():
            pred = fwd(model, **full_inputs)
        save_pred(name, pred)
        del model, pred
        torch.cuda.empty_cache()
    except Exception as e:
        print('[real-export][skip]', name, repr(e), flush=True)

try:
    print('[real-export] building dgsmp', flush=True)
    from architecture.DGSMP import HSI_CS
    model = HSI_CS(28, 4).to(device).eval()
    ck = torch.load(str(SAM / 'pth/dgsmp/dgsmp.pth'), map_location=device, weights_only=False)
    state = ck.get('model', ck.get('state_dict', ck)) if isinstance(ck, dict) else ck
    if any(k.startswith('module.') for k in state.keys()):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    with torch.no_grad():
        pred = model(meas_raw)
        if isinstance(pred, (list, tuple)):
            pred = pred[-1]
    save_pred('dgsmp', pred.clamp(0,1))
except Exception as e:
    print('[real-export][skip] dgsmp', repr(e), flush=True)

try:
    print('[real-export] building smile_m', flush=True)
    old = os.getcwd(); os.chdir(str(E2E))
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['SMILE_CONFIG'] = 'configs/runtime_friend/progressive_222_step2_estimate_evolve_v2.yaml'
    sys.path.insert(0, str(E2E))
    from train import build_model as smile_build_model
    model = smile_build_model().to(device).eval()
    ckpt = torch.load(str(E2E / 'result/model/2026_07_13_17_46_16_progressive_222_step2_estimate_evolve_v2/best_psnr.pth'), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'], strict=True)
    mask_b = torch.from_numpy(mask3d_np).unsqueeze(0).to(device).float()
    with torch.no_grad():
        pred = model(meas_raw, mask_b)
        if isinstance(pred, (list, tuple)):
            pred = pred[-1]
    save_pred('smile_m', pred.clamp(0,1))
    os.chdir(old)
except Exception as e:
    print('[real-export][skip] smile_m', repr(e), flush=True)

with open(OUT / 'meta.json', 'w', encoding='utf-8') as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print('[real-export] done', OUT, meta, flush=True)


