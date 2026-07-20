#!/usr/bin/env python
import os, sys, json, importlib.util
from pathlib import Path
import numpy as np
import torch

OUT = Path('/tmp/smile2_scene7_e2e_preds')
OUT.mkdir(parents=True, exist_ok=True)
SAM = Path('/tmp/sam_e2e_eval')
E2E = Path('.')
PY_E2E = str(E2E)

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

os.chdir(str(SAM))
sys.path.insert(0, str(SAM))
ts = load_module('sam_test_sota', SAM / 'test_sota.py')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('[export] device', device, torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu', flush=True)

# Load the exact simulated Scene 7 from the npy file to avoid mat/npy duplicate
# ordering in the generic loader.
scene_path = Path(ts.cfg['test_path']) / 'scene07.npy'
scene_np = np.load(str(scene_path)).astype('float32')
if scene_np.ndim == 3 and scene_np.shape[-1] == ts.nC:
    scene_np = np.transpose(scene_np, (2, 0, 1))
test_data = torch.from_numpy(scene_np).unsqueeze(0)
scene_names = ['scene07']
mask3d = ts.load_mask(ts.cfg['mask_path'], nC=ts.nC)
idx = 0
scene_name = 'scene07'
meas_raw, shift_mask, PhiPhiT, meas_H, mask3d_batch = ts.prepare_inputs(test_data, mask3d, step=ts.STEP)
full_inputs = dict(
    meas_raw=meas_raw[idx:idx+1], shift_mask=shift_mask[idx:idx+1], PhiPhiT=PhiPhiT[idx:idx+1],
    meas_H=meas_H[idx:idx+1], mask3d_batch=mask3d_batch[idx:idx+1], gt_cuda=test_data[idx:idx+1].cuda(),
)
np.save(OUT / 'gt.npy', test_data[idx].detach().cpu().numpy().astype('float32'))
np.save(OUT / 'measurement.npy', meas_raw[idx].detach().cpu().numpy().astype('float32'))
np.save(OUT / 'mask.npy', mask3d.detach().cpu().numpy().astype('float32'))
meta = {'scene_index': 7, 'scene_name': scene_name, 'models': []}

def save_pred(name, pred):
    arr = pred[0].detach().float().cpu().numpy().clip(0, 1).astype('float32')
    np.save(OUT / f'{name}.npy', arr)
    meta['models'].append(name)
    print('[export] saved', name, arr.shape, float(arr.min()), float(arr.max()), flush=True)

# Models registered in test_sota.py and allowed for this figure: E2E/single-pass only.
allowed = ['lambda_net', 'tsa_net', 'mst_l', 'cst_l', 'birnat']
registry = {name: (build, kw, fwd) for name, build, kw, fwd in ts.ALL_MODELS}
for name in allowed:
    try:
        build, kw, fwd = registry[name]
        print('[export] building', name, flush=True)
        model = build(**kw)
        ts._core_model(model).eval()
        with torch.no_grad():
            pred = fwd(model, **full_inputs)
        save_pred(name, pred)
        del model, pred
        torch.cuda.empty_cache()
    except Exception as e:
        print('[export][skip]', name, repr(e), flush=True)

# DGSMP: present in sam package but not registered by test_sota.
try:
    print('[export] building dgsmp', flush=True)
    from architecture.DGSMP import HSI_CS
    model = HSI_CS(28, 4).to(device).eval()
    ck = torch.load(str(SAM / 'pth/dgsmp/dgsmp.pth'), map_location=device, weights_only=False)
    state = ck.get('model', ck.get('state_dict', ck)) if isinstance(ck, dict) else ck
    if any(k.startswith('module.') for k in state.keys()):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print('[export] dgsmp missing', len(missing), 'unexpected', len(unexpected), flush=True)
    with torch.no_grad():
        # DGSMP consumes raw 2D CASSI measurement [N,H,W'] and internally y2x().
        pred = model(full_inputs['meas_raw'])
        if isinstance(pred, (list, tuple)):
            pred = pred[-1]
    save_pred('dgsmp', pred.clamp(0, 1))
    del model, pred
    torch.cuda.empty_cache()
except Exception as e:
    print('[export][skip] dgsmp', repr(e), flush=True)

# SMILE-M from e2e project.
try:
    print('[export] building smile_m', flush=True)
    old_cwd = os.getcwd()
    os.chdir(str(E2E))
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['SMILE_CONFIG'] = 'configs/runtime_friend/progressive_222_step2_estimate_evolve_v2.yaml'
    if str(E2E) not in sys.path:
        sys.path.insert(0, str(E2E))
    from dataset import load_mask as smile_load_mask
    from train import build_model as smile_build_model, cfg as smile_cfg
    from model.e2e import cassi_measure, phi_phi_t, shift_cube
    model = smile_build_model().to(device).eval()
    ckpt = torch.load(str(E2E / 'result/model/2026_07_13_17_46_16_progressive_222_step2_estimate_evolve_v2/best_psnr.pth'), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'], strict=True)
    gt = test_data[idx:idx+1].to(device).float()
    mask_b = mask3d.unsqueeze(0).to(device).float()
    y = cassi_measure(gt, mask_b)
    shifted = shift_cube(mask_b)
    ppt = phi_phi_t(mask_b)
    with torch.no_grad():
        pred = model(y, mask_b, shifted, ppt)
        if isinstance(pred, (list, tuple)):
            pred = pred[-1]
    save_pred('smile_m', pred.clamp(0, 1))
    os.chdir(old_cwd)
except Exception as e:
    print('[export][skip] smile_m', repr(e), flush=True)

with open(OUT / 'meta.json', 'w', encoding='utf-8') as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print('[export] done', OUT, meta, flush=True)

