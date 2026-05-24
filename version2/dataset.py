"""
dataset.py — 数据加载、mask 生成、CASSI 仿真、数据增强

支持 .npy 和 .mat 两种格式，优先读 .npy。
数据路径从 config.yaml 读取。
"""

import json
import os
import random
import numpy as np
import torch
import scipy.io as sio
import yaml
from pathlib import Path

import time

# 从 config.yaml 读取路径
_cfg_path = Path(__file__).parent / 'config.yaml'
with open(_cfg_path, 'r') as _f:
    _cfg = yaml.safe_load(_f)

DATA_ROOT  = Path(_cfg['data_root'])
TRAIN_PATH = Path(_cfg['train_path'])
TEST_PATH  = Path(_cfg['test_path'])
MASK_PATH  = Path(_cfg['mask_path'])


# ──────────────────────────────────────────────
# 基础 CASSI 操作
# ──────────────────────────────────────────────

def shift(inputs, step=2):
    """[B, nC, H, W] -> [B, nC, H, W+(nC-1)*step]"""
    B, nC, H, W = inputs.shape
    out = torch.zeros(B, nC, H, W + (nC - 1) * step, device=inputs.device, dtype=inputs.dtype)
    for i in range(nC):
        out[:, i, :, step * i: step * i + W] = inputs[:, i, :, :]
    return out


def shift_back(inputs, step=2, nC=28):
    """[B, H, W_shifted] -> [B, nC, H, W]"""
    B, H, W_shifted = inputs.shape
    W = W_shifted - (nC - 1) * step
    out = torch.zeros(B, nC, H, W, device=inputs.device, dtype=inputs.dtype)
    for i in range(nC):
        out[:, i, :, :] = inputs[:, :, step * i: step * i + W]
    return out


def shift_3d(mask3d, step=2):
    """[nC, H, W] -> [nC, H, W+(nC-1)*step]"""
    nC, H, W = mask3d.shape
    out = torch.zeros(nC, H, W + (nC - 1) * step, device=mask3d.device, dtype=mask3d.dtype)
    for c in range(nC):
        out[c, :, c * step: c * step + W] = mask3d[c, :, :]
    return out


def gen_meas(gt, mask3d, input_setting='H'):
    """在线生成 CASSI 测量值。"""
    nC = gt.shape[1]
    masked = mask3d * gt
    shifted = shift(masked, step=2)
    meas = torch.sum(shifted, dim=1)

    if input_setting == 'Y':
        return meas
    meas_norm = meas / nC * 2
    H_out = shift_back(meas_norm, step=2, nC=nC)
    if input_setting == 'HM':
        return H_out * mask3d
    return H_out


def gen_meas_unfolding(gt, mask3d, step=2):
    """生成 unfolding 模型的原始测量值 g。"""
    nC, H, W = gt.shape
    masked = mask3d * gt
    shifted = shift_3d(masked, step)
    g = torch.sum(shifted, dim=0, keepdim=True)

    mask_shifted = shift_3d(mask3d, step)
    PhiPhiT = torch.sum(mask_shifted ** 2, dim=0, keepdim=True)
    PhiPhiT[PhiPhiT == 0] = 1.0

    return g, PhiPhiT


# ──────────────────────────────────────────────
# Mask 加载
# ──────────────────────────────────────────────

def load_mask(mask_path=None, nC=28):
    """加载 mask 文件，返回 mask3d [nC, H, W] tensor。"""
    if mask_path is None:
        mask_path = str(MASK_PATH)
    path = Path(mask_path)
    if path.is_dir():
        npy = path / 'mask.npy'
        mat = path / 'mask.mat'
        if npy.exists():
            path = npy
        elif mat.exists():
            path = mat
        else:
            raise FileNotFoundError(f"mask.npy 或 mask.mat 不存在于 {mask_path}")

    if path.suffix == '.npy':
        mask = np.load(str(path)).astype(np.float32)
    else:
        data = sio.loadmat(str(path))
        mask = data['mask'].astype(np.float32)

    mask3d = np.tile(mask[:, :, np.newaxis], (1, 1, nC))
    mask3d = np.transpose(mask3d, (2, 0, 1))
    return torch.from_numpy(mask3d)


def prepare_mask(mask3d, batch_size, device='cuda'):
    """将 mask3d [nC, H, W] 扩展为训练用的 batch。"""
    nC, H, W = mask3d.shape
    mask3d_batch = mask3d.unsqueeze(0).expand(batch_size, -1, -1, -1).to(device).float()
    shift_mask = shift(mask3d_batch, step=2)
    return mask3d_batch, shift_mask


# ──────────────────────────────────────────────
# 数据加载
# ──────────────────────────────────────────────

def load_training(data_path=None, max_scenes=205, wavelengths=None):
    """加载训练集，返回 list of ndarray，每个 (H, W, nC) float32 [0,1]。"""
    if data_path is None:
        data_path = str(TRAIN_PATH)
    start_time = time.time()
    print(f"starting at: {time.ctime(start_time)}")

    path = Path(data_path)
    data_name = path.name

    info_path  = path / f'{data_name}_info.json'
    bands_path = path / f'{data_name}_bands.json'

    need_scan = not (info_path.exists() and bands_path.exists())
    if not need_scan:
        print(f"[dataset] JSON 已存在，跳过扫描: {info_path.name} / {bands_path.name}")

    files = sorted(
        path.iterdir(),
        key=lambda x: int(''.join(c for c in x.stem if c.isdigit()) or '0')
    )

    imgs = []
    band_sum   = None
    scene_info = []

    for f in files:
        if f.suffix not in ('.npy', '.mat'):
            continue
        digits = ''.join(c for c in f.stem if c.isdigit())
        if not digits:
            continue
        if int(digits) > max_scenes:
            continue

        try:
            print(f"Loading: {f.name}")
            if f.suffix == '.npy':
                img = np.load(str(f)).astype(np.float32)
            else:
                d = sio.loadmat(str(f))
                if 'img_expand' in d:
                    img = (d['img_expand'] / 65536.).astype(np.float32)
                else:
                    img = (d['img'] / 65536.).astype(np.float32)
            imgs.append(img)

            if need_scan:
                nC_img = img.shape[-1]
                if band_sum is None:
                    band_sum = np.zeros(nC_img, dtype=np.float64)
                band_sum += img.mean(axis=(0, 1))

                wl_min = int(wavelengths[0])           if wavelengths else None
                wl_max = int(wavelengths[nC_img - 1])  if wavelengths and len(wavelengths) >= nC_img else None
                scene_info.append({
                    'filename':         f.name,
                    'num_bands':        nC_img,
                    'wavelength_min_nm': wl_min,
                    'wavelength_max_nm': wl_max,
                })

        except Exception as e:
            print(f"  [WARNING] 加载失败 {f.name}: {e}")

    print(f"训练集加载完成：{len(imgs)} 个场景，路径: {data_path}")
    print(f"finished after {time.time() - start_time:.2f} s")

    if need_scan and imgs:
        nC_final  = imgs[0].shape[-1]
        band_mean = (band_sum / len(imgs)).tolist()

        info_data = {
            'data_name':  data_name,
            'num_scenes': len(imgs),
            'num_bands':  nC_final,
            'scenes':     scene_info,
        }
        with open(info_path, 'w', encoding='utf-8') as fp:
            json.dump(info_data, fp, indent=2, ensure_ascii=False)
        print(f"[dataset] 已写入 {info_path}")

        wl_list = [int(w) for w in wavelengths[:nC_final]] if wavelengths else None
        bands_data = {
            'data_name':      data_name,
            'num_bands':      nC_final,
            'wavelengths_nm': wl_list,
            'band_mean':      [round(v, 4) for v in band_mean],
        }
        with open(bands_path, 'w', encoding='utf-8') as fp:
            json.dump(bands_data, fp, indent=2, ensure_ascii=False)
        print(f"[dataset] 已写入 {bands_path}")

    return imgs


def load_test(test_path=None, nC=28):
    """加载测试集，返回 tensor [N, nC, H, W] float32。"""
    if test_path is None:
        test_path = str(TEST_PATH)
    path = Path(test_path)
    files = sorted(f for f in path.iterdir() if f.suffix in ('.npy', '.mat'))

    scenes = []
    for f in files:
        if f.suffix == '.npy':
            img = np.load(str(f)).astype(np.float32)
        else:
            img = sio.loadmat(str(f))['img'].astype(np.float32)

        if img.ndim == 3 and img.shape[-1] == nC:
            img = np.transpose(img, (2, 0, 1))
        scenes.append(img)

    data = np.stack(scenes, axis=0)
    return torch.from_numpy(data)


# ──────────────────────────────────────────────
# 数据增强
# ──────────────────────────────────────────────

def _augment(x):
    rot = random.randint(0, 3)
    if rot:
        x = torch.rot90(x, rot, dims=(1, 2))
    if random.randint(0, 1):
        x = torch.flip(x, dims=(1,))
    if random.randint(0, 1):
        x = torch.flip(x, dims=(2,))
    return x


def _splice_four(train_data, crop_half, nC=28):
    crop = crop_half * 2
    indices = np.random.randint(0, len(train_data), 4)
    patches = []
    for idx in indices:
        img = train_data[idx]
        h, w, _ = img.shape
        x0 = np.random.randint(0, h - crop_half)
        y0 = np.random.randint(0, w - crop_half)
        p = img[x0:x0+crop_half, y0:y0+crop_half, :]
        patches.append(torch.from_numpy(np.transpose(p, (2, 0, 1))).float())

    out = torch.zeros(nC, crop, crop)
    out[:, :crop_half, :crop_half] = patches[0]
    out[:, :crop_half, crop_half:] = patches[1]
    out[:, crop_half:, :crop_half] = patches[2]
    out[:, crop_half:, crop_half:] = patches[3]
    return out


def shuffle_crop(train_data, batch_size, crop_size=256, augment=True, device='cuda', nC=28):
    half = batch_size // 2
    gt_batch = []

    indices = np.random.choice(len(train_data), half, replace=True)
    for idx in indices:
        img = train_data[idx]
        h, w, _ = img.shape
        x0 = np.random.randint(0, h - crop_size)
        y0 = np.random.randint(0, w - crop_size)
        patch = img[x0:x0+crop_size, y0:y0+crop_size, :]
        t = torch.from_numpy(np.transpose(patch, (2, 0, 1))).float()
        if augment:
            t = _augment(t)
        gt_batch.append(t)

    for _ in range(batch_size - half):
        t = _splice_four(train_data, crop_size // 2, nC=nC)
        if augment:
            t = _augment(t)
        gt_batch.append(t)

    return torch.stack(gt_batch, dim=0).to(device).float()
