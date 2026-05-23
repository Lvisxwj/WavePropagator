"""
test.py — WaveMST 测试入口

修改 CONFIG 区域，然后运行：
    python test.py
"""

import json
import os
import time
import torch
import numpy as np
from pathlib import Path

import physics   # 用于 init_wavelengths


# ════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════
MODEL_INDEX   = 1      # 0: WaveMST_3D  1: WaveMST_KG  2: WaveMST_Parallel  3: WaveMST_Mamba  4: WaveMST_Phys  5: Helmholtzformer  6: WaveMST_Helm
CHECKPOINT    = 'result/model/2026_05_02_00_40_50_3d_wpo_kg/best.pth'
# /data5/SCI/xieweijie/CASSI/result/model/2026_05_05_13_01_33_h2_gamma_main
GPU_ID        = '0'
INPUT_SETTING = 'H'
NUM_BANDS     = 28       # 默认值，main() 中从 bands.json 覆盖
DIM           = 28
STAGE         = 2
NUM_BLOCKS    = [2, 2, 2]
MASK_MODE     = 'D'      # 'A' / 'B' / 'D'（需与训练时一致）
FUSION        = 'gate'   # 仅 Model 2 有效：'gate' / 'add' / 'linear'

DATA_ROOT   = Path('./dataset')
TRAIN_PATH  = DATA_ROOT / 'CAVE_1024_npy'           # 用于定位 bands.json
TRAIN_PATH_FALLBACK = DATA_ROOT / 'CAVE_1024' / 'cave_1024_28'
TEST_PATH   = DATA_ROOT / 'TSA_simu_data' / 'Truth'
MASK_PATH   = DATA_ROOT / 'TSA_simu_data'
RESULT_ROOT = Path('./result')
# ════════════════════════════════════════════

os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

MODELS = {
    0: ('WaveMST_3D',       '3d_wpo_pure'),
    1: ('WaveMST_KG',       '3d_wpo_kg'),
    2: ('WaveMST_Parallel', '3d_wpo_smsa'),
    3: ('WaveMST_Mamba',    '2d_wpo_mamba'),
    4: ('WaveMST_Phys',     'h2_alpha_phys'),
    5: ('Helmholtzformer',  'h1_gamma_helm_pure'),
    6: ('WaveMST_Helm',     'h2_gamma_main'),
}


def build_model(index):
    if index == 0:
        from wpo3d import WaveMST_3D
        return WaveMST_3D(dim=DIM, stage=STAGE, num_blocks=NUM_BLOCKS,
                          mask_mode=MASK_MODE)
    elif index == 1:
        from wpo3d import WaveMST_KG
        return WaveMST_KG(dim=DIM, stage=STAGE, num_blocks=NUM_BLOCKS)
    elif index == 2:
        from wpo_smsa import WaveMST_Parallel
        return WaveMST_Parallel(dim=DIM, stage=STAGE, num_blocks=NUM_BLOCKS,
                                mask_mode=MASK_MODE, fusion=FUSION)
    elif index == 3:
        from wpo_mamba import WaveMST_Mamba
        return WaveMST_Mamba(dim=DIM, stage=STAGE, num_blocks=NUM_BLOCKS,
                             mask_mode=MASK_MODE)
    elif index == 4:
        from wpo3d_phys import WaveMST_Phys
        return WaveMST_Phys(dim=DIM, stage=STAGE, num_blocks=NUM_BLOCKS,
                            mask_mode=MASK_MODE)
    elif index == 5:
        from helm_pure import Helmholtzformer
        return Helmholtzformer(dim=DIM, stage=STAGE, num_blocks=NUM_BLOCKS,
                               mask_mode=MASK_MODE)
    elif index == 6:
        from wpo3d_helm import WaveMST_Helm
        return WaveMST_Helm(dim=DIM, stage=STAGE, num_blocks=NUM_BLOCKS,
                            mask_mode=MASK_MODE)
    raise ValueError(f"无效 MODEL_INDEX: {index}")


def main():
    global NUM_BANDS, DIM

    from dataset import load_test, load_mask, prepare_mask, gen_meas
    from loss import (torch_psnr, torch_ssim, torch_sam,
                      torch_freq_amp_err, torch_freq_band_err, count_params)

    # ── 从 bands.json 读取 NUM_BANDS，同步 physics.WAVELENGTHS ──
    train_path = TRAIN_PATH if TRAIN_PATH.exists() else TRAIN_PATH_FALLBACK
    bands_path = train_path / f'{train_path.name}_bands.json'
    if bands_path.exists():
        with open(bands_path, 'r', encoding='utf-8') as fp:
            bdata = json.load(fp)
        NUM_BANDS = bdata.get('num_bands', NUM_BANDS)
        DIM = NUM_BANDS
        physics.init_wavelengths(str(bands_path))
        print(f"[test] NUM_BANDS={NUM_BANDS}（来自 {bands_path.name}）")
    else:
        print(f"[test] 警告：{bands_path} 不存在，使用默认 NUM_BANDS={NUM_BANDS}")

    # ── 模型 ──
    model = build_model(MODEL_INDEX).cuda()
    model.load_state_dict(torch.load(CHECKPOINT, map_location='cuda'))
    model.eval()
    print(f"模型: {MODELS[MODEL_INDEX][0]}  参数量: {count_params(model):.2f}M")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"MASK_MODE: {MASK_MODE}")

    # ── 数据 ──
    test_data = load_test(str(TEST_PATH), nC=NUM_BANDS).cuda().float()
    N = test_data.shape[0]
    mask3d = load_mask(str(MASK_PATH), nC=NUM_BANDS).cuda()
    mask3d_test, shift_mask_test = prepare_mask(mask3d, N)

    # ── 推理 ──
    with torch.no_grad():
        meas = gen_meas(test_data, mask3d_test, INPUT_SETTING)
        pred = model(meas, shift_mask_test)   # [N, nC, H, W]

    # ── 评估 ──
    psnr_list, ssim_list, sam_list = [], [], []
    freq_amp_list, low_freq_list, high_freq_list = [], [], []

    for i in range(N):
        p = pred[i]
        g = test_data[i]
        psnr_list.append(torch_psnr(p, g).item())
        ssim_list.append(torch_ssim(p, g).item())
        sam_list.append(torch_sam(p, g).item())
        freq_amp_list.append(torch_freq_amp_err(p, g).item())
        fd = torch_freq_band_err(p, g)
        low_freq_list.append(fd['low_freq_err'])
        high_freq_list.append(fd['high_freq_err'])

        print(f"  Scene {i+1:02d}: PSNR={psnr_list[-1]:.2f}  "
              f"SSIM={ssim_list[-1]:.4f}  SAM={sam_list[-1]:.4f}  "
              f"FreqAmpErr={freq_amp_list[-1]:.5f}")

    def _mean(lst): return sum(lst) / len(lst)

    print(f"\n  平均:    PSNR={_mean(psnr_list):.2f}  "
          f"SSIM={_mean(ssim_list):.4f}  SAM={_mean(sam_list):.4f}")
    print(f"           FreqAmpErr={_mean(freq_amp_list):.5f}  "
          f"LowFreqErr={_mean(low_freq_list):.5f}  "
          f"HighFreqErr={_mean(high_freq_list):.5f}")

    # ── 保存结果 ──
    time_str = time.strftime('%Y_%m_%d_%H_%M_%S')
    out_dir = RESULT_ROOT / 'show' / f"{time_str}_{MODELS[MODEL_INDEX][1]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(out_dir / 'pred.npy'), pred.cpu().numpy())
    np.save(str(out_dir / 'gt.npy'),   test_data.cpu().numpy())
    print(f"\n结果保存到: {out_dir}")


if __name__ == '__main__':
    main()
