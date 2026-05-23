"""
test.py — WaveMST 测试入口（stage4: ML-WPO 混合架构）

修改 CONFIG 区域，然后运行：
    python test.py
"""

import json
import os
import time
import torch
import numpy as np
from pathlib import Path

import physics


# ════════════════════════════════════════════
# CONFIG（需与训练时一致）
# ════════════════════════════════════════════
MODEL_INDEX   = 7
CHECKPOINT    = 'result/model/FILL_IN/best.pth'
GPU_ID        = '0'
INPUT_SETTING = 'H'
NUM_BANDS     = 28
DIM           = 28
STAGE         = 3
NUM_BLOCKS    = [2, 2, 2]
CROP_SIZE     = 256

# ML 层（需与训练时一致）
ML_TYPE      = 'dwconv_ca'
UNET_MODE    = 'symmetric'

# Unfolding（需与训练时一致）
NUM_STAGES          = 5
SHARE_STAGE_WEIGHTS = True

# 色散
USE_DISPERSIVE       = False
USE_DISPERSIVE_BLOCK = False

DATA_ROOT   = Path('../../dataset')
TRAIN_PATH  = DATA_ROOT / 'CAVE_1024_npy'
TRAIN_PATH_FALLBACK = DATA_ROOT / 'CAVE_1024' / 'cave_1024_28'
TEST_PATH   = DATA_ROOT / 'TSA_simu_data' / 'Truth'
MASK_PATH   = DATA_ROOT / 'TSA_simu_data'
RESULT_ROOT = Path('./result')
# ════════════════════════════════════════════

os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

IS_UNFOLDING = MODEL_INDEX >= 7

MODELS = {
    0: ('WaveMST_ML',          'ml_wpo_e2e'),
    1: ('WaveMST_ML_KG',       'ml_kg_e2e'),
    7: ('WaveMST_ML_Unfold',   'ml_wpo_unfold'),
    8: ('WaveMST_ML_KG_Unfold','ml_kg_unfold'),
}


def build_model(index):
    use_kg = index in [1, 8]

    if index in [0, 1]:
        from wpo3d import WaveMST_ML
        return WaveMST_ML(
            dim=DIM, stage=STAGE, num_blocks=NUM_BLOCKS,
            ml_type=ML_TYPE, unet_mode=UNET_MODE,
            use_kg=use_kg,
            use_dispersive_block=USE_DISPERSIVE_BLOCK,
        )
    elif index in [7, 8]:
        from wpo3d_unfold import WaveMST_ML_Unfold
        return WaveMST_ML_Unfold(
            dim=DIM, stage=STAGE, num_blocks=NUM_BLOCKS,
            num_stages=NUM_STAGES, share_weights=SHARE_STAGE_WEIGHTS,
            ml_type=ML_TYPE, unet_mode=UNET_MODE,
            use_kg=use_kg,
            size=CROP_SIZE, len_shift=2,
            use_dispersive=USE_DISPERSIVE,
            use_dispersive_block=USE_DISPERSIVE_BLOCK,
        )
    raise ValueError(f"无效 MODEL_INDEX: {index}")


def main():
    global NUM_BANDS, DIM

    from dataset import load_test, load_mask, prepare_mask, gen_meas, gen_meas_unfolding
    from loss import (torch_psnr, torch_ssim, torch_sam,
                      torch_freq_amp_err, torch_freq_band_err, count_params)

    # ── 从 bands.json 读取 NUM_BANDS ──
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
    print(f"模型: {MODELS[MODEL_INDEX][0]}  ML层: {ML_TYPE}  U-Net: {UNET_MODE}")
    print(f"参数量: {count_params(model):.2f}M  Checkpoint: {CHECKPOINT}")
    if IS_UNFOLDING:
        print(f"  Unfolding: stages={NUM_STAGES}  share_weights={SHARE_STAGE_WEIGHTS}")

    # ── 数据 ──
    test_data = load_test(str(TEST_PATH), nC=NUM_BANDS).cuda().float()
    N = test_data.shape[0]
    mask3d = load_mask(str(MASK_PATH), nC=NUM_BANDS).cuda()
    mask3d_test, shift_mask_test = prepare_mask(mask3d, N)

    # ── 推理 ──
    with torch.no_grad():
        if IS_UNFOLDING:
            from unfolding_ops import compute_PhiPhiT
            # PhiPhiT 预缓存（mask 固定）
            PhiPhiT_single = compute_PhiPhiT(mask3d, len_shift=2).cuda()  # [1, H, W']
            PhiPhiT = PhiPhiT_single.expand(N, -1, -1, -1)

            g_list = []
            for i in range(N):
                g_i, _ = gen_meas_unfolding(test_data[i], mask3d, step=2)
                g_list.append(g_i)
            g = torch.stack(g_list, dim=0).cuda()

            outputs = model(g, input_mask=(mask3d_test, PhiPhiT))
            pred = outputs[-1]

            # 打印每个 stage 的 PSNR 演化
            print("\n  Stage-wise PSNR:")
            for k, out_k in enumerate(outputs):
                psnr_k = []
                for i in range(N):
                    psnr_k.append(torch_psnr(out_k[i], test_data[i]).item())
                mean_k = sum(psnr_k) / len(psnr_k)
                print(f"    Stage {k+1}: PSNR={mean_k:.2f}")
            print()
        else:
            meas = gen_meas(test_data, mask3d_test, INPUT_SETTING)
            pred = model(meas, shift_mask_test)

    # ── 评估 ──
    psnr_list, ssim_list, sam_list = [], [], []
    freq_amp_list, low_freq_list, high_freq_list = [], [], []

    for i in range(N):
        p = pred[i]
        g_ref = test_data[i]
        psnr_list.append(torch_psnr(p, g_ref).item())
        ssim_list.append(torch_ssim(p, g_ref).item())
        sam_list.append(torch_sam(p, g_ref).item())
        freq_amp_list.append(torch_freq_amp_err(p, g_ref).item())
        fd = torch_freq_band_err(p, g_ref)
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
