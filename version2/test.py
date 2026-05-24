"""
test.py — 测试入口
读取 config.yaml 和 __init__.py 的配置
"""

import os
import sys
import yaml
from pathlib import Path

# 读取配置 & 设置 GPU（必须在 import torch 之前）
_script_dir = Path(__file__).parent
with open(_script_dir / 'config.yaml', 'r') as _f:
    cfg = yaml.safe_load(_f)

os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu_id']

import time
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(_script_dir))

from __init__ import (
    USE_KG, WPO_FBGW_MODE, USE_SWIN_WPO, SWIN_WINDOW_SIZE,
    USE_UNFOLDING, USE_AHQS, NUM_STAGES, SHARE_STAGE_WEIGHTS, BEST_CKPT,
)


# ──────────────────────────────────────────────
# 模型构建
# ──────────────────────────────────────────────

def build_model():
    if USE_UNFOLDING:
        from model.unfolding import WPO_Unfold
        return WPO_Unfold(
            dim=cfg['dim'],
            unet_stage=cfg['unet_stage'],
            num_blocks=cfg['num_blocks'],
            use_kg=USE_KG,
            num_stages=NUM_STAGES,
            share_weights=SHARE_STAGE_WEIGHTS,
            use_swin_wpo=USE_SWIN_WPO,
            swin_window_size=SWIN_WINDOW_SIZE,
            fbgw_mode=WPO_FBGW_MODE,
            size=cfg['crop_size'],
            use_ahqs=USE_AHQS,
        )
    else:
        from model.wpo3d import WaveMST_3D, WaveMST_KG
        cls = WaveMST_KG if USE_KG else WaveMST_3D
        return cls(
            dim=cfg['dim'],
            stage=cfg['unet_stage'],
            num_blocks=cfg['num_blocks'],
            use_swin_wpo=USE_SWIN_WPO,
            swin_window_size=SWIN_WINDOW_SIZE,
            fbgw_mode=WPO_FBGW_MODE,
        )


# ──────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────

def main():
    from dataset import load_test, load_mask, prepare_mask, gen_meas, gen_meas_unfolding
    from loss import torch_psnr, torch_ssim, torch_sam, count_params
    from model.utils import compute_PhiPhiT

    NUM_BANDS = cfg['num_bands']

    if not BEST_CKPT:
        print("错误：请在 __init__.py 中设置 BEST_CKPT 路径")
        return

    # 模型
    model = build_model().cuda()
    model.load_state_dict(torch.load(BEST_CKPT, map_location='cuda'))
    model.eval()
    print(f"参数量: {count_params(model):.2f}M")
    print(f"Checkpoint: {BEST_CKPT}")
    if USE_UNFOLDING:
        print(f"  Unfolding: stages={NUM_STAGES}  share_weights={SHARE_STAGE_WEIGHTS}")

    # 数据
    test_data = load_test(cfg['test_path'], nC=NUM_BANDS).cuda().float()
    N = test_data.shape[0]
    mask3d = load_mask(cfg['mask_path'], nC=NUM_BANDS).cuda()
    mask3d_test, shift_mask_test = prepare_mask(mask3d, N)

    # 推理
    t0 = time.time()
    with torch.no_grad():
        if USE_UNFOLDING:
            PhiPhiT_single = compute_PhiPhiT(mask3d, len_shift=2).cuda()
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
            meas = gen_meas(test_data, mask3d_test, cfg.get('input_setting', 'H'))
            pred = model(meas, shift_mask_test)

    elapsed = time.time() - t0

    # 评估
    psnr_list, ssim_list, sam_list = [], [], []
    for i in range(N):
        p = pred[i]
        g_ref = test_data[i]
        psnr_list.append(torch_psnr(p, g_ref).item())
        ssim_list.append(torch_ssim(p, g_ref).item())
        sam_list.append(torch_sam(p, g_ref).item())

        print(f"  Scene {i+1:02d}: PSNR={psnr_list[-1]:.2f}  "
              f"SSIM={ssim_list[-1]:.4f}  SAM={sam_list[-1]:.4f}")

    def _mean(lst): return sum(lst) / len(lst)

    print(f"\n  平均:    PSNR={_mean(psnr_list):.2f}  "
          f"SSIM={_mean(ssim_list):.4f}  SAM={_mean(sam_list):.4f}  "
          f"Time: {elapsed:.1f}s")

    # 保存结果
    time_str = time.strftime('%Y_%m_%d_%H_%M_%S')
    tag = 'wpo_unfold' if USE_UNFOLDING else 'wpo_e2e'
    if USE_KG:
        tag += '_kg'
    out_dir = Path('result/show') / f"{time_str}_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(out_dir / 'pred.npy'), pred.cpu().numpy())
    np.save(str(out_dir / 'gt.npy'),   test_data.cpu().numpy())
    print(f"\n结果保存到: {out_dir}")


if __name__ == '__main__':
    main()
