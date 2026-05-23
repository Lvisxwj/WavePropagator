"""
train.py — WaveMST 训练入口

修改 CONFIG 区域的参数，用 MODEL_INDEX 选择模型，然后直接运行：
    python train.py
"""

import json
import os
import time
import torch
import torch.nn.functional as F
from pathlib import Path

import physics   # 需在 load_training 之后调用 physics.init_wavelengths


# ════════════════════════════════════════════
# CONFIG（在此修改所有超参数）
# ════════════════════════════════════════════
MODEL_INDEX  = 6       # 0-3: WaveMST 系列  4: WaveMST_Phys  5: Helmholtzformer  6: WaveMST_Helm
GPU_ID       = '0'
BATCH_SIZE   = 8
MAX_EPOCH    = 300
LR           = 4e-4
SCHEDULER    = 'CosineAnnealingLR'   # 'CosineAnnealingLR' 或 'MultiStepLR'
MILESTONES   = [50, 100, 150, 200, 250]
EPOCH_SAMPLE = 5000     # 每个 epoch 采样总数
CROP_SIZE    = 256
NUM_BANDS    = 28       # 默认值，main() 中从 bands.json 覆盖
DIM          = 28
STAGE        = 2
NUM_BLOCKS   = [2, 2, 2]
MASK_MODE    = 'A'      # 'A' / 'B' / 'D'
FUSION       = 'gate'   # 仅 Model 2 有效：'gate' / 'add' / 'linear'
INPUT_SETTING = 'H'     # 'H' / 'HM' / 'Y'
SAVE_THRESH  = 28.0     # PSNR 达到该值才保存 checkpoint

# 数据路径
DATA_ROOT  = Path('./dataset')
TRAIN_PATH = DATA_ROOT / 'CAVE_1024_npy'
TRAIN_PATH_FALLBACK = DATA_ROOT / 'CAVE_1024' / 'cave_1024_28'
TEST_PATH  = DATA_ROOT / 'TSA_simu_data' / 'Truth'
MASK_PATH  = DATA_ROOT / 'TSA_simu_data'

RESULT_ROOT = Path('./result')
# ════════════════════════════════════════════

MODELS = {
    0: ('WaveMST_3D',       '3d_wpo_pure'),
    1: ('WaveMST_KG',       '3d_wpo_kg'),
    2: ('WaveMST_Parallel', '3d_wpo_smsa'),
    3: ('WaveMST_Mamba',    '2d_wpo_mamba'),
    4: ('WaveMST_Phys',     'h2_alpha_phys'),
    5: ('Helmholtzformer',  'h1_gamma_helm_pure'),
    6: ('WaveMST_Helm',     'h2_gamma_main'),
}

os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID


# ──────────────────────────────────────────────
# 时间工具
# ──────────────────────────────────────────────

def _fmt_time(t: float) -> str:
    """将 Unix 时间戳格式化为 月.日.时:分（24 小时制）"""
    s = time.localtime(t)
    return f"{s.tm_mon}.{s.tm_mday}.{s.tm_hour}:{s.tm_min:02d}"


# ──────────────────────────────────────────────
# 模型构建
# ──────────────────────────────────────────────

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
        # Helmholtzformer 不支持 mask_mode='D'，若设置了 D 会在构建时抛出 ValueError
        return Helmholtzformer(dim=DIM, stage=STAGE, num_blocks=NUM_BLOCKS,
                               mask_mode=MASK_MODE)
    elif index == 6:
        from wpo3d_helm import WaveMST_Helm
        return WaveMST_Helm(dim=DIM, stage=STAGE, num_blocks=NUM_BLOCKS,
                            mask_mode=MASK_MODE)
    else:
        raise ValueError(f"无效 MODEL_INDEX: {index}")


# ──────────────────────────────────────────────
# 训练 / 测试
# ──────────────────────────────────────────────

def train_epoch(epoch, model, optimizer, train_set,
                mask3d_batch, input_mask, batch_num):
    from dataset import shuffle_crop, gen_meas
    from loss import rmse_loss

    model.train()
    total_loss = 0.
    t0 = time.time()

    for _ in range(batch_num):
        gt = shuffle_crop(train_set, BATCH_SIZE, CROP_SIZE,
                          device='cuda', nC=NUM_BANDS).float()
        meas = gen_meas(gt, mask3d_batch, INPUT_SETTING)

        optimizer.zero_grad()
        pred = model(meas, input_mask)
        loss = rmse_loss(pred, gt)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    elapsed = time.time() - t0
    avg = total_loss / batch_num
    lr_now = optimizer.param_groups[0]['lr']
    print(f"[Epoch {epoch:03d}] Loss: {avg:.6f}  Time: {elapsed:.1f}s  LR: {lr_now:.2e}")
    return avg


def test_epoch(epoch, model, test_data, mask3d_test, input_mask_test):
    from dataset import gen_meas
    from loss import torch_psnr, torch_ssim, torch_sam, torch_freq_amp_err, torch_freq_band_err

    model.eval()
    with torch.no_grad():
        gt   = test_data.cuda().float()
        meas = gen_meas(gt, mask3d_test, INPUT_SETTING)
        pred = model(meas, input_mask_test)

    psnr_list, ssim_list, sam_list = [], [], []
    freq_amp_list, low_freq_list, high_freq_list = [], [], []

    for i in range(pred.shape[0]):
        p = pred[i].detach()
        g = gt[i]
        psnr_list.append(torch_psnr(p, g).item())
        ssim_list.append(torch_ssim(p, g).item())
        sam_list.append(torch_sam(p, g).item())
        freq_amp_list.append(torch_freq_amp_err(p, g).item())
        fd = torch_freq_band_err(p, g)
        low_freq_list.append(fd['low_freq_err'])
        high_freq_list.append(fd['high_freq_err'])

    def _mean(lst): return sum(lst) / len(lst)

    psnr_mean  = _mean(psnr_list)
    ssim_mean  = _mean(ssim_list)
    sam_mean   = _mean(sam_list)
    freq_amp   = _mean(freq_amp_list)
    low_freq   = _mean(low_freq_list)
    high_freq  = _mean(high_freq_list)

    print(f"         Test → PSNR: {psnr_mean:.2f}  SSIM: {ssim_mean:.4f}  SAM: {sam_mean:.4f}")
    print(f"                FreqAmpErr: {freq_amp:.5f}  "
          f"LowFreqErr: {low_freq:.5f}  HighFreqErr: {high_freq:.5f}")
    return pred.detach().cpu(), psnr_mean, ssim_mean


# ──────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────

def main():
    global NUM_BANDS, DIM

    from dataset import load_training, load_test, load_mask, prepare_mask
    from loss import count_params, count_flops

    # ── 数据加载 ──
    train_path = TRAIN_PATH if TRAIN_PATH.exists() and any(TRAIN_PATH.iterdir()) \
                 else TRAIN_PATH_FALLBACK
    print(f"训练集路径: {train_path}")

    # load_training 会在首次运行时生成 bands.json / info.json
    train_set = load_training(str(train_path), max_scenes=205,
                              wavelengths=physics.WAVELENGTHS)

    # ── 从 bands.json 获取 NUM_BANDS，同步更新 physics.WAVELENGTHS ──
    bands_path = train_path / f'{train_path.name}_bands.json'
    if bands_path.exists():
        with open(bands_path, 'r', encoding='utf-8') as fp:
            bdata = json.load(fp)
        NUM_BANDS = bdata.get('num_bands', NUM_BANDS)
        DIM = NUM_BANDS
        physics.init_wavelengths(str(bands_path))
        print(f"[train] NUM_BANDS={NUM_BANDS}（来自 {bands_path.name}）")
    else:
        print(f"[train] 警告：{bands_path} 不存在，使用默认 NUM_BANDS={NUM_BANDS}")

    test_data = load_test(str(TEST_PATH), nC=NUM_BANDS)   # [N, nC, H, W]

    # ── Mask 准备 ──
    mask3d = load_mask(str(MASK_PATH), nC=NUM_BANDS).cuda()
    mask3d_train, shift_mask_train = prepare_mask(mask3d, BATCH_SIZE)
    mask3d_test,  shift_mask_test  = prepare_mask(mask3d, test_data.shape[0])

    # ── 模型 ──
    model = build_model(MODEL_INDEX).cuda()
    n_params = count_params(model)
    print(f"模型: {MODELS[MODEL_INDEX][0]}  参数量: {n_params:.2f}M")

    # ── 优化器 & 调度器 ──
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    if SCHEDULER == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=MAX_EPOCH, eta_min=1e-6)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=MILESTONES, gamma=0.5)

    # ── 输出目录 ──
    time_str = time.strftime('%Y_%m_%d_%H_%M_%S')
    save_dir = RESULT_ROOT / 'model' / f"{time_str}_{MODELS[MODEL_INDEX][1]}"
    save_dir.mkdir(parents=True, exist_ok=True)

    batch_num   = EPOCH_SAMPLE // BATCH_SIZE
    best_psnr   = 0.
    train_start = time.time()

    # ── 打印训练开始时间 ──
    print(f"训练开始: {_fmt_time(train_start)}")

    for epoch in range(1, MAX_EPOCH + 1):
        train_epoch(epoch, model, optimizer, train_set,
                    mask3d_train, shift_mask_train, batch_num)

        _, psnr_mean, ssim_mean = test_epoch(
            epoch, model, test_data, mask3d_test, shift_mask_test)

        scheduler.step()

        if psnr_mean > best_psnr:
            best_psnr = psnr_mean
            if psnr_mean > SAVE_THRESH:
                ckpt = save_dir / 'best.pth'
                torch.save(model.state_dict(), str(ckpt))
                print(f"  ★ 新最优: PSNR={psnr_mean:.2f}  SSIM={ssim_mean:.4f}  → {ckpt}")

        # 每 50 epoch 保存 checkpoint 并打印时间估计
        if epoch % 50 == 0:
            torch.save(model.state_dict(), str(save_dir / f'epoch_{epoch:03d}.pth'))
            now   = time.time()
            avg_t = (now - train_start) / epoch          # 每 epoch 平均耗时（秒）
            eta   = now + avg_t * (MAX_EPOCH - epoch)    # 预计结束时间戳
            print(f"[Epoch {epoch:03d}] 当前时间: {_fmt_time(now)}"
                  f"  预计结束: {_fmt_time(eta)}")


if __name__ == '__main__':
    main()
