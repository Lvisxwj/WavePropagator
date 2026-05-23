"""
train.py — WaveMST 训练入口（stage2: 支持 unfolding）

修改 CONFIG 区域的参数，用 MODEL_INDEX 选择模型，然后直接运行：
    python train.py
"""

import json
import os
import time
import torch
import torch.nn.functional as F
from pathlib import Path

import physics


# ════════════════════════════════════════════
# CONFIG（在此修改所有超参数）
# ════════════════════════════════════════════
MODEL_INDEX  = 7       # 0: WaveMST_3D  1: WaveMST_KG  7: Unfold_3D  8: Unfold_KG
GPU_ID       = '2'
BATCH_SIZE   = 2
MAX_EPOCH    = 300
LR           = 4e-4
SCHEDULER    = 'CosineAnnealingLR'
MILESTONES   = [50, 100, 150, 200, 250]
EPOCH_SAMPLE = 5000
CROP_SIZE    = 256
NUM_BANDS    = 28
DIM          = 28
STAGE        = 3
NUM_BLOCKS   = [2, 2, 2]
MASK_MODE    = 'A'
FUSION       = 'gate'
INPUT_SETTING = 'H'
SAVE_THRESH  = 28.0

# Unfolding 专用配置（仅对 MODEL_INDEX >= 7 生效）
NUM_STAGES          = 5       # unfolding stage 数：3/5/7/9
# SHARE_STAGE_WEIGHTS = False   # True 所有 stage 共享 WPO 权重
SHARE_STAGE_WEIGHTS = True
MULTI_STAGE_LOSS    = True    # 多 stage 加权损失（DPU 风格）

# 数据路径（相对于 stage2/ 目录，数据集在上级的 dataset/）
DATA_ROOT  = Path('../dataset')
TRAIN_PATH = DATA_ROOT / 'CAVE_1024_npy'
TRAIN_PATH_FALLBACK = DATA_ROOT / 'CAVE_1024' / 'cave_1024_28'
TEST_PATH  = DATA_ROOT / 'TSA_simu_data' / 'Truth'
MASK_PATH  = DATA_ROOT / 'TSA_simu_data'

RESULT_ROOT = Path('./result')
# ════════════════════════════════════════════

MODELS = {
    0: ('WaveMST_3D',       '3d_wpo_pure'),
    1: ('WaveMST_KG',       '3d_wpo_kg'),
    7: ('WaveMST_3D_Unfold', '3d_wpo_unfold'),
    8: ('WaveMST_KG_Unfold', '3d_wpo_kg_unfold'),
}

os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

IS_UNFOLDING = MODEL_INDEX >= 7


# ──────────────────────────────────────────────
# 时间工具
# ──────────────────────────────────────────────

def _fmt_time(t: float) -> str:
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
    elif index == 7:
        from wpo3d_unfold import WaveMST_3D_Unfold
        return WaveMST_3D_Unfold(
            dim=DIM, stage=STAGE, num_blocks=NUM_BLOCKS,
            num_stages=NUM_STAGES,
            share_weights=SHARE_STAGE_WEIGHTS,
            use_kg=False,
            mask_mode=MASK_MODE,
            size=CROP_SIZE,
            len_shift=2,
        )
    elif index == 8:
        from wpo3d_unfold import WaveMST_KG_Unfold
        return WaveMST_KG_Unfold(
            dim=DIM, stage=STAGE, num_blocks=NUM_BLOCKS,
            num_stages=NUM_STAGES,
            share_weights=SHARE_STAGE_WEIGHTS,
            mask_mode=MASK_MODE,
            size=CROP_SIZE,
            len_shift=2,
        )
    raise ValueError(f"无效 MODEL_INDEX: {index}")


# ──────────────────────────────────────────────
# 损失函数
# ──────────────────────────────────────────────

def rmse_loss(pred, gt):
    return torch.sqrt(F.mse_loss(pred, gt))


def multi_stage_loss(outputs, gt):
    """DPU 风格多 stage 加权 RMSE。"""
    K = len(outputs)
    loss = rmse_loss(outputs[-1], gt)
    if K >= 2:
        loss = loss + 0.7 * rmse_loss(outputs[-2], gt)
    if K >= 3:
        loss = loss + 0.5 * rmse_loss(outputs[-3], gt)
    if K >= 4:
        loss = loss + 0.3 * rmse_loss(outputs[-4], gt)
    return loss


# ──────────────────────────────────────────────
# 训练 / 测试
# ──────────────────────────────────────────────

def train_epoch(epoch, model, optimizer, train_set,
                mask3d_batch, shift_mask_train, batch_num):
    from dataset import shuffle_crop, gen_meas, gen_meas_unfolding
    from unfolding_ops import compute_PhiPhiT

    model.train()
    total_loss = 0.
    t0 = time.time()

    for _ in range(batch_num):
        gt = shuffle_crop(train_set, BATCH_SIZE, CROP_SIZE,
                          device='cuda', nC=NUM_BANDS).float()

        if IS_UNFOLDING:
            # 为每个样本生成 g 和 PhiPhiT
            B = gt.shape[0]
            g_list, ppt_list = [], []
            for b in range(B):
                g_b, ppt_b = gen_meas_unfolding(gt[b], mask3d_batch[0], step=2)
                g_list.append(g_b)
                ppt_list.append(ppt_b)
            g = torch.stack(g_list, dim=0).cuda()           # [B, 1, H, W']
            PhiPhiT = torch.stack(ppt_list, dim=0).cuda()   # [B, 1, H, W']

            optimizer.zero_grad()
            outputs = model(g, input_mask=(mask3d_batch, PhiPhiT))

            if MULTI_STAGE_LOSS:
                loss = multi_stage_loss(outputs, gt)
            else:
                loss = rmse_loss(outputs[-1], gt)
        else:
            meas = gen_meas(gt, mask3d_batch, INPUT_SETTING)
            optimizer.zero_grad()
            pred = model(meas, shift_mask_train)
            loss = rmse_loss(pred, gt)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    elapsed = time.time() - t0
    avg = total_loss / batch_num
    lr_now = optimizer.param_groups[0]['lr']
    print(f"[Epoch {epoch:03d}] Loss: {avg:.6f}  Time: {elapsed:.1f}s  LR: {lr_now:.2e}")
    return avg


def test_epoch(epoch, model, test_data, mask3d_test, shift_mask_test, mask3d_single):
    from dataset import gen_meas, gen_meas_unfolding
    from loss import torch_psnr, torch_ssim, torch_sam, torch_freq_amp_err, torch_freq_band_err
    from unfolding_ops import compute_PhiPhiT

    model.eval()
    with torch.no_grad():
        gt = test_data.cuda().float()

        if IS_UNFOLDING:
            N = gt.shape[0]
            g_list, ppt_list = [], []
            for i in range(N):
                g_i, ppt_i = gen_meas_unfolding(gt[i], mask3d_single.cuda(), step=2)
                g_list.append(g_i)
                ppt_list.append(ppt_i)
            g = torch.stack(g_list, dim=0).cuda()
            PhiPhiT = torch.stack(ppt_list, dim=0).cuda()

            outputs = model(g, input_mask=(mask3d_test, PhiPhiT))
            pred = outputs[-1]
        else:
            meas = gen_meas(gt, mask3d_test, INPUT_SETTING)
            pred = model(meas, shift_mask_test)

    psnr_list, ssim_list, sam_list = [], [], []
    freq_amp_list, low_freq_list, high_freq_list = [], [], []

    for i in range(pred.shape[0]):
        p = pred[i].detach()
        g_i = gt[i]
        psnr_list.append(torch_psnr(p, g_i).item())
        ssim_list.append(torch_ssim(p, g_i).item())
        sam_list.append(torch_sam(p, g_i).item())
        freq_amp_list.append(torch_freq_amp_err(p, g_i).item())
        fd = torch_freq_band_err(p, g_i)
        low_freq_list.append(fd['low_freq_err'])
        high_freq_list.append(fd['high_freq_err'])

    def _mean(lst): return sum(lst) / len(lst)

    psnr_mean  = _mean(psnr_list)
    ssim_mean  = _mean(ssim_list)
    sam_mean   = _mean(sam_list)
    freq_amp   = _mean(freq_amp_list)
    low_freq   = _mean(low_freq_list)
    high_freq  = _mean(high_freq_list)

    print(f"         Test -> PSNR: {psnr_mean:.2f}  SSIM: {ssim_mean:.4f}  SAM: {sam_mean:.4f}")
    print(f"                FreqAmpErr: {freq_amp:.5f}  "
          f"LowFreqErr: {low_freq:.5f}  HighFreqErr: {high_freq:.5f}")
    return pred.detach().cpu(), psnr_mean, ssim_mean


# ──────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────

def main():
    global NUM_BANDS, DIM

    from dataset import load_training, load_test, load_mask, prepare_mask
    from loss import count_params

    # ── 数据加载 ──
    train_path = TRAIN_PATH if TRAIN_PATH.exists() and any(TRAIN_PATH.iterdir()) \
                 else TRAIN_PATH_FALLBACK
    print(f"训练集路径: {train_path}")

    train_set = load_training(str(train_path), max_scenes=205,
                              wavelengths=physics.WAVELENGTHS)

    # ── 从 bands.json 获取 NUM_BANDS ──
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

    test_data = load_test(str(TEST_PATH), nC=NUM_BANDS)

    # ── Mask 准备 ──
    mask3d = load_mask(str(MASK_PATH), nC=NUM_BANDS).cuda()
    mask3d_train, shift_mask_train = prepare_mask(mask3d, BATCH_SIZE)
    mask3d_test,  shift_mask_test  = prepare_mask(mask3d, test_data.shape[0])

    # ── 模型 ──
    model = build_model(MODEL_INDEX).cuda()
    n_params = count_params(model)
    model_name = MODELS[MODEL_INDEX][0]
    print(f"模型: {model_name}  参数量: {n_params:.2f}M")
    if IS_UNFOLDING:
        print(f"  Unfolding: stages={NUM_STAGES}  share_weights={SHARE_STAGE_WEIGHTS}"
              f"  multi_stage_loss={MULTI_STAGE_LOSS}")

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
    tag = MODELS[MODEL_INDEX][1]
    if IS_UNFOLDING:
        tag += f'_stg{NUM_STAGES}'
        if SHARE_STAGE_WEIGHTS:
            tag += '_share'
    save_dir = RESULT_ROOT / 'model' / f"{time_str}_{tag}"
    save_dir.mkdir(parents=True, exist_ok=True)

    batch_num   = EPOCH_SAMPLE // BATCH_SIZE
    best_psnr   = 0.
    train_start = time.time()

    print(f"训练开始: {_fmt_time(train_start)}")

    for epoch in range(1, MAX_EPOCH + 1):
        train_epoch(epoch, model, optimizer, train_set,
                    mask3d_train, shift_mask_train, batch_num)

        _, psnr_mean, ssim_mean = test_epoch(
            epoch, model, test_data, mask3d_test, shift_mask_test, mask3d)

        scheduler.step()

        if psnr_mean > best_psnr:
            best_psnr = psnr_mean
            if psnr_mean > SAVE_THRESH:
                ckpt = save_dir / 'best.pth'
                torch.save(model.state_dict(), str(ckpt))
                print(f"  * 新最优: PSNR={psnr_mean:.2f}  SSIM={ssim_mean:.4f}  -> {ckpt}")

        if epoch % 50 == 0:
            torch.save(model.state_dict(), str(save_dir / f'epoch_{epoch:03d}.pth'))
            now   = time.time()
            avg_t = (now - train_start) / epoch
            eta   = now + avg_t * (MAX_EPOCH - epoch)
            print(f"[Epoch {epoch:03d}] 当前时间: {_fmt_time(now)}"
                  f"  预计结束: {_fmt_time(eta)}")


if __name__ == '__main__':
    main()
