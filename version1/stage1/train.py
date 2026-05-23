"""
train.py — WaveMST 训练入口

修改 CONFIG 区域的参数，用 MODEL_INDEX 选择模型，然后直接运行：
    python train.py
"""

import os
import time
import torch
import torch.nn.functional as F
from pathlib import Path


# ════════════════════════════════════════════
# CONFIG（在此修改所有超参数）
# ════════════════════════════════════════════
MODEL_INDEX  = 2       # 0: WaveMST_3D  1: WaveMST_KG  2: WaveMST_Parallel  3: WaveMST_Mamba
GPU_ID       = '2'
BATCH_SIZE   = 8
MAX_EPOCH    = 300
LR           = 4e-4
SCHEDULER    = 'CosineAnnealingLR'   # 'CosineAnnealingLR' 或 'MultiStepLR'
MILESTONES   = [50, 100, 150, 200, 250]
EPOCH_SAMPLE = 5000     # 每个 epoch 采样总数
CROP_SIZE    = 256
NUM_BANDS    = 28
DIM          = 28
STAGE        = 2
NUM_BLOCKS   = [2, 2, 2]
MASK_MODE    = 'A'      # 'A' / 'B' / 'D'
FUSION       = 'gate'   # 仅 Model 2 有效：'gate' / 'add' / 'linear'
INPUT_SETTING = 'H'     # 'H' / 'HM' / 'Y'
SAVE_THRESH  = 28.0     # PSNR 达到该值才保存 checkpoint

# 数据路径
DATA_ROOT  = Path('./dataset')
TRAIN_PATH = DATA_ROOT / 'CAVE_1024_npy'       # 优先 npy；不存在则自动尝试 mat
TRAIN_PATH_FALLBACK = DATA_ROOT / 'CAVE_1024' / 'cave_1024_28'
TEST_PATH  = DATA_ROOT / 'TSA_simu_data' / 'Truth'
MASK_PATH  = DATA_ROOT / 'TSA_simu_data'        # 含 mask.npy 或 mask.mat

RESULT_ROOT = Path('./result')
# ════════════════════════════════════════════

MODELS = {
    0: ('WaveMST_3D',       '3d_wpo_pure'),
    1: ('WaveMST_KG',       '3d_wpo_kg'),
    2: ('WaveMST_Parallel', '3d_wpo_smsa'),
    3: ('WaveMST_Mamba',    '2d_wpo_mamba'),
}

os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID


# ──────────────────────────────────────────────
# 模型构建
# ──────────────────────────────────────────────

def build_model(index):
    if index == 0:
        # from xieweijie.CASSI.wpo3d import WaveMST_3D
        from wpo3d import WaveMST_3D
        return WaveMST_3D(dim=DIM, stage=STAGE, num_blocks=NUM_BLOCKS,
                          mask_mode=MASK_MODE)
    elif index == 1:
        # from xieweijie.CASSI.wpo3d import WaveMST_KG
        from wpo3d import WaveMST_KG
        return WaveMST_KG(dim=DIM, stage=STAGE, num_blocks=NUM_BLOCKS)
    elif index == 2:
        # from xieweijie.CASSI.wpo_smsa import WaveMST_Parallel
        from wpo_smsa import WaveMST_Parallel
        return WaveMST_Parallel(dim=DIM, stage=STAGE, num_blocks=NUM_BLOCKS,
                                mask_mode=MASK_MODE, fusion=FUSION)
    elif index == 3:
        # from xieweijie.CASSI.wpo_mamba import WaveMST_Mamba
        from wpo_mamba import WaveMST_Mamba
        return WaveMST_Mamba(dim=DIM, stage=STAGE, num_blocks=NUM_BLOCKS,
                             mask_mode=MASK_MODE)
    else:
        raise ValueError(f"无效 MODEL_INDEX: {index}")


# ──────────────────────────────────────────────
# 训练 / 测试
# ──────────────────────────────────────────────

def train_epoch(epoch, model, optimizer, train_set,
                mask3d_batch, input_mask, batch_num):
    # from xieweijie.CASSI.dataset import shuffle_crop, gen_meas
    # from xieweijie.CASSI.loss import rmse_loss
    from dataset import shuffle_crop, gen_meas
    from loss import rmse_loss

    model.train()
    total_loss = 0.
    t0 = time.time()

    for _ in range(batch_num):
        gt = shuffle_crop(train_set, BATCH_SIZE, CROP_SIZE).cuda().float()
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
    # from xieweijie.CASSI.dataset import gen_meas
    # from xieweijie.CASSI.loss import torch_psnr, torch_ssim
    from dataset import gen_meas
    from loss import torch_psnr, torch_ssim

    model.eval()
    with torch.no_grad():
        gt = test_data.cuda().float()
        meas = gen_meas(gt, mask3d_test, INPUT_SETTING)
        pred = model(meas, input_mask_test)

    psnr_list, ssim_list = [], []
    for i in range(pred.shape[0]):
        p = torch_psnr(pred[i].detach(), gt[i])
        s = torch_ssim(pred[i].detach(), gt[i])
        psnr_list.append(p.item())
        ssim_list.append(s.item())

    psnr_mean = sum(psnr_list) / len(psnr_list)
    ssim_mean = sum(ssim_list) / len(ssim_list)
    print(f"         Test → PSNR: {psnr_mean:.2f}  SSIM: {ssim_mean:.4f}")
    return pred.detach().cpu(), psnr_mean, ssim_mean


# ──────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────

def main():
    from dataset import load_training, load_test, load_mask, prepare_mask
    from loss import count_params, count_flops
    # from xieweijie.CASSI.dataset import load_training, load_test, load_mask, prepare_mask
    # from xieweijie.CASSI.loss import count_params, count_flops

    # ── 数据加载 ──
    train_path = TRAIN_PATH if TRAIN_PATH.exists() and any(TRAIN_PATH.iterdir()) \
                 else TRAIN_PATH_FALLBACK
    print(f"训练集路径: {train_path}")
    train_set = load_training(str(train_path), max_scenes=205)
    test_data = load_test(str(TEST_PATH))   # [N, 28, 256, 256]

    # ── Mask 准备 ──
    mask3d = load_mask(str(MASK_PATH)).cuda()          # [28, 256, 256]
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

    batch_num = EPOCH_SAMPLE // BATCH_SIZE
    best_psnr = 0.

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

        # 每 50 epoch 保存一次常规 checkpoint
        if epoch % 50 == 0:
            torch.save(model.state_dict(), str(save_dir / f'epoch_{epoch:03d}.pth'))


if __name__ == '__main__':
    main()
