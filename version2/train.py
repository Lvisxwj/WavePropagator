"""
train.py — 训练入口
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

sys.path.insert(0, str(_script_dir))

from __init__ import (
    USE_KG, WPO_FBGW_MODE, USE_SWIN_WPO, SWIN_WINDOW_SIZE,
    USE_UNFOLDING, USE_AHQS, NUM_STAGES, SHARE_STAGE_WEIGHTS, MULTI_STAGE_LOSS,
    DEBUG_FORWARD,
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
            debug=DEBUG_FORWARD,
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


def print_config(model):
    from loss import count_params
    print("=" * 60)
    print(f"当前配置组合:")
    print(f"  KG方程:     {'是' if USE_KG else '否'}")
    print(f"  FBGW:       {WPO_FBGW_MODE}")
    print(f"  Swin-WPO:   {'是 (ws=' + str(SWIN_WINDOW_SIZE) + ')' if USE_SWIN_WPO else '否'}")
    if USE_UNFOLDING:
        algo = 'A-HQS+动量' if USE_AHQS else 'GAP'
        print(f"  展开:       {NUM_STAGES} stage, "
              f"{'共享' if SHARE_STAGE_WEIGHTS else '独立'}权重, {algo}")
        print(f"  多阶段损失: {'是' if MULTI_STAGE_LOSS else '否'}")
    else:
        print(f"  展开:       无 (端到端)")
    print(f"  参数量:     {count_params(model):.2f}M")
    print("=" * 60)


# ──────────────────────────────────────────────
# 损失函数
# ──────────────────────────────────────────────

def rmse_loss(pred, gt):
    return torch.sqrt(F.mse_loss(pred, gt))


def multi_stage_loss(outputs, gt):
    K = len(outputs)
    loss = rmse_loss(outputs[-1], gt)
    if K >= 2: loss = loss + 0.7 * rmse_loss(outputs[-2], gt)
    if K >= 3: loss = loss + 0.5 * rmse_loss(outputs[-3], gt)
    if K >= 4: loss = loss + 0.3 * rmse_loss(outputs[-4], gt)
    return loss


# ──────────────────────────────────────────────
# 时间工具
# ──────────────────────────────────────────────

def _fmt_time(t):
    s = time.localtime(t)
    return f"{s.tm_mon}.{s.tm_mday}.{s.tm_hour}:{s.tm_min:02d}"


# ──────────────────────────────────────────────
# 训练 / 测试
# ──────────────────────────────────────────────

def train_epoch(epoch, model, optimizer, train_set,
                mask3d_batch, shift_mask_train, batch_num,
                scaler=None, PhiPhiT_cached=None):
    from dataset import shuffle_crop, gen_meas, gen_meas_unfolding

    model.train()
    total_loss = 0.
    t0 = time.time()
    BATCH_SIZE = cfg['batch_size']
    CROP_SIZE = cfg['crop_size']
    NUM_BANDS = cfg['num_bands']
    USE_AMP = cfg.get('use_amp', False)

    for _ in range(batch_num):
        gt = shuffle_crop(train_set, BATCH_SIZE, CROP_SIZE,
                          device='cuda', nC=NUM_BANDS).float()

        if USE_UNFOLDING:
            B = gt.shape[0]
            g_list = []
            for b in range(B):
                g_b, _ = gen_meas_unfolding(gt[b], mask3d_batch[0], step=2)
                g_list.append(g_b)
            g = torch.stack(g_list, dim=0).cuda()

            if PhiPhiT_cached is not None:
                PhiPhiT = PhiPhiT_cached.expand(B, -1, -1, -1)
            else:
                ppt_list = []
                for b in range(B):
                    _, ppt_b = gen_meas_unfolding(gt[b], mask3d_batch[0], step=2)
                    ppt_list.append(ppt_b)
                PhiPhiT = torch.stack(ppt_list, dim=0).cuda()

            optimizer.zero_grad()

            if USE_AMP and scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(g, input_mask=(mask3d_batch, PhiPhiT))
                    if MULTI_STAGE_LOSS:
                        loss = multi_stage_loss(outputs, gt)
                    else:
                        loss = rmse_loss(outputs[-1], gt)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(g, input_mask=(mask3d_batch, PhiPhiT))
                if MULTI_STAGE_LOSS:
                    loss = multi_stage_loss(outputs, gt)
                else:
                    loss = rmse_loss(outputs[-1], gt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
        else:
            meas = gen_meas(gt, mask3d_batch, cfg.get('input_setting', 'H'))
            optimizer.zero_grad()

            if USE_AMP and scaler is not None:
                with torch.cuda.amp.autocast():
                    pred = model(meas, shift_mask_train)
                    loss = rmse_loss(pred, gt)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(meas, shift_mask_train)
                loss = rmse_loss(pred, gt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

        total_loss += loss.item()

    elapsed = time.time() - t0
    avg = total_loss / batch_num
    lr_now = optimizer.param_groups[0]['lr']
    print(f"[Epoch {epoch:03d}] Loss: {avg:.6f}  Time: {elapsed:.1f}s  LR: {lr_now:.2e}")
    return avg


def test_epoch(epoch, model, test_data, mask3d_test, shift_mask_test, mask3d_single,
               PhiPhiT_cached=None):
    from dataset import gen_meas, gen_meas_unfolding
    from loss import torch_psnr, torch_ssim, torch_sam

    model.eval()
    t0 = time.time()
    with torch.no_grad():
        gt = test_data.cuda().float()

        if USE_UNFOLDING:
            N = gt.shape[0]
            g_list = []
            for i in range(N):
                g_i, _ = gen_meas_unfolding(gt[i], mask3d_single.cuda(), step=2)
                g_list.append(g_i)
            g = torch.stack(g_list, dim=0).cuda()

            if PhiPhiT_cached is not None:
                PhiPhiT = PhiPhiT_cached.expand(N, -1, -1, -1)
            else:
                ppt_list = []
                for i in range(N):
                    _, ppt_i = gen_meas_unfolding(gt[i], mask3d_single.cuda(), step=2)
                    ppt_list.append(ppt_i)
                PhiPhiT = torch.stack(ppt_list, dim=0).cuda()

            outputs = model(g, input_mask=(mask3d_test, PhiPhiT))
            pred = outputs[-1]
        else:
            meas = gen_meas(gt, mask3d_test, cfg.get('input_setting', 'H'))
            pred = model(meas, shift_mask_test)

    psnr_list, ssim_list, sam_list = [], [], []
    for i in range(pred.shape[0]):
        p = pred[i].detach()
        g_i = gt[i]
        psnr_list.append(torch_psnr(p, g_i).item())
        ssim_list.append(torch_ssim(p, g_i).item())
        sam_list.append(torch_sam(p, g_i).item())

    def _mean(lst): return sum(lst) / len(lst)

    psnr_mean = _mean(psnr_list)
    ssim_mean = _mean(ssim_list)
    sam_mean  = _mean(sam_list)
    elapsed   = time.time() - t0

    print(f"         Test -> PSNR: {psnr_mean:.2f}  SSIM: {ssim_mean:.4f}  "
          f"SAM: {sam_mean:.4f}  Time: {elapsed:.1f}s")
    return pred.detach().cpu(), psnr_mean, ssim_mean


# ──────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────

def main():
    from dataset import load_training, load_test, load_mask, prepare_mask
    from loss import count_params
    from model.utils import compute_PhiPhiT

    BATCH_SIZE = cfg['batch_size']
    MAX_EPOCH = cfg['max_epoch']
    LR = cfg['learning_rate']
    CROP_SIZE = cfg['crop_size']
    NUM_BANDS = cfg['num_bands']
    SAVE_THRESH = cfg.get('save_thresh', 28.0)
    EPOCH_SAMPLE = cfg.get('epoch_sample', 5000)
    USE_AMP = cfg.get('use_amp', False)

    # 数据加载
    train_path = cfg['train_path']
    print(f"训练集路径: {train_path}")
    train_set = load_training(str(train_path), max_scenes=205)

    test_data = load_test(cfg['test_path'], nC=NUM_BANDS)

    # Mask 准备
    mask3d = load_mask(cfg['mask_path'], nC=NUM_BANDS).cuda()
    mask3d_train, shift_mask_train = prepare_mask(mask3d, BATCH_SIZE)
    mask3d_test,  shift_mask_test  = prepare_mask(mask3d, test_data.shape[0])

    # PhiPhiT 预缓存
    PhiPhiT_cached = None
    if USE_UNFOLDING:
        PhiPhiT_cached = compute_PhiPhiT(mask3d, len_shift=2).cuda()
        print(f"[train] PhiPhiT 已缓存: shape={list(PhiPhiT_cached.shape)}")

    # 模型
    model = build_model().cuda()
    print_config(model)

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler() if USE_AMP else None
    if USE_AMP:
        print(f"  混合精度训练: ON")

    # 优化器 & 调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    scheduler_name = cfg.get('scheduler', 'CosineAnnealingLR')
    if scheduler_name == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=MAX_EPOCH, eta_min=1e-6)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[50, 100, 150, 200, 250], gamma=0.5)

    # 输出目录
    time_str = time.strftime('%Y_%m_%d_%H_%M_%S')
    tag = 'wpo_unfold' if USE_UNFOLDING else 'wpo_e2e'
    if USE_KG:
        tag += '_kg'
    if USE_UNFOLDING:
        tag += f'_stg{NUM_STAGES}'
        if SHARE_STAGE_WEIGHTS:
            tag += '_share'
    save_dir = Path('result/model') / f"{time_str}_{tag}"
    save_dir.mkdir(parents=True, exist_ok=True)

    batch_num   = EPOCH_SAMPLE // BATCH_SIZE
    best_psnr   = 0.
    train_start = time.time()

    print(f"训练开始: {_fmt_time(train_start)}")

    for epoch in range(1, MAX_EPOCH + 1):
        train_epoch(epoch, model, optimizer, train_set,
                    mask3d_train, shift_mask_train, batch_num,
                    scaler=scaler, PhiPhiT_cached=PhiPhiT_cached)

        _, psnr_mean, ssim_mean = test_epoch(
            epoch, model, test_data, mask3d_test, shift_mask_test, mask3d,
            PhiPhiT_cached=PhiPhiT_cached)

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
