"""
viz.py — WaveMST 可视化工具

函数：
    show_bands(img, bands, title, save_path)        — 多波段灰度图
    show_rgb(img, r, g, b, save_path)               — 伪彩色 RGB 合成
    show_spectrum(imgs, labels, pos, save_path)      — 光谱曲线对比
    show_comparison(pred, gt, band, save_path)       — pred vs GT 并排
    show_error_map(pred, gt, band, save_path)        — 误差热图
    show_freq_magnitude(img, title, save_path)       — 频域幅度谱（log 尺度）
    show_freq_comparison(pred, gt, band, save_path)  — 频域并排对比
    show_rapsd(pred, gt, save_path)                  — 径向平均功率谱曲线
    show_all(pred, gt, scene_idx, save_dir)          — 一键输出全部图

用法：
    from viz import show_all
    show_all(pred, gt, scene_idx=0, save_dir='result/show/viz/')
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')   # 无头模式，不弹窗
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


# ──────────────────────────────────────────────
# 内部工具
# ──────────────────────────────────────────────

def _to_np(x):
    """tensor / ndarray → numpy [C, H, W] float32 [0,1]"""
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)
    return np.clip(x, 0, 1)


def _save_or_show(fig, save_path):
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ──────────────────────────────────────────────
# 多波段灰度图
# ──────────────────────────────────────────────

def show_bands(img, bands=None, title='', save_path=None):
    """
    显示若干波段的灰度图像。

    img:   [C, H, W] tensor 或 ndarray
    bands: 要显示的波段列表（默认均匀取 6 个）
    """
    img = _to_np(img)
    C = img.shape[0]
    if bands is None:
        bands = np.linspace(0, C - 1, 6, dtype=int).tolist()

    n = len(bands)
    fig, axes = plt.subplots(1, n, figsize=(n * 3, 3))
    if n == 1:
        axes = [axes]
    for ax, b in zip(axes, bands):
        ax.imshow(img[b], cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'Band {b}', fontsize=9)
        ax.axis('off')
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ──────────────────────────────────────────────
# 伪彩色 RGB
# ──────────────────────────────────────────────

def show_rgb(img, r=20, g=12, b=4, save_path=None, title=''):
    """
    伪彩色 RGB 合成（默认 NIR/Red/Green 对应 band 20/12/4）。

    img: [C, H, W]
    """
    img = _to_np(img)
    rgb = np.stack([img[r], img[g], img[b]], axis=-1)
    for c in range(3):
        lo, hi = np.percentile(rgb[:, :, c], [2, 98])
        rgb[:, :, c] = np.clip((rgb[:, :, c] - lo) / (hi - lo + 1e-8), 0, 1)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(rgb)
    ax.set_title(title or f'RGB (R={r} G={g} B={b})', fontsize=10)
    ax.axis('off')
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ──────────────────────────────────────────────
# 光谱曲线对比
# ──────────────────────────────────────────────

def show_spectrum(imgs, labels, positions, save_path=None, wavelengths=None):
    """
    在指定空间位置绘制多张图的光谱曲线。

    imgs:        list of [C, H, W]（可以是 pred 和 gt）
    labels:      list of str（图例名）
    positions:   list of (row, col)  要提取光谱的像素坐标
    wavelengths: 波长轴标签（默认 1~C）
    """
    imgs = [_to_np(im) for im in imgs]
    C = imgs[0].shape[0]
    wl = wavelengths if wavelengths is not None else list(range(1, C + 1))

    n_pos = len(positions)
    fig, axes = plt.subplots(1, n_pos, figsize=(n_pos * 4, 3.5), squeeze=False)
    colors = ['#e74c3c', '#2980b9', '#27ae60', '#8e44ad']

    for j, (row, col) in enumerate(positions):
        ax = axes[0][j]
        for i, (img, lab) in enumerate(zip(imgs, labels)):
            spec = img[:, row, col]
            ax.plot(wl, spec, label=lab, color=colors[i % len(colors)], linewidth=1.5)
        ax.set_title(f'Pixel ({row},{col})', fontsize=9)
        ax.set_xlabel('Band')
        ax.set_ylabel('Intensity')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_path)


# ──────────────────────────────────────────────
# pred vs GT 并排对比
# ──────────────────────────────────────────────

def show_comparison(pred, gt, band=14, save_path=None, title=''):
    """
    单波段 pred / GT / 误差图 三联展示。

    pred, gt: [C, H, W]
    band:     要显示的波段索引
    """
    pred = _to_np(pred)
    gt   = _to_np(gt)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    vmin, vmax = 0, 1

    axes[0].imshow(gt[band],   cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'GT  band={band}', fontsize=10)
    axes[0].axis('off')

    axes[1].imshow(pred[band], cmap='gray', vmin=vmin, vmax=vmax)
    axes[1].set_title(f'Pred band={band}', fontsize=10)
    axes[1].axis('off')

    err = np.abs(pred[band] - gt[band])
    im  = axes[2].imshow(err, cmap='hot', vmin=0, vmax=0.1)
    axes[2].set_title(f'|Error| band={band}', fontsize=10)
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ──────────────────────────────────────────────
# 误差热图（平均所有波段）
# ──────────────────────────────────────────────

def show_error_map(pred, gt, save_path=None, title=''):
    """
    展示所有波段平均绝对误差的空间分布热图。

    pred, gt: [C, H, W]
    """
    pred = _to_np(pred)
    gt   = _to_np(gt)
    err  = np.abs(pred - gt).mean(axis=0)   # [H, W]

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    im = ax.imshow(err, cmap='hot', vmin=0, vmax=err.max())
    ax.set_title(title or 'Mean Absolute Error', fontsize=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ──────────────────────────────────────────────
# 频域幅度谱（log 尺度）
# ──────────────────────────────────────────────

def show_freq_magnitude(img, bands=None, title='', save_path=None):
    """
    展示图像若干波段的频域幅度谱（log(1 + |FFT_shifted|)，DC 居中）。

    img:   [C, H, W] tensor 或 ndarray
    bands: 要显示的波段列表（默认均匀取 6 个）
    """
    img = _to_np(img)
    C = img.shape[0]
    if bands is None:
        bands = np.linspace(0, C - 1, 6, dtype=int).tolist()

    n = len(bands)
    fig, axes = plt.subplots(1, n, figsize=(n * 3, 3))
    if n == 1:
        axes = [axes]

    for ax, b in zip(axes, bands):
        fft = np.fft.fftshift(np.fft.fft2(img[b]))
        mag = np.log1p(np.abs(fft))
        ax.imshow(mag, cmap='inferno')
        ax.set_title(f'Band {b}', fontsize=9)
        ax.axis('off')

    fig.suptitle(title or 'Freq Magnitude (log)', fontsize=11)
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ──────────────────────────────────────────────
# 频域并排对比（pred vs GT，单波段）
# ──────────────────────────────────────────────

def show_freq_comparison(pred, gt, band=14, save_path=None, title=''):
    """
    单波段频域幅度谱：pred / GT / 差值图 三联展示。

    pred, gt: [C, H, W]
    """
    pred = _to_np(pred)
    gt   = _to_np(gt)

    def _fft_mag(x_band):
        return np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(x_band))))

    mag_pred = _fft_mag(pred[band])
    mag_gt   = _fft_mag(gt[band])
    mag_diff = np.abs(mag_pred - mag_gt)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    axes[0].imshow(mag_gt,   cmap='inferno')
    axes[0].set_title(f'GT Freq  band={band}', fontsize=10)
    axes[0].axis('off')

    axes[1].imshow(mag_pred, cmap='inferno')
    axes[1].set_title(f'Pred Freq  band={band}', fontsize=10)
    axes[1].axis('off')

    im = axes[2].imshow(mag_diff, cmap='hot')
    axes[2].set_title(f'|Diff|  band={band}', fontsize=10)
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(title or f'Frequency Domain Comparison  band={band}', fontsize=11)
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ──────────────────────────────────────────────
# 径向平均功率谱密度（RAPSD）对比曲线
# ──────────────────────────────────────────────

def show_rapsd(pred, gt, save_path=None, title=''):
    """
    绘制 pred 和 GT 的径向平均功率谱密度（RAPSD）曲线对比。

    pred, gt: [C, H, W]
    """
    pred = _to_np(pred)
    gt   = _to_np(gt)
    C    = pred.shape[0]

    def _rapsd_np(img):
        """img: [C, H, W] → 1D RAPSD ndarray"""
        H, W = img.shape[1], img.shape[2]
        power = np.abs(np.fft.fftshift(
            np.fft.fft2(img), axes=(-2, -1))) ** 2     # [C, H, W]
        power_mean = power.mean(0)                      # [H, W]
        cy, cx = H // 2, W // 2
        y_idx = np.arange(H) - cy
        x_idx = np.arange(W) - cx
        r = np.sqrt(y_idx[:, None] ** 2 + x_idx[None, :] ** 2).astype(int)
        r_max = r.max() + 1
        rapsd = np.zeros(r_max)
        count = np.zeros(r_max)
        np.add.at(rapsd, r.flatten(), power_mean.flatten())
        np.add.at(count, r.flatten(), 1)
        return rapsd / np.maximum(count, 1)

    rapsd_pred = _rapsd_np(pred)
    rapsd_gt   = _rapsd_np(gt)
    freq_axis  = np.arange(len(rapsd_gt))

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.semilogy(freq_axis, rapsd_gt,   label='GT',   color='#2980b9', linewidth=1.8)
    ax.semilogy(freq_axis, rapsd_pred, label='Pred', color='#e74c3c', linewidth=1.8, linestyle='--')
    ax.set_xlabel('Radial Frequency (pixel)', fontsize=10)
    ax.set_ylabel('Power (log scale)', fontsize=10)
    ax.set_title(title or 'Radially Averaged Power Spectral Density', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ──────────────────────────────────────────────
# 一键输出全部图
# ──────────────────────────────────────────────

def show_all(pred, gt, scene_idx=0, save_dir='result/show/viz'):
    """
    对第 scene_idx 个场景输出所有可视化图。

    pred, gt: [N, C, H, W] tensor 或 ndarray
    """
    if hasattr(pred, 'detach'):
        pred = pred.detach().cpu().numpy()
    if hasattr(gt, 'detach'):
        gt = gt.detach().cpu().numpy()

    p = pred[scene_idx]   # [C, H, W]
    g = gt[scene_idx]

    save_dir = Path(save_dir) / f'scene_{scene_idx:02d}'
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. 多波段灰度
    show_bands(p, title=f'Pred Scene {scene_idx}',
               save_path=str(save_dir / 'pred_bands.png'))
    show_bands(g, title=f'GT Scene {scene_idx}',
               save_path=str(save_dir / 'gt_bands.png'))

    # 2. 伪彩色 RGB
    show_rgb(p, save_path=str(save_dir / 'pred_rgb.png'), title='Pred RGB')
    show_rgb(g, save_path=str(save_dir / 'gt_rgb.png'),   title='GT RGB')

    # 3. 光谱曲线（5 个位置）
    H, W = p.shape[1], p.shape[2]
    positions = [
        (H // 2,     W // 2),
        (H // 4,     W // 4),
        (3 * H // 4, W // 4),
        (H // 4,     3 * W // 4),
        (3 * H // 4, 3 * W // 4),
    ]
    show_spectrum([p, g], ['Pred', 'GT'], positions,
                  save_path=str(save_dir / 'spectra.png'))

    # 4. 单波段对比（中间波段）
    mid = p.shape[0] // 2
    show_comparison(p, g, band=mid,
                    save_path=str(save_dir / f'compare_band{mid}.png'),
                    title=f'Scene {scene_idx}')

    # 5. 误差热图
    show_error_map(p, g, save_path=str(save_dir / 'error_map.png'),
                   title=f'Scene {scene_idx} MAE')

    # 6. 频域幅度谱
    show_freq_magnitude(p, title=f'Pred Freq Scene {scene_idx}',
                        save_path=str(save_dir / 'pred_freq.png'))
    show_freq_magnitude(g, title=f'GT Freq Scene {scene_idx}',
                        save_path=str(save_dir / 'gt_freq.png'))

    # 7. 频域并排对比（中间波段）
    show_freq_comparison(p, g, band=mid,
                         save_path=str(save_dir / f'freq_compare_band{mid}.png'),
                         title=f'Scene {scene_idx}')

    # 8. 径向功率谱对比
    show_rapsd(p, g, save_path=str(save_dir / 'rapsd.png'),
               title=f'Scene {scene_idx} RAPSD')

    print(f"可视化图像保存到: {save_dir}")
