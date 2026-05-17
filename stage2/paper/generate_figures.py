"""
generate_figures.py — 生成论文级架构图

运行方式：
    cd G:\MachineLearning\CASSI\stage2\paper
    python generate_figures.py

输出：
    fig1_wpo3d_block.pdf/png     — WPO3D Block 内部结构
    fig2_wavemst_unet.pdf/png    — WaveMST-3D U-Net 整体架构
    fig3_unfolding.pdf/png       — Deep Unfolding (GAP) 框架
    fig4_enhanced.pdf/png        — 增强版：源项注入 + 色散修正
    fig5_wave_modulation.pdf/png — 波动方程频域调制细节
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ArrowStyle
import matplotlib.patheffects as pe
import numpy as np
import os

# ═══════════════════════════════════════════════
# 全局样式
# ═══════════════════════════════════════════════

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'mathtext.fontset': 'cm',
    'axes.linewidth': 0.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# 颜色方案
C = {
    'embed':     '#E8E8E8',   # 灰 - embedding / stem
    'wpo':       '#FFB347',   # 橙 - WPO3D 核心
    'wpo_dark':  '#E8941A',   # 深橙 - WPO border
    'fft':       '#FF6B6B',   # 红 - FFT
    'ffn':       '#98D8A0',   # 绿 - FFN
    'norm':      '#87CEEB',   # 浅蓝 - LayerNorm
    'conv':      '#C4C4C4',   # 灰 - Conv
    'mask':      '#DDA0DD',   # 紫 - Mask 操作
    'gd':        '#87CEEB',   # 蓝 - GD step
    'prior':     '#FFB347',   # 橙 - Prior (WaveMST)
    'source':    '#FF9999',   # 红 - Source injection
    'disp':      '#B19CD9',   # 紫 - Dispersive
    'rho':       '#FFFFB0',   # 黄 - ParaEstimator
    'output':    '#C8E6C9',   # 浅绿 - 输出
    'input':     '#BBDEFB',   # 浅蓝 - 输入
    'residual':  '#FFFFFF',   # 白 - 残差
    'formula':   '#FFF8E1',   # 米黄 - 公式背景
    'stage_bg':  '#F5F5F5',   # 浅灰 - stage 背景
    'down':      '#90CAF9',   # 蓝 - 下采样
    'up':        '#A5D6A7',   # 绿 - 上采样
    'skip':      '#FFCC80',   # 橙 - skip connection
    'text':      '#333333',
    'arrow':     '#555555',
}


def rounded_box(ax, x, y, w, h, label, color, fontsize=8,
                edgecolor=None, text_color='#333333', lw=0.8,
                alpha=1.0, style='round,pad=0.05', zorder=2,
                bold=False, math=False):
    """画一个圆角矩形 + 居中文字"""
    if edgecolor is None:
        edgecolor = _darken(color, 0.3)
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=style,
        facecolor=color, edgecolor=edgecolor,
        linewidth=lw, alpha=alpha, zorder=zorder,
    )
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    if math:
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=fontsize, color=text_color, zorder=zorder+1,
                fontweight=weight)
    else:
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=fontsize, color=text_color, zorder=zorder+1,
                fontweight=weight)
    return box


def _darken(hex_color, factor=0.2):
    """把颜色变暗"""
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    darker = tuple(max(0, int(c * (1 - factor))) for c in rgb)
    return '#{:02x}{:02x}{:02x}'.format(*darker)


def arrow(ax, x1, y1, x2, y2, color='#555555', lw=1.0,
          style='->', head_width=0.06, shrinkA=0, shrinkB=0, zorder=3):
    """画箭头"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle=style, color=color,
                    lw=lw, shrinkA=shrinkA, shrinkB=shrinkB,
                ),
                zorder=zorder)


def plus_circle(ax, x, y, r=0.12, fontsize=10):
    """画 ⊕ 符号"""
    circle = plt.Circle((x, y), r, fill=True, facecolor='white',
                         edgecolor='#555555', linewidth=0.8, zorder=4)
    ax.add_patch(circle)
    ax.text(x, y, '+', ha='center', va='center', fontsize=fontsize,
            color='#555555', fontweight='bold', zorder=5)


def mul_circle(ax, x, y, r=0.12, fontsize=10):
    """画 ⊙ 符号"""
    circle = plt.Circle((x, y), r, fill=True, facecolor='white',
                         edgecolor='#555555', linewidth=0.8, zorder=4)
    ax.add_patch(circle)
    ax.text(x, y, r'$\odot$', ha='center', va='center', fontsize=fontsize,
            color='#555555', zorder=5)


def bracket_text(ax, x, y, text, fontsize=7, color='#666666'):
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=color, style='italic')


# ═══════════════════════════════════════════════
# Figure 1: WPO3D Block 内部结构
# ═══════════════════════════════════════════════

def fig1_wpo3d_block():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2),
                             gridspec_kw={'width_ratios': [1.6, 1]})

    # ── (a) WPO3D 核心模块 ──
    ax = axes[0]
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-1.2, 4.0)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(a) WPO3D — 3D Wave Propagation Operator', fontsize=10,
                 fontweight='bold', pad=10)

    bw, bh = 1.2, 0.55  # box width, height

    # 输入 x
    rounded_box(ax, 0, 1.5, 0.8, bh, r'$\mathbf{x}$', C['input'], fontsize=9, bold=True, math=True)

    # Mask Gate
    rounded_box(ax, 1.3, 2.3, 1.4, bh, 'Mask Gate', C['mask'], fontsize=8)
    ax.text(2.0, 2.0, r'$\mathsf{gate} = \epsilon + (1{-}\epsilon)\,\mathbf{M}$',
            fontsize=6.5, ha='center', color='#666666')
    # mask input
    rounded_box(ax, 1.3, 3.2, 1.4, bh, r'Mask $\mathbf{M}$', C['mask'],
                fontsize=8, alpha=0.6)
    arrow(ax, 2.0, 3.2, 2.0, 2.85)

    # Phi, Psi
    rounded_box(ax, 1.3, 0.5, 0.6, bh, r'$\Phi$', '#FFE0B2', fontsize=9, bold=True)
    rounded_box(ax, 2.1, 0.5, 0.6, bh, r'$\Psi$', '#FFE0B2', fontsize=9, bold=True)

    arrow(ax, 0.8, 1.77, 1.3, 1.77)  # x → split
    # x to Phi, Psi
    ax.plot([0.8, 1.05, 1.05], [1.77, 1.77, 0.77], color=C['arrow'], lw=0.8)
    arrow(ax, 1.05, 0.77, 1.3, 0.77)
    ax.plot([1.05, 1.05], [0.77, 0.77], color=C['arrow'], lw=0.8)
    ax.plot([1.05, 1.85, 1.85], [0.77, 0.77, 0.77], color=C['arrow'], lw=0.8)
    # Mask gate → multiply
    arrow(ax, 1.6, 2.3, 1.6, 1.05)
    arrow(ax, 2.4, 2.3, 2.4, 1.05)

    # u0, v0
    bracket_text(ax, 1.6, 1.2, r'$u_0$', fontsize=8, color='#E65100')
    bracket_text(ax, 2.4, 1.2, r'$v_0$', fontsize=8, color='#E65100')

    # 3D rFFT
    rounded_box(ax, 3.2, 1.5, 1.3, bh, '3D rFFT', C['fft'], fontsize=9, bold=True)
    arrow(ax, 1.9, 0.77, 3.2, 1.77)
    arrow(ax, 2.7, 0.77, 3.2, 1.77)

    # Wave Modulation (核心!)
    wm_x, wm_w = 5.0, 2.8
    rounded_box(ax, wm_x, 1.2, wm_w, 1.1, '', C['wpo'], fontsize=9,
                edgecolor=C['wpo_dark'], lw=1.5, style='round,pad=0.08')
    ax.text(wm_x + wm_w/2, 1.95, 'Wave Modulation', ha='center', va='center',
            fontsize=9, fontweight='bold', color='#5D4037')
    ax.text(wm_x + wm_w/2, 1.5, r'$e^{-\frac{\alpha t}{2}}\!\left[\hat{u}_0 \cos\omega_d t + \frac{\hat{v}_0 + \frac{\alpha}{2}\hat{u}_0}{\omega_d}\sin\omega_d t\right]$',
            fontsize=7.5, ha='center', va='center', color='#4E342E',
            bbox=dict(boxstyle='round,pad=0.15', facecolor=C['formula'],
                      edgecolor='#DDD0A0', lw=0.5))

    arrow(ax, 4.5, 1.77, 5.0, 1.77)

    # Learnable params annotation
    ax.text(wm_x + wm_w/2, 0.85, r'$\alpha,\, v_s,\, v_\lambda,\, t$ (learnable)',
            fontsize=6.5, ha='center', color='#888888', style='italic')

    # 3D irFFT
    rounded_box(ax, 8.3, 1.5, 1.3, bh, '3D irFFT', C['fft'], fontsize=9, bold=True)
    arrow(ax, 7.8, 1.77, 8.3, 1.77)

    # Output projection
    rounded_box(ax, 10.0, 2.3, 0.7, bh, 'LN', C['norm'], fontsize=8)
    rounded_box(ax, 10.0, 1.5, 0.7, bh, 'SiLU', '#E8E8E8', fontsize=8)
    rounded_box(ax, 10.0, 0.6, 0.85, bh, 'Conv1x1', C['conv'], fontsize=7.5)

    arrow(ax, 9.6, 1.77, 10.0, 2.57)  # irFFT → LN
    # x skip to SiLU
    ax.annotate('', xy=(10.0, 1.77), xytext=(0.8, 1.77),
                arrowprops=dict(arrowstyle='->', color='#999999',
                                lw=0.7, linestyle='--',
                                connectionstyle='arc3,rad=-0.3'))
    ax.text(5.5, 3.3, r'$\mathbf{x}$ (skip)', fontsize=7, color='#999999',
            style='italic', ha='center')

    mul_circle(ax, 10.35, 2.0, r=0.1, fontsize=8)
    arrow(ax, 10.35, 2.3, 10.35, 2.1)
    arrow(ax, 10.35, 1.9, 10.35, 1.15)
    arrow(ax, 10.35, 1.5, 10.35, 1.2)

    # 输出
    rounded_box(ax, 10.7, 0.6, 0.7, bh, 'out', C['output'], fontsize=9, bold=True)
    arrow(ax, 10.85, 0.6, 10.7+0.35, 0.6)

    # ── (b) WPO3D Block ──
    ax = axes[1]
    ax.set_xlim(-0.3, 4.5)
    ax.set_ylim(-0.5, 6.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(b) WPO3D Block', fontsize=10, fontweight='bold', pad=10)

    cx = 1.5  # center x
    bw2, bh2 = 1.5, 0.5

    # 输入
    rounded_box(ax, cx, 0, bw2, bh2, r'Input $\mathbf{x}$', C['input'],
                fontsize=8, bold=True)

    # LN1
    rounded_box(ax, cx, 0.9, bw2, bh2, 'LayerNorm', C['norm'], fontsize=8)
    arrow(ax, cx+bw2/2, 0.5, cx+bw2/2, 0.9)

    # WPO3D
    rounded_box(ax, cx, 1.8, bw2, 0.65, 'WPO3D', C['wpo'], fontsize=9,
                bold=True, edgecolor=C['wpo_dark'], lw=1.2)
    arrow(ax, cx+bw2/2, 1.4, cx+bw2/2, 1.8)

    # mask input to WPO3D
    rounded_box(ax, 3.5, 1.9, 0.8, 0.45, 'Mask', C['mask'], fontsize=7, alpha=0.7)
    arrow(ax, 3.5, 2.12, cx+bw2, 2.12)

    # Add1
    plus_circle(ax, cx+bw2/2, 2.85, r=0.13)
    arrow(ax, cx+bw2/2, 2.45, cx+bw2/2, 2.72)
    # skip from input
    ax.plot([cx-0.15, cx-0.15, cx+bw2/2-0.13],
            [0.25, 2.85, 2.85], color=C['arrow'], lw=0.7, linestyle='--')

    # LN2
    rounded_box(ax, cx, 3.3, bw2, bh2, 'LayerNorm', C['norm'], fontsize=8)
    arrow(ax, cx+bw2/2, 2.98, cx+bw2/2, 3.3)

    # FFN
    rounded_box(ax, cx, 4.2, bw2, 0.65, 'FFN', C['ffn'], fontsize=9, bold=True)
    arrow(ax, cx+bw2/2, 3.8, cx+bw2/2, 4.2)

    # FFN detail
    ax.text(cx+bw2+0.15, 4.52, 'Conv1x1 → GELU\n→ DWConv3x3 → GELU\n→ Conv1x1',
            fontsize=5.5, color='#666666', va='center',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='#F0F0F0',
                      edgecolor='#CCCCCC', lw=0.4))

    # Add2
    plus_circle(ax, cx+bw2/2, 5.25, r=0.13)
    arrow(ax, cx+bw2/2, 4.85, cx+bw2/2, 5.12)
    # skip from Add1
    ax.plot([cx-0.15, cx-0.15, cx+bw2/2-0.13],
            [2.85, 5.25, 5.25], color=C['arrow'], lw=0.7, linestyle='--')

    # Output
    rounded_box(ax, cx, 5.7, bw2, bh2, 'Output', C['output'],
                fontsize=8, bold=True)
    arrow(ax, cx+bw2/2, 5.38, cx+bw2/2, 5.7)

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════
# Figure 2: WaveMST-3D U-Net 整体架构
# ═══════════════════════════════════════════════

def fig2_wavemst_unet():
    fig, ax = plt.subplots(1, 1, figsize=(11, 4.5))
    ax.set_xlim(-0.5, 14)
    ax.set_ylim(-1.2, 4.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('WaveMST-3D Architecture (U-Net Backbone with WPO3D Blocks)',
                 fontsize=11, fontweight='bold', pad=8)

    bw, bh = 1.0, 0.5

    # ── 输入 ──
    rounded_box(ax, 0, 2.0, 1.0, bh, r'Input', C['input'], fontsize=8, bold=True)
    ax.text(0.5, 1.7, r'$[B, 28, H, W]$', fontsize=6, ha='center', color='#888')

    # ── Embedding ──
    rounded_box(ax, 1.4, 2.0, 1.2, bh, 'Conv3x3\nLeakyReLU', C['embed'], fontsize=7)
    arrow(ax, 1.0, 2.25, 1.4, 2.25)

    # ── Encoder level 1 ──
    ex1 = 3.0
    rounded_box(ax, ex1, 2.0, 1.5, bh, r'WPO3D Block $\times 2$', C['wpo'],
                fontsize=7, edgecolor=C['wpo_dark'])
    arrow(ax, 2.6, 2.25, ex1, 2.25)

    # Downsample 1
    ds1 = 4.8
    rounded_box(ax, ds1, 2.0, 0.8, bh, r'$\downarrow$ 2x', C['down'], fontsize=7.5, bold=True)
    arrow(ax, ex1+1.5, 2.25, ds1, 2.25)

    # ── Encoder level 2 ──
    ex2 = 6.0
    rounded_box(ax, ex2, 1.0, 1.5, bh, r'WPO3D Block $\times 2$', C['wpo'],
                fontsize=7, edgecolor=C['wpo_dark'])
    ax.plot([ds1+0.4, ds1+0.4, ex2], [2.0, 1.25, 1.25], color=C['arrow'], lw=0.8)
    arrow(ax, ds1+0.4, 1.25, ex2, 1.25, style='->')

    # Downsample 2
    ds2_x = 7.8
    rounded_box(ax, ds2_x, 1.0, 0.8, bh, r'$\downarrow$ 2x', C['down'], fontsize=7.5, bold=True)
    arrow(ax, ex2+1.5, 1.25, ds2_x, 1.25)

    # ── Bottleneck ──
    bn_x = 6.3
    rounded_box(ax, bn_x, 0.0, 1.8, 0.55, r'WPO3D Block $\times 2$', C['wpo'],
                fontsize=7.5, edgecolor=C['wpo_dark'], bold=True)
    ax.text(bn_x+0.9, -0.3, 'Bottleneck', fontsize=7, ha='center',
            color='#777', style='italic')
    ax.plot([ds2_x+0.4, ds2_x+0.4, bn_x+0.9, bn_x+0.9],
            [1.0, 0.28, 0.28, 0.55], color=C['arrow'], lw=0.8)

    # ── Decoder level 2 ──
    us2 = 8.5
    rounded_box(ax, us2, 1.0, 0.8, bh, r'$\uparrow$ 2x', C['up'], fontsize=7.5, bold=True)
    ax.plot([bn_x+1.8, us2+0.4, us2+0.4], [0.28, 0.28, 1.0],
            color=C['arrow'], lw=0.8)

    # Skip connection 2
    ax.annotate('', xy=(us2+0.8, 1.25), xytext=(ex2+1.5, 1.25),
                arrowprops=dict(arrowstyle='->', color=C['skip'],
                                lw=1.5, linestyle='-',
                                connectionstyle='arc3,rad=0.35'))
    ax.text(8.1, 1.7, 'skip', fontsize=6, color='#CC8800', style='italic')

    # Cat + Conv + Blocks
    dx2 = 9.6
    rounded_box(ax, dx2, 1.0, 0.5, bh, 'Cat\n1x1', C['conv'], fontsize=6)
    arrow(ax, us2+0.8, 1.25, dx2, 1.25)

    dx2b = 10.3
    rounded_box(ax, dx2b, 1.0, 1.5, bh, r'WPO3D Block $\times 2$', C['wpo'],
                fontsize=7, edgecolor=C['wpo_dark'])
    arrow(ax, dx2+0.5, 1.25, dx2b, 1.25)

    # ── Decoder level 1 ──
    us1 = 10.3
    rounded_box(ax, us1, 2.0, 0.8, bh, r'$\uparrow$ 2x', C['up'], fontsize=7.5, bold=True)
    ax.plot([dx2b+1.5, dx2b+1.5+0.2], [1.25, 1.25], color=C['arrow'], lw=0.8)
    ax.plot([dx2b+1.5+0.2, dx2b+1.5+0.2, us1+0.4], [1.25, 2.25, 2.25],
            color=C['arrow'], lw=0.8)

    # Skip connection 1
    ax.annotate('', xy=(us1+0.8, 2.25), xytext=(ex1+1.5, 2.25),
                arrowprops=dict(arrowstyle='->', color=C['skip'],
                                lw=1.5, linestyle='-',
                                connectionstyle='arc3,rad=-0.35'))
    ax.text(7.5, 3.2, 'skip', fontsize=6, color='#CC8800', style='italic')

    # Cat + Conv + Blocks dec1
    ddx1 = 11.3
    rounded_box(ax, ddx1, 2.0, 0.5, bh, 'Cat\n1x1', C['conv'], fontsize=6)
    arrow(ax, us1+0.8, 2.25, ddx1, 2.25)

    ddx1b = 12.0
    rounded_box(ax, ddx1b, 2.0, 1.5, bh, r'WPO3D Block $\times 2$', C['wpo'],
                fontsize=7, edgecolor=C['wpo_dark'])
    arrow(ax, ddx1+0.5, 2.25, ddx1b, 2.25)

    # ── Output ──
    out_x = 12.0
    rounded_box(ax, out_x, 3.2, 1.0, bh, 'Conv3x3', C['conv'], fontsize=7)
    ax.plot([ddx1b+0.75, ddx1b+0.75, out_x+0.5], [2.5, 3.45, 3.45],
            color=C['arrow'], lw=0.8)

    plus_circle(ax, 13.3, 3.45, r=0.12)
    arrow(ax, out_x+1.0, 3.45, 13.17, 3.45)
    # global skip from input
    ax.annotate('', xy=(13.3, 3.33), xytext=(0.5, 2.5),
                arrowprops=dict(arrowstyle='->', color='#AAAAAA',
                                lw=0.7, linestyle='--',
                                connectionstyle='arc3,rad=-0.25'))
    ax.text(6, 4.0, r'Global Residual ($+\mathbf{x}$)', fontsize=7,
            color='#999999', ha='center', style='italic')

    rounded_box(ax, 13.0, 2.8, 0.8, bh, 'Output', C['output'], fontsize=8, bold=True)
    arrow(ax, 13.3, 3.33, 13.3, 3.3)

    # Mask 路径标注
    ax.text(0.5, 3.7, r'Mask $\mathbf{M}$: 与特征并行下采样 (Conv4x4↓ + Sigmoid)',
            fontsize=7, color='#9C27B0', style='italic',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='#F3E5F5',
                      edgecolor='#CE93D8', lw=0.4))

    # 维度标注
    ax.text(3.75, 1.7, r'$C$', fontsize=6.5, ha='center', color='#666')
    ax.text(6.75, 0.7, r'$2C$', fontsize=6.5, ha='center', color='#666')
    ax.text(7.2, -0.3, r'$4C$', fontsize=6.5, ha='center', color='#666')

    return fig


# ═══════════════════════════════════════════════
# Figure 3: Deep Unfolding (GAP) 框架
# ═══════════════════════════════════════════════

def fig3_unfolding():
    fig, ax = plt.subplots(1, 1, figsize=(13, 5.5))
    ax.set_xlim(-1, 14.5)
    ax.set_ylim(-2.5, 5.0)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Deep Unfolding Framework (K-stage GAP with WaveMST-3D Prior)',
                 fontsize=11, fontweight='bold', pad=8)

    bw, bh = 1.0, 0.55

    # ── 输入 ──
    rounded_box(ax, -0.8, 2.5, 0.9, 0.7, r'$\mathbf{g}$'+'\nmeas.', C['input'],
                fontsize=8, bold=True)
    rounded_box(ax, -0.8, 1.3, 0.9, 0.7, r'$\mathbf{\Phi}$'+'\nmask', C['mask'],
                fontsize=8, bold=True)

    # ── 初始化 ──
    init_x = 0.6
    rounded_box(ax, init_x, 2.2, 1.8, 1.3, '', C['stage_bg'],
                fontsize=7, edgecolor='#CCCCCC', lw=0.8,
                style='round,pad=0.1')
    ax.text(init_x+0.9, 3.3, 'Initialize', fontsize=8, ha='center',
            fontweight='bold', color='#555')
    rounded_box(ax, init_x+0.15, 2.85, 1.5, 0.35,
                r'shift_back($\mathbf{g}$/$C$$\times$2)', '#E3F2FD', fontsize=6.5)
    rounded_box(ax, init_x+0.15, 2.35, 1.5, 0.35,
                r'Conv1x1([$\mathbf{f}_0$, $\Phi$])', C['conv'], fontsize=6.5)

    arrow(ax, 0.1, 2.85, init_x, 2.85)
    arrow(ax, 0.1, 1.65, init_x+0.15, 2.35)

    # ── Stage 背景 ──
    stage_x = 3.0
    stage_w = 7.8
    rect = FancyBboxPatch((stage_x, -1.8), stage_w, 5.8,
                           boxstyle='round,pad=0.15',
                           facecolor='#FAFAFA', edgecolor='#BDBDBD',
                           linewidth=1.2, linestyle='--', zorder=0)
    ax.add_patch(rect)
    ax.text(stage_x+stage_w/2, 4.2, r'$\times\, K$ stages $(k = 1, \ldots, K)$',
            fontsize=10, ha='center', fontweight='bold', color='#555')

    # ── GD Step ──
    gd_x = 3.5
    rounded_box(ax, gd_x, 1.5, 3.2, 2.2, '', '#E3F2FD',
                fontsize=7, edgecolor='#90CAF9', lw=1.0,
                style='round,pad=0.1', alpha=0.7)
    ax.text(gd_x+1.6, 3.5, 'GD Step (Data Fidelity)', fontsize=8.5,
            ha='center', fontweight='bold', color='#1565C0')

    # rho estimator
    rounded_box(ax, gd_x+0.1, 2.7, 1.2, 0.45, r'$\rho_k$ Estimator', C['rho'],
                fontsize=7)
    ax.text(gd_x+0.7, 2.4, r'CNN $\rightarrow$ scalar', fontsize=5.5,
            ha='center', color='#888')

    # GD formula
    ax.text(gd_x+1.6, 1.85,
            r'$\mathbf{z} = \mathbf{f} + \rho_k \cdot \Phi^\top\!\frac{\mathbf{g} - \Phi\mathbf{f}}{\Phi\Phi^\top}$',
            fontsize=9, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.15', facecolor=C['formula'],
                      edgecolor='#DDD0A0', lw=0.5))

    # Phi_f, residual
    ax.text(gd_x+1.6, 1.15,
            r'$\Phi\mathbf{f}$: shift $\circ$ mask $\circ$ sum'
            r'$\quad$'
            r'$\Phi^\top\mathbf{r}$: broadcast $\circ$ mask $\circ$ shift\_back',
            fontsize=5.5, ha='center', color='#666')

    # ── Prior Step ──
    pr_x = 7.2
    rounded_box(ax, pr_x, 1.6, 2.5, 1.8, '', '#FFF3E0',
                fontsize=7, edgecolor='#FFB74D', lw=1.0,
                style='round,pad=0.1', alpha=0.7)
    ax.text(pr_x+1.25, 3.2, 'Prior Step (Physics Prior)', fontsize=8.5,
            ha='center', fontweight='bold', color='#E65100')

    rounded_box(ax, pr_x+0.25, 2.0, 2.0, 0.8, 'WaveMST-3D\n(WPO3D U-Net)',
                C['wpo'], fontsize=8, bold=True, edgecolor=C['wpo_dark'], lw=1.2)

    ax.text(pr_x+1.25, 1.7,
            r'$\mathbf{f}^{(k)} = \mathrm{WPO3D}(\mathbf{z},\, \Phi)$',
            fontsize=7.5, ha='center', color='#555')

    # Arrows in stage
    arrow(ax, 2.4, 2.55, gd_x, 2.55)  # init → GD
    arrow(ax, gd_x+3.2, 2.55, pr_x, 2.55)  # GD → Prior
    ax.text(gd_x+3.2+0.15, 2.75, r'$\mathbf{z}$', fontsize=9, color='#E65100',
            fontweight='bold')
    ax.text(gd_x-0.3, 2.75, r'$\mathbf{f}$', fontsize=9, color='#1565C0',
            fontweight='bold')

    # ── 反馈箭头 ──
    ax.annotate('', xy=(gd_x, 0.8), xytext=(pr_x+2.5, 0.8),
                arrowprops=dict(arrowstyle='->', color='#888888',
                                lw=1.2, linestyle='-',
                                connectionstyle='arc3,rad=0.0'))
    ax.text(gd_x+3.5, 0.5, r'$\mathbf{f}^{(k)} \rightarrow$ next stage',
            fontsize=7, ha='center', color='#888')

    # ── 输出 ──
    out_x = 11.5
    rounded_box(ax, out_x, 2.2, 1.3, 0.7, r'$\mathbf{f}^{(K)}$' + '\nOutput',
                C['output'], fontsize=8, bold=True)
    arrow(ax, pr_x+2.5, 2.55, out_x, 2.55, lw=1.5, color='#E65100')

    # ── Multi-stage Loss ──
    rounded_box(ax, 3.5, -1.5, 7.0, 0.8, '', C['formula'],
                edgecolor='#DDD0A0', lw=0.8, style='round,pad=0.1')
    ax.text(7.0, -1.1,
            r'$\mathcal{L} = \sqrt{\mathrm{MSE}(\mathbf{f}^K, \mathrm{GT})} '
            r'+ 0.7\sqrt{\mathrm{MSE}(\mathbf{f}^{K\!-\!1}, \mathrm{GT})} '
            r'+ 0.5\sqrt{\mathrm{MSE}(\mathbf{f}^{K\!-\!2}, \mathrm{GT})} '
            r'+ 0.3\sqrt{\mathrm{MSE}(\mathbf{f}^{K\!-\!3}, \mathrm{GT})}$',
            fontsize=7, ha='center', va='center', color='#555')
    ax.text(7.0, -1.65, 'Multi-Stage Loss (DPU-style)', fontsize=7,
            ha='center', color='#888', style='italic')

    # Φ^T annotation
    ax.text(0.1, 0.8, r'$\Phi\Phi^\top$: precomputed',
            fontsize=6.5, color='#666', style='italic')

    return fig


# ═══════════════════════════════════════════════
# Figure 4: 增强版 Unfolding（源项注入 + 色散修正）
# ═══════════════════════════════════════════════

def fig4_enhanced():
    fig, ax = plt.subplots(1, 1, figsize=(14, 6.5))
    ax.set_xlim(-1.5, 15)
    ax.set_ylim(-2.5, 6.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Enhanced Unfolding: Source Injection + Dispersive Correction',
                 fontsize=11, fontweight='bold', pad=8)

    bw, bh = 1.0, 0.55

    # ── 输入 ──
    rounded_box(ax, -1.0, 3.5, 0.9, 0.7, r'$\mathbf{g}$'+'\nmeas.', C['input'],
                fontsize=8, bold=True)
    rounded_box(ax, -1.0, 2.2, 0.9, 0.7, r'$\Phi$'+'\nmask', C['mask'],
                fontsize=8, bold=True)

    # ── 初始化 ──
    init_x = 0.4
    rounded_box(ax, init_x, 2.8, 1.6, 1.2, '', C['stage_bg'],
                fontsize=7, edgecolor='#CCCCCC')
    ax.text(init_x+0.8, 3.8, 'Init', fontsize=8, ha='center',
            fontweight='bold', color='#555')
    rounded_box(ax, init_x+0.1, 3.2, 1.4, 0.35,
                r'shift_back($\mathbf{g}$)', '#E3F2FD', fontsize=6.5)
    rounded_box(ax, init_x+0.1, 2.9, 1.4, 0.3,
                r'Conv1x1([$\mathbf{f}_0$,$\Phi$])', C['conv'], fontsize=6)

    arrow(ax, -0.1, 3.85, init_x, 3.4)
    arrow(ax, -0.1, 2.55, init_x+0.1, 2.9)

    # ── Precompute PhiT_g ──
    rounded_box(ax, -1.0, 5.0, 1.6, 0.6, r'$\Phi^\top\mathbf{g}$'+'\n(precomputed)',
                C['source'], fontsize=7, alpha=0.7)
    arrow(ax, -0.55, 4.2, -0.55, 5.0)

    # ── Stage 背景 ──
    stage_x = 2.5
    stage_w = 10.0
    rect = FancyBboxPatch((stage_x, -1.5), stage_w, 7.2,
                           boxstyle='round,pad=0.15',
                           facecolor='#FAFAFA', edgecolor='#BDBDBD',
                           linewidth=1.2, linestyle='--', zorder=0)
    ax.add_patch(rect)
    ax.text(stage_x+stage_w/2, 5.9, r'$\times\, K$ stages',
            fontsize=10, ha='center', fontweight='bold', color='#555')

    # ── GD Step ──
    gd_x = 3.0
    rounded_box(ax, gd_x, 2.5, 2.8, 2.0, '', '#E3F2FD',
                fontsize=7, edgecolor='#90CAF9', lw=1.0,
                style='round,pad=0.08', alpha=0.6)
    ax.text(gd_x+1.4, 4.3, 'GD Step', fontsize=8.5,
            ha='center', fontweight='bold', color='#1565C0')

    # rho
    rounded_box(ax, gd_x+0.1, 3.7, 1.1, 0.4, r'$\rho_k$ Est.', C['rho'], fontsize=7)

    # GD formula
    ax.text(gd_x+1.4, 2.9,
            r'$\mathbf{z} = \mathbf{f} + \rho_k \Phi^\top\!\frac{\mathbf{g} - \Phi\mathbf{f}}{\Phi\Phi^\top}$',
            fontsize=8, ha='center',
            bbox=dict(boxstyle='round,pad=0.12', facecolor=C['formula'],
                      edgecolor='#DDD0A0', lw=0.4))

    # ── 模块 A：源项注入（新增！）──
    src_x = 6.2
    rounded_box(ax, src_x, 2.8, 2.2, 1.6, '', '#FFEBEE',
                fontsize=7, edgecolor='#EF9A9A', lw=1.2,
                style='round,pad=0.08', alpha=0.8)
    ax.text(src_x+1.1, 4.2, r'Source Injection', fontsize=8.5,
            ha='center', fontweight='bold', color='#C62828')
    ax.text(src_x+1.1, 4.55, r'Module A', fontsize=6.5,
            ha='center', color='#E57373', style='italic')

    rounded_box(ax, src_x+0.1, 3.4, 2.0, 0.4, r'Cat[$\mathbf{z}$, $\Phi^\top\mathbf{g}$]',
                '#FFCDD2', fontsize=7)
    rounded_box(ax, src_x+0.1, 2.9, 2.0, 0.4, r'Conv1x1 $\rightarrow \mathbf{z}^\prime$',
                '#FFCDD2', fontsize=7)

    # PhiT_g arrow
    ax.annotate('', xy=(src_x+1.1, 4.55), xytext=(-0.2, 5.3),
                arrowprops=dict(arrowstyle='->', color='#E57373',
                                lw=1.0, linestyle='-',
                                connectionstyle='arc3,rad=-0.15'))

    # ── Prior Step ──
    pr_x = 8.8
    rounded_box(ax, pr_x, 2.8, 1.8, 1.6, '', '#FFF3E0',
                fontsize=7, edgecolor='#FFB74D', lw=1.0,
                style='round,pad=0.08', alpha=0.7)
    ax.text(pr_x+0.9, 4.2, 'Prior Step', fontsize=8.5,
            ha='center', fontweight='bold', color='#E65100')

    rounded_box(ax, pr_x+0.15, 3.15, 1.5, 0.8, 'WaveMST\n3D', C['wpo'],
                fontsize=8, bold=True, edgecolor=C['wpo_dark'], lw=1.2)
    ax.text(pr_x+0.9, 2.95, r'$\mathbf{f} = \mathrm{WPO}(\mathbf{z}^\prime, \Phi)$',
            fontsize=6.5, ha='center', color='#555')

    # ── 模块 C：色散修正（新增！）──
    disp_x = 11.0
    rounded_box(ax, disp_x, 2.5, 2.0, 2.2, '', '#EDE7F6',
                fontsize=7, edgecolor='#B39DDB', lw=1.2,
                style='round,pad=0.08', alpha=0.8)
    ax.text(disp_x+1.0, 4.5, r'Dispersive', fontsize=8.5,
            ha='center', fontweight='bold', color='#4527A0')
    ax.text(disp_x+1.0, 4.85, r'Module C', fontsize=6.5,
            ha='center', color='#7E57C2', style='italic')

    rounded_box(ax, disp_x+0.1, 3.7, 1.8, 0.4, r'$\delta v(\mathbf{r})$ Net',
                '#D1C4E9', fontsize=7)
    ax.text(disp_x+1.0, 3.45, r'DWConv$\rightarrow$ReLU$\rightarrow$Conv1x1$\rightarrow$Tanh',
            fontsize=5, ha='center', color='#666')

    rounded_box(ax, disp_x+0.1, 2.6, 1.8, 0.5, r'$\nabla^2\mathbf{f}$ (Laplacian)',
                '#D1C4E9', fontsize=7)
    ax.text(disp_x+1.0, 2.35, r'fixed kernel, reflect pad', fontsize=5,
            ha='center', color='#888', style='italic')

    # Formula
    ax.text(disp_x+1.0, -0.4,
            r'$\mathbf{f}_\mathrm{out} = \mathbf{f} + \gamma \cdot \delta v(\mathbf{r}) \cdot \nabla^2\mathbf{f}$',
            fontsize=8, ha='center',
            bbox=dict(boxstyle='round,pad=0.12', facecolor='#EDE7F6',
                      edgecolor='#B39DDB', lw=0.5))

    # ── Arrows ──
    arrow(ax, 2.0, 3.5, gd_x, 3.5)          # init → GD
    arrow(ax, gd_x+2.8, 3.5, src_x, 3.5)    # GD → Source
    arrow(ax, src_x+2.2, 3.5, pr_x, 3.5)     # Source → Prior
    arrow(ax, pr_x+1.8, 3.5, disp_x, 3.5)    # Prior → Disp

    # Labels on arrows
    ax.text(gd_x+2.8+0.15, 3.75, r'$\mathbf{z}$', fontsize=9,
            color='#1565C0', fontweight='bold')
    ax.text(src_x+2.2+0.1, 3.75, r'$\mathbf{z}^\prime$', fontsize=9,
            color='#C62828', fontweight='bold')
    ax.text(pr_x+1.8+0.1, 3.75, r'$\mathbf{f}$', fontsize=9,
            color='#E65100', fontweight='bold')

    # ── 反馈 ──
    ax.annotate('', xy=(gd_x, 1.7), xytext=(disp_x+2.0, 1.7),
                arrowprops=dict(arrowstyle='->', color='#888888',
                                lw=1.2, connectionstyle='arc3,rad=0.0'))
    ax.text(7.5, 1.35, r'$\mathbf{f}_\mathrm{out}^{(k)} \rightarrow$ next stage',
            fontsize=7.5, ha='center', color='#888')

    # ── 输出 ──
    rounded_box(ax, 13.2, 3.1, 1.2, 0.8, r'$\mathbf{f}^{(K)}$', C['output'],
                fontsize=10, bold=True)
    arrow(ax, disp_x+2.0, 3.5, 13.2, 3.5, lw=1.5, color='#4527A0')

    # ── 标注各模块状态 ──
    ax.text(gd_x+1.4, 2.2, r'unchanged', fontsize=6, ha='center',
            color='#999', style='italic')
    ax.text(pr_x+0.9, 2.6, r'unchanged', fontsize=6, ha='center',
            color='#999', style='italic')
    ax.text(src_x+1.1, 2.6, r'NEW', fontsize=7, ha='center',
            color='#C62828', fontweight='bold')
    ax.text(disp_x+1.0, 2.1, r'NEW', fontsize=7, ha='center',
            color='#4527A0', fontweight='bold')

    # ── Multi-stage Loss ──
    rounded_box(ax, 3.5, -1.3, 6.5, 0.6, '', C['formula'],
                edgecolor='#DDD0A0', lw=0.6, style='round,pad=0.08')
    ax.text(6.75, -1.0,
            r'$\mathcal{L} = \sum_{j=K\!-\!3}^{K} w_j \sqrt{\mathrm{MSE}(\mathbf{f}^{(j)}, \mathrm{GT})}$'
            r'$\quad w = [0.3,\, 0.5,\, 0.7,\, 1.0]$',
            fontsize=7, ha='center', color='#555')

    return fig


# ═══════════════════════════════════════════════
# Figure 5: 波动方程频域调制细节
# ═══════════════════════════════════════════════

def fig5_wave_modulation():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5),
                             gridspec_kw={'width_ratios': [1.3, 1]})

    # ── (a) 频域调制管线 ──
    ax = axes[0]
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(-1.5, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(a) Frequency-Domain Wave Modulation Pipeline', fontsize=9.5,
                 fontweight='bold', pad=8)

    cx = 4.5  # center
    bw = 2.0

    # 输入
    rounded_box(ax, cx-bw/2, 4.0, bw, 0.5, r'$\mathbf{x}$ [B, C, H, W]',
                C['input'], fontsize=7.5)
    arrow(ax, cx, 4.0, cx, 3.6)

    # Mask Gate
    rounded_box(ax, cx-bw/2-0.8, 3.1, 1.3, 0.5, r'$\Phi(\mathbf{x})$',
                '#FFE0B2', fontsize=7.5)
    rounded_box(ax, cx+bw/2-0.5, 3.1, 1.3, 0.5, r'$\Psi(\mathbf{x})$',
                '#FFE0B2', fontsize=7.5)
    rounded_box(ax, cx+bw/2+1.0, 3.1, 1.0, 0.5, r'gate($\mathbf{M}$)',
                C['mask'], fontsize=6.5)

    # multiply
    ax.text(cx-bw/2-0.15, 2.75, r'$\times$', fontsize=10, ha='center',
            color='#E65100', fontweight='bold')
    ax.text(cx+bw/2+0.15, 2.75, r'$\times$', fontsize=10, ha='center',
            color='#E65100', fontweight='bold')

    # u0, v0
    rounded_box(ax, cx-bw/2-0.8, 2.1, 1.3, 0.45, r'$u_0$', '#FFCC80', fontsize=8, bold=True)
    rounded_box(ax, cx+bw/2-0.5, 2.1, 1.3, 0.45, r'$v_0$', '#FFCC80', fontsize=8, bold=True)

    # 3D rFFT
    rounded_box(ax, cx-bw/2, 1.2, bw, 0.55, '3D rFFT', C['fft'],
                fontsize=8.5, bold=True)
    ax.text(cx, 0.95, r'pad to $2^n$ (28$\to$32)', fontsize=6,
            ha='center', color='#888', style='italic')
    arrow(ax, cx-0.5, 2.1, cx-0.5, 1.75)
    arrow(ax, cx+0.5, 2.1, cx+0.5, 1.75)

    # Wave Mod
    rounded_box(ax, cx-bw/2-0.5, 0.0, bw+1.0, 0.7, '', C['wpo'],
                edgecolor=C['wpo_dark'], lw=1.2)
    ax.text(cx, 0.35, 'Damped Wave Equation\nClosed-Form Solution',
            fontsize=7.5, ha='center', fontweight='bold', color='#5D4037')
    arrow(ax, cx, 1.2, cx, 0.7)

    # irFFT
    rounded_box(ax, cx-bw/2, -1.0, bw, 0.55, '3D irFFT', C['fft'],
                fontsize=8.5, bold=True)
    ax.text(cx, -1.3, r'truncate to $C$ channels', fontsize=6,
            ha='center', color='#888', style='italic')
    arrow(ax, cx, 0.0, cx, -0.45)

    # ── (b) 闭式解分支 ──
    ax = axes[1]
    ax.set_xlim(-0.5, 8)
    ax.set_ylim(-1.5, 6)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(b) Closed-Form Solution Branches', fontsize=9.5,
                 fontweight='bold', pad=8)

    # omega_sq 计算
    rounded_box(ax, 0.5, 5.0, 5.5, 0.6, '', C['formula'], edgecolor='#DDD0A0')
    ax.text(3.25, 5.3,
            r'$\omega^2 = (2\pi)^2\!\left[v_s^2(f_H^2 + f_W^2) + v_\lambda^2 f_C^2\right]$',
            fontsize=8, ha='center', color='#555')

    # eta
    ax.text(3.25, 4.5, r'$\eta = \omega^2 - (\alpha/2)^2$',
            fontsize=8, ha='center', color='#555')
    arrow(ax, 3.25, 5.0, 3.25, 4.7)

    # branch
    ax.text(1.2, 3.8, r'$\eta \geq 0$', fontsize=8, color='#1B5E20',
            fontweight='bold', ha='center')
    ax.text(5.3, 3.8, r'$\eta < 0$', fontsize=8, color='#B71C1C',
            fontweight='bold', ha='center')

    # Underdamped
    rounded_box(ax, 0, 2.5, 2.4, 1.0, '', '#E8F5E9', edgecolor='#66BB6A')
    ax.text(1.2, 3.25, 'Underdamped', fontsize=7.5, ha='center',
            fontweight='bold', color='#2E7D32')
    ax.text(1.2, 2.85, r'$\omega_d = \sqrt{\eta}$', fontsize=7, ha='center',
            color='#555')
    ax.text(1.2, 2.6, r'$\cos(\omega_d t),\; \frac{\sin(\omega_d t)}{\omega_d}$',
            fontsize=7, ha='center', color='#555')

    # Overdamped
    rounded_box(ax, 4.1, 2.5, 2.4, 1.0, '', '#FFEBEE', edgecolor='#EF5350')
    ax.text(5.3, 3.25, 'Overdamped', fontsize=7.5, ha='center',
            fontweight='bold', color='#C62828')
    ax.text(5.3, 2.85, r'$\gamma = \sqrt{-\eta}$', fontsize=7, ha='center',
            color='#555')
    ax.text(5.3, 2.6, r'$\cosh(\gamma t),\; \frac{\sinh(\gamma t)}{\gamma}$',
            fontsize=7, ha='center', color='#555')

    arrow(ax, 2.0, 4.3, 1.2, 3.5)
    arrow(ax, 4.5, 4.3, 5.3, 3.5)

    # Final formula
    rounded_box(ax, 0.3, 0.8, 6.0, 1.2, '', C['formula'], edgecolor='#DDD0A0', lw=0.8)
    ax.text(3.3, 1.65, 'Combined Output (torch.where)', fontsize=7.5,
            ha='center', fontweight='bold', color='#555')
    ax.text(3.3, 1.15,
            r'$\hat{u}_\mathrm{out} = e^{-\frac{\alpha t}{2}}\!'
            r'\left[\hat{u}_0 \cdot \mathrm{cs} + '
            r'(\hat{v}_0 + \frac{\alpha}{2}\hat{u}_0)\cdot\mathrm{sinc}\right]$',
            fontsize=8, ha='center', color='#333')

    arrow(ax, 1.2, 2.5, 2.5, 2.0)
    arrow(ax, 5.3, 2.5, 4.0, 2.0)

    # decay annotation
    ax.text(3.3, 0.5, r'decay $= e^{-\alpha t/2}$: learned damping',
            fontsize=6.5, ha='center', color='#888', style='italic')

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════
# Figure 6: ParaEstimator 细节
# ═══════════════════════════════════════════════

def fig6_para_estimator():
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
    ax.set_xlim(-0.5, 6)
    ax.set_ylim(-0.5, 7)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(r'ParaEstimator: $\rho_k$ Prediction Network', fontsize=10,
                 fontweight='bold', pad=8)

    cx = 2.5
    bw = 2.5

    # input
    rounded_box(ax, cx-bw/2, 6.0, bw, 0.5, r'$\mathbf{f}^{(k)}$ [B, 28, H, W]',
                C['input'], fontsize=7.5)

    # Conv 1x1 fusion
    rounded_box(ax, cx-bw/2, 5.0, bw, 0.5, 'Conv1x1 (28 → 32) + ReLU',
                C['conv'], fontsize=7)
    arrow(ax, cx, 6.0, cx, 5.5)

    # AvgPool
    rounded_box(ax, cx-bw/2, 4.0, bw, 0.5, 'AdaptiveAvgPool2d(1)',
                '#E3F2FD', fontsize=7)
    arrow(ax, cx, 5.0, cx, 4.5)

    # MLP
    rounded_box(ax, cx-bw/2, 2.5, bw, 1.2, '', '#FFF8E1', edgecolor='#FFE082')
    ax.text(cx, 3.45, 'MLP', fontsize=8, ha='center', fontweight='bold', color='#F57F17')
    ax.text(cx, 3.05, 'Conv1x1(32→32) + ReLU', fontsize=6.5, ha='center', color='#666')
    ax.text(cx, 2.75, 'Conv1x1(32→32) + ReLU', fontsize=6.5, ha='center', color='#666')
    ax.text(cx, 2.45, 'Conv1x1(32→1)', fontsize=6.5, ha='center', color='#666')
    arrow(ax, cx, 4.0, cx, 3.7)

    # + bias
    plus_circle(ax, cx, 1.9, r=0.15)
    arrow(ax, cx, 2.5, cx, 2.05)
    ax.text(cx+0.7, 1.9, r'+ bias (learnable)', fontsize=6.5,
            va='center', color='#888', style='italic')

    # output
    rounded_box(ax, cx-bw/4, 0.8, bw/2, 0.55,
                r'$\rho_k$' + '\n[B,1,1,1]', C['rho'], fontsize=8, bold=True)
    arrow(ax, cx, 1.75, cx, 1.35)

    return fig


# ═══════════════════════════════════════════════
# 主函数：生成所有图
# ═══════════════════════════════════════════════

def main():
    try:
        out_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        out_dir = r'G:\MachineLearning\CASSI\stage2\paper'
    os.makedirs(out_dir, exist_ok=True)

    figures = [
        ('fig1_wpo3d_block',      fig1_wpo3d_block),
        ('fig2_wavemst_unet',     fig2_wavemst_unet),
        ('fig3_unfolding',        fig3_unfolding),
        ('fig4_enhanced',         fig4_enhanced),
        ('fig5_wave_modulation',  fig5_wave_modulation),
        ('fig6_para_estimator',   fig6_para_estimator),
    ]

    for name, func in figures:
        print(f"Generating {name}...")
        fig = func()
        for ext in ['pdf', 'png']:
            path = os.path.join(out_dir, f'{name}.{ext}')
            fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
        print(f"  -> {name}.pdf / {name}.png")

    print("\nDone! All figures saved to:", out_dir)


if __name__ == '__main__':
    main()
