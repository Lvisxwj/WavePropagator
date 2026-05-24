"""
__init__.py — 模型选择开关 + Unfolding 配置 + Checkpoint 路径

train.py 和 test.py 共享此文件。修改这里的开关即可切换模型配置。
"""

# ── 模型核心选择 ──
USE_KG = False                     # True → KG 方程（mask_mode='D'），False → 纯 WPO（mask_mode='A'）

# ── WPO FBGW 频带引导加权 ──
wpo_inside = ['none', 'snr_adaptive', 'learnable_band']
WPO_FBGW_MODE = wpo_inside[0]            # 'none' / 'snr_adaptive' / 'learnable_band'

# ── Swin-WPO 窗口传播 ──
USE_SWIN_WPO = False               # True → 64×64 窗内传播 + shift window
SWIN_WINDOW_SIZE = 64              # 窗大小（不能小于 56 = CASSI shift 跨度）

# ── Unfolding 配置 ──
USE_UNFOLDING = True               # True → deep unfolding，False → 端到端
USE_AHQS = False                   # True → A-HQS（动量 + Phi_eff 修正），False → GAP（与 version1 一致）
NUM_STAGES = 5                     # unfolding stage 数
SHARE_STAGE_WEIGHTS = True         # True → 所有 stage 共享 prior 权重
MULTI_STAGE_LOSS = True            # True → DPU 风格多 stage 加权损失

# ── Checkpoint（test.py 使用）──
BEST_CKPT = ''                     # 训练完成后填入 best.pth 路径
