"""
physics.py — CAVE 数据集物理常数与波数预计算

提供 28 波段物理波长、归一化波数 k_phys = λ_min/λ_b ∈ [0.66, 1.0]，
以及 Beer-Lambert 吸收所需的归一化 1/λ。

被 wpo3d_phys.py / helm_pure.py / wpo3d_helm.py 共用。

使用方式：
    # train.py 在 load_training() 之后调用一次
    import physics
    physics.init_wavelengths('dataset/CAVE_1024_npy/CAVE_1024_npy_bands.json')
    # 此后 physics.WAVELENGTHS 即为 json 中的值
"""

import json
import torch
import torch.nn.functional as F
from pathlib import Path

# 硬编码 fallback：CAVE 28 波段波长（nm）
_WAVELENGTHS_FALLBACK = [
    453, 457, 462, 467, 472, 476, 481, 486, 491, 496, 502, 507,
    515, 526, 537, 547, 558, 569, 580, 590, 600, 611, 622, 633,
    644, 655, 668, 681,
]  # nm，共 28 个

# 模块级波长列表，由 init_wavelengths() 初始化，默认 fallback
WAVELENGTHS = list(_WAVELENGTHS_FALLBACK)


def init_wavelengths(bands_json_path):
    """
    从 [data_name]_bands.json 读取 wavelengths_nm 字段，更新模块级 WAVELENGTHS。
    若文件不存在或字段缺失，保持 fallback 硬编码值并打印警告。

    train.py 在 load_training() 之后调用一次即可。
    """
    global WAVELENGTHS
    path = Path(bands_json_path)
    if not path.exists():
        print(f"[physics] 警告：{bands_json_path} 不存在，使用硬编码波长 fallback。")
        return
    try:
        with open(path, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        wl = data.get('wavelengths_nm')
        if not wl:
            print(f"[physics] 警告：{bands_json_path} 缺少 wavelengths_nm，使用 fallback。")
            return
        WAVELENGTHS = [int(w) for w in wl]
        print(f"[physics] WAVELENGTHS 已从 {path.name} 加载，共 {len(WAVELENGTHS)} 个波段。")
    except Exception as e:
        print(f"[physics] 警告：读取 {bands_json_path} 失败：{e}，使用 fallback。")


def get_k_phys_for_dim(dim: int, num_bands: int = None) -> torch.Tensor:
    """
    归一化物理波数：k = λ_min / λ_b ∈ [0.66, 1.0]，shape [dim]。

    当 dim == num_bands 时直接返回物理值；
    当 dim != num_bands（深层 UNet 通道翻倍）时线性插值。
    """
    if num_bands is None:
        num_bands = len(WAVELENGTHS)
    wls = torch.tensor(WAVELENGTHS[:num_bands], dtype=torch.float32)
    k = wls.min() / wls
    if dim == num_bands:
        return k
    k = F.interpolate(k.view(1, 1, -1), size=dim,
                      mode='linear', align_corners=True).view(dim)
    return k


def get_inv_lambda_for_dim(dim: int, num_bands: int = None) -> torch.Tensor:
    """
    归一化 1/λ（均值归一化为 1），shape [dim]。
    用于 Beer-Lambert 吸收因子的波长依赖项。
    """
    if num_bands is None:
        num_bands = len(WAVELENGTHS)
    wls = torch.tensor(WAVELENGTHS[:num_bands], dtype=torch.float32)
    inv_lam = 1.0 / wls
    inv_lam = inv_lam / inv_lam.mean()
    if dim == num_bands:
        return inv_lam
    inv_lam = F.interpolate(inv_lam.view(1, 1, -1), size=dim,
                            mode='linear', align_corners=True).view(dim)
    return inv_lam
