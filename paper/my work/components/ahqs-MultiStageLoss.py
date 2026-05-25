"""
ahqs-MultiStageLoss.py — 多 stage RMSE 加权损失
对应代码：version2/train.py::multi_stage_loss
对应公式：(1.38) L = Σ w_k RMSE(f^k, GT), w_K=1, w_{K-1}=0.7, w_{K-2}=0.5, w_{K-3}=0.3
颜色：背景 #e6f1ff（Part III）
"""

import torch
import torch.nn.functional as F


def rmse(p, g):
    return torch.sqrt(F.mse_loss(p, g))


def multi_stage_loss(outputs, gt):
    K = len(outputs)
    loss = rmse(outputs[-1], gt)
    if K >= 2: loss = loss + 0.7 * rmse(outputs[-2], gt)
    if K >= 3: loss = loss + 0.5 * rmse(outputs[-3], gt)
    if K >= 4: loss = loss + 0.3 * rmse(outputs[-4], gt)
    return loss
