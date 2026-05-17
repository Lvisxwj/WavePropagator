"""
mst.py — MST baseline 模型（精简自 MST/simulation/train_code/architecture/MST.py）

包含：MaskGuidedMechanism, MS_MSA, FeedForward, MSAB, MST
不修改任何逻辑，作为 baseline 对照和 Model 2 (wpo_smsa.py) 的组件复用。
"""

import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.init import _calculate_fan_in_and_fan_out


# ──────────────────────────────────────────────
# 初始化工具
# ──────────────────────────────────────────────

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# ──────────────────────────────────────────────
# shift_back（用于 MaskGuidedMechanism 内部）
# ──────────────────────────────────────────────

def shift_back(inputs, step=2):
    """[B, nC, H, W_shifted] → [B, nC, H, H]（按分辨率自适应步长，out-of-place）"""
    B, nC, H, W_shifted = inputs.shape
    down_sample = 256 // H
    step_eff = float(step) / float(down_sample * down_sample)
    W = H
    out = torch.zeros(B, nC, H, W, device=inputs.device, dtype=inputs.dtype)
    for i in range(nC):
        out[:, i, :, :] = inputs[:, i, :, int(step_eff * i): int(step_eff * i) + W]
    return out


# ──────────────────────────────────────────────
# GELU 兼容包装
# ──────────────────────────────────────────────

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


# ──────────────────────────────────────────────
# MST 模块
# ──────────────────────────────────────────────

class MaskGuidedMechanism(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.conv1 = nn.Conv2d(n_feat, n_feat, 1, bias=True)
        self.conv2 = nn.Conv2d(n_feat, n_feat, 1, bias=True)
        self.depth_conv = nn.Conv2d(n_feat, n_feat, 5, padding=2, bias=True, groups=n_feat)

    def forward(self, mask_shift):
        # mask_shift: [B, C, H, W_shifted]
        mask_shift = self.conv1(mask_shift)
        attn_map = torch.sigmoid(self.depth_conv(self.conv2(mask_shift)))
        res = mask_shift * attn_map
        mask_shift = res + mask_shift
        mask_emb = shift_back(mask_shift)   # [B, C, H, H]
        return mask_emb


class MS_MSA(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.mm = MaskGuidedMechanism(dim)
        self.dim = dim

    def forward(self, x_in, mask=None):
        """
        x_in: [B, H, W, C]   (channels-last)
        mask: [1, H, W_shifted, C] 或 [1, C, H, W_shifted]（会自动处理）
        返回: [B, H, W, C]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)

        # mask 处理：支持 [1, C, H, W_shifted] 或 [1, H, W_shifted, C]
        if mask.dim() == 4 and mask.shape[1] == c:
            mask_in = mask  # [1, C, H, W_shifted]
        else:
            mask_in = mask.permute(0, 3, 1, 2)  # [1, H, W_shifted, C] → [1, C, H, W_shifted]

        mask_attn = self.mm(mask_in).permute(0, 2, 3, 1)  # [1, H, W, C]
        mask_attn = mask_attn[0].expand(b, h, w, c)        # [B, H, W, C]

        q, k, v, mask_attn = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
            (q_inp, k_inp, v_inp, mask_attn.flatten(1, 2))
        )
        v = v * mask_attn

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1)) * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v  # [B, heads, d, hw]
        x = x.permute(0, 3, 1, 2).reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return out_c + out_p


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """x: [B, H, W, C] → [B, H, W, C]"""
        return self.net(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)


class MSAB(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                MS_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim)),
            ])
            for _ in range(num_blocks)
        ])

    def forward(self, x, mask):
        """x: [B, C, H, W],  mask: [B, C, H, W_shifted]  →  [B, C, H, W]"""
        x = x.permute(0, 2, 3, 1)           # [B, H, W, C]
        mask_cf = mask[:1]                   # [1, C, H, W_shifted]  只用第0个
        for attn, ff in self.blocks:
            x = attn(x, mask=mask_cf) + x
            x = ff(x) + x
        return x.permute(0, 3, 1, 2)        # [B, C, H, W]


class MST(nn.Module):
    def __init__(self, dim=28, stage=2, num_blocks=[2, 2, 2]):
        super().__init__()
        self.dim = dim
        self.stage = stage

        self.embedding = nn.Conv2d(28, dim, 3, 1, 1, bias=False)

        self.encoder_layers = nn.ModuleList()
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                MSAB(dim=dim_stage, num_blocks=num_blocks[i],
                     dim_head=dim, heads=dim_stage // dim),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),   # fea down
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),   # mask down
            ]))
            dim_stage *= 2

        self.bottleneck = MSAB(dim=dim_stage, dim_head=dim,
                                heads=dim_stage // dim, num_blocks=num_blocks[-1])

        self.decoder_layers = nn.ModuleList()
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, 2, 2, 0),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                MSAB(dim=dim_stage // 2, num_blocks=num_blocks[stage - 1 - i],
                     dim_head=dim, heads=(dim_stage // 2) // dim),
            ]))
            dim_stage //= 2

        self.mapping = nn.Conv2d(self.dim, 28, 3, 1, 1, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, mask=None):
        """
        x:    [B, 28, H, W]
        mask: [B, 28, H, W_shifted]（shifted mask for MS_MSA）
        """
        if mask is None:
            mask = torch.zeros(1, 28, x.shape[2], x.shape[3] + 54, device=x.device)

        fea = self.lrelu(self.embedding(x))

        fea_encoder = []
        masks = []
        for msab, fea_down, mask_down in self.encoder_layers:
            fea = msab(fea, mask)
            masks.append(mask)
            fea_encoder.append(fea)
            fea = fea_down(fea)
            mask = mask_down(mask)

        fea = self.bottleneck(fea, mask)

        for i, (fea_up, fusion, msab) in enumerate(self.decoder_layers):
            fea = fea_up(fea)
            fea = fusion(torch.cat([fea, fea_encoder[self.stage - 1 - i]], dim=1))
            mask = masks[self.stage - 1 - i]
            fea = msab(fea, mask)

        return self.mapping(fea) + x
