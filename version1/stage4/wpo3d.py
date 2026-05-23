"""
wpo3d.py — Model 0 (WaveMST_3D) 和 Model 1 (WaveMST_KG)

核心：3D Wave Propagation Operator (WPO3D)
  - 各向异性阻尼波动方程的频域闭式解
  - 处理欠阻尼（cos/sin）和过阻尼（cosh/sinh）两种情况
  - Mask 软门控（方案 A，默认）或 Klein-Gordon Born 修正（方案 D）

U-Net 骨架参照 MST，WPO3D 替代 S-MSA。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mask_ops import MaskGateA, MaskSourceB, MaskKleinGordonD


# ──────────────────────────────────────────────
# LayerNorm（channels-first）
# ──────────────────────────────────────────────

class LayerNorm2d(nn.LayerNorm):
    """对 [B, C, H, W] 做 LayerNorm（在 C 维上）"""
    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


# ──────────────────────────────────────────────
# FFN（参照 WaveFormer Mlp，channels-first）
# ──────────────────────────────────────────────

class FFN(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        hidden = dim * mult
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1, bias=False),
        )

    def forward(self, x):
        """x: [B, C, H, W]"""
        return self.net(x)


# ──────────────────────────────────────────────
# 核心：WPO3D
# ──────────────────────────────────────────────

def _next_power_of_2(n):
    """返回 >= n 的最小 2 的幂"""
    if n <= 0:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


# 全局开关：是否将 FFT 维度 pad 到 2 的幂（cuFFT 对 2^n 最高效）
FFT_PAD_TO_POW2 = True


class WPO3D(nn.Module):
    """
    3D Wave Propagation Operator。

    输入: x [B, C, H, W],  mask_spatial [B, C, H, W]
    输出: [B, C, H, W]

    可学习参数（每层独立）：
        alpha — 阻尼系数
        vs    — 空间波速
        vl    — 光谱波速（作用在 C/通道维度）
        t     — 传播时间步长

    参数用 softplus 保证正值。
    """

    def __init__(self, dim, mask_mode='A', eps=0.1, use_dispersive=False):
        super().__init__()
        self.dim = dim
        self.use_dispersive = use_dispersive

        # 可学习物理参数
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.vs    = nn.Parameter(torch.tensor(1.0))
        self.vl    = nn.Parameter(torch.tensor(0.5))
        self.t     = nn.Parameter(torch.tensor(1.0))

        # mask 机制
        self.mask_mode = mask_mode
        if mask_mode == 'A':
            self.mask_op = MaskGateA(dim, eps=eps)
        elif mask_mode == 'B':
            self.mask_op = MaskSourceB(dim)
        elif mask_mode == 'D':
            self.mask_op = MaskKleinGordonD(dim, eps=eps)
        else:
            raise ValueError(f"mask_mode 必须是 'A', 'B', 'D'，得到 '{mask_mode}'")

        # Block 级色散修正（iFFT 后、输出投影前）
        if use_dispersive:
            from enhancement_ops import DispersionCorrector
            self.dispersion = DispersionCorrector(dim)

        # 输出投影
        self.out_norm   = LayerNorm2d(dim)
        self.out_linear = nn.Conv2d(dim, dim, 1, bias=False)

    def _get_effective_params(self):
        alpha = F.softplus(self.alpha)
        vs    = F.softplus(self.vs)
        vl    = F.softplus(self.vl)
        t     = F.softplus(self.t)
        return alpha, vs, vl, t

    def _build_freq_grid(self, C, H, W, device):
        """构建 omega_sq: [C, H, W//2+1]"""
        fc = torch.fft.fftfreq(C, device=device)    # [C]
        fh = torch.fft.fftfreq(H, device=device)    # [H]
        fw = torch.fft.rfftfreq(W, device=device)   # [W//2+1]
        fc = fc[:, None, None]    # [C, 1, 1]
        fh = fh[None, :, None]    # [1, H, 1]
        fw = fw[None, None, :]    # [1, 1, W//2+1]
        return fc, fh, fw

    def _wave_modulate(self, u0_fft, v0_fft, alpha, vs, vl, t, C, H, W, device):
        """
        在频域做波动方程调制，返回 out_fft [B, C, H, W//2+1]。
        同时返回 sinc_term 和 decay（供方案 D 使用）。
        """
        fc, fh, fw = self._build_freq_grid(C, H, W, device)
        pi2 = (2 * math.pi) ** 2
        omega_sq = pi2 * (vs ** 2 * (fh ** 2 + fw ** 2) + vl ** 2 * fc ** 2)
        # [C, H, W//2+1]

        eta = omega_sq - (alpha / 2) ** 2   # > 0: 欠阻尼，< 0: 过阻尼

        # 欠阻尼
        pos = eta.clamp(min=0)
        omega_d = torch.sqrt(pos + 1e-30)   # 安全 sqrt
        cos_term  = torch.cos(omega_d * t)
        sinc_term_pos = torch.sin(omega_d * t) / (omega_d + 1e-8)

        # 过阻尼
        neg = (-eta).clamp(min=0)
        gamma = torch.sqrt(neg + 1e-30)
        cosh_term = torch.cosh(gamma * t)
        sinch_term_neg = torch.sinh(gamma * t) / (gamma + 1e-8)

        is_under = (eta >= 0)
        cs   = torch.where(is_under, cos_term,  cosh_term)
        sinc = torch.where(is_under, sinc_term_pos, sinch_term_neg)

        decay = torch.exp(-alpha * t / 2)

        # 闭式解：decay * [u0*cs + (v0 + alpha/2 * u0) * sinc]
        out_fft = decay * (u0_fft * cs + (v0_fft + alpha / 2 * u0_fft) * sinc)
        return out_fft, sinc, decay

    def forward(self, x, mask_spatial):
        B, C, H, W = x.shape
        alpha, vs, vl, t = self._get_effective_params()

        # ── mask 操作生成 u0, v0 ──
        if self.mask_mode == 'A':
            u0, v0 = self.mask_op(x, mask_spatial)
            source = None
            m_sq   = None
        elif self.mask_mode == 'B':
            u0, v0, source = self.mask_op(x, mask_spatial)
            m_sq = None
        else:  # 'D'
            u0, v0, m_sq = self.mask_op(x, mask_spatial)
            source = None

        # ── FFT 维度（可 pad 到 2 的幂加速 cuFFT）──
        if FFT_PAD_TO_POW2:
            C_fft = _next_power_of_2(C)
            H_fft = H   # H=256 已是 2^8
            W_fft = W   # W=256 已是 2^8
        else:
            C_fft, H_fft, W_fft = C, H, W

        # ── 3D rFFT ──
        u0_fft = torch.fft.rfftn(u0, s=(C_fft, H_fft, W_fft), dim=(-3, -2, -1))
        v0_fft = torch.fft.rfftn(v0, s=(C_fft, H_fft, W_fft), dim=(-3, -2, -1))

        # ── 频域调制 ──
        out_fft, sinc_term, decay = self._wave_modulate(
            u0_fft, v0_fft, alpha, vs, vl, t, C_fft, H_fft, W_fft, x.device
        )

        # ── 方案 B：叠加源项 ──
        if self.mask_mode == 'B' and source is not None:
            src_fft = torch.fft.rfftn(source, s=(C_fft, H_fft, W_fft), dim=(-3, -2, -1))
            out_fft = out_fft + src_fft * sinc_term * decay * self.mask_op.get_source_weight()

        # ── 3D irFFT（截回原始尺寸）──
        out = torch.fft.irfftn(out_fft, s=(C_fft, H_fft, W_fft), dim=(-3, -2, -1))
        if C_fft != C:
            out = out[:, :C, :, :]

        # ── 方案 D：Born 修正 ──
        if self.mask_mode == 'D' and m_sq is not None:
            out = self.mask_op.apply_correction(out, m_sq, sinc_term[:C], decay, C, H, W)

        # ── Block 级色散修正（波传播后、投影前）──
        if self.use_dispersive:
            out = self.dispersion(out)

        # ── 输出投影：LayerNorm → Linear → SiLU gate ──
        # 用 z（WaveFormer 风格的门控分支，这里简化为 x 本身）
        out = self.out_norm(out)
        out = out * F.silu(x)
        out = self.out_linear(out)
        return out


# ──────────────────────────────────────────────
# WPO3D Block = LN + WPO3D + Residual + LN + FFN + Residual
# ──────────────────────────────────────────────

class WPO3DBlock(nn.Module):
    def __init__(self, dim, mask_mode='A', use_dispersive=False):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.wpo   = WPO3D(dim, mask_mode=mask_mode, use_dispersive=use_dispersive)
        self.norm2 = LayerNorm2d(dim)
        self.ffn   = FFN(dim)

    def forward(self, x, mask_spatial):
        x = x + self.wpo(self.norm1(x), mask_spatial)
        x = x + self.ffn(self.norm2(x))
        return x


# ──────────────────────────────────────────────
# 工具：把 shifted mask → spatial mask
# ──────────────────────────────────────────────

def shifted_mask_to_spatial(shift_mask):
    """
    shift_mask: [B, C, H, W_shifted]
    返回: spatial [B, C, H, W]（截取前 H 列）
    """
    H = shift_mask.shape[2]
    return shift_mask[:, :, :, :H]


# ──────────────────────────────────────────────
# WaveMST_3D — Model 0（主推）
# ──────────────────────────────────────────────

class WaveMST_3D(nn.Module):
    """
    U-Net 骨架（参照 MST），WPO3D Block 替代 S-MSA。

    Model 0 (use_kg=False): 纯 3D WPO
    Model 1 (use_kg=True):  3D WPO + Klein-Gordon Born（实际通过 mask_mode='D' 实现）
    """

    def __init__(self, dim=28, stage=2, num_blocks=[2, 2, 2],
                 mask_mode='A', use_kg=False, use_dispersive_block=False):
        super().__init__()
        self.dim   = dim
        self.stage = stage

        # use_kg → 强制 mask_mode='D'
        if use_kg:
            mask_mode = 'D'
        self.mask_mode = mask_mode

        # 输入嵌入
        self.embedding = nn.Conv2d(28, dim, 3, 1, 1, bias=False)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

        # Encoder
        self.encoder_layers = nn.ModuleList()
        dim_stage = dim
        for i in range(stage):
            blocks = nn.ModuleList([
                WPO3DBlock(dim_stage, mask_mode, use_dispersive=use_dispersive_block)
                for _ in range(num_blocks[i])
            ])
            fea_down  = nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False)
            mask_down = nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False)
            self.encoder_layers.append(nn.ModuleList([blocks, fea_down, mask_down]))
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = nn.ModuleList([
            WPO3DBlock(dim_stage, mask_mode, use_dispersive=use_dispersive_block)
            for _ in range(num_blocks[-1])
        ])

        # Decoder
        self.decoder_layers = nn.ModuleList()
        for i in range(stage):
            fea_up  = nn.ConvTranspose2d(dim_stage, dim_stage // 2, 2, 2, 0)
            fusion  = nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False)
            blocks  = nn.ModuleList([
                WPO3DBlock(dim_stage // 2, mask_mode, use_dispersive=use_dispersive_block)
                for _ in range(num_blocks[stage - 1 - i])
            ])
            self.decoder_layers.append(nn.ModuleList([fea_up, fusion, blocks]))
            dim_stage //= 2

        # 输出映射
        self.mapping = nn.Conv2d(self.dim, 28, 3, 1, 1, bias=False)

    def forward(self, x, input_mask):
        """
        x:          [B, 28, H, W]  CASSI 初始化估计（H setting）
        input_mask: [B, 28, H, W_shifted] shifted mask，或 [B, 28, H, W] spatial mask

        内部把 shifted mask 截取为 spatial mask 传给 WPO3D。
        """
        # 准备 spatial mask
        H = x.shape[2]
        if input_mask.shape[-1] > H:
            # shifted mask，截取前 H 列得到 spatial mask
            mask_spatial = input_mask[:, :, :, :H]
        else:
            mask_spatial = input_mask

        # Embedding
        fea = self.lrelu(self.embedding(x))

        # Encoder
        fea_encoder = []
        masks_enc   = []
        for blocks, fea_down, mask_down in self.encoder_layers:
            for blk in blocks:
                fea = blk(fea, mask_spatial)
            fea_encoder.append(fea)
            masks_enc.append(mask_spatial)
            fea = fea_down(fea)
            # mask 下采样后 sigmoid 保持 [0,1]
            mask_spatial = torch.sigmoid(mask_down(mask_spatial))

        # Bottleneck
        for blk in self.bottleneck:
            fea = blk(fea, mask_spatial)

        # Decoder
        for i, (fea_up, fusion, blocks) in enumerate(self.decoder_layers):
            fea = fea_up(fea)
            fea = fusion(torch.cat([fea, fea_encoder[self.stage - 1 - i]], dim=1))
            mask_spatial = masks_enc[self.stage - 1 - i]
            for blk in blocks:
                fea = blk(fea, mask_spatial)

        return self.mapping(fea) + x


class WaveMST_KG(WaveMST_3D):
    """Model 1 — WaveMST_3D(use_kg=True) 的别名"""
    def __init__(self, dim=28, stage=2, num_blocks=[2, 2, 2], mask_mode='A',
                 use_dispersive_block=False):
        super().__init__(dim=dim, stage=stage, num_blocks=num_blocks,
                         mask_mode=mask_mode, use_kg=True,
                         use_dispersive_block=use_dispersive_block)


# ──────────────────────────────────────────────
# ML_WPO_Block = ML层 → WPO层 → FFN（三段残差）
# ──────────────────────────────────────────────

class ML_WPO_Block(nn.Module):
    """ML-WPO 交替 Block：ML 层 → WPO 层 → FFN

    ML 层提取局部空间-光谱特征，WPO 层做物理全局传播，FFN 做通道混合。
    三者各有残差连接。
    """

    def __init__(self, dim, ml_type='dwconv_ca', mask_mode='A', use_dispersive=False):
        super().__init__()
        from ml_layers import build_ml_layer

        # ML 层
        self.norm_ml = LayerNorm2d(dim)
        self.ml = build_ml_layer(ml_type, dim)

        # WPO 层
        self.norm_wpo = LayerNorm2d(dim)
        self.wpo = WPO3D(dim, mask_mode=mask_mode, use_dispersive=use_dispersive)

        # FFN
        self.norm_ffn = LayerNorm2d(dim)
        self.ffn = FFN(dim)

    def forward(self, x, mask_spatial):
        x = x + self.ml(self.norm_ml(x))
        x = x + self.wpo(self.norm_wpo(x), mask_spatial)
        x = x + self.ffn(self.norm_ffn(x))
        return x


# ──────────────────────────────────────────────
# WaveMST_ML — ML-WPO 混合模型（U-Net 骨架）
# ──────────────────────────────────────────────

class WaveMST_ML(nn.Module):
    """ML-WPO 混合模型

    U-Net 骨架，Block 类型由 unet_mode 决定：
      - 'symmetric':    所有层用 ML_WPO_Block
      - 'asymmetric':   encoder 用 ML_WPO_Block，decoder 用 WPO3DBlock
      - 'alternating':  同一层内 ML_WPO_Block 与 WPO3DBlock 交替排列

    forward 签名与 WaveMST_3D 完全一致，可无缝替换。
    """

    def __init__(self, dim=28, stage=2, num_blocks=[2, 2, 2],
                 ml_type='dwconv_ca', unet_mode='symmetric',
                 use_kg=False, use_dispersive_block=False):
        super().__init__()
        self.dim = dim
        self.stage = stage
        self.unet_mode = unet_mode

        # WPO 固定用 mask_mode='A'，KG 固定用 'D'
        mask_mode = 'D' if use_kg else 'A'
        self.mask_mode = mask_mode

        def _ml_block(d):
            return ML_WPO_Block(d, ml_type, mask_mode, use_dispersive=use_dispersive_block)

        def _wpo_block(d):
            return WPO3DBlock(d, mask_mode, use_dispersive=use_dispersive_block)

        def _make_blocks(d, n, is_encoder_side=True):
            """生成 n 个 block 的列表"""
            if unet_mode == 'symmetric':
                return nn.ModuleList([_ml_block(d) for _ in range(n)])
            elif unet_mode == 'asymmetric':
                if is_encoder_side:
                    return nn.ModuleList([_ml_block(d) for _ in range(n)])
                else:
                    return nn.ModuleList([_wpo_block(d) for _ in range(n)])
            elif unet_mode == 'alternating':
                # 偶数索引 = ML_WPO_Block，奇数索引 = WPO3DBlock
                return nn.ModuleList([
                    _ml_block(d) if j % 2 == 0 else _wpo_block(d)
                    for j in range(n)
                ])
            else:
                raise ValueError(f"未知 unet_mode: {unet_mode}")

        # Embedding
        self.embedding = nn.Conv2d(28, dim, 3, 1, 1, bias=False)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

        # Encoder
        self.encoder_layers = nn.ModuleList()
        dim_stage = dim
        for i in range(stage):
            blocks = _make_blocks(dim_stage, num_blocks[i], is_encoder_side=True)
            fea_down = nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False)
            mask_down = nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False)
            self.encoder_layers.append(nn.ModuleList([blocks, fea_down, mask_down]))
            dim_stage *= 2

        # Bottleneck（始终用 ML_WPO_Block）
        self.bottleneck = nn.ModuleList([
            _ml_block(dim_stage) for _ in range(num_blocks[-1])
        ])

        # Decoder
        self.decoder_layers = nn.ModuleList()
        for i in range(stage):
            fea_up = nn.ConvTranspose2d(dim_stage, dim_stage // 2, 2, 2, 0)
            fusion = nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False)
            blocks = _make_blocks(dim_stage // 2, num_blocks[stage - 1 - i], is_encoder_side=False)
            self.decoder_layers.append(nn.ModuleList([fea_up, fusion, blocks]))
            dim_stage //= 2

        # 输出映射
        self.mapping = nn.Conv2d(self.dim, 28, 3, 1, 1, bias=False)

    def forward(self, x, input_mask):
        H = x.shape[2]
        if input_mask.shape[-1] > H:
            mask_spatial = input_mask[:, :, :, :H]
        else:
            mask_spatial = input_mask

        fea = self.lrelu(self.embedding(x))

        # Encoder
        fea_encoder = []
        masks_enc = []
        for blocks, fea_down, mask_down in self.encoder_layers:
            for blk in blocks:
                fea = blk(fea, mask_spatial)
            fea_encoder.append(fea)
            masks_enc.append(mask_spatial)
            fea = fea_down(fea)
            mask_spatial = torch.sigmoid(mask_down(mask_spatial))

        # Bottleneck
        for blk in self.bottleneck:
            fea = blk(fea, mask_spatial)

        # Decoder
        for i, (fea_up, fusion, blocks) in enumerate(self.decoder_layers):
            fea = fea_up(fea)
            fea = fusion(torch.cat([fea, fea_encoder[self.stage - 1 - i]], dim=1))
            mask_spatial = masks_enc[self.stage - 1 - i]
            for blk in blocks:
                fea = blk(fea, mask_spatial)

        return self.mapping(fea) + x
