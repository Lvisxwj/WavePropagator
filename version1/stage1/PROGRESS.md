# WaveMST 开发进度

> 每次会话开始先读本文件，结束前更新进度。

---

## 项目简介

将 WaveFormer (AAAI 2026) 的波动方程传播算子 (WPO) 迁移到 CASSI 高光谱重建任务，
替代/增强 MST (CVPR 2022) 的 Spectral-wise Self-Attention (S-MSA)。

**参考文档**：`WaveMST_Technical_Handoff.md`（架构设计）、`WaveMST_Analysis_Claude_Code.md`（数学推导）

---

## 目录结构规划

```
WaveMST/           ← 待创建（与 MST/、WaveFormer/ 同级）
├── dataset.py     ← 数据加载、mask、CASSI 仿真
├── mst.py         ← MST baseline（从参考代码精简复制）
├── wpo3d.py       ← Model 0 (WaveMST_3D) & Model 1 (WaveMST_KG)
├── wpo_smsa.py    ← Model 2 (WaveMST_Parallel, WPO 并联 S-MSA)
├── wpo_mamba.py   ← Model 3 (WaveMST_Mamba, 2D WPO + 1D Mamba)
├── mask_ops.py    ← 三种 mask 机制 (A/B/D)
├── loss.py        ← PSNR, SSIM, SAM, 参数量/FLOPs
├── train.py       ← 训练入口（CONFIG 区域，模型索引选择）
├── test.py        ← 测试入口
└── viz.py         ← 可视化
```

---

## 数据集情况

| 目录 | 内容 | 状态 |
|------|------|------|
| `dataset/CAVE_1024/cave_1024_28/` | scene1~242.mat，训练集，(1024,1024,28)，key='img'或'img_expand'，值/65536 | 原始 .mat |
| `dataset/CAVE_1024_npy/` | 待转换的 .npy 训练集 | **空，需转换** |
| `dataset/CAVE_512_mat/` | scene01~30.mat，512 版训练集 | 原始 .mat |
| `dataset/CAVE_512_npy/` | 待转换的 .npy 512训练集 | **空，需转换** |
| `dataset/TSA_simu_data/Truth/` | scene01~10.mat，测试集，(256,256,28)，key='img' | 原始 .mat |
| `dataset/TSA_simu_data/mask.mat` | 仿真 mask，(256,256)，key='mask' | 原始 .mat |
| `dataset/mask/mask.npy` | 已存在的 mask npy | ✅ 已有 |

**转换脚本**：`dataset/mat2npy.py`（已完成）

---

## 开发进度

### 已完成

- [x] 阅读并理解 `WaveMST_Technical_Handoff.md` 和 `WaveMST_Analysis_Claude_Code.md`
- [x] `dataset/mat2npy.py` — mat→npy 轻量转换脚本
- [x] `dataset.py` — shift/shift_back/gen_meas/load_training/load_test/load_mask/prepare_mask/shuffle_crop
- [x] `mst.py` — MST baseline（MaskGuidedMechanism, MS_MSA, FeedForward, MSAB, MST）
- [x] `loss.py` — rmse_loss, torch_psnr, torch_ssim, torch_sam, count_params, count_flops
- [x] `mask_ops.py` — MaskGateA, MaskSourceB, MaskKleinGordonD
- [x] `wpo3d.py` — WPO3D, WPO3DBlock, WaveMST_3D (Model 0), WaveMST_KG (Model 1)
- [x] `wpo_smsa.py` — WPO_SMSA_Block, WaveMST_Parallel (Model 2)
- [x] `train.py` — 训练入口（CONFIG 区域，完整训练循环）
- [x] `test.py` — 测试入口（PSNR/SSIM/SAM 评估，保存 npy）

- [x] `wpo_mamba.py` — WPO2D + SpectralMamba (mamba_ssm / SimpleSSM fallback) + WaveMST_Mamba (Model 3)
- [x] `viz.py` — show_bands / show_rgb / show_spectrum / show_comparison / show_error_map / show_all
- [x] `TUTORIAL.md` — 运行手册（数据准备、训练、测试、调参、报错处理）
- [x] `architect.md` — Pipeline / 实现细节 / 模型图提示词

### Bug 修复记录（2026-04-23）

| 文件 | Bug | 修复 |
|------|-----|------|
| wpo3d.py | 未使用的 `cassi_shift_back` 导入 | 删除 |
| mst.py | `shift_back` 原地修改 tensor（破坏 autograd） | 改为 out-of-place |
| mst.py | MSAB.forward 中 `mask_p` 定义但未使用 | 删除，直接用 `mask[:1]` |
| wpo_smsa.py | `MS_MSA(heads=1)` 在高 stage 不正确 | 加 `base_dim` 参数，`heads=dim//base_dim` |

### 待完成

- [ ] 跑通验证：先 `python dataset/mat2npy.py`，再 `python train.py`（MODEL_INDEX=0）

### 优先级

1. `dataset/mat2npy.py` ✅ → 先转数据
2. `dataset.py` → 数据管道通了才能跑模型
3. `mst.py` + `loss.py` → baseline 能跑起来
4. `mask_ops.py` → WPO 依赖
5. `wpo3d.py` → 主模型（Model 0）
6. `train.py` + `test.py` → 完整训练测试

---

## 关键实现要点（备忘）

### WPO3D 核心
- 3D FFT：`torch.fft.rfftn(x, dim=(-3,-2,-1))`，输出 shape `[B, C, H, W//2+1]`
- 频率网格：`freq_c=fftfreq(C)`, `freq_h=fftfreq(H)`, `freq_w=rfftfreq(W)`
- NaN 防护：`eta = omega_sq - (alpha/2)^2`，分 `eta>0`（欠阻尼 cos/sin）和 `eta<0`（过阻尼 cosh/sinh）
- 参数用 `F.softplus` 保证正值
- Mask 软门控：`gate = 0.1 + 0.9 * mask_spatial`

### CASSI 仿真
- `shift`: `[B,28,256,256]` → `[B,28,256,310]`，每波段右移 2i 像素
- `shift_back`: `[B,256,310]` → `[B,28,256,256]`，从第 2i 列截取 256 列
- 测量值 H：`shift_back(sum(shift(mask3d*gt))/28*2)`

### 模型索引
- 0: WaveMST_3D（纯 3D WPO，主推）
- 1: WaveMST_KG（3D WPO + Klein-Gordon Born）
- 2: WaveMST_Parallel（WPO 并联 S-MSA）
- 3: WaveMST_Mamba（2D WPO + 1D Mamba）

---

## 会话记录

| 日期 | 完成内容 |
|------|---------|
| 2026-04-23 | 阅读技术文档，创建进度文件，完成 `dataset/mat2npy.py` |
| 2026-04-23 | 完成 `dataset.py` / `mst.py` / `loss.py` / `mask_ops.py` / `wpo3d.py` / `wpo_smsa.py` / `train.py` / `test.py` |
| 2026-04-23 | 代码审查修 4 个 bug；完成 `wpo_mamba.py` / `viz.py` / `TUTORIAL.md` / `architect.md` |
