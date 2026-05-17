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
- [x] `physics.py` — CAVE 波长表、get_k_phys_for_dim、get_inv_lambda_for_dim
- [x] `helmholtz_ops.py` — HelmholtzInverseOp（H1-γ 核心）、BeerLambertAbsorption（H2-γ Step 3）
- [x] `wpo3d_phys.py` — WPO3DPhys / WPO3DPhysBlock / WaveMST_Phys（Model 4, H2-α）
- [x] `helm_pure.py` — HelmBlock / Helmholtzformer（Model 5, H1-γ）
- [x] `wpo3d_helm.py` — WPO3DHelmBlock / WaveMST_Helm（Model 6, H2-γ 主推）

### Bug 修复记录（2026-04-23）

| 文件 | Bug | 修复 |
|------|-----|------|
| wpo3d.py | 未使用的 `cassi_shift_back` 导入 | 删除 |
| mst.py | `shift_back` 原地修改 tensor（破坏 autograd） | 改为 out-of-place |
| mst.py | MSAB.forward 中 `mask_p` 定义但未使用 | 删除，直接用 `mask[:1]` |
| wpo_smsa.py | `MS_MSA(heads=1)` 在高 stage 不正确 | 加 `base_dim` 参数，`heads=dim//base_dim` |

### 工程改进（2026-05-04）

- [x] `dataset.py` — band 数软编码（nC 参数化）；`load_training` 边加载边扫描，首次生成 `[data_name]_info.json` 和 `[data_name]_bands.json`，再次运行幂等检查跳过扫描
- [x] `physics.py` — `WAVELENGTHS_CAVE_28` → `_WAVELENGTHS_FALLBACK` + 模块级 `WAVELENGTHS`；新增 `init_wavelengths(bands_json_path)` 从 json 动态读取；`num_bands` 默认值改为 `len(WAVELENGTHS)`
- [x] `train.py` — `load_training` 后读取 `bands.json` 覆盖 `NUM_BANDS`/`DIM`，调用 `physics.init_wavelengths`；所有 nC 传参软化；新增 `_fmt_time()` 时间格式化；训练开始打印时间，每 50 epoch 打印当前时间和预计结束时间
- [x] `loss.py` — 新增 `torch_freq_amp_err`（频域幅度 MSE）、`torch_rapsd`（径向平均功率谱）、`torch_freq_band_err`（低/高频分层误差）；`test_epoch` 打印 SAM + 三项频域指标
- [x] `viz.py` — 新增 `show_freq_magnitude`、`show_freq_comparison`、`show_rapsd`；`show_all` 扩展步骤 6/7/8 输出频域可视化

### 待完成

- [ ] Model 4 (H2-α) 训练验证：`MODEL_INDEX=4`，确认 PSNR > 30 dB（50 epoch 后）
- [ ] Model 6 (H2-γ) 训练验证：`MODEL_INDEX=6`，预期超过 Model 4
- [ ] Model 5 (H1-γ) 训练验证：`MODEL_INDEX=5`，作为消融基准
- [ ] 完整 300 epoch 对比实验：Model 0 vs 4 vs 5 vs 6，写论文 Table

### 优先级（当前阶段）

1. 等待 Model 0/1/2/4 训练完成（目前正在跑）
2. 用 Model 6 验证三合一框架的效果
3. Model 5 最后跑（作为消融对照）

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
- 0: WaveMST_3D（纯 3D WPO，主推）→ wpo3d.py
- 1: WaveMST_KG（3D WPO + Klein-Gordon Born）→ wpo3d.py
- 2: WaveMST_Parallel（WPO 并联 S-MSA）→ wpo_smsa.py
- 3: WaveMST_Mamba（2D WPO + 1D Mamba）→ wpo_mamba.py
- 4: WaveMST_Phys（H2-α，物理波数 WPO）→ wpo3d_phys.py
- 5: Helmholtzformer（H1-γ，纯稳态亥姆霍兹）→ helm_pure.py
- 6: WaveMST_Helm（H2-γ，三合一主推）→ wpo3d_helm.py

---

## 会话记录

| 日期 | 完成内容 |
|------|---------|
| 2026-04-23 | 阅读技术文档，创建进度文件，完成 `dataset/mat2npy.py` |
| 2026-04-23 | 完成 `dataset.py` / `mst.py` / `loss.py` / `mask_ops.py` / `wpo3d.py` / `wpo_smsa.py` / `train.py` / `test.py` |
| 2026-04-23 | 代码审查修 4 个 bug；完成 `wpo_mamba.py` / `viz.py` / `TUTORIAL.md` / `architect.md` |
| 2026-04-29 | Model 0 训练跑通；修复 CUDA 兼容性检查（fail fast，避免 629s 后才报错） |
| 2026-04-30 | 完成 Helmholtz 系列（Model 4/5/6）：`physics.py` / `helmholtz_ops.py` / `wpo3d_phys.py` / `helm_pure.py` / `wpo3d_helm.py`；更新 `train.py` 新增索引 4/5/6 |
| 2026-05-04 | 工程改进：dataset/physics/train/loss/viz 五文件软编码+频域指标+时间打印+JSON元数据 |
