# Figure Checklist — SMILE² 论文图片生成清单

> 对应 `main_zh.tex` 中的 9 个占位图。标注每张图的内容、生成方式、所需输入。

---

## Figure 1 — SMILE² 总体框架图

| 项目 | 说明 |
|------|------|
| **内容** | 左→右：输入 $g, \Phi$ → Init Conv → $f^0$ → K 个 Stage 循环 → 输出 $f^K$ + Multi-Stage Loss |
| **Stage 内部** | LDE → A-HQS GD Step → DAG 净化 → SWAP → LRB |
| **跨 Part 连线** | $\sigma \dashrightarrow \alpha_{\text{eff}}$（虚线），$\Delta\Phi \dashrightarrow \Phi_{\text{eff}}$（虚线），$w \to z_{\text{clean}}$（实线） |
| **生成方式** | **手绘**（PPT/Figma/Draw.io），不需要代码 |
| **状态** | 待你手绘 |

---

## Figure 2 — SWAP Block 内部结构

| 项目 | 说明 |
|------|------|
| **内容** | LN → WPO3D（MI → 3D rFFT → Wave Modulate → AdaSpec → 3D irFFT → SiLU gate → Conv）→ Res → LN → FFN → Res |
| **生成方式** | **手绘**（PPT/Figma） |
| **参考代码** | `version2/model/wpo3d.py: WPO3DBlock` |
| **状态** | 待你手绘 |

---

## Figure 3 — LDE 内部结构

| 项目 | 说明 |
|------|------|
| **内容** | 三路分支：SEC→$\Delta\Phi$、DAG→$w$、NLE→$\sigma$；$\sigma$ 虚线连至 SWAP 的 $\alpha_{\text{eff}}$ |
| **生成方式** | **手绘**（PPT/Figma） |
| **参考代码** | `version2/model/degradation.py: DegradationEstimation` |
| **状态** | 待你手绘 |

---

## Figure 4 — 视觉重建对比（KAIST 测试场景）

| 项目 | 说明 |
|------|------|
| **内容** | 每行一个测试场景，列：GT / MST / DAUHST / DPU / SSR / SMILE² / Error Map |
| **选取** | 3-4 个场景（含纹理丰富区 + 光谱敏感区） |
| **生成方式** | Python 脚本 |
| **所需输入** | |
| - GT | KAIST 测试集 `.npy`（已有，路径见 `config.yaml` 的 `test_data_path`） |
| - 各方法重建结果 | 需跑 `test.py` 得到 SMILE² 的 `.pth` 输出；其他方法需从其公开 repo 获取或自行推理 |
| - Error Map | `abs(recon - GT)` 取均值或某个波段，归一化后 colormap |

| **前置条件** | 需要训练好的 SMILE² checkpoint (.pth) + 完成 `test.py` 推理 |
| **状态** | 待训练完成后生成 |
```python
import numpy as np, matplotlib.pyplot as plt
gt = np.load('test_gt.npy')  # (scene, H, W, 28)
ours = np.load('smile2_recon.npy')
# 选 RGB 合成波段 (e.g., band 5,15,25) 做伪彩色
# subplot grid: rows=scenes, cols=methods
``` 
---

## Figure 5 — 光谱曲线对比

| 项目 | 说明 |
|------|------|
| **内容** | 选 2-3 个像素点，绘制 GT 和各方法在 28 波段上的光谱响应曲线 |
| **生成方式** | Python (matplotlib) |
| **所需输入** | |
| - GT | 同 Figure 4 |
| - 各方法重建 | 同 Figure 4（需要精确到像素级的重建结果） |

| **前置条件** | 同 Figure 4 |
| **状态** | 待训练完成后生成 |
```python
pixel = (x, y)  # 手动选取代表性像素
for method in methods:
    plt.plot(range(28), recon[method][pixel[0], pixel[1], :], label=method)
plt.plot(range(28), gt[pixel[0], pixel[1], :], 'k--', label='GT')
``` 
---

## Figure 6 — SWAP 频域行为可视化

| 项目 | 说明 |
|------|------|
| **内容** | (a) 欠阻尼/过阻尼区的频率分布图 ($\eta > 0$ vs $\eta < 0$) (b) Cs/Sn 调制函数在不同 $\alpha$ 下的频率响应 (c) AdaSpec 权重 $W$ 的频域分布 |
| **生成方式** | Python 脚本（纯数学计算 + 从 .pth 提取参数） |
| **所需输入** | |
| - 训练好的模型 | `.pth` checkpoint，提取 `alpha, v_s, v_lambda, t` 参数 |
| - 计算公式 | $\omega_0^2 = v_s^2(\omega_x^2+\omega_y^2) + v_\lambda^2 \omega_\lambda^2$，$\eta = \omega_0^2 - (\alpha/2)^2$ |
| **前置条件** | 需要训练好的 .pth（即使只训练几个 epoch 也可以，只要参数已学习） |
| **是否需要修改代码** | 否，直接加载 .pth 读取参数即可 |
| **状态** | 待训练完成后生成 |

```python
import torch
ckpt = torch.load('best.pth', map_location='cpu')
# 提取 wpo3d 模块参数
alpha = ckpt['model']['stage0.swap.alpha'].item()
vs = ckpt['model']['stage0.swap.vs'].item()
vl = ckpt['model']['stage0.swap.vl'].item()
t_val = ckpt['model']['stage0.swap.t'].item()

# 构建频率网格
wx = np.linspace(-pi, pi, 256)
wy = np.linspace(-pi, pi, 256)
wl = np.linspace(-pi, pi, 28)
omega0_sq = vs**2 * (wx**2 + wy**2) + vl**2 * wl**2
eta = omega0_sq - (alpha/2)**2

# (a) 绘制 eta 的正负分布
# (b) 绘制 Cs(eta,t), Sn(eta,t) 曲线
# (c) 提取 AdaSpec 的 SNR-adaptive 权重
``` 


---

## Figure 7 — 收敛曲线

| 项目 | 说明 |
|------|------|
| **内容** | (a) PSNR vs. Epoch：SMILE² vs. version1 baseline (b) Stage-wise PSNR：展示各 stage 输出质量逐步提升 |
| **生成方式** | Python (matplotlib) |
| **所需输入** | |
| - (a) 训练日志 | `version2/logs/` 下的 tensorboard 或 csv 日志（训练时自动生成） |
| - (b) Stage-wise | 需要**修改代码**：在 `test.py` 中保存每个 stage 的中间输出并计算 PSNR |
| **代码修改说明** | 在 `unfolding.py` 的 `WPO_Unfold.forward()` 中，`outputs` 列表已保存各 stage 输出。在 `test.py` 中对 `outputs` 的每个元素计算 PSNR 即可 |
| **前置条件** | 完成完整训练 + baseline 训练（或使用公开的 baseline 结果） |
| **状态** | 待训练完成后生成 |
```python
# (b) stage-wise
model.eval()
with torch.no_grad():
    outputs = model(meas, mask)  # outputs: list of K tensors
    for k, f_k in enumerate(outputs):
        psnr_k = compute_psnr(f_k, gt)
        print(f"Stage {k}: PSNR = {psnr_k:.2f}")
``` 
---

## Figure 8 — DAG 权重可视化

| 项目 | 说明 |
|------|------|
| **内容** | 将 $w \in (0,1)^{H \times W \times \Lambda}$ 在 $\Lambda$ 维取均值后显示为热力图；对比 mask $M$、$\Phi^*$、$w$ 的空间分布 |
| **生成方式** | Python (matplotlib + forward hook) |
| **所需输入** | |
| - 训练好的模型 | `.pth` checkpoint |
| - Mask | `mask.npy`（已有，路径见 `config.yaml`） |
| - 测试图像 | 任一测试场景的 measurement |
| **代码修改说明** | 需要**修改代码**添加 forward hook 或在 `degradation.py` 的 `DegradationEstimation.forward()` 中保存中间变量 `w` |
| **前置条件** | 需要训练好的 .pth + 一张测试图 |
| **状态** | 待训练完成后生成 |
```python
# 方法1: register_forward_hook
dag_weights = {}
def hook_fn(module, input, output):
    dag_weights['w'] = output[1]  # w 是 LDE 第二个输出

model.lde.register_forward_hook(hook_fn)
model(meas, mask)

w = dag_weights['w'].mean(dim=-1).squeeze().cpu().numpy()  # (H, W)
plt.imshow(w, cmap='hot')
``` 
---

## Figure 9 — 参数效率对比（散点图）

| 项目 | 说明 |
|------|------|
| **内容** | 散点图：x=Params(M), y=PSNR(dB)，标注 MST, CST, DAUHST, RDLUF, DPU, SSR, SMILE² |
| **生成方式** | Python (matplotlib) |
| **所需输入** | |
| - 各方法数据 | 从论文中收集（Params 和 PSNR），硬编码到脚本中 |
| - SMILE² 数据 | 模型参数量：`sum(p.numel() for p in model.parameters())`；PSNR：测试结果 |
| **前置条件** | 需要 SMILE² 的最终 PSNR 结果 + 参数量统计 |
| **是否需要修改代码** | 否 |
| **状态** | 待实验完成后生成 |
```python
methods = {
    'MST':    (2.03, 32.07),
    'CST':    (3.00, 32.91),
    'DAUHST': (6.15, 34.25),
    'RDLUF':  (1.89, 33.51),
    'DPU':    (5.80, 34.60),
    'SSR':    (3.20, 34.81),
    'SMILE2': (X.XX, XX.XX),  # 待填
}
for name, (params, psnr) in methods.items():
    plt.scatter(params, psnr)
    plt.annotate(name, (params, psnr))
``` 
---

## 总结

| Figure | 类型 | 需要 .pth? | 需要改代码? | 优先级 |
|--------|------|-----------|------------|--------|
| Fig 1 | 手绘框架图 | 否 | 否 | 高（投稿必须） |
| Fig 2 | 手绘结构图 | 否 | 否 | 高 |
| Fig 3 | 手绘结构图 | 否 | 否 | 高 |
| Fig 4 | Python 生成 | 是 | 否 | 高（核心实验） |
| Fig 5 | Python 生成 | 是 | 否 | 中 |
| Fig 6 | Python 生成 | 是 | 否 | 高（展示物理特性） |
| Fig 7 | Python 生成 | 是 | 轻微 | 中 |
| Fig 8 | Python 生成 | 是 | 轻微 | 中（ablation 辅助） |
| Fig 9 | Python 生成 | 是 | 否 | 低（数据收集即可） |

### 建议工作顺序
1. **现在可做**：Fig 1, 2, 3（手绘）
2. **训练完成后**：Fig 6（只需加载参数）→ Fig 4, 5（需完整推理）→ Fig 7, 8（需 hook/日志）→ Fig 9（收集数据）
