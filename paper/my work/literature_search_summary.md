# SMILE² 文献搜索总结报告

> 生成日期：2026-05-26 | 服务于 main_zh.tex 引言与相关工作的完善

---

## 一、搜索结果汇总

### 1. WaveFormer（波方程视觉算子）

| 字段 | 内容 |
|------|------|
| **确认标题** | WaveFormer: Frequency-Time Decoupled Vision Modeling with Wave Equation |
| **作者** | Zishan Shu, Juntong Wu, Wei Yan, Xudong Liu, Hongyu Zhang, Chang Liu, Youdong Mao, Jie Chen |
| **年份/发表** | 2026，arXiv preprint（2026-01-13提交，无会议录用确认） |
| **ArXiv ID** | [arXiv:2601.08602](https://arxiv.org/abs/2601.08602) |
| **核心贡献** | 将特征图视为受欠阻尼波动方程支配的空间信号，推导频域闭式解（WPO算子），O(N log N)复杂度替代O(N²)注意力 |
| **⚠️ 重要：任务域** | **通用视觉任务（ImageNet分类、COCO检测、ADE20K分割），不是 CASSI/HSI。** 在论文中应明确标注为"通用视觉方法"，用于证明 PDE 算子可行性，不可声称"该方法应用于 CASSI" |
| **HSI 领域的 WaveFormer** | IEEE GRSL 2024 有一篇同名 WaveFormer 用于 HSI **分类**任务（非重建），还有 IEEE JSTSP 2025 的 Waveformer 用于 HSI **去马赛克**（非 CASSI）。**CASSI 重建领域不存在同名方法。** |

**BibTeX citekey 更新：** `zhuang2024waveformer` → `shu2026waveformer`

---

### 2. vHeat（热方程视觉算子）

| 字段 | 内容 |
|------|------|
| **确认标题** | vHeat: Building Vision Models upon Heat Conduction |
| **作者** | Zhaozhi Wang, Yue Liu, Yunfan Liu, Hongtian Yu, Yaowei Wang, Qixiang Ye, Yunjie Tian |
| **年份/发表** | 2024，arXiv preprint（ICLR 2025 投稿后撤回，无正式发表） |
| **ArXiv ID** | [arXiv:2405.16555](https://arxiv.org/abs/2405.16555) |
| **核心贡献** | 将图像patch建模为热源，通过热传导方程（DCT/IDCT实现）计算相关性，O(N^1.5)复杂度 |
| **⚠️ 重要：任务域** | **通用视觉任务（ImageNet图像识别），不是 CASSI/HSI。** 论文中应明确标注为"通用视觉方法"。 |
| **⚠️ 注意** | 原 bib 中以"Anonymous"记录为"Heat-Former"，实为 vHeat，作者完全不同 |

**BibTeX citekey 更新：** `heatformer2024` → `wang2024vheat`

---

### 3. Phy-CoSF（物理引导的 CASSI 重建）

| 字段 | 内容 |
|------|------|
| **确认标题** | Phy-CoSF: Physics-Guided Continuous Spectral Fields Reconstruction and Super-Resolution for Snapshot Compressive Imaging |
| **作者** | Wudi Chen, Zhiyuan Zha, Xin Yuan, Shigang Wang, Bihan Wen, Jiantao Zhou, Gang Yan, Zipei Fan, Ce Zhu |
| **年份/发表** | 2026，**ICML 2026**（arXiv:2605.13583，2026-05-13提交） |
| **PSNR (KAIST)** | 39.80 dB（9stg），vs RDLUF 39.57 dB（+0.23 dB） |
| **核心贡献** | A-HQS展开框架 + 隐式神经表示（INR）+ Fourier Mamba频域特征提取，首个结合物理引导展开与连续光谱场建模的CASSI方法 |
| **关键缺口** | **无显式退化估计**（无ΔΦ修正、无σ估计），Fourier Mamba是**学习的频域特征提取器**而非PDE闭式解，β_k动量是启发式而非严格Nesterov |

**BibTeX citekey 更新：** `phycosf2025` → `chen2026phycosf`

---

### 4. CA²UN（协同注意力加速展开网络）

| 字段 | 内容 |
|------|------|
| **确认标题** | Lightweight Accelerated Unfolding Network With Collaborative Attention for Snapshot Spectral Compressive Imaging |
| **作者** | Mengjie Qin, Yuchao Feng |
| **年份/发表** | **IET Image Processing**, 2025, Volume 19, Issue 1 |
| **DOI** | [10.1049/ipr2.70024](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/ipr2.70024) |
| **核心贡献** | DADN（退化感知动态网络）+ SGLB（全局-局部协同块）+ A-HQS Nesterov动量加速 |
| **⚠️ 注意** | 原 bib 标题为"Content-Aware"，实为"Collaborative Attention"，作者也为占位符 |

**BibTeX citekey 更新：** `ca2un2025` → `qin2025ca2un`

---

### 5. DERNN-LNLT（退化估计RNN）

| 字段 | 内容 |
|------|------|
| **确认标题** | Degradation Estimation Recurrent Neural Network with Local and Non-Local Priors for Compressive Spectral Imaging |
| **作者** | Yubo Dong, Dahua Gao, Yuyan Li, Guangming Shi, Danhua Liu |
| **年份/发表** | 2024（arXiv:2311.08808，2023-11提交；AAAI 2024待核实） |
| **PSNR (KAIST)** | 39.93 dB (9stg)，40.33 dB (9stg*)，vs DAUHST 38.36 dB（+1.57 dB） |
| **核心贡献** | DEN同时估计ΔΦ（sensing error）和σ（噪声水平），跨stage共享权重的RNN结构，参数1.04M |
| **⚠️ 注意** | 原 bib 标题有误（写的是"Learnable Nonlinear Local Transform"，实为"Local and Non-Local Priors"） |

**BibTeX citekey 更新：** `dernnlnlt2023` → `dong2024dernnlnlt`

---

### 6. DPU（双先验展开）—— 作者需修正

| 字段 | 内容 |
|------|------|
| **确认标题** | Dual Prior Unfolding for Snapshot Compressive Imaging |
| **实际第一作者** | Jiancheng **Zhang** (GitHub: ZhangJC-2k/DPU) |
| **完整作者** | Jiancheng Zhang, Haijin Zeng, Jiezhang Cao, Yongyong Chen, Dengxiu Yu, Yin-Ping Zhao |
| **年份/发表** | CVPR 2024, pp. 25742–25752 |
| **PSNR (KAIST)** | 40.33 dB（5stg）/40.52 dB（9stg） |
| **⚠️ 注意** | 原 bib 第一作者写为"Xu, Jiancheng"，实为"Zhang, Jiancheng" |

**BibTeX citekey 更新：** `xu2024dpu` → `zhang2024dpu`

---

### 7. SSR（光谱-空间校正）—— 作者需修正

| 字段 | 内容 |
|------|------|
| **确认标题** | Improving Spectral Snapshot Reconstruction with Spectral-Spatial Rectification |
| **实际第一作者** | Jiancheng **Zhang** (同DPU作者组) |
| **完整作者** | Jiancheng Zhang, Haijin Zeng, Yongyong Chen, Dengxiu Yu, Yin-Ping Zhao |
| **年份/发表** | CVPR 2024, pp. 25817–25826 |
| **PSNR (KAIST)** | 40.47 dB (SSR-L) |
| **⚠️ 注意** | 原 bib 第一作者写为"Li, Jiamian"，实为"Zhang, Jiancheng" |

**BibTeX citekey 更新：** `li2024ssr` → `zhang2024ssr`

---

### 8. 新增：DADF-Net（傅里叶+退化感知，最接近我们的现有工作）

| 字段 | 内容 |
|------|------|
| **标题** | Degradation-Aware Dynamic Fourier-Based Network for Spectral Compressive Imaging |
| **发表** | IEEE Transactions on Multimedia (TMM), 2023 |
| **DOI** | [10.1109/TMM.2023.3304450](https://ieeexplore.ieee.org/document/10214675/) |
| **核心贡献** | 从初始化HSI估计退化特征图，用动态Fourier处理指导退化感知重建 |
| **关键缺口** | 端到端网络（非深度展开），Fourier处理是学习特征提取器（非PDE闭式解），无Nesterov动量 |
| **引用价值** | 正面承认"退化+Fourier"这一方向有效，但证明我们的方法更完整 |

**新增 BibTeX citekey：** `dong2023dadfnet`

---

### 9. 新增：RDFNet（FISTA/Nesterov加速展开）

| 字段 | 内容 |
|------|------|
| **标题** | Regional Dynamic FISTA-Net for Spectral Snapshot Compressive Imaging |
| **作者** | Shiyun Zhou, Tingfa Xu, Shaocong Dong, Jianan Li |
| **发表** | IEEE Transactions on Computational Imaging, 2023 |
| **DOI** | [10.1109/TCI.2023.3237175](https://ieeexplore.ieee.org/document/10012515/) |
| **ArXiv** | arXiv:2302.02519 |
| **核心贡献** | 展开FISTA（Nesterov加速近端梯度）用于CASSI，O(1/k²)收敛率 |
| **关键缺口** | 无显式退化估计，无PDE闭式传播算子，先验是学习CNN去噪器 |
| **引用价值** | 支撑我们"Nesterov动量O(1/k²)"的claim，可作为路线B的第三引用 |

**新增 BibTeX citekey：** `zhou2023rdfnet`

---

## 二、关键差距分析（支撑 SMILE² 贡献独特性的核心证据）

| 方法 | 显式退化估计<br>（ΔΦ, w, σ） | 频域闭式<br>物理传播 | 二阶<br>Nesterov加速 |
|------|:---:|:---:|:---:|
| DAUHST (NeurIPS'22) | △ 仅标量 | × | × |
| RDLUF (CVPR'23) | ✓ (ΔΦ) | × | × |
| DERNN-LNLT (2024) | ✓ (ΔΦ+σ) | × | × |
| DPU (CVPR'24) | × | × | × |
| SSR (CVPR'24) | × | × | × |
| DADF-Net (TMM'23) | ✓ 学习特征图 | △ 学习Fourier | × |
| WaveFormer (2026) | × | ✓ 波方程 | × |
| vHeat (2024) | × | ✓ 热传导方程 | × |
| Phy-CoSF (ICML'26) | × | △ Fourier Mamba | △ A-HQS β_k |
| RDFNet (TCI'23) | × | × | ✓ FISTA |
| **SMILE² (本文)** | **✓✓ SEC+DAG+NLE** | **✓✓ 3D波方程** | **✓✓ Nesterov** |

**结论：没有任何现有方法同时覆盖三个维度。**  
最接近的是 Phy-CoSF（ICML 2026）：有 A-HQS 的近似动量，有学习的频域特征，但**无退化估计**，且 Fourier Mamba 不是 PDE 闭式解。

---

## 三、推荐的新论文逻辑链条

### 当前逻辑链问题

原有的"三点局限"框架：
1. 先验算子缺乏物理结构
2. 退化建模与传播先验从未真正同框
3. 初始场退化被忽视

**问题：** 点2和点3高度重叠，论证力度弱，且缺乏具体数字支撑。

### 推荐改进：两条路线框架

```
背景：CASSI深度展开已成主流
  ↓
路线A（物理算子派）：WaveFormer/vHeat → O(N log N)，但直接操作退化污染的GD输出
  ↓ 问题：FFT全局算子将局部退化错误扩散至全空间，放大成像退化
路线B（退化感知派）：RDLUF/DERNN-LNLT → +1-2 dB，但先验是Transformer黑盒
  ↓ 问题：退化信息只条件化去噪器，从未注入物理传播过程本身
核心断层：两条路线从未耦合
  ↓ 关键洞察：物理传播算子接受退化输入会全局扩散错误；解法是估计-演化双向耦合
提出 SMILE²：LDE(退化估计) ↔ SWAP(波传播) 双向耦合
  ↓ LDE净化SWAP的输入；σ反馈耦合至SWAP阻尼系数
贡献：首个同时做到（a）显式退化三合一估计 + (b) PDE闭式传播 + (c) Nesterov加速的方法
```

这个框架的优势：
- **有具体数字**：KAIST PSNR 38.36→39.93 dB（仅靠退化估计+Transformer）；vs 我们更小参数量更优性能
- **有对立面**：两条路线各有合理性，我们是融合而非否定
- **差距清晰**：差距表可视化，无法被反驳

---

## 四、BibTeX 需要修改的条目汇总

| 原 citekey | 新 citekey | 修改内容 |
|-----------|-----------|---------|
| `xu2024dpu` | `zhang2024dpu` | 修正第一作者名 |
| `li2024ssr` | `zhang2024ssr` | 修正第一作者名 |
| `zhuang2024waveformer` | `shu2026waveformer` | 修正所有字段，arXiv:2601.08602 |
| `heatformer2024` | `wang2024vheat` | 修正为vHeat，arXiv:2405.16555 |
| `phycosf2025` | `chen2026phycosf` | 修正作者，ICML 2026，arXiv:2605.13583 |
| `ca2un2025` | `qin2025ca2un` | 修正标题和作者，IET 2025 |
| `dernnlnlt2023` | `dong2024dernnlnlt` | 修正标题和年份 |
| 新增 | `dong2023dadfnet` | DADF-Net, TMM 2023 |
| 新增 | `zhou2023rdfnet` | RDFNet, TCI 2023 |
