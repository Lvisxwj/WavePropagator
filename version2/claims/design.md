1. 三合一退化估计模块
2. wpo设置三个选项，1方案 A：基于信噪比的自适应权重（零参数）；2方案 B：可学习的频带权重；C：原始的，什么都不加
3. 降维度：暂时不考虑，后续再说
4. 换A-HQS
5. 轻量局部增强
6. swin选项
具体的
切成小窗64×64做 WPO 传播：
窗内：WPO 只需建模 64×64 范围的波传播——局部关联更精准
窗间：Swin 风格 shift 或 skip connection 融合——补全跨窗信息，具体的参照DPU和swin的做法，设计一个完整的方法

目前代码情况
linux服务器/data5/SCI/xieweijie/CASSI，以下是linux服务器的目录格式，本地的和linux服务器的一致，仅少了dataset目录
```
CASSI/
├── dataset          # 所有数据集，npy格式
├── version1         # 以前的代码，内涵多个stage目录，是之前做的消融对比和验证
├── version2         # ★ 最新的工作目录，需要先从version1/stage4中复制需要的文件
```
version2具体的新改造：
# ════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════
train 和 test的这些内容全部放到config.yaml里面。如：
GPU_ID       = '2'
BATCH_SIZE   = 8  train有关
MAX_EPOCH    = 300
LR           = 4e-4
SCHEDULER    = 'CosineAnnealingLR'
MILESTONES   = [50, 100, 150, 200, 250]
EPOCH_SAMPLE = 5000
CROP_SIZE    = 256
NUM_BANDS    = 28
DIM          = 28
STAGE        = 3       # U-Net encoder 层数
NUM_BLOCKS   = [2, 2, 2]
INPUT_SETTING = 'H'
SAVE_THRESH  = 28.0

# ── 时空优化 ──
USE_AMP       = False 一定默认false
CACHE_PHIPHIT = True

# ── 色散介质 删除，以实验证明效果非常差



# 数据路径，
数据集目录永远在/data5/SCI/xieweijie/CASSI/dataset，这部分的逻辑都不变，生成的结果放置的都是当前工作目录，这些逻辑也不变，都可以统一放到yaml里

model字典，刚说明的几个选项开关，都保留，但是放到init.py里面，train和test共享，这样我只需要改一遍就行了。Unfolding 配置等也放到init里面
NUM_STAGES          = 5
SHARE_STAGE_WEIGHTS = True
MULTI_STAGE_LOSS    = True

test会用到best pth的路径，这里也放到init里面，每次跑好了我会自行修改这部分内容，就可以精准无误了

version2下面只需要train test init dataset4个py。然后model 放到 version2/model里面，其他的都各自的归一个类，帮我整理好














