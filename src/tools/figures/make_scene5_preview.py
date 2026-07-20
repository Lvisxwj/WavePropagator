from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

P = Path('paper/my work/e2e/figure/figuredraft/4/scene5_e2e/smile2_scene5_e2e_preds')
OUT = Path('paper/my work/e2e/figure/figuredraft/4/scene5_e2e/outputs')
OUT.mkdir(parents=True, exist_ok=True)
gt = np.load(P / 'gt.npy')

def robust01(x, p1=1, p99=99):
    lo, hi = np.percentile(x, [p1, p99])
    return np.clip((x-lo)/(hi-lo+1e-8), 0, 1)

rgb = np.stack([robust01(gt[25]), robust01(gt[17]), robust01(gt[5])], -1) ** 0.85
fig, ax = plt.subplots(figsize=(6,6), dpi=160)
ax.imshow(rgb)
ax.set_xticks(np.arange(0,257,32))
ax.set_yticks(np.arange(0,257,32))
ax.grid(color='yellow', linewidth=0.6, alpha=0.8)
ax.set_xlim(0,256); ax.set_ylim(256,0)
fig.savefig(OUT / 'scene05_rgb_coord_preview.png', bbox_inches='tight')
plt.close(fig)

