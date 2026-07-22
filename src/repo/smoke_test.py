"""One real-size forward/backward step for memory and finite-value validation."""

import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.environ.get("SMILE_CONFIG", ROOT / "configs/compact.yaml"))
if not CONFIG_PATH.is_absolute():
    CONFIG_PATH = ROOT / CONFIG_PATH
with open(CONFIG_PATH, "r", encoding="utf-8") as fp:
    cfg = yaml.safe_load(fp)
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu_id"])

import torch
from dataset import load_mask
from loss import rmse_loss
from model.smile import SMILE2, cassi_measure, phi_phi_t, shift_cube


device = torch.device("cuda:0")
model = SMILE2(
    dim=cfg["dim"], unet_stage=cfg["unet_stage"], num_blocks=cfg["num_blocks"],
    use_spatial_content_modulation=cfg["use_spatial_content_modulation"], use_perchannel=cfg["use_perchannel"],
    use_spectral_wave=cfg.get("use_spectral_wave", True),
    post_block=cfg["post_block"], ffn_mult=cfg["ffn_mult"],
    swp_variant=cfg.get("swp_variant", "full"),
    gradient_checkpointing=cfg.get("gradient_checkpointing", False),
    input_mode=cfg["input_mode"], output_dc=cfg["output_dc"],
    dc_gamma_init=cfg.get("dc_gamma_init", 0.30), bands=cfg["num_bands"],
    input_adapter=cfg.get("input_adapter", "none"),
    wavelength_cutoff_init=cfg.get("wavelength_cutoff_init", 0.28),
    wave_param_mode=cfg.get("wave_param_mode", "free"),
    wave_basis_count=cfg.get("wave_basis_count", 3),
    num_field_outputs=cfg.get("num_field_outputs", 1),
    share_estimator_evolver_weights=cfg.get("share_estimator_evolver_weights", True),
    return_intermediate_fields=cfg.get("return_intermediate_fields", False),
).to(device)
b = int(os.environ.get("SMOKE_BATCH", cfg["batch_size"]))
gt = torch.rand(b, cfg["num_bands"], cfg["crop_size"], cfg["crop_size"], device=device)
mask = load_mask(cfg["mask_path"], nC=cfg["num_bands"]).to(device).float()
mask = mask.unsqueeze(0).expand(b, -1, -1, -1)
shifted_mask = shift_cube(mask)
ppt = phi_phi_t(mask)
y = cassi_measure(gt, mask)
torch.cuda.reset_peak_memory_stats()
outputs = model(y, mask, shifted_mask, ppt)
pred = outputs[-1] if isinstance(outputs, (list, tuple)) else outputs
loss = rmse_loss(pred, gt)
loss.backward()
finite = torch.isfinite(loss).item() and all(
    p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()
)
print(
    f"SMOKE_OK={finite} exp={cfg['experiment_name']} params="
    f"{sum(p.numel() for p in model.parameters())/1e6:.4f}M "
    f"loss={loss.item():.6f} peak_mem={torch.cuda.max_memory_allocated()/1024**3:.3f}GiB"
)
if not finite:
    raise SystemExit(2)


