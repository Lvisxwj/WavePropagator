"""Real-size forward/backward throughput and peak-memory scan."""

import gc
import os
import time
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


def build_model():
    return SMILE2(
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
        field_process_mode=cfg.get("field_process_mode", "plain"),
    )


def final_output(output):
    if isinstance(output, (list, tuple)):
        return output[-1]
    return output


def main():
    device = torch.device("cuda:0")
    model = build_model().to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    mask_single = load_mask(cfg["mask_path"], nC=cfg["num_bands"]).to(device).float()
    batches = [int(x) for x in os.environ.get("SCAN_BATCHES", "4,6,8,10,12,14,16").split(",")]
    repeats = int(os.environ.get("SCAN_REPEATS", "2"))
    print("batch,peak_gib,seconds_per_step,samples_per_second,status", flush=True)

    for batch in batches:
        gt = mask = shifted = ppt = y = pred = loss = None
        try:
            gt = torch.rand(
                batch, cfg["num_bands"], cfg["crop_size"], cfg["crop_size"], device=device
            )
            mask = mask_single.unsqueeze(0).expand(batch, -1, -1, -1)
            shifted = shift_cube(mask)
            ppt = phi_phi_t(mask)
            y = cassi_measure(gt, mask)

            # Warm-up also allocates Adam states, matching real training more closely.
            optimizer.zero_grad(set_to_none=True)
            loss = rmse_loss(final_output(model(y, mask, shifted, ppt)), gt)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()

            torch.cuda.reset_peak_memory_stats(device)
            start = time.time()
            for _ in range(repeats):
                optimizer.zero_grad(set_to_none=True)
                pred = final_output(model(y, mask, shifted, ppt))
                loss = rmse_loss(pred, gt)
                loss.backward()
                optimizer.step()
            torch.cuda.synchronize()
            elapsed = (time.time() - start) / repeats
            peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            print(f"{batch},{peak:.3f},{elapsed:.4f},{batch/elapsed:.3f},ok", flush=True)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            print(f"{batch},nan,nan,nan,oom", flush=True)
            break
        finally:
            optimizer.zero_grad(set_to_none=True)
            del gt, mask, shifted, ppt, y, pred, loss
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()


