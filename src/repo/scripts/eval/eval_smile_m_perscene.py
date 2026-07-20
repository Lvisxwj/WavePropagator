#!/usr/bin/env python
import csv
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")
os.environ.setdefault(
    "SMILE_CONFIG",
    "configs/runtime_friend/progressive_222_step2_estimate_evolve_v2.yaml",
)

import torch

from dataset import load_mask, load_test
from train import build_model, cfg, count_params, test_epoch


def main():
    root = Path(__file__).resolve().parent
    ckpt_path = root / "result/model/2026_07_13_17_46_16_progressive_222_step2_estimate_evolve_v2/best_psnr.pth"
    out_dir = root / "result/eval_smile_m"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "smile_m_perscene_best_psnr.csv"
    out_summary = out_dir / "smile_m_summary_best_psnr.csv"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print(f"[eval] config={os.environ['SMILE_CONFIG']}", flush=True)
    print(f"[eval] ckpt={ckpt_path}", flush=True)
    print(f"[eval] device={device} cuda={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}", flush=True)

    test_data = load_test(cfg["test_path"], nC=cfg["num_bands"])
    mask = load_mask(cfg["mask_path"], nC=cfg["num_bands"])
    model = build_model().to(device)

    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    print(f"[eval] ckpt_epoch={state.get('epoch')} metrics={state.get('metrics')}", flush=True)

    (psnr, ssim, sam), per_scene = test_epoch(model, test_data, mask, device)
    params_m = count_params(model)

    with out_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["model", "scene", "psnr", "ssim", "sam_rad", "params_M"])
        writer.writeheader()
        for row in per_scene:
            writer.writerow(
                {
                    "model": "SMILE-M",
                    "scene": f"scene{int(row['scene']):02d}",
                    "psnr": f"{float(row['psnr']):.4f}",
                    "ssim": f"{float(row['ssim']):.6f}",
                    "sam_rad": f"{float(row['sam']):.6f}",
                    "params_M": f"{params_m:.4f}",
                }
            )
    with out_summary.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["model", "params_M", "gflops", "psnr", "ssim", "sam_rad", "ckpt_epoch"])
        writer.writeheader()
        writer.writerow(
            {
                "model": "SMILE-M",
                "params_M": f"{params_m:.4f}",
                "gflops": "28.281",
                "psnr": f"{psnr:.4f}",
                "ssim": f"{ssim:.6f}",
                "sam_rad": f"{sam:.6f}",
                "ckpt_epoch": state.get("epoch"),
            }
        )
    print(f"[eval] wrote {out_csv}", flush=True)
    print(f"[eval] wrote {out_summary}", flush=True)


if __name__ == "__main__":
    main()

