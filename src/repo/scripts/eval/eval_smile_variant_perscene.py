#!/usr/bin/env python
import csv
import os
from pathlib import Path

import torch

from dataset import load_mask, load_test
from train import build_model, cfg, count_params, test_epoch


def _collapse_tsa_duplicates(rows):
    # A800 TSA_simu_data/Truth may contain both .mat and .npy copies, producing
    # 20 rows for the 10 benchmark scenes. Prior verified convention: use 0::2.
    if len(rows) == 20:
        return rows[0::2]
    return rows


def _avg(rows, key):
    return sum(float(r[key]) for r in rows) / max(1, len(rows))


def main():
    root = Path(__file__).resolve().parent
    model_name = os.environ["SMILE_MODEL_NAME"]
    ckpt_path = root / os.environ["SMILE_CKPT"]
    out_dir = root / "result/eval_smile_variants"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = os.environ.get("SMILE_OUT_PREFIX", model_name.lower().replace("-", "_"))
    gflops = os.environ.get("SMILE_GFLOPS", "")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    print(f"[eval] model={model_name}", flush=True)
    print(f"[eval] config={os.environ.get('SMILE_CONFIG')}", flush=True)
    print(f"[eval] ckpt={ckpt_path}", flush=True)
    print(f"[eval] device={device}", flush=True)

    test_data = load_test(cfg["test_path"], nC=cfg["num_bands"])
    mask = load_mask(cfg["mask_path"], nC=cfg["num_bands"])
    model = build_model().to(device)

    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    print(f"[eval] ckpt_epoch={state.get('epoch')} metrics={state.get('metrics')}", flush=True)

    (_, _, _), per_scene_raw = test_epoch(model, test_data, mask, device)
    per_scene = _collapse_tsa_duplicates(per_scene_raw)
    params_m = count_params(model)
    psnr = _avg(per_scene, "psnr")
    ssim = _avg(per_scene, "ssim")
    sam = _avg(per_scene, "sam")

    out_csv = out_dir / f"{out_prefix}_perscene_best_psnr_10scene.csv"
    out_summary = out_dir / f"{out_prefix}_summary_best_psnr.csv"

    with out_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["model", "scene", "psnr", "ssim", "sam_rad", "params_M"])
        writer.writeheader()
        for i, row in enumerate(per_scene, start=1):
            writer.writerow(
                {
                    "model": model_name,
                    "scene": f"scene{i:02d}",
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
                "model": model_name,
                "params_M": f"{params_m:.4f}",
                "gflops": gflops,
                "psnr": f"{psnr:.4f}",
                "ssim": f"{ssim:.6f}",
                "sam_rad": f"{sam:.6f}",
                "ckpt_epoch": state.get("epoch"),
            }
        )

    print(f"[eval] raw_rows={len(per_scene_raw)} used_rows={len(per_scene)}", flush=True)
    print(f"[eval] AVG {psnr:.4f} {ssim:.6f} {sam:.6f}", flush=True)
    print(f"[eval] wrote {out_csv}", flush=True)
    print(f"[eval] wrote {out_summary}", flush=True)


if __name__ == "__main__":
    main()

