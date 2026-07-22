#!/usr/bin/env python
"""Measure SMILE-S/M/L runtime and peak CUDA memory on a CUDA GPU."""

import csv
import os
import time
from pathlib import Path

import torch


ROOT = Path(os.environ.get("SMILE_ROOT", ".")).resolve()
OUT = ROOT / "result" / "runtime_memory_smile.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

VARIANTS = [
    {
        "model": "SMILE-S",
        "config": "configs/smile_s.yaml",
        "ckpt": os.environ.get("SMILE_S_CKPT"),
        "gflops": 28.281,
    },
    {
        "model": "SMILE-M",
        "config": "configs/smile_m.yaml",
        "ckpt": os.environ.get(
            "SMILE_M_CKPT",
            str((ROOT / "../../evidence/checkpoints/SMILE-M/SMILE-M_best_psnr.pth").resolve()),
        ),
        "gflops": 28.281,
    },
    {
        "model": "SMILE-L",
        "config": "configs/smile_l.yaml",
        "ckpt": os.environ.get("SMILE_L_CKPT"),
        "gflops": 38.601,
    },
]


def clear_cuda():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def load_train_module(config_rel):
    import importlib
    import sys

    os.environ["SMILE_CONFIG"] = str(ROOT / config_rel)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if "train" in sys.modules:
        del sys.modules["train"]
    return importlib.import_module("train")


def main():
    torch.backends.cudnn.benchmark = True
    rows = []
    for spec in VARIANTS:
        print(f"[measure] {spec['model']}", flush=True)
        ckpt_path = Path(spec["ckpt"]).resolve() if spec.get("ckpt") else None
        if ckpt_path is None or not ckpt_path.exists():
            print(f"[skip] {spec['model']} checkpoint not provided or not found", flush=True)
            continue
        tr = load_train_module(spec["config"])
        device = torch.device("cuda:0")
        model = tr.build_model().to(device)
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
        model.load_state_dict(state, strict=True)
        model.eval()
        params_m = tr.count_params(model)

        test_data = tr.load_test(tr.cfg["test_path"], nC=tr.cfg["num_bands"])
        if test_data.shape[0] == 20:
            test_data = test_data[::2].contiguous()
        else:
            test_data = test_data[:10].contiguous()
        mask = tr.load_mask(tr.cfg["mask_path"], nC=tr.cfg["num_bands"])

        def one_forward(i):
            gt = test_data[i : i + 1].to(device).float()
            mask_b = tr.expand_mask(mask, 1, device)
            shifted_mask = tr.shift_cube(mask_b)
            ppt = tr.phi_phi_t(mask_b)
            y = tr.cassi_measure(gt, mask_b)
            outputs = model(y, mask_b, shifted_mask, ppt)
            pred = tr.final_output(outputs)
            return pred

        with torch.no_grad():
            _ = one_forward(0)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        clear_cuda()
        t0 = time.perf_counter()
        with torch.no_grad():
            for i in range(test_data.shape[0]):
                pred = one_forward(i)
                del pred
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        rows.append(
            {
                "model": spec["model"],
                "params_M": params_m,
                "gflops": spec["gflops"],
                "scenes": int(test_data.shape[0]),
                "time_total_s": elapsed,
                "time_per_scene_ms": elapsed / max(1, int(test_data.shape[0])) * 1000.0,
                "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
                "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
            }
        )
        del model
        torch.cuda.empty_cache()

    with open(OUT, "w", newline="", encoding="utf-8") as fp:
        if not rows:
            raise RuntimeError("No runtime rows were measured; provide SMILE_*_CKPT paths.")
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[done] {OUT}", flush=True)


if __name__ == "__main__":
    main()



