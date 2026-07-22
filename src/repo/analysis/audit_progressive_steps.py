#!/usr/bin/env python
"""Audit progressive E2E intermediate steps.

This script is inference-only. It is intended for checking whether a trained
2-step not-share model naturally separates into an initial-state estimation /
purification step and a later evolution/refinement step.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((a - b) ** 2) + 1e-12).item())


def rel_rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = torch.sqrt(torch.mean(b ** 2) + 1e-12)
    return float((torch.sqrt(torch.mean((a - b) ** 2) + 1e-12) / denom).item())


def spectral_energy(x: torch.Tensor) -> np.ndarray:
    """Mean residual energy by spectral FFT mode for one [C,H,W] cube."""
    spec = torch.fft.fft(x.float(), dim=0)
    return spec.abs().square().mean(dim=(1, 2)).detach().cpu().numpy()


def spatial_band_energy(x: torch.Tensor) -> dict[str, float]:
    """Coarse 2D spatial frequency energy split for one [C,H,W] cube."""
    _, h, w = x.shape
    fy = torch.fft.fftfreq(h, device=x.device).view(h, 1)
    fx = torch.fft.fftfreq(w, device=x.device).view(1, w)
    radius = torch.sqrt(fx ** 2 + fy ** 2)
    fft = torch.fft.fft2(x.float(), dim=(-2, -1))
    power = fft.abs().square().mean(dim=0)
    total = power.mean().clamp_min(1e-12)
    return {
        "spatial_low_frac": float(power[radius <= 0.10].mean().div(total).item()),
        "spatial_mid_frac": float(power[(radius > 0.10) & (radius <= 0.25)].mean().div(total).item()),
        "spatial_high_frac": float(power[radius > 0.25].mean().div(total).item()),
    }


def load_state(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> dict:
    try:
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(str(ckpt_path), map_location=device)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[load] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        if missing:
            print("[load] missing sample:", missing[:10], flush=True)
        if unexpected:
            print("[load] unexpected sample:", unexpected[:10], flush=True)
    return ckpt if isinstance(ckpt, dict) else {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--num-scenes", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    ckpt_path = Path(args.ckpt).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # dataset.py reads SMILE_CONFIG at import time.
    os.environ["SMILE_CONFIG"] = str(config_path)
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from dataset import load_mask, load_test  # noqa: WPS433
    from loss import torch_psnr, torch_sam, torch_ssim  # noqa: WPS433
    from model.smile import SMILE2, cassi_measure, phi_phi_t, shift_back, shift_cube  # noqa: WPS433

    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = SMILE2(
        dim=cfg["dim"],
        unet_stage=cfg["unet_stage"],
        num_blocks=cfg["num_blocks"],
        use_spatial_content_modulation=cfg["use_spatial_content_modulation"],
        use_perchannel=cfg["use_perchannel"],
        use_spectral_wave=cfg.get("use_spectral_wave", True),
        post_block=cfg["post_block"],
        ffn_mult=cfg["ffn_mult"],
        input_mode=cfg["input_mode"],
        output_dc=cfg["output_dc"],
        dc_gamma_init=cfg.get("dc_gamma_init", 0.30),
        swp_variant=cfg.get("swp_variant", "full"),
        gradient_checkpointing=False,
        bands=cfg["num_bands"],
        input_adapter=cfg.get("input_adapter", "none"),
        wavelength_cutoff_init=cfg.get("wavelength_cutoff_init", 0.28),
        wave_param_mode=cfg.get("wave_param_mode", "free"),
        wave_basis_count=cfg.get("wave_basis_count", 3),
        num_field_outputs=cfg.get("num_field_outputs", 1),
        share_estimator_evolver_weights=cfg.get("share_estimator_evolver_weights", True),
        return_intermediate_fields=True,
    ).to(device)
    model.eval()
    ckpt = load_state(model, ckpt_path, device)

    test = load_test(cfg["test_path"]).float()
    test = test[: min(args.num_scenes, test.shape[0])]
    mask_single = load_mask(cfg["mask_path"], nC=cfg["num_bands"]).float()

    per_stage_rows: list[dict] = []
    consistency_rows: list[dict] = []
    delta_rows: list[dict] = []
    spectral_rows: list[dict] = []
    spatial_rows: list[dict] = []
    stage_values: dict[str, dict[str, list[float]]] = {}

    stage_order = ["H0", "U1", "U2"]
    for scene_idx in range(test.shape[0]):
        gt = test[scene_idx:scene_idx + 1].to(device)
        mask = mask_single.unsqueeze(0).to(device)
        shifted = shift_cube(mask)
        ppt = phi_phi_t(mask)
        y = cassi_measure(gt, mask)
        h0 = shift_back(y / cfg["num_bands"] * 2.0, cfg["num_bands"], cfg.get("shift_step", 2))
        with torch.no_grad():
            outputs = model(y, mask, shifted, ppt)
        if not isinstance(outputs, (list, tuple)) or len(outputs) < 2:
            raise RuntimeError("Expected at least two progressive outputs.")
        cubes = {"H0": h0[0], "U1": outputs[0][0], "U2": outputs[1][0]}

        for stage in stage_order:
            cube = cubes[stage]
            psnr = float(torch_psnr(cube, gt[0]).item())
            ssim = float(torch_ssim(cube, gt[0]).item())
            sam = float(torch_sam(cube, gt[0]).item())
            rec_y = cassi_measure(cube.unsqueeze(0), mask)[0]
            row = {
                "scene": scene_idx + 1,
                "stage": stage,
                "psnr": psnr,
                "ssim": ssim,
                "sam": sam,
                "gt_rmse": rmse(cube, gt[0]),
                "meas_rmse": rmse(rec_y, y[0]),
                "meas_rel_rmse": rel_rmse(rec_y, y[0]),
            }
            per_stage_rows.append(row)
            consistency_rows.append({
                "scene": scene_idx + 1,
                "stage": stage,
                "meas_rmse": row["meas_rmse"],
                "meas_rel_rmse": row["meas_rel_rmse"],
            })
            stage_values.setdefault(stage, {"psnr": [], "ssim": [], "sam": [], "gt_rmse": [], "meas_rel_rmse": []})
            for key in stage_values[stage]:
                stage_values[stage][key].append(row[key])

            residual = cube - gt[0]
            e = spectral_energy(residual)
            e_norm = e / max(float(e.sum()), 1e-12)
            for mode_idx, (energy, energy_norm) in enumerate(zip(e, e_norm)):
                freq = np.fft.fftfreq(cfg["num_bands"])[mode_idx]
                spectral_rows.append({
                    "scene": scene_idx + 1,
                    "stage": stage,
                    "mode": mode_idx,
                    "frequency_cycles_per_band": float(freq),
                    "energy": float(energy),
                    "energy_norm": float(energy_norm),
                })
            spatial = spatial_band_energy(residual)
            spatial.update({"scene": scene_idx + 1, "stage": stage})
            spatial_rows.append(spatial)

        for name, a, b in [
            ("U1_minus_H0", cubes["U1"], cubes["H0"]),
            ("U2_minus_U1", cubes["U2"], cubes["U1"]),
            ("U2_minus_H0", cubes["U2"], cubes["H0"]),
        ]:
            delta_rows.append({
                "scene": scene_idx + 1,
                "delta": name,
                "rmse": rmse(a, b),
                "mean_abs": float((a - b).abs().mean().item()),
                "max_abs": float((a - b).abs().max().item()),
            })
        print(f"[audit] scene {scene_idx + 1}/{test.shape[0]}", flush=True)

    save_csv(outdir / "per_scene_stage_metrics.csv", per_stage_rows)
    save_csv(outdir / "measurement_consistency.csv", consistency_rows)
    save_csv(outdir / "stage_deltas.csv", delta_rows)
    save_csv(outdir / "residual_spectral_energy.csv", spectral_rows)
    save_csv(outdir / "residual_spatial_energy.csv", spatial_rows)

    summary_rows = []
    for stage in stage_order:
        row = {"stage": stage}
        for key, values in stage_values[stage].items():
            row[key] = mean(values)
        summary_rows.append(row)
    save_csv(outdir / "stage_summary.csv", summary_rows)

    h0, u1, u2 = (summary_rows[0], summary_rows[1], summary_rows[2])
    psnr_h0_u1 = u1["psnr"] - h0["psnr"]
    psnr_u1_u2 = u2["psnr"] - u1["psnr"]
    sam_h0_u1 = h0["sam"] - u1["sam"]
    sam_u1_u2 = u1["sam"] - u2["sam"]
    meas_h0_u1 = h0["meas_rel_rmse"] - u1["meas_rel_rmse"]
    meas_u1_u2 = u1["meas_rel_rmse"] - u2["meas_rel_rmse"]

    evidence = {
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_metrics": ckpt.get("metrics"),
        "num_scenes": int(test.shape[0]),
        "stage_summary": summary_rows,
        "gains": {
            "psnr_H0_to_U1": psnr_h0_u1,
            "psnr_U1_to_U2": psnr_u1_u2,
            "sam_H0_to_U1_positive_is_better": sam_h0_u1,
            "sam_U1_to_U2_positive_is_better": sam_u1_u2,
            "meas_rel_rmse_H0_to_U1_positive_is_better": meas_h0_u1,
            "meas_rel_rmse_U1_to_U2_positive_is_better": meas_u1_u2,
        },
    }
    (outdir / "audit_summary.json").write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    support_lines = []
    if psnr_h0_u1 > 5.0 and sam_h0_u1 > 0.05:
        support_lines.append("U1 strongly improves reconstruction quality over the dirty shift-back field.")
    if psnr_u1_u2 > 0.5:
        support_lines.append("U2 still provides a large refinement/evolution gain after U1.")
    if meas_h0_u1 > 0:
        support_lines.append("U1 improves measurement consistency relative to H0.")
    else:
        support_lines.append("U1 does not improve measurement consistency relative to H0; its role is not a pure data-fidelity correction.")
    if sam_u1_u2 > 0.02:
        support_lines.append("U2 materially improves spectral-angle quality after U1.")

    verdict = "partial_support"
    if psnr_h0_u1 > 5.0 and psnr_u1_u2 > 0.8:
        verdict = "support_estimate_then_evolve"
    if psnr_u1_u2 < 0.2:
        verdict = "weak_step2_evidence"

    md = [
        "# Progressive Step Audit: 222 step2 not-share",
        "",
        f"- Checkpoint: `{ckpt_path}`",
        f"- Checkpoint epoch: `{ckpt.get('epoch')}`",
        f"- Scenes: {test.shape[0]}",
        f"- Verdict: **{verdict}**",
        "",
        "## Stage summary",
        "",
        "| stage | PSNR | SSIM | SAM | GT RMSE | measurement rel-RMSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        md.append(
            f"| {row['stage']} | {row['psnr']:.4f} | {row['ssim']:.6f} | "
            f"{row['sam']:.6f} | {row['gt_rmse']:.6f} | {row['meas_rel_rmse']:.6f} |"
        )
    md.extend([
        "",
        "## Step gains",
        "",
        f"- H0 → U1: PSNR {psnr_h0_u1:+.4f} dB, SAM {sam_h0_u1:+.6f} lower-is-better gain.",
        f"- U1 → U2: PSNR {psnr_u1_u2:+.4f} dB, SAM {sam_u1_u2:+.6f} lower-is-better gain.",
        f"- H0 → U1 measurement rel-RMSE change: {meas_h0_u1:+.6f} positive means better.",
        f"- U1 → U2 measurement rel-RMSE change: {meas_u1_u2:+.6f} positive means better.",
        "",
        "## Interpretation",
        "",
    ])
    for line in support_lines:
        md.append(f"- {line}")
    md.extend([
        "",
        "## Design implication",
        "",
        "- If U1 mainly improves quality but not measurement consistency, a future Estimate/Purify step should be a learned dirty-field purifier, not a hard data-consistency or DU-style LDE clone.",
        "- If U2 keeps a large PSNR/SAM gain, keep the second step as a full SWAP evolution/refinement backbone.",
        "- Do not add hard physical constraints solely for narrative; use these diagnostics to decide the role split.",
        "",
        "## Files",
        "",
        "- `per_scene_stage_metrics.csv`",
        "- `measurement_consistency.csv`",
        "- `stage_deltas.csv`",
        "- `residual_spectral_energy.csv`",
        "- `residual_spatial_energy.csv`",
        "- `stage_summary.csv`",
        "- `audit_summary.json`",
    ])
    (outdir / "audit_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(evidence["gains"], indent=2), flush=True)
    print(f"[audit] wrote {outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


