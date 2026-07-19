"""Pre-training audits for stage-trajectory KD, residual low rank, and checkpoints."""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--mode", required=True, choices=("trajectory", "svd", "checkpoints"))
parser.add_argument("--gpu", required=True, type=int)
parser.add_argument("--samples", type=int, default=64)
parser.add_argument("--batch", type=int, default=8)
parser.add_argument("--output", default="analysis/results/next_round_20260707")
args = parser.parse_args()

config_path = ROOT / "configs" / f"audit_gpu{args.gpu}.yaml"
if not config_path.is_file():
    raise FileNotFoundError(config_path)
os.environ["SMILE_CONFIG"] = str(config_path)
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

import train_distill as td


OUT = ROOT / args.output
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda:0")


def mean_std(values):
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std())}


def sample_rmse(pred, target):
    return (pred - target).square().flatten(1).mean(1).sqrt()


def sample_psnr(pred, target):
    mse = (pred - target).square().flatten(1).mean(1).clamp_min(1e-12)
    return -10.0 * torch.log10(mse)


def sample_sam(pred, target, eps=1e-8):
    dot = (pred * target).sum(1)
    denom = pred.square().sum(1).sqrt() * target.square().sum(1).sqrt()
    cosine = (dot / denom.clamp_min(eps)).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(cosine).flatten(1).mean(1)


def sample_cosine(left, right, eps=1e-12):
    left = left.flatten(1)
    right = right.flatten(1)
    return (left * right).sum(1) / (left.norm(dim=1) * right.norm(dim=1)).clamp_min(eps)


def prepare_training():
    td.seed_everything(int(td.cfg.get("seed", 2026)))
    train_set = td.load_training(td.cfg["train_path"], max_scenes=205)
    mask = td.load_mask(td.cfg["mask_path"], nC=int(td.cfg["num_bands"]))
    return train_set, mask


def next_batch(train_set, mask, batch_size, cassi_measure, phi_phi_t, shift_cube):
    gt = td.shuffle_crop(
        train_set, batch_size, int(td.cfg["crop_size"]),
        device=DEVICE, nC=int(td.cfg["num_bands"]),
    ).float()
    mask_b = td.expand_mask(mask, batch_size, DEVICE)
    shifted = shift_cube(mask_b)
    ppt = phi_phi_t(mask_b)
    y = cassi_measure(gt, mask_b)
    return gt, mask_b, shifted, ppt, y


def run_trajectory():
    teacher = td.build_teacher(DEVICE)
    _, cassi_measure, phi_phi_t, shift_cube, _ = td.import_student_components()
    train_set, mask = prepare_training()
    stage_values = defaultdict(list)
    increment_values = defaultdict(list)
    weights = [0.3 / 2.5, 0.5 / 2.5, 0.7 / 2.5, 1.0 / 2.5]
    seen = 0
    started = time.time()

    with torch.no_grad():
        while seen < args.samples:
            batch = min(args.batch, args.samples - seen)
            gt, mask_b, _, ppt, y = next_batch(
                train_set, mask, batch, cassi_measure, phi_phi_t, shift_cube
            )
            outputs = teacher(
                y.unsqueeze(1), input_mask=(mask_b, ppt.unsqueeze(1))
            )
            if len(outputs) != 5:
                raise RuntimeError(f"expected 5 teacher stages, got {len(outputs)}")

            for index, output in enumerate(outputs, start=1):
                stage_values[f"t{index}_psnr"].extend(sample_psnr(output, gt).cpu().tolist())
                stage_values[f"t{index}_rmse"].extend(sample_rmse(output, gt).cpu().tolist())
                stage_values[f"t{index}_sam"].extend(sample_sam(output, gt).cpu().tolist())

            weighted = torch.zeros_like(outputs[0])
            for index in range(1, 5):
                previous, current = outputs[index - 1], outputs[index]
                delta = current - previous
                ideal = gt - previous
                weighted = weighted + weights[index - 1] * delta
                prev_rmse = sample_rmse(previous, gt)
                curr_rmse = sample_rmse(current, gt)
                prefix = f"d{index + 1}"
                increment_values[f"{prefix}_cos_ideal"].extend(
                    sample_cosine(delta, ideal).cpu().tolist()
                )
                increment_values[f"{prefix}_rmse_improvement"].extend(
                    (prev_rmse - curr_rmse).cpu().tolist()
                )
                increment_values[f"{prefix}_norm_ratio"].extend(
                    (delta.flatten(1).norm(dim=1) /
                     ideal.flatten(1).norm(dim=1).clamp_min(1e-12)).cpu().tolist()
                )

            late_ideal = gt - outputs[0]
            increment_values["weighted_cos_late_ideal"].extend(
                sample_cosine(weighted, late_ideal).cpu().tolist()
            )
            increment_values["weighted_norm_ratio"].extend(
                (weighted.flatten(1).norm(dim=1) /
                 late_ideal.flatten(1).norm(dim=1).clamp_min(1e-12)).cpu().tolist()
            )
            weighted_norm = weighted.flatten(1).norm(dim=1).clamp_min(1e-12)
            late_norm = late_ideal.flatten(1).norm(dim=1)
            scaled_weighted = weighted * (late_norm / weighted_norm).view(-1, 1, 1, 1)
            trajectory_target = outputs[0] + scaled_weighted
            increment_values["trajectory_target_psnr"].extend(
                sample_psnr(trajectory_target, gt).cpu().tolist()
            )
            increment_values["trajectory_target_sam"].extend(
                sample_sam(trajectory_target, gt).cpu().tolist()
            )
            increment_values["trajectory_target_vs_t5_rmse"].extend(
                sample_rmse(trajectory_target, outputs[-1]).cpu().tolist()
            )
            telescoped = sum((outputs[i] - outputs[i - 1] for i in range(1, 5)), torch.zeros_like(outputs[0]))
            increment_values["telescoping_max_error"].append(
                float((telescoped - (outputs[-1] - outputs[0])).abs().max().cpu())
            )
            seen += batch
            print(f"[trajectory] {seen}/{args.samples}", flush=True)

    stage_summary = {key: mean_std(value) for key, value in stage_values.items()}
    increment_summary = {key: mean_std(value) for key, value in increment_values.items()}
    result = {
        "samples": seen,
        "seconds": time.time() - started,
        "stage": stage_summary,
        "increment": increment_summary,
    }
    (OUT / "trajectory.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with open(OUT / "trajectory_stages.csv", "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["metric", "mean", "std"])
        for key, stats in sorted(stage_summary.items()):
            writer.writerow([key, stats["mean"], stats["std"]])
    with open(OUT / "trajectory_increments.csv", "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["metric", "mean", "std"])
        for key, stats in sorted(increment_summary.items()):
            writer.writerow([key, stats["mean"], stats["std"]])
    print(json.dumps(result, indent=2), flush=True)


class CovarianceAccumulator:
    def __init__(self, channels=28):
        self.count = 0
        self.sum = torch.zeros(channels, dtype=torch.float64)
        self.xtx = torch.zeros(channels, channels, dtype=torch.float64)

    def update(self, tensor):
        matrix = tensor.permute(0, 2, 3, 1).reshape(-1, tensor.shape[1]).float()
        self.count += matrix.shape[0]
        self.sum += matrix.sum(0).double().cpu()
        self.xtx += (matrix.T @ matrix).double().cpu()

    def summarize(self):
        mean = self.sum / self.count
        uncentered = self.xtx / self.count
        centered = uncentered - torch.outer(mean, mean)
        summaries = {}
        for name, covariance in (("uncentered", uncentered), ("centered", centered)):
            values = torch.linalg.eigvalsh(covariance).flip(0).clamp_min(0)
            energy = values / values.sum().clamp_min(1e-20)
            cumulative = energy.cumsum(0)
            summaries[name] = {
                "eigenvalues": values.tolist(),
                "cumulative": cumulative.tolist(),
                "rank_energy": {str(rank): float(cumulative[rank - 1]) for rank in (4, 6, 8, 11)},
                "rank90": int(torch.searchsorted(cumulative, 0.90).item() + 1),
                "rank95": int(torch.searchsorted(cumulative, 0.95).item() + 1),
                "rank99": int(torch.searchsorted(cumulative, 0.99).item() + 1),
            }
        return summaries


def run_svd():
    teacher = td.build_teacher(DEVICE)
    E2ESMILE, cassi_measure, phi_phi_t, shift_cube, _ = td.import_student_components()
    student = td.build_student(E2ESMILE).to(DEVICE)
    td.load_student_initialization(student, DEVICE)
    student.eval()
    train_set, mask = prepare_training()
    accumulators = {"gt_minus_student": CovarianceAccumulator(), "teacher_minus_student": CovarianceAccumulator()}
    seen = 0
    started = time.time()

    with torch.no_grad():
        while seen < args.samples:
            batch = min(args.batch, args.samples - seen)
            gt, mask_b, shifted, ppt, y = next_batch(
                train_set, mask, batch, cassi_measure, phi_phi_t, shift_cube
            )
            student_output = student(y, mask_b, shifted, ppt)
            teacher_output = teacher(
                y.unsqueeze(1), input_mask=(mask_b, ppt.unsqueeze(1))
            )[-1]
            accumulators["gt_minus_student"].update(gt - student_output)
            accumulators["teacher_minus_student"].update(teacher_output - student_output)
            seen += batch
            print(f"[svd] {seen}/{args.samples}", flush=True)

    result = {
        "samples": seen,
        "pixels": accumulators["gt_minus_student"].count,
        "seconds": time.time() - started,
        "residuals": {name: accumulator.summarize() for name, accumulator in accumulators.items()},
    }
    (OUT / "residual_svd.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with open(OUT / "residual_svd.csv", "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["residual", "covariance", "rank", "cumulative_energy"])
        for residual, variants in result["residuals"].items():
            for covariance, summary in variants.items():
                for rank, value in summary["rank_energy"].items():
                    writer.writerow([residual, covariance, rank, value])
    print(json.dumps(result, indent=2), flush=True)


def derivative_errors(prediction, target):
    pred_d1 = prediction[1:] - prediction[:-1]
    gt_d1 = target[1:] - target[:-1]
    pred_d2 = prediction[2:] - 2.0 * prediction[1:-1] + prediction[:-2]
    gt_d2 = target[2:] - 2.0 * target[1:-1] + target[:-2]
    d1_mae = (pred_d1 - gt_d1).abs().mean()
    d2_mae = (pred_d2 - gt_d2).abs().mean()
    threshold = torch.quantile(gt_d2.abs().flatten(), 0.80)
    high_mask = gt_d2.abs() >= threshold
    curvature_mae = (pred_d2 - gt_d2).abs()[high_mask].mean()
    return float(d1_mae), float(d2_mae), float(curvature_mae)


def load_state(model, path):
    checkpoint = torch.load(str(path), map_location=DEVICE)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state, strict=True)


def run_checkpoints():
    E2ESMILE, cassi_measure, phi_phi_t, shift_cube, _ = td.import_student_components()
    test_data = td.load_test(td.cfg["test_path"], nC=int(td.cfg["num_bands"]))
    mask = td.load_mask(td.cfg["mask_path"], nC=int(td.cfg["num_bands"]))
    registry = [
        ("base244", [2, 4, 4], "result/model/2026_06_28_18_58_00_e2e_capacity/best.pth"),
        ("k0_last", [2, 4, 4], "result/distill/2026_07_04_13_26_28_kd_k0_gt_only_244/last.pth"),
        ("k1_psnr", [2, 4, 4], "result/distill/2026_07_03_19_43_02_kd_k1_output_244/best.pth"),
        ("k1_last", [2, 4, 4], "result/distill/2026_07_03_19_43_02_kd_k1_output_244/last.pth"),
        ("k1a_psnr", [2, 4, 4], "result/distill/2026_07_04_23_01_53_kd_k1a_output_only_244/best_psnr.pth"),
        ("k1a_sam", [2, 4, 4], "result/distill/2026_07_04_23_01_53_kd_k1a_output_only_244/best_sam.pth"),
        ("k1b_sam", [2, 4, 4], "result/distill/2026_07_05_12_18_49_kd_k1b_spectral_only_244/best_sam.pth"),
        ("d1_sam", [2, 4, 4], "result/distill/2026_07_05_08_32_52_kd_d1_gt_derivative_244/best_sam.pth"),
        ("capacity266_psnr", [2, 6, 6], "result/model/2026_07_01_19_35_01_e2e_capacity_266/best.pth"),
        ("capacity266_sam_e230", [2, 6, 6], "result/model/2026_07_01_19_35_01_e2e_capacity_266/stopped_last_e230_psnr35.9484_sam0.104816.pth"),
    ]
    rows = []
    summaries = []
    for name, blocks, relative in registry:
        path = ROOT / relative
        if not path.is_file():
            print(f"[skip missing] {path}", flush=True)
            continue
        model = E2ESMILE(
            dim=28, unet_stage=2, num_blocks=blocks,
            use_sicmb=True, use_perchannel=True, post_block="ffn", ffn_mult=4,
            input_mode="H", output_dc=False, wpo_variant="full",
            gradient_checkpointing=False, bands=28,
        ).to(DEVICE).eval()
        load_state(model, path)
        model_rows = []
        torch.cuda.reset_peak_memory_stats(DEVICE)
        started = time.time()
        with torch.no_grad():
            for scene in range(len(test_data)):
                gt = test_data[scene:scene + 1].to(DEVICE).float()
                mask_b = td.expand_mask(mask, 1, DEVICE)
                shifted = shift_cube(mask_b)
                ppt = phi_phi_t(mask_b)
                y = cassi_measure(gt, mask_b)
                prediction = model(y, mask_b, shifted, ppt)
                d1_mae, d2_mae, curvature_mae = derivative_errors(prediction[0], gt[0])
                row = {
                    "model": name, "scene": scene + 1,
                    "psnr": float(td.torch_psnr(prediction[0], gt[0])),
                    "ssim": float(td.torch_ssim(prediction[0], gt[0])),
                    "sam": float(td.torch_sam(prediction[0], gt[0])),
                    "d1_mae": d1_mae, "d2_mae": d2_mae,
                    "high_curvature_d2_mae": curvature_mae,
                }
                rows.append(row)
                model_rows.append(row)
        summary = {"model": name, "checkpoint": str(path)}
        for metric in ("psnr", "ssim", "sam", "d1_mae", "d2_mae", "high_curvature_d2_mae"):
            summary[metric] = float(np.mean([row[metric] for row in model_rows]))
        summary["seconds"] = time.time() - started
        summary["peak_memory_gib"] = torch.cuda.max_memory_allocated(DEVICE) / 1024 ** 3
        summaries.append(summary)
        print(summary, flush=True)
        del model
        torch.cuda.empty_cache()

    psnr_wins = defaultdict(int)
    sam_wins = defaultdict(int)
    for scene in range(1, len(test_data) + 1):
        scene_rows = [row for row in rows if row["scene"] == scene]
        psnr_wins[max(scene_rows, key=lambda row: row["psnr"])["model"]] += 1
        sam_wins[min(scene_rows, key=lambda row: row["sam"])["model"]] += 1
    for summary in summaries:
        summary["psnr_scene_wins"] = psnr_wins[summary["model"]]
        summary["sam_scene_wins"] = sam_wins[summary["model"]]

    with open(OUT / "checkpoint_per_scene.csv", "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(OUT / "checkpoint_summary.csv", "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (OUT / "checkpoint_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    torch.cuda.set_device(DEVICE)
    if args.mode == "trajectory":
        run_trajectory()
    elif args.mode == "svd":
        run_svd()
    else:
        run_checkpoints()


