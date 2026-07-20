"""Controlled K0/K1/D1 fine-tuning for SMILE² E2E experiments.

K0: GT-only continuation from the frozen student initialization checkpoint.
K1: the same continuation plus final-output and spectral-angle distillation
    from a frozen 5-stage unfolding teacher.
D1: GT-only continuation plus robust first/second spectral derivatives.

The teacher and student repositories both expose a top-level ``model`` package.
The teacher is therefore built first in an isolated import window; its modules
are then removed from ``sys.modules`` before importing the E2E student package.
The instantiated teacher keeps valid references to its original class globals.
"""

import os
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.environ.get("SMILE_CONFIG", ROOT / "configs/distill_k1.yaml"))
if not CONFIG_PATH.is_absolute():
    CONFIG_PATH = ROOT / CONFIG_PATH
with open(CONFIG_PATH, "r", encoding="utf-8") as fp:
    cfg = yaml.safe_load(fp)

# Must be set before importing torch.
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu_id"])

import json
import random
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F

from dataset import load_mask, load_test, load_training, shuffle_crop
from loss import rmse_loss, torch_psnr, torch_sam, torch_ssim


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _clear_model_package():
    for name in list(sys.modules):
        if name == "model" or name.startswith("model."):
            del sys.modules[name]


def build_teacher(device):
    """Build the exact version3_fixedLDE teacher without polluting imports."""
    teacher_root = Path(cfg["teacher_root"]).resolve()
    if not teacher_root.is_dir():
        raise FileNotFoundError(f"teacher_root not found: {teacher_root}")

    _clear_model_package()
    sys.path.insert(0, str(teacher_root))
    try:
        from model.unfolding import WPO_Unfold

        teacher = WPO_Unfold(
            dim=int(cfg["num_bands"]),
            unet_stage=int(cfg.get("teacher_unet_stage", 2)),
            num_blocks=list(cfg.get("teacher_num_blocks", [2, 2, 2])),
            use_kg=False,
            num_stages=int(cfg.get("teacher_num_stages", 5)),
            share_weights=True,
            fbgw_mode="none",
            size=int(cfg["crop_size"]),
            len_shift=2,
            use_ahqs=True,
            use_sab=False,
            debug=False,
        )
        state = torch.load(str(cfg["teacher_checkpoint"]), map_location="cpu")
        teacher.load_state_dict(state, strict=True)
        teacher.to(device).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    finally:
        try:
            sys.path.remove(str(teacher_root))
        except ValueError:
            pass

    # Keep the instantiated object, but free the package name for the student.
    _clear_model_package()
    return teacher


def import_student_components():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from model.e2e import E2ESMILE, cassi_measure, phi_phi_t, shift_cube
    from model.wpo3d import WPO3D

    return E2ESMILE, cassi_measure, phi_phi_t, shift_cube, WPO3D


def build_student(model_class):
    return model_class(
        dim=int(cfg["dim"]),
        unet_stage=int(cfg["unet_stage"]),
        num_blocks=list(cfg["num_blocks"]),
        use_sicmb=bool(cfg["use_sicmb"]),
        use_perchannel=bool(cfg["use_perchannel"]),
        use_spectral_wave=bool(cfg.get("use_spectral_wave", True)),
        post_block=cfg["post_block"],
        ffn_mult=int(cfg["ffn_mult"]),
        input_mode=cfg["input_mode"],
        output_dc=bool(cfg["output_dc"]),
        dc_gamma_init=float(cfg.get("dc_gamma_init", 0.30)),
        wpo_variant=cfg.get("wpo_variant", "full"),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", False)),
        bands=int(cfg["num_bands"]),
        low_rank_residual_rank=int(cfg.get("low_rank_residual_rank", 0)),
        low_rank_gamma_init=float(cfg.get("low_rank_gamma_init", 0.0)),
    )


def load_student_initialization(student, device):
    checkpoint = torch.load(str(cfg["student_checkpoint"]), map_location=device)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    allow_new_head = bool(cfg.get("allow_new_low_rank_head", False))
    if allow_new_head:
        incompatible = student.load_state_dict(state, strict=False)
        invalid_missing = [key for key in incompatible.missing_keys if not key.startswith("low_rank_head.")]
        if invalid_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                f"invalid checkpoint mismatch: missing={invalid_missing}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        print(f"Initialized new low-rank head; missing keys={len(incompatible.missing_keys)}", flush=True)
    else:
        student.load_state_dict(state, strict=True)
    return int(checkpoint.get("epoch", -1)) if isinstance(checkpoint, dict) else -1


def count_params(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) / 1e6


def expand_mask(mask, batch_size, device):
    return mask.unsqueeze(0).expand(batch_size, -1, -1, -1).to(device).float()


def spectral_cosine_loss(prediction, target, eps=1e-8):
    prediction = F.normalize(prediction, p=2, dim=1, eps=eps)
    target = F.normalize(target, p=2, dim=1, eps=eps)
    return (1.0 - (prediction * target).sum(dim=1)).mean()


def spectral_derivative_loss(prediction, target, order=1, loss_type="charbonnier", eps=1e-3):
    """Robust loss between finite differences on the explicit 28-band axis."""
    if prediction.ndim != 4 or target.ndim != 4:
        raise ValueError("spectral derivative loss expects [B,C,H,W] tensors")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    if order == 1:
        pred_diff = prediction[:, 1:] - prediction[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
    elif order == 2:
        pred_diff = prediction[:, 2:] - 2.0 * prediction[:, 1:-1] + prediction[:, :-2]
        target_diff = target[:, 2:] - 2.0 * target[:, 1:-1] + target[:, :-2]
    else:
        raise ValueError(f"spectral derivative order must be 1 or 2, got {order}")
    error = pred_diff - target_diff
    if loss_type == "charbonnier":
        return torch.sqrt(error.square() + float(eps) ** 2).mean()
    if loss_type == "l1":
        return error.abs().mean()
    if loss_type == "rmse":
        return torch.sqrt(error.square().mean() + 1e-12)
    raise ValueError(f"derivative_loss_type must be charbonnier/l1/rmse, got {loss_type!r}")


@torch.no_grad()
def teacher_forward(teacher, y, mask_b, ppt):
    microbatch = max(1, int(cfg.get("teacher_microbatch", 1)))
    predictions = []
    for start in range(0, y.shape[0], microbatch):
        end = min(start + microbatch, y.shape[0])
        outputs = teacher(
            y[start:end].unsqueeze(1),
            input_mask=(mask_b[start:end], ppt[start:end].unsqueeze(1)),
        )
        predictions.append(outputs[-1].detach())
    return torch.cat(predictions, dim=0)


@torch.no_grad()
def teacher_forward_stages(teacher, y, mask_b, ppt):
    microbatch = max(1, int(cfg.get("teacher_microbatch", 1)))
    stage_chunks = None
    for start in range(0, y.shape[0], microbatch):
        end = min(start + microbatch, y.shape[0])
        outputs = teacher(
            y[start:end].unsqueeze(1),
            input_mask=(mask_b[start:end], ppt[start:end].unsqueeze(1)),
        )
        if stage_chunks is None:
            stage_chunks = [[] for _ in outputs]
        for index, output in enumerate(outputs):
            stage_chunks[index].append(output.detach())
    return [torch.cat(chunks, dim=0) for chunks in stage_chunks]


def normalized_trajectory_target(stages, weights=(0.3, 0.5, 0.7, 1.0), eps=1e-8):
    """Scale a weighted T2..T5 increment direction to the full T1->T5 norm."""
    if len(stages) != 5:
        raise ValueError(f"trajectory KD expects 5 stages, got {len(stages)}")
    weight_tensor = stages[0].new_tensor(weights)
    weight_tensor = weight_tensor / weight_tensor.sum()
    raw = torch.zeros_like(stages[0])
    for index, weight in enumerate(weight_tensor, start=1):
        raw = raw + weight * (stages[index] - stages[index - 1])
    full = stages[-1] - stages[0]
    raw_norm = raw.flatten(1).norm(dim=1).clamp_min(eps)
    full_norm = full.flatten(1).norm(dim=1)
    scale = (full_norm / raw_norm).view(-1, 1, 1, 1)
    scaled = raw * scale
    target = stages[0] + scaled
    cosine = spectral_cosine_loss(scaled, full)
    return target.detach(), scale.detach(), cosine.detach()


def save_checkpoint(
    path, epoch, student, optimizer, scheduler,
    best_psnr, best_sam, initial_epoch, metrics=None,
):
    torch.save(
        {
            "epoch": epoch,
            "student_initial_epoch": initial_epoch,
            "model": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_psnr": best_psnr,
            "best_sam": best_sam,
            "metrics": metrics,
            "config": cfg,
        },
        str(path),
    )


def print_config(student, teacher, initial_epoch):
    mode = cfg["distill_mode"].lower()
    has_derivatives = bool(
        float(cfg.get("gt_d1_weight", 0.0)) or float(cfg.get("gt_d2_weight", 0.0))
    )
    if mode == "k0" and has_derivatives:
        mode_description = "GT + spectral derivatives"
    elif mode == "k0":
        mode_description = "GT only"
    else:
        mode_description = "GT + teacher KD"
    print("=" * 72, flush=True)
    print(f"Experiment:       {cfg['experiment_name']}", flush=True)
    print(f"Mode:             {mode} ({mode_description})", flush=True)
    print(f"GPU:              physical {cfg['gpu_id']} (local cuda:0)", flush=True)
    print(f"Student:          blocks={cfg['num_blocks']} params={count_params(student):.4f}M", flush=True)
    print(f"Student init:     epoch={initial_epoch} {cfg['student_checkpoint']}", flush=True)
    print(f"Teacher:          {'disabled' if teacher is None else cfg['teacher_checkpoint']}", flush=True)
    print(f"Batch/samples:    {cfg['batch_size']} / {cfg['epoch_sample']} per epoch", flush=True)
    print(f"Fine-tune:        {cfg['max_epoch']} epochs lr={float(cfg['learning_rate']):.2e}", flush=True)
    print(
        f"Loss weights:     GT=1.0 output={float(cfg.get('kd_output_weight', 0.0)):.3f} "
        f"spectral={float(cfg.get('kd_spectral_weight', 0.0)):.3f} "
        f"D1={float(cfg.get('gt_d1_weight', 0.0)):.3f} "
        f"D2={float(cfg.get('gt_d2_weight', 0.0)):.3f} "
        f"warmup={int(cfg.get('derivative_warmup_epochs', 0))}ep "
        f"deriv={cfg.get('derivative_loss_type', 'charbonnier')}",
        flush=True,
    )
    print(
        f"Trajectory KD:    output={float(cfg.get('trajectory_output_weight', 0.0)):.3f} "
        f"trajectory={float(cfg.get('trajectory_kd_weight', 0.0)):.3f}; "
        f"curriculum={float(cfg.get('stage_curriculum_weight', 0.0)):.3f}; "
        f"low-rank={int(cfg.get('low_rank_residual_rank', 0))}",
        flush=True,
    )
    print(f"Config:           {CONFIG_PATH}", flush=True)
    print("=" * 72, flush=True)


def print_physics_stats(student, wpo_class):
    alpha_all, vs_all = [], []
    for module in student.modules():
        if isinstance(module, wpo_class):
            alpha, vs, _, _ = module._get_effective_params()
            alpha_all.append(alpha.detach().reshape(-1).cpu())
            vs_all.append(vs.detach().reshape(-1).cpu())
    if alpha_all:
        alpha = torch.cat(alpha_all)
        vs = torch.cat(vs_all)
        print(
            f"  [PerCh] alpha mean={alpha.mean():.4f} std={alpha.std():.4f} "
            f"CV={alpha.std()/(alpha.mean()+1e-12):.3f}; "
            f"vs mean={vs.mean():.4f} std={vs.std():.4f} "
            f"CV={vs.std()/(vs.mean()+1e-12):.3f}",
            flush=True,
        )
    low_rank_head = getattr(student, "low_rank_head", None)
    if low_rank_head is not None:
        print(
            f"  [LowRank] rank={low_rank_head.rank} "
            f"gamma={torch.tanh(low_rank_head.gamma_raw).detach().item():.6f}",
            flush=True,
        )


def train_epoch(
    epoch,
    student,
    teacher,
    optimizer,
    train_set,
    mask,
    device,
    cassi_measure,
    phi_phi_t,
    shift_cube,
):
    student.train()
    if teacher is not None:
        teacher.eval()
    batch_size = int(cfg["batch_size"])
    batch_num = int(cfg["epoch_sample"]) // batch_size
    mask_b = expand_mask(mask, batch_size, device)
    shifted_mask = shift_cube(mask_b)
    ppt = phi_phi_t(mask_b)
    totals = {
        "loss": 0.0, "gt": 0.0, "out": 0.0, "spectral": 0.0,
        "d1": 0.0, "d2": 0.0, "trajectory": 0.0,
        "trajectory_scale": 0.0, "trajectory_cosine": 0.0,
        "curriculum": 0.0, "curriculum_stage": 0.0,
    }
    start_time = time.time()
    torch.cuda.reset_peak_memory_stats(device)

    for batch_idx in range(batch_num):
        gt = shuffle_crop(
            train_set,
            batch_size,
            int(cfg["crop_size"]),
            device=device,
            nC=int(cfg["num_bands"]),
        ).float()
        y = cassi_measure(gt, mask_b)

        trajectory_weight = float(cfg.get("trajectory_kd_weight", 0.0))
        trajectory_output_weight = float(cfg.get("trajectory_output_weight", 0.0))
        curriculum_weight = float(cfg.get("stage_curriculum_weight", 0.0))
        if teacher is not None and (trajectory_weight > 0 or curriculum_weight > 0):
            teacher_stages = teacher_forward_stages(teacher, y, mask_b, ppt)
            target = teacher_stages[-1]
            if trajectory_weight > 0:
                trajectory_target, trajectory_scale, trajectory_cosine = normalized_trajectory_target(
                    teacher_stages
                )
            else:
                trajectory_target = None
                trajectory_scale = None
                trajectory_cosine = None
            milestones = list(cfg.get("stage_curriculum_milestones", [5, 10, 15]))
            curriculum_index = min(4, 1 + sum(epoch > int(milestone) for milestone in milestones))
            curriculum_target = teacher_stages[curriculum_index]
        else:
            teacher_stages = None
            target = teacher_forward(teacher, y, mask_b, ppt) if teacher is not None else None
            trajectory_target = None
            trajectory_scale = None
            trajectory_cosine = None
            curriculum_target = None
            curriculum_index = 0
        optimizer.zero_grad(set_to_none=True)
        prediction = student(y, mask_b, shifted_mask, ppt)
        loss_gt = rmse_loss(prediction, gt)

        if target is None:
            loss_out = prediction.new_zeros(())
            loss_spectral = prediction.new_zeros(())
        else:
            loss_out = rmse_loss(prediction, target)
            loss_spectral = spectral_cosine_loss(prediction, target)

        if trajectory_target is None:
            loss_trajectory = prediction.new_zeros(())
            trajectory_scale_mean = 0.0
            trajectory_cosine_value = 0.0
        else:
            loss_trajectory = rmse_loss(prediction, trajectory_target)
            trajectory_scale_mean = float(trajectory_scale.mean())
            trajectory_cosine_value = float(trajectory_cosine)

        loss_curriculum = (
            rmse_loss(prediction, curriculum_target)
            if curriculum_target is not None else prediction.new_zeros(())
        )

        d1_weight = float(cfg.get("gt_d1_weight", 0.0))
        d2_weight = float(cfg.get("gt_d2_weight", 0.0))
        derivative_loss_type = str(cfg.get("derivative_loss_type", "charbonnier")).lower()
        derivative_eps = float(cfg.get("derivative_charbonnier_eps", 1e-3))
        loss_d1 = spectral_derivative_loss(
            prediction, gt, order=1, loss_type=derivative_loss_type, eps=derivative_eps
        ) if d1_weight else prediction.new_zeros(())
        loss_d2 = spectral_derivative_loss(
            prediction, gt, order=2, loss_type=derivative_loss_type, eps=derivative_eps
        ) if d2_weight else prediction.new_zeros(())
        warmup_epochs = int(cfg.get("derivative_warmup_epochs", 0))
        derivative_scale = min(1.0, epoch / warmup_epochs) if warmup_epochs > 0 else 1.0

        loss = (
            loss_gt
            + float(cfg.get("kd_output_weight", 0.0)) * loss_out
            + float(cfg.get("kd_spectral_weight", 0.0)) * loss_spectral
            + derivative_scale * d1_weight * loss_d1
            + derivative_scale * d2_weight * loss_d2
            + trajectory_output_weight * loss_out
            + trajectory_weight * loss_trajectory
            + curriculum_weight * loss_curriculum
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at epoch={epoch} batch={batch_idx}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), float(cfg["grad_clip"]))
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite gradient at epoch={epoch} batch={batch_idx}")
        optimizer.step()

        totals["loss"] += loss.item()
        totals["gt"] += loss_gt.item()
        totals["out"] += loss_out.item()
        totals["spectral"] += loss_spectral.item()
        totals["d1"] += loss_d1.item()
        totals["d2"] += loss_d2.item()
        totals["trajectory"] += loss_trajectory.item()
        totals["trajectory_scale"] += trajectory_scale_mean
        totals["trajectory_cosine"] += trajectory_cosine_value
        totals["curriculum"] += loss_curriculum.item()
        totals["curriculum_stage"] += curriculum_index + 1 if curriculum_target is not None else 0

        interval = int(cfg.get("log_interval", 0))
        if interval and (batch_idx + 1) % interval == 0:
            elapsed = time.time() - start_time
            rate = elapsed / (batch_idx + 1)
            eta = rate * (batch_num - batch_idx - 1)
            print(
                f"  [E{epoch:03d}] {batch_idx+1}/{batch_num} "
                f"loss={totals['loss']/(batch_idx+1):.5f} "
                f"gt={totals['gt']/(batch_idx+1):.5f} "
                f"out={totals['out']/(batch_idx+1):.5f} "
                f"spec={totals['spectral']/(batch_idx+1):.5f} "
                f"d1={totals['d1']/(batch_idx+1):.5f} "
                f"d2={totals['d2']/(batch_idx+1):.5f} "
                f"traj={totals['trajectory']/(batch_idx+1):.5f} "
                f"curr={totals['curriculum']/(batch_idx+1):.5f} "
                f"T={curriculum_index+1 if curriculum_target is not None else 0} "
                f"dwarm={derivative_scale:.2f} "
                f"{rate:.3f}s/it ETA={eta/60:.1f}min",
                flush=True,
            )

    elapsed = time.time() - start_time
    peak_memory_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    averages = {key: value / batch_num for key, value in totals.items()}
    print(
        f"[Epoch {epoch:03d}] Loss={averages['loss']:.6f} GT={averages['gt']:.6f} "
        f"KD={averages['out']:.6f} Spec={averages['spectral']:.6f} "
        f"D1={averages['d1']:.6f} D2={averages['d2']:.6f} "
        f"Traj={averages['trajectory']:.6f} "
        f"TrajScale={averages['trajectory_scale']:.3f} "
        f"TrajCosLoss={averages['trajectory_cosine']:.4f} "
        f"Curr={averages['curriculum']:.6f} "
        f"CurrStage={averages['curriculum_stage']:.1f} "
        f"Time={elapsed:.1f}s PeakMem={peak_memory_gb:.2f}GiB "
        f"LR={optimizer.param_groups[0]['lr']:.2e}",
        flush=True,
    )
    return averages


@torch.no_grad()
def test_student(student, test_data, mask, device, cassi_measure, phi_phi_t, shift_cube):
    student.eval()
    psnrs, ssims, sams = [], [], []
    start_time = time.time()
    for start in range(0, len(test_data), int(cfg.get("test_batch_size", 1))):
        gt = test_data[start:start + int(cfg.get("test_batch_size", 1))].to(device).float()
        mask_b = expand_mask(mask, gt.shape[0], device)
        shifted_mask = shift_cube(mask_b)
        ppt = phi_phi_t(mask_b)
        y = cassi_measure(gt, mask_b)
        prediction = student(y, mask_b, shifted_mask, ppt)
        for idx in range(prediction.shape[0]):
            psnrs.append(torch_psnr(prediction[idx], gt[idx]).item())
            ssims.append(torch_ssim(prediction[idx], gt[idx]).item())
            sams.append(torch_sam(prediction[idx], gt[idx]).item())
    means = tuple(sum(values) / len(values) for values in (psnrs, ssims, sams))
    print(
        f"         Student -> PSNR={means[0]:.4f} SSIM={means[1]:.6f} "
        f"SAM={means[2]:.6f} Time={time.time()-start_time:.1f}s",
        flush=True,
    )
    return means


@torch.no_grad()
def verify_teacher(teacher, test_data, mask, device, cassi_measure, phi_phi_t):
    if teacher is None:
        return None
    teacher.eval()
    psnrs, ssims, sams = [], [], []
    for idx in range(len(test_data)):
        gt = test_data[idx:idx + 1].to(device).float()
        mask_b = expand_mask(mask, 1, device)
        ppt = phi_phi_t(mask_b)
        y = cassi_measure(gt, mask_b)
        prediction = teacher_forward(teacher, y, mask_b, ppt)
        psnrs.append(torch_psnr(prediction[0], gt[0]).item())
        ssims.append(torch_ssim(prediction[0], gt[0]).item())
        sams.append(torch_sam(prediction[0], gt[0]).item())
    means = tuple(sum(values) / len(values) for values in (psnrs, ssims, sams))
    print(
        f"         Teacher -> PSNR={means[0]:.4f} SSIM={means[1]:.6f} SAM={means[2]:.6f}",
        flush=True,
    )
    minimum = float(cfg.get("teacher_verify_min_psnr", 38.0))
    if means[0] < minimum:
        raise RuntimeError(f"teacher verification failed: PSNR {means[0]:.4f} < {minimum:.4f}")
    return means


def main():
    mode = str(cfg["distill_mode"]).lower()
    if mode not in {"k0", "k1"}:
        raise ValueError("distill_mode must be 'k0' or 'k1'")

    seed_everything(int(cfg.get("seed", 2026)))
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    teacher = build_teacher(device) if mode == "k1" else None
    E2ESMILE, cassi_measure, phi_phi_t, shift_cube, WPO3D = import_student_components()
    student = build_student(E2ESMILE).to(device)
    initial_epoch = load_student_initialization(student, device)

    print(f"starting at: {time.ctime()}", flush=True)
    train_set = load_training(cfg["train_path"], max_scenes=205)
    test_data = load_test(cfg["test_path"], nC=int(cfg["num_bands"]))
    mask = load_mask(cfg["mask_path"], nC=int(cfg["num_bands"]))
    print_config(student, teacher, initial_epoch)

    optimizer = torch.optim.Adam(
        student.parameters(), lr=float(cfg["learning_rate"]), betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cfg.get("scheduler_t_max", cfg["max_epoch"])),
        eta_min=float(cfg.get("eta_min", 1e-6)),
    )

    run_stamp = time.strftime("%Y_%m_%d_%H_%M_%S")
    save_dir = ROOT / "result" / "distill" / f"{run_stamp}_{cfg['experiment_name']}"
    save_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(CONFIG_PATH), str(save_dir / "config.yaml"))
    with open(save_dir / "run_meta.json", "w", encoding="utf-8") as fp:
        json.dump(
            {
                "experiment": cfg["experiment_name"],
                "mode": mode,
                "student_checkpoint": cfg["student_checkpoint"],
                "teacher_checkpoint": cfg.get("teacher_checkpoint", ""),
                "torch": torch.__version__,
                "cuda_device": torch.cuda.get_device_name(0),
                "student_parameters_m": count_params(student),
            },
            fp,
            indent=2,
            ensure_ascii=False,
        )

    if bool(cfg.get("verify_teacher", mode == "k1")):
        verify_teacher(teacher, test_data, mask, device, cassi_measure, phi_phi_t)
    baseline_psnr, baseline_ssim, baseline_sam = test_student(
        student, test_data, mask, device, cassi_measure, phi_phi_t, shift_cube
    )
    best_psnr = baseline_psnr
    best_sam = baseline_sam
    baseline_metrics = {"psnr": baseline_psnr, "ssim": baseline_ssim, "sam": baseline_sam}
    save_checkpoint(
        save_dir / "initial.pth", 0, student, optimizer, scheduler,
        best_psnr, best_sam, initial_epoch, baseline_metrics,
    )
    save_checkpoint(
        save_dir / "best_psnr.pth", 0, student, optimizer, scheduler,
        best_psnr, best_sam, initial_epoch, baseline_metrics,
    )
    save_checkpoint(
        save_dir / "best_sam.pth", 0, student, optimizer, scheduler,
        best_psnr, best_sam, initial_epoch, baseline_metrics,
    )
    # Compatibility alias for older evaluation scripts.
    save_checkpoint(
        save_dir / "best.pth", 0, student, optimizer, scheduler,
        best_psnr, best_sam, initial_epoch, baseline_metrics,
    )
    print(
        f"Baseline: PSNR={baseline_psnr:.4f} SSIM={baseline_ssim:.6f} SAM={baseline_sam:.6f}",
        flush=True,
    )
    print(f"Save dir: {save_dir}", flush=True)

    for epoch in range(1, int(cfg["max_epoch"]) + 1):
        try:
            train_epoch(
                epoch,
                student,
                teacher,
                optimizer,
                train_set,
                mask,
                device,
                cassi_measure,
                phi_phi_t,
                shift_cube,
            )
        except FloatingPointError as exc:
            torch.save(
                {"epoch": epoch, "model": student.state_dict(), "error": str(exc)},
                save_dir / "nan_abort.pth",
            )
            print(f"[ABORT] {exc}", flush=True)
            raise

        psnr, ssim, sam = test_student(
            student, test_data, mask, device, cassi_measure, phi_phi_t, shift_cube
        )
        print_physics_stats(student, WPO3D)
        scheduler.step()
        metrics = {"psnr": psnr, "ssim": ssim, "sam": sam}
        is_best_psnr = psnr > best_psnr
        is_best_sam = sam < best_sam
        if is_best_psnr:
            best_psnr = psnr
        if is_best_sam:
            best_sam = sam

        save_checkpoint(
            save_dir / "last.pth", epoch, student, optimizer, scheduler,
            best_psnr, best_sam, initial_epoch, metrics,
        )
        if is_best_psnr:
            save_checkpoint(
                save_dir / "best_psnr.pth", epoch, student, optimizer, scheduler,
                best_psnr, best_sam, initial_epoch, metrics,
            )
            save_checkpoint(
                save_dir / "best.pth", epoch, student, optimizer, scheduler,
                best_psnr, best_sam, initial_epoch, metrics,
            )
            print(f"  * New PSNR best: PSNR={psnr:.4f} SSIM={ssim:.6f} SAM={sam:.6f}", flush=True)
        if is_best_sam:
            save_checkpoint(
                save_dir / "best_sam.pth", epoch, student, optimizer, scheduler,
                best_psnr, best_sam, initial_epoch, metrics,
            )
            print(f"  * New SAM best:  PSNR={psnr:.4f} SSIM={ssim:.6f} SAM={sam:.6f}", flush=True)
        if epoch % int(cfg.get("save_every", 10)) == 0:
            save_checkpoint(
                save_dir / f"epoch_{epoch:03d}.pth",
                epoch,
                student,
                optimizer,
                scheduler,
                best_psnr,
                best_sam,
                initial_epoch,
                metrics,
            )

    print(
        f"finished at: {time.ctime()} best_psnr={best_psnr:.4f} best_sam={best_sam:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

