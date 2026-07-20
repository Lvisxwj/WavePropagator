"""YAML-driven single-GPU trainer for the three E2E experiments."""

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.environ.get("SMILE_CONFIG", ROOT / "configs/compact.yaml"))
if not CONFIG_PATH.is_absolute():
    CONFIG_PATH = ROOT / CONFIG_PATH
with open(CONFIG_PATH, "r", encoding="utf-8") as fp:
    cfg = yaml.safe_load(fp)

# Explicit smoke/debug overrides. They are written into resolved_config/run_meta.
_ENV_OVERRIDES = {
    "SMILE_EXPERIMENT_NAME": ("experiment_name", str),
    "SMILE_BATCH_SIZE": ("batch_size", int),
    "SMILE_MAX_EPOCH": ("max_epoch", int),
    "SMILE_EPOCH_SAMPLE": ("epoch_sample", int),
}
for _env_name, (_cfg_name, _cast) in _ENV_OVERRIDES.items():
    if _env_name in os.environ:
        cfg[_cfg_name] = _cast(os.environ[_env_name])

# Must happen before importing torch.
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu_id"])

import json
import math
import random
import shutil
import time

import numpy as np
import torch

from dataset import load_mask, load_test, load_training, shuffle_crop
from loss import rmse_loss, torch_psnr, torch_sam, torch_ssim
from model.e2e import E2ESMILE, cassi_measure, phi_phi_t, shift_cube
from model.wpo3d import WPO3D
from diagnostics import (
    append_csv, atomic_write_json, collect_model_stats, probe_forward,
    rng_state, source_manifest,
)


class NumericalTrainingError(FloatingPointError):
    def __init__(self, message, snapshot=None):
        super().__init__(message)
        self.snapshot = snapshot or {}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model():
    return E2ESMILE(
        dim=cfg["dim"],
        unet_stage=cfg["unet_stage"],
        num_blocks=cfg["num_blocks"],
        use_sicmb=cfg["use_sicmb"],
        use_perchannel=cfg["use_perchannel"],
        use_spectral_wave=cfg.get("use_spectral_wave", True),
        post_block=cfg["post_block"],
        ffn_mult=cfg["ffn_mult"],
        input_mode=cfg["input_mode"],
        output_dc=cfg["output_dc"],
        dc_gamma_init=cfg.get("dc_gamma_init", 0.30),
        wpo_variant=cfg.get("wpo_variant", "full"),
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        bands=cfg["num_bands"],
        input_adapter=cfg.get("input_adapter", "none"),
        wavelength_cutoff_init=cfg.get("wavelength_cutoff_init", 0.28),
        wave_param_mode=cfg.get("wave_param_mode", "free"),
        wave_basis_count=cfg.get("wave_basis_count", 3),
        use_mask_gate=cfg.get("use_mask_gate", True),
        progressive_steps=cfg.get("progressive_steps", 1),
        progressive_share=cfg.get("progressive_share", True),
        return_intermediates=cfg.get("return_intermediates", False),
        progressive_role_mode=cfg.get("progressive_role_mode", "plain"),
        disable_sfevolver=cfg.get("disable_sfevolver", False),
    )


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def expand_mask(mask, batch_size, device):
    return mask.unsqueeze(0).expand(batch_size, -1, -1, -1).to(device).float()


def print_config(model):
    print("=" * 68, flush=True)
    print(f"Experiment:      {cfg['experiment_name']}", flush=True)
    print(f"GPU:             physical {cfg['gpu_id']} (local cuda:0)", flush=True)
    print(f"E2E:             single pass, no unfolding/LDE/rho/delta-Phi", flush=True)
    print(f"Backbone:        2-level dim={cfg['dim']} blocks={cfg['num_blocks']}", flush=True)
    print(
        f"SWAP:            SiCMB={cfg['use_sicmb']} PerCh={cfg['use_perchannel']} "
        f"SpectralWave={cfg.get('use_spectral_wave', True)} "
        f"MaskGateA={cfg.get('use_mask_gate', True)}",
        flush=True,
    )
    print(f"WPO variant:     {cfg.get('wpo_variant', 'full')}", flush=True)
    print(
        f"Wave parameters: {cfg.get('wave_param_mode', 'free')} "
        f"basis={cfg.get('wave_basis_count', 3)}",
        flush=True,
    )
    print(f"Input adapter:   {cfg.get('input_adapter', 'none')}", flush=True)
    print(
        f"Progressive:     steps={cfg.get('progressive_steps', 1)} "
        f"share={cfg.get('progressive_share', True)} "
        f"role={cfg.get('progressive_role_mode', 'plain')} "
        f"disable_sfevolver={cfg.get('disable_sfevolver', False)} "
        f"loss_weights={cfg.get('progressive_loss_weights', [])}",
        flush=True,
    )
    print(f"Grad checkpoint: {cfg.get('gradient_checkpointing', False)}", flush=True)
    print(f"Post block:      {cfg['post_block']} mult={cfg['ffn_mult']}", flush=True)
    print(f"Input/DC:        {cfg['input_mode']} / output_dc={cfg['output_dc']}", flush=True)
    print(f"Batch/samples:   {cfg['batch_size']} / {cfg['epoch_sample']} per epoch", flush=True)
    print(f"Parameters:      {count_params(model):.4f} M", flush=True)
    print(f"Config:          {CONFIG_PATH}", flush=True)
    print("=" * 68, flush=True)


def save_checkpoint(path, epoch, model, optimizer, scheduler, best_psnr, best_sam, metrics):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_psnr": best_psnr,
        "best_sam": best_sam,
        "metrics": metrics,
        "config": cfg,
        "rng_state": rng_state(),
    }, str(path))


def final_output(output):
    if isinstance(output, (list, tuple)):
        return output[-1]
    return output


def progressive_loss(outputs, gt):
    if not isinstance(outputs, (list, tuple)):
        return rmse_loss(outputs, gt)
    weights = cfg.get("progressive_loss_weights", None)
    if weights is None or len(weights) == 0:
        weights = [0.0] * (len(outputs) - 1) + [1.0]
    weights = [float(w) for w in weights]
    if len(weights) != len(outputs):
        raise ValueError(
            f"progressive_loss_weights length {len(weights)} != outputs length {len(outputs)}"
        )
    loss = outputs[-1].new_tensor(0.0)
    for weight, output in zip(weights, outputs):
        if weight:
            loss = loss + weight * rmse_loss(output, gt)
    return loss


def print_physics_stats(model):
    rows = []
    for name, module in model.named_modules():
        if isinstance(module, WPO3D):
            alpha, vs, _, _ = module._get_effective_params()
            rows.append((name, module, alpha.detach().reshape(-1).cpu(), vs.detach().reshape(-1).cpu()))
    if rows:
        alpha = torch.cat([row[2] for row in rows])
        vs = torch.cat([row[3] for row in rows])
        scopes = {"mode" if row[2].numel() > 1 else "scalar" for row in rows}
        scope = scopes.pop() if len(scopes) == 1 else "mixed"
        print(
            f"  [Wave:{scope}] layers={len(rows)} alpha mean={alpha.mean():.4f} std={alpha.std():.4f} "
            f"CV={alpha.std()/(alpha.mean()+1e-12):.3f}; "
            f"vs mean={vs.mean():.4f} std={vs.std():.4f} "
            f"CV={vs.std()/(vs.mean()+1e-12):.3f}",
            flush=True,
        )
    if model.dc_gamma is not None:
        print(f"  [DC] gamma={model.dc_gamma.item():.5f}", flush=True)
    if getattr(model, "estimate_step_beta", None) is not None:
        beta = model.estimate_step_beta
        if beta is not None:
            print(f"  [EstimateStep] beta={beta.item():.5f}", flush=True)


def train_epoch(epoch, model, optimizer, scaler, train_set, mask, device, save_dir=None):
    model.train()
    batch_size = int(cfg["batch_size"])
    batch_num = int(cfg["epoch_sample"]) // batch_size
    mask_b = expand_mask(mask, batch_size, device)
    shifted_mask = shift_cube(mask_b)
    ppt = phi_phi_t(mask_b)
    total = 0.0
    grad_total = 0.0
    start = time.time()
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
        optimizer.zero_grad(set_to_none=True)

        if cfg.get("use_amp", False):
            with torch.cuda.amp.autocast():
                outputs = model(y, mask_b, shifted_mask, ppt)
                pred = final_output(outputs)
                loss = progressive_loss(outputs, gt)
            if not torch.isfinite(pred).all() or not torch.isfinite(loss):
                raise NumericalTrainingError(
                    f"non-finite AMP output/loss at epoch={epoch} batch={batch_idx}",
                    {
                        "batch_idx": batch_idx,
                        "gt": gt[:2].detach().half().cpu(),
                        "measurement": y[:2].detach().half().cpu(),
                        "pred": pred[:2].detach().half().cpu(),
                    },
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            if not torch.isfinite(grad_norm):
                raise NumericalTrainingError(
                    f"non-finite AMP gradient at epoch={epoch} batch={batch_idx}",
                    {"batch_idx": batch_idx},
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(y, mask_b, shifted_mask, ppt)
            pred = final_output(outputs)
            if not torch.isfinite(pred).all():
                raise NumericalTrainingError(
                    f"non-finite model output at epoch={epoch} batch={batch_idx}",
                    {
                        "batch_idx": batch_idx,
                        "gt": gt[:2].detach().half().cpu(),
                        "measurement": y[:2].detach().half().cpu(),
                        "pred": pred[:2].detach().half().cpu(),
                    },
                )
            loss = progressive_loss(outputs, gt)
            if not torch.isfinite(loss):
                raise NumericalTrainingError(
                    f"non-finite loss at epoch={epoch} batch={batch_idx}",
                    {
                        "batch_idx": batch_idx,
                        "gt": gt[:2].detach().half().cpu(),
                        "measurement": y[:2].detach().half().cpu(),
                        "pred": pred[:2].detach().half().cpu(),
                    },
                )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            if not torch.isfinite(grad_norm):
                bad_parameters = [
                    name for name, parameter in model.named_parameters()
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                ]
                raise NumericalTrainingError(
                    f"non-finite gradient at epoch={epoch} batch={batch_idx}",
                    {
                        "batch_idx": batch_idx,
                        "bad_parameters": bad_parameters[:50],
                        "gt": gt[:2].detach().half().cpu(),
                        "measurement": y[:2].detach().half().cpu(),
                    },
                )
            optimizer.step()

        total += loss.item()
        grad_total += float(grad_norm.detach().cpu())
        interval = int(cfg.get("log_interval", 0))
        if interval and (batch_idx + 1) % interval == 0:
            elapsed = time.time() - start
            rate = elapsed / (batch_idx + 1)
            eta = rate * (batch_num - batch_idx - 1)
            print(
                f"  [E{epoch:03d}] {batch_idx+1}/{batch_num} "
                f"loss={total/(batch_idx+1):.5f} {rate:.3f}s/it "
                f"ETA={eta/60:.1f}min",
                flush=True,
            )
            if save_dir is not None:
                atomic_write_json(Path(save_dir) / "progress.json", {
                    "epoch": epoch, "batch": batch_idx + 1, "batches": batch_num,
                    "loss": total / (batch_idx + 1), "seconds_per_step": rate,
                    "eta_minutes": eta / 60.0, "updated_at": time.time(),
                })

    elapsed = time.time() - start
    avg = total / batch_num
    peak_memory_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    samples_per_s = (batch_num * batch_size) / max(elapsed, 1e-6)
    print(
        f"[Epoch {epoch:03d}] Loss: {avg:.6f} Time: {elapsed:.1f}s "
        f"LR: {optimizer.param_groups[0]['lr']:.2e}",
        flush=True,
    )
    return {
        "train_loss": avg,
        "grad_norm": grad_total / batch_num,
        "epoch_time_s": elapsed,
        "samples_per_s": samples_per_s,
        "peak_memory_gb": peak_memory_gb,
    }


@torch.no_grad()
def test_epoch(model, test_data, mask, device):
    model.eval()
    psnrs, ssims, sams = [], [], []
    step_psnrs, step_ssims, step_sams = None, None, None
    test_bs = int(cfg.get("test_batch_size", 1))
    start = time.time()
    for start_idx in range(0, len(test_data), test_bs):
        gt = test_data[start_idx:start_idx + test_bs].to(device).float()
        mask_b = expand_mask(mask, gt.shape[0], device)
        shifted_mask = shift_cube(mask_b)
        ppt = phi_phi_t(mask_b)
        y = cassi_measure(gt, mask_b)
        outputs = model(y, mask_b, shifted_mask, ppt)
        if isinstance(outputs, (list, tuple)) and cfg.get("test_all_outputs", True):
            if step_psnrs is None:
                step_psnrs = [[] for _ in outputs]
                step_ssims = [[] for _ in outputs]
                step_sams = [[] for _ in outputs]
            for step_idx, output in enumerate(outputs):
                for i in range(output.shape[0]):
                    step_psnrs[step_idx].append(torch_psnr(output[i], gt[i]).item())
                    step_ssims[step_idx].append(torch_ssim(output[i], gt[i]).item())
                    step_sams[step_idx].append(torch_sam(output[i], gt[i]).item())
        pred = final_output(outputs)
        for i in range(pred.shape[0]):
            psnrs.append(torch_psnr(pred[i], gt[i]).item())
            ssims.append(torch_ssim(pred[i], gt[i]).item())
            sams.append(torch_sam(pred[i], gt[i]).item())
    means = tuple(sum(values) / len(values) for values in (psnrs, ssims, sams))
    print(
        f"         Test -> PSNR: {means[0]:.4f} SSIM: {means[1]:.6f} "
        f"SAM: {means[2]:.6f} Time: {time.time()-start:.1f}s",
        flush=True,
    )
    if step_psnrs is not None:
        step_summary = []
        for step_idx in range(len(step_psnrs)):
            step_summary.append(
                f"s{step_idx+1}: {sum(step_psnrs[step_idx])/len(step_psnrs[step_idx]):.3f}/"
                f"{sum(step_sams[step_idx])/len(step_sams[step_idx]):.4f}"
            )
        print("         Steps -> " + "  ".join(step_summary) + "  (PSNR/SAM)", flush=True)
    per_scene = [
        {"scene": idx + 1, "psnr": psnrs[idx], "ssim": ssims[idx], "sam": sams[idx]}
        for idx in range(len(psnrs))
    ]
    return means, per_scene


def run_diagnostic_probe(epoch, model, test_data, mask, device, save_dir, save_arrays=False):
    model.eval()
    gt = test_data[:1].to(device).float()
    mask_b = expand_mask(mask, 1, device)
    shifted_mask = shift_cube(mask_b)
    ppt = phi_phi_t(mask_b)
    y = cassi_measure(gt, mask_b)
    pred, stats = probe_forward(model, y, mask_b, shifted_mask, ppt)
    stats.update({"epoch": epoch, "scene": 1})
    diag_dir = save_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(diag_dir / f"probe_epoch_{epoch:03d}.json", stats)
    if save_arrays:
        np.savez_compressed(
            str(diag_dir / f"probe_epoch_{epoch:03d}.npz"),
            pred=pred.detach().half().cpu().numpy(),
            gt=gt.detach().half().cpu().numpy(),
            residual=(pred - gt).detach().half().cpu().numpy(),
        )


def main():
    seed_everything(int(cfg.get("seed", 2026)))
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    print(f"starting at: {time.ctime()}", flush=True)
    train_set = load_training(cfg["train_path"], max_scenes=205)
    test_data = load_test(cfg["test_path"], nC=cfg["num_bands"])
    mask = load_mask(cfg["mask_path"], nC=cfg["num_bands"])

    model = build_model().to(device)
    print_config(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]), betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(cfg["max_epoch"]), eta_min=1e-6
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.get("use_amp", False))

    run_stamp = time.strftime("%Y_%m_%d_%H_%M_%S")
    save_dir = ROOT / "result" / "model" / f"{run_stamp}_{cfg['experiment_name']}"
    save_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(CONFIG_PATH), str(save_dir / "config.yaml"))
    with open(save_dir / "run_meta.json", "w", encoding="utf-8") as fp:
        json.dump({
            "experiment": cfg["experiment_name"],
            "config_path": str(CONFIG_PATH),
            "torch": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0),
            "parameters_m": count_params(model),
            "resolved_config": cfg,
            "source": source_manifest(ROOT, [
                "train.py", "diagnostics.py", "model/e2e.py",
                "model/wpo3d.py", "model/candidates.py", "dataset.py", "loss.py",
            ]),
        }, fp, indent=2, ensure_ascii=False)

    metric_fields = [
        "epoch", "train_loss", "psnr", "ssim", "sam", "lr", "grad_norm",
        "epoch_time_s", "samples_per_s", "peak_memory_gb",
    ]
    scene_fields = ["epoch", "scene", "psnr", "ssim", "sam"]
    probe_epochs = {int(value) for value in cfg.get(
        "probe_epochs", [0, 1, 5, 10, 20, 40, 60, 100, 150, 200, 250, 300]
    )}
    probe_array_epochs = {int(value) for value in cfg.get(
        "probe_array_epochs", [0, 20, 60, 120, 200, 300]
    )}
    atomic_write_json(save_dir / "status.json", {
        "status": "initialized", "epoch": 0, "updated_at": time.time(),
    })
    start_epoch, best_psnr, best_sam = 1, 0.0, float("inf")
    resume = str(cfg.get("resume", "")).strip()
    if resume:
        state = torch.load(resume, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        best_psnr = float(state.get("best_psnr", 0.0))
        best_sam = float(state.get("best_sam", float("inf")))
        print(f"Resumed from {resume}, next epoch={start_epoch}", flush=True)

    if 0 in probe_epochs and start_epoch == 1:
        run_diagnostic_probe(
            0, model, test_data, mask, device, save_dir, 0 in probe_array_epochs
        )

    print(f"Save dir: {save_dir}", flush=True)
    for epoch in range(start_epoch, int(cfg["max_epoch"]) + 1):
        try:
            train_metrics = train_epoch(
                epoch, model, optimizer, scaler, train_set, mask, device, save_dir
            )
        except FloatingPointError as exc:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng_state": rng_state(),
                "error": str(exc),
                "snapshot": getattr(exc, "snapshot", {}),
                "model_stats": collect_model_stats(model),
                "config": cfg,
            }, save_dir / "nan_abort.pth")
            atomic_write_json(save_dir / "status.json", {
                "status": "failed", "epoch": epoch, "error": str(exc),
                "updated_at": time.time(),
            })
            print(f"[ABORT] {exc}", flush=True)
            raise

        (psnr, ssim, sam), per_scene = test_epoch(model, test_data, mask, device)
        print_physics_stats(model)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        metrics = {
            **train_metrics, "psnr": psnr, "ssim": ssim, "sam": sam, "lr": lr,
        }
        append_csv(
            save_dir / "metrics.csv", {"epoch": epoch, **metrics}, metric_fields
        )
        for scene_row in per_scene:
            append_csv(
                save_dir / "per_scene.csv", {"epoch": epoch, **scene_row}, scene_fields
            )
        if epoch in probe_epochs:
            run_diagnostic_probe(
                epoch, model, test_data, mask, device, save_dir,
                epoch in probe_array_epochs,
            )
        is_best_psnr = psnr > best_psnr
        is_best_sam = sam < best_sam
        if is_best_psnr:
            best_psnr = psnr
            print(f"  * New best: PSNR={psnr:.4f} SSIM={ssim:.6f} SAM={sam:.6f}", flush=True)
        if is_best_sam:
            best_sam = sam
            print(f"  * New best SAM: {sam:.6f} (PSNR={psnr:.4f})", flush=True)
        save_checkpoint(
            save_dir / "last.pth", epoch, model, optimizer, scheduler,
            best_psnr, best_sam, metrics,
        )
        if is_best_psnr and psnr >= float(cfg["save_thresh"]):
            save_checkpoint(
                save_dir / "best_psnr.pth", epoch, model, optimizer, scheduler,
                best_psnr, best_sam, metrics,
            )
            save_checkpoint(
                save_dir / "best.pth", epoch, model, optimizer, scheduler,
                best_psnr, best_sam, metrics,
            )
        if is_best_sam:
            save_checkpoint(
                save_dir / "best_sam.pth", epoch, model, optimizer, scheduler,
                best_psnr, best_sam, metrics,
            )
        if epoch % int(cfg.get("save_every", 25)) == 0:
            save_checkpoint(
                save_dir / f"epoch_{epoch:03d}.pth", epoch, model, optimizer, scheduler,
                best_psnr, best_sam, metrics,
            )
        atomic_write_json(save_dir / "status.json", {
            "status": "running", "epoch": epoch, "metrics": metrics,
            "best_psnr": best_psnr, "best_sam": best_sam,
            "updated_at": time.time(),
        })

    atomic_write_json(save_dir / "status.json", {
        "status": "finished", "epoch": int(cfg["max_epoch"]),
        "best_psnr": best_psnr, "best_sam": best_sam,
        "updated_at": time.time(),
    })
    print(
        f"finished at: {time.ctime()} best_psnr={best_psnr:.4f} best_sam={best_sam:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

