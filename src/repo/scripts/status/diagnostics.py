"""Low-overhead structured diagnostics for E2E training."""

import csv
import hashlib
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch


def atomic_write_json(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
    tmp.replace(path)


def append_csv(path, row, fieldnames):
    path = Path(path)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest(root, paths):
    root = Path(root)
    result = {}
    for rel in paths:
        path = root / rel
        if path.exists():
            result[str(rel).replace("\\", "/")] = sha256_file(path)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(root), stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        commit = None
    return {"git_commit": commit, "sha256": result}


def rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _tensor_stats(value):
    value = value.detach().float().reshape(-1).cpu()
    return {
        "count": int(value.numel()),
        "mean": float(value.mean()),
        "std": float(value.std(unbiased=False)),
        "min": float(value.min()),
        "max": float(value.max()),
    }


def _symmetry_error(value):
    value = value.detach().float().reshape(-1)
    if value.numel() <= 1:
        return 0.0
    indices = (-torch.arange(value.numel(), device=value.device)) % value.numel()
    return float(((value - value[indices]).abs().mean() / (value.abs().mean() + 1e-8)).cpu())


def final_output(output):
    if isinstance(output, (list, tuple)):
        return output[-1]
    return output


def collect_model_stats(model):
    stats = {"wpo": {}, "candidate": {}}
    for name, module in model.named_modules():
        if module.__class__.__name__ == "WPO3D":
            alpha, vs, vl, t = module._get_effective_params()
            stats["wpo"][name] = {
                "wave_param_mode": getattr(module, "wave_param_mode", "free"),
                "parameter_scope": "frequency_mode" if alpha.numel() > 1 else "layer_scalar",
                "alpha": _tensor_stats(alpha),
                "vs": _tensor_stats(vs),
                "alpha_symmetry_error": _symmetry_error(alpha),
                "vs_symmetry_error": _symmetry_error(vs),
                "vl": float(vl.detach().float().cpu()),
                "t": float(t.detach().float().cpu()),
            }
        diagnostic_fn = getattr(module, "diagnostic_stats", None)
        if callable(diagnostic_fn):
            values = diagnostic_fn()
            if values:
                stats["candidate"][name] = values
    return stats


@torch.no_grad()
def probe_forward(model, y, mask, shifted_mask, ppt):
    """One fixed forward with temporary WPO hooks; no persistent log spam."""
    activations = {}
    handles = []

    def make_hook(name):
        def hook(_module, inputs, output):
            x = inputs[0].detach().float()
            z = output.detach().float()
            x_rms = x.square().mean().sqrt()
            z_rms = z.square().mean().sqrt()
            activations[name] = {
                "input_rms": float(x_rms.cpu()),
                "output_rms": float(z_rms.cpu()),
                "output_input_ratio": float((z_rms / (x_rms + 1e-8)).cpu()),
                "output_absmax": float(z.abs().max().cpu()),
            }
        return hook

    for name, module in model.named_modules():
        if module.__class__.__name__ == "WPO3D":
            handles.append(module.register_forward_hook(make_hook(name)))
    try:
        pred = final_output(model(y, mask, shifted_mask, ppt))
    finally:
        for handle in handles:
            handle.remove()
    return pred, {"activations": activations, **collect_model_stats(model)}

