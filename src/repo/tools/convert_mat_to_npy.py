#!/usr/bin/env python3
"""Convert CASSI/HSI .mat datasets to the .npy layout used by e2e.v1.

The trainer can read .mat directly, but pre-converting to .npy avoids repeated
SciPy MATLAB parsing and makes startup much faster on shared servers.

Supported inputs:
  - Training scenes: keys usually `img_expand` or `img`, shape [H,W,28].
    Values larger than 2 are assumed to be 16-bit-like and divided by 65536.
  - Test Truth scenes: key usually `img`; shape is preserved except HWC inputs
    remain HWC, which `dataset.load_test()` already handles.
  - Mask: key `mask`, saved as mask.npy if requested.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io as sio


SCENE_KEYS = ("img_expand", "img", "truth", "gt", "data")


def pick_array(mat: dict, keys: tuple[str, ...]) -> tuple[str, np.ndarray]:
    for key in keys:
        value = mat.get(key)
        if isinstance(value, np.ndarray) and value.ndim >= 2:
            return key, value
    visible = sorted(k for k in mat.keys() if not k.startswith("__"))
    raise KeyError(f"none of keys {keys} found; visible keys={visible}")


def normalize_scene(arr: np.ndarray, kind: str) -> np.ndarray:
    arr = np.asarray(arr)
    # Drop MATLAB singleton dimensions when safe.
    arr = np.squeeze(arr)
    arr = arr.astype(np.float32, copy=False)
    if kind == "train":
        if float(np.nanmax(arr)) > 2.0:
            arr = arr / 65536.0
    elif kind == "test":
        # Test truth files in many CASSI repos are already [0,1], but some
        # mirrors store uint16.  Use the same conservative rule.
        if float(np.nanmax(arr)) > 2.0:
            arr = arr / 65536.0
    return np.ascontiguousarray(arr.astype(np.float32, copy=False))


def convert_scenes(src: Path, dst: Path, kind: str, overwrite: bool) -> dict:
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.mat"), key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or 0))
    if not files:
        raise FileNotFoundError(f"no .mat files found under {src}")

    rows = []
    for mat_path in files:
        out_path = dst / f"{mat_path.stem}.npy"
        if out_path.exists() and not overwrite:
            rows.append({"file": mat_path.name, "out": out_path.name, "status": "skip"})
            continue
        mat = sio.loadmat(str(mat_path))
        key, arr = pick_array(mat, SCENE_KEYS)
        arr = normalize_scene(arr, kind)
        np.save(str(out_path), arr)
        rows.append({
            "file": mat_path.name,
            "out": out_path.name,
            "key": key,
            "shape": list(arr.shape),
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
            "status": "ok",
        })
        print(f"[{kind}] {mat_path.name} -> {out_path.name} key={key} shape={arr.shape}", flush=True)
    return {"src": str(src), "dst": str(dst), "kind": kind, "count": len(rows), "items": rows}


def convert_mask(src: Path, dst_dir: Path, overwrite: bool) -> dict:
    dst_dir.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        src = src / "mask.mat"
    out_path = dst_dir / "mask.npy"
    if out_path.exists() and not overwrite:
        return {"src": str(src), "out": str(out_path), "status": "skip"}
    mat = sio.loadmat(str(src))
    key, mask = pick_array(mat, ("mask", "Phi", "phi"))
    mask = np.squeeze(mask).astype(np.float32, copy=False)
    np.save(str(out_path), np.ascontiguousarray(mask))
    print(f"[mask] {src.name} -> {out_path} key={key} shape={mask.shape}", flush=True)
    return {"src": str(src), "out": str(out_path), "key": key, "shape": list(mask.shape), "status": "ok"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-mat", type=Path, help="directory containing CAVE training .mat scenes")
    parser.add_argument("--train-out", type=Path, help="output directory for training .npy scenes")
    parser.add_argument("--test-mat", type=Path, help="directory containing TSA Truth .mat scenes")
    parser.add_argument("--test-out", type=Path, help="output directory for test .npy scenes")
    parser.add_argument("--mask-mat", type=Path, help="mask.mat file or directory containing mask.mat")
    parser.add_argument("--mask-out-dir", type=Path, help="directory to write mask.npy")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    manifest = {}
    if args.train_mat and args.train_out:
        manifest["train"] = convert_scenes(args.train_mat, args.train_out, "train", args.overwrite)
    if args.test_mat and args.test_out:
        manifest["test"] = convert_scenes(args.test_mat, args.test_out, "test", args.overwrite)
    if args.mask_mat and args.mask_out_dir:
        manifest["mask"] = convert_mask(args.mask_mat, args.mask_out_dir, args.overwrite)

    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[manifest] {args.manifest}", flush=True)


if __name__ == "__main__":
    main()


