#!/usr/bin/env python3
"""Generate runtime YAML configs with server-local paths/GPU/batch settings."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_batch(value: str) -> dict[str, int]:
    result = {}
    if not value:
        return result
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        name, batch = item.split("=", 1)
        result[name.strip()] = int(batch)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--test-path", required=True)
    parser.add_argument("--mask-path", required=True)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument(
        "--templates",
        default="smile_s.yaml,smile_m.yaml,smile_l.yaml",
    )
    parser.add_argument(
        "--batches",
        default="",
        help="optional per-experiment override, e.g. smile_s=16,smile_m=12",
    )
    parser.add_argument("--max-epoch", type=int, default=None)
    parser.add_argument("--epoch-sample", type=int, default=None)
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    out_dir = args.out_dir or (project_dir / "configs" / "generated")
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_overrides = parse_batch(args.batches)

    written = []
    for template_name in [x.strip() for x in args.templates.split(",") if x.strip()]:
        src = project_dir / "configs" / template_name
        cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
        cfg["gpu_id"] = str(args.gpu_id)
        cfg["train_path"] = str(Path(args.train_path).resolve())
        cfg["test_path"] = str(Path(args.test_path).resolve())
        cfg["mask_path"] = str(Path(args.mask_path).resolve())
        if cfg.get("experiment_name") in batch_overrides:
            cfg["batch_size"] = int(batch_overrides[cfg["experiment_name"]])
        if args.max_epoch is not None:
            cfg["max_epoch"] = int(args.max_epoch)
        if args.epoch_sample is not None:
            cfg["epoch_sample"] = int(args.epoch_sample)
        dst = out_dir / template_name
        dst.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        written.append(dst)
        print(f"[config] {src.name} -> {dst}", flush=True)

    print("[done] runtime configs:")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()


