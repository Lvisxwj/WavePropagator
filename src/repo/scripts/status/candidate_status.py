"""Print compact status for the three architecture candidates."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NAMES = (
    "candidate_mask_init_244",
    "candidate_wavelength_rec_244",
    "candidate_ms_mode_swap_244",
)


for name in NAMES:
    matches = sorted((ROOT / "result" / "model").glob("*_%s" % name))
    if not matches:
        print("%-32s not started" % name)
        continue
    run = matches[-1]
    status_path = run / "status.json"
    if not status_path.exists():
        print("%-32s %s (no status.json)" % (name, run.name))
        continue
    status = json.loads(status_path.read_text(encoding="utf-8"))
    metrics = status.get("metrics", {})
    print(
        "%-32s status=%-11s E=%3s PSNR=%7s SAM=%8s best=%7s/%8s"
        % (
            name,
            status.get("status", "?"),
            status.get("epoch", "?"),
            ("%.4f" % metrics["psnr"]) if "psnr" in metrics else "-",
            ("%.6f" % metrics["sam"]) if "sam" in metrics else "-",
            ("%.4f" % status["best_psnr"]) if "best_psnr" in status else "-",
            ("%.6f" % status["best_sam"]) if "best_sam" in status else "-",
        )
    )

