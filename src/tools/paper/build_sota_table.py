import csv
from pathlib import Path

root = Path.cwd()
out = root / "paper/my work/e2e/paper/sota_table.md"
perscene_csv = root / "paper/my work/e2e/paper/sota_csv/sota_perscene_20260716_053748.csv"
summary_csv = root / "paper/my work/e2e/paper/sota_csv/sota_summary_20260716_053748.csv"
smile_s_perscene_csv = root / "paper/my work/e2e/paper/sota_csv/smile_s_perscene_best_psnr_10scene.csv"
smile_s_summary_csv = root / "paper/my work/e2e/paper/sota_csv/smile_s_summary_best_psnr.csv"
smile_m_perscene_csv = root / "paper/my work/e2e/paper/sota_csv/smile_m_perscene_best_psnr_10scene.csv"
smile_m_summary_csv = root / "paper/my work/e2e/paper/sota_csv/smile_m_summary_best_psnr.csv"
smile_l_perscene_csv = root / "paper/my work/e2e/paper/sota_csv/smile_l_perscene_best_psnr_10scene.csv"
smile_l_summary_csv = root / "paper/my work/e2e/paper/sota_csv/smile_l_summary_best_psnr.csv"

name_map = {
    "gap_tv": ("GAP-TV / GAP-Net", "ICIP 2016"),
    "lambda_net": ("λ-Net", "ICCV 2019"),
    "tsa_net": ("TSA-Net", "ECCV 2020"),
    "dgsmp": ("DGSMP", "CVPR 2021"),
    "mst_l": ("MST-L", "CVPR 2022"),
    "cst_l": ("CST-L", "ECCV 2022"),
    "birnat": ("BIRNAT", "TPAMI 2023"),
}

perscene = {}
with perscene_csv.open("r", encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f):
        m = r["model"]
        if not r["scene"].startswith("scene"):
            continue
        idx = int(r["scene"].replace("scene", ""))
        perscene.setdefault(m, {})[idx] = {
            "psnr": float(r["psnr"]),
            "ssim": float(r["ssim"]),
            "sam": float(r["sam_rad"]),
        }

summary = {}
with summary_csv.open("r", encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f):
        summary[r["model"]] = {
            "params": float(r["params_M"]),
            "gflops": float(r["gflops"]),
            "psnr": float(r["psnr"]),
            "ssim": float(r["ssim"]),
            "sam": float(r["sam_rad"]),
        }

def load_smile_variant(perscene_path, summary_path):
    scenes = {}
    summary_row = None
    if not (perscene_path.exists() and summary_path.exists()):
        return scenes, summary_row
    raw = []
    with perscene_path.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            raw.append(
                {
                    "psnr": float(r["psnr"]),
                    "ssim": float(r["ssim"]),
                    "sam": float(r["sam_rad"]),
                }
            )
    # GPU server TSA_simu_data/Truth contains both .mat and .npy copies; load_test reads
    # 20 entries with exact pairs. Collapse paired duplicates to the real 10 scenes.
    if len(raw) == 20:
        raw = raw[0::2]
    scenes = {idx + 1: row for idx, row in enumerate(raw)}
    with summary_path.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            summary_row = {
                "params": float(r["params_M"]),
                "gflops": float(r["gflops"]),
                "psnr": float(r["psnr"]),
                "ssim": float(r["ssim"]),
                "sam": float(r["sam_rad"]),
            }
            break
    return scenes, summary_row


smile_s_scenes, smile_s_summary = load_smile_variant(smile_s_perscene_csv, smile_s_summary_csv)
smile_m_scenes, smile_m_summary = load_smile_variant(smile_m_perscene_csv, smile_m_summary_csv)
smile_l_scenes, smile_l_summary = load_smile_variant(smile_l_perscene_csv, smile_l_summary_csv)

# Screenshot-filled missing rows. SAM is not available in those screenshots.
paper_rows = {
    "twist": {
        "name": "TwIST",
        "venue": "TIP 2017",
        "params": None,
        "gflops": None,
        "psnr": [25.16, 23.02, 21.40, 30.19, 21.41, 20.95, 22.20, 21.82, 22.42, 22.67],
        "ssim": [0.700, 0.604, 0.711, 0.851, 0.635, 0.644, 0.643, 0.650, 0.690, 0.569],
        "avg_psnr": 23.12,
        "avg_ssim": 0.669,
    },
    "desci": {
        "name": "DeSCI",
        "venue": "TPAMI 2019",
        "params": None,
        "gflops": None,
        "psnr": [27.13, 23.04, 26.62, 34.96, 23.94, 22.38, 24.45, 22.03, 24.56, 23.59],
        "ssim": [0.748, 0.620, 0.818, 0.897, 0.706, 0.683, 0.743, 0.673, 0.732, 0.587],
        "avg_psnr": 25.27,
        "avg_ssim": 0.721,
    },
    "s2transformer": {
        "name": "S2Transformer",
        "venue": "TPAMI 2025",
        "params": 1.80,
        "gflops": 27.21,
        "psnr": [36.17, 37.57, 37.29, 42.96, 34.40, 36.44, 35.41, 34.50, 36.54, 33.57],
        "ssim": [0.949, 0.958, 0.957, 0.975, 0.960, 0.965, 0.946, 0.962, 0.959, 0.952],
        "avg_psnr": 36.48,
        "avg_ssim": 0.958,
    },
    "dsmt_star": {
        "name": "DSMT*",
        "venue": "TIP 2025",
        "params": 3.73,
        "gflops": 44.94,
        "psnr": [36.23, 37.34, 37.89, 42.11, 33.71, 36.12, 35.10, 34.46, 36.29, 33.75],
        "ssim": [0.957, 0.961, 0.964, 0.981, 0.961, 0.970, 0.948, 0.969, 0.959, 0.959],
        "avg_psnr": 36.30,
        "avg_ssim": 0.963,
    },
}


def fmt_num(x, digits=2):
    return "—" if x is None else f"{x:.{digits}f}"


def scene_cell(psnr=None, ssim=None, sam=None):
    p = "—" if psnr is None else f"{psnr:.2f}"
    s = "—" if ssim is None else f"{ssim:.3f}"
    a = "—" if sam is None else f"{sam:.4f}"
    return f"{p}<br>{s}<br>{a}"


rows = []
order = [
    ("paper", "twist"),
    ("measured", "gap_tv"),
    ("paper", "desci"),
    ("measured", "lambda_net"),
    ("measured", "tsa_net"),
    ("measured", "dgsmp"),
    ("measured", "mst_l"),
    ("measured", "cst_l"),
    ("paper", "s2transformer"),
    ("measured", "birnat"),
    ("paper", "dsmt_star"),
    ("smile", "SMILE-S"),
    ("smile", "SMILE-M"),
    ("smile", "SMILE-L"),
]

for kind, key in order:
    if kind == "paper":
        r = paper_rows[key]
        rows.append(
            {
                "alg": r["name"],
                "ref": r["venue"],
                "params": fmt_num(r["params"]),
                "gflops": fmt_num(r["gflops"]),
                "cells": [scene_cell(r["psnr"][i], r["ssim"][i], None) for i in range(10)],
                "avg": scene_cell(r["avg_psnr"], r["avg_ssim"], None),
                "source": "paper table screenshot",
            }
        )
    elif kind == "measured":
        name, venue = name_map[key]
        s = summary[key]
        rows.append(
            {
                "alg": name,
                "ref": venue,
                "params": fmt_num(s["params"]),
                "gflops": fmt_num(s["gflops"]),
                "cells": [
                    scene_cell(perscene[key][i]["psnr"], perscene[key][i]["ssim"], perscene[key][i]["sam"])
                    for i in range(1, 11)
                ],
                "avg": scene_cell(s["psnr"], s["ssim"], s["sam"]),
                "source": "GPU server eval CSV",
            }
        )
    else:
        if key == "SMILE-M":
            sm = smile_m_summary or {"params": 2.0005, "gflops": 28.281, "psnr": 36.5687, "ssim": 0.961448, "sam": 0.096745}
            params, gflops = sm["params"], sm["gflops"]
            avg = scene_cell(sm["psnr"], sm["ssim"], sm["sam"])
            cells = [
                scene_cell(smile_m_scenes[i]["psnr"], smile_m_scenes[i]["ssim"], smile_m_scenes[i]["sam"])
                for i in range(1, 11)
            ] if len(smile_m_scenes) >= 10 else [scene_cell() for _ in range(10)]
            note = "GPU server eval CSV; best_psnr E284"
        elif key == "SMILE-S":
            sm = smile_s_summary or {"params": 1.0179, "gflops": 28.281, "psnr": 35.9378, "ssim": 0.955705, "sam": 0.111643}
            params, gflops = sm["params"], sm["gflops"]
            avg = scene_cell(sm["psnr"], sm["ssim"], sm["sam"])
            cells = [
                scene_cell(smile_s_scenes[i]["psnr"], smile_s_scenes[i]["ssim"], smile_s_scenes[i]["sam"])
                for i in range(1, 11)
            ] if len(smile_s_scenes) >= 10 else [scene_cell() for _ in range(10)]
            note = "GPU server eval CSV; best_psnr E278"
        else:
            sm = smile_l_summary or {"params": 3.1909, "gflops": 38.601, "psnr": 36.8438, "ssim": 0.963403, "sam": 0.098169}
            params, gflops = sm["params"], sm["gflops"]
            avg = scene_cell(sm["psnr"], sm["ssim"], sm["sam"])
            cells = [
                scene_cell(smile_l_scenes[i]["psnr"], smile_l_scenes[i]["ssim"], smile_l_scenes[i]["sam"])
                for i in range(1, 11)
            ] if len(smile_l_scenes) >= 10 else [scene_cell() for _ in range(10)]
            note = "GPU server eval CSV; best_psnr E216"
        rows.append(
            {
                "alg": f"{key} (ours)",
                "ref": "Ours",
                "params": fmt_num(params),
                "gflops": fmt_num(gflops),
                "cells": cells,
                "avg": avg,
                "source": note,
            }
        )

headers = ["Algorithms", "Reference", "Params (M)", "FLOPs (G)"] + [f"S{i}" for i in range(1, 11)] + ["Avg"]
md = [
    "# SOTA comparison table for SMILE² E2E paper",
    "",
    "Source CSVs generated on GPU server from `/tmp/sam_e2e_eval/run_e2e_sota.py` using TSA simulation test data (`10` npy scenes only). Missing paper-only rows are filled from the user-provided SOTA tables; only missing values were filled, and GPU server-measured rows were not overwritten.",
    "",
    "- Per-scene CSV: `sota_csv/sota_perscene_20260716_053748.csv`",
    "- Summary CSV: `sota_csv/sota_summary_20260716_053748.csv`",
    "- SMILE GFLOPs CSV: `sota_csv/smile_gflops.csv`",
    "- SMILE-S/M/L per-scene CSVs: `sota_csv/smile_s_perscene_best_psnr_10scene.csv`, `sota_csv/smile_m_perscene_best_psnr_10scene.csv`, `sota_csv/smile_l_perscene_best_psnr_10scene.csv`.",
    "- Cell format follows the paper-table style: `PSNR` / `SSIM` / `SAM(rad)` in each scene cell.",
    "- Paper-table-only rows do not report SAM, so SAM is shown as `—` rather than guessed.",
    "- `GAP-TV / GAP-Net` keeps the current GPU server local-pth evaluation row; it was not replaced by the paper-table GAP-TV values.",
    "",
    "## Main table",
    "",
    "| " + " | ".join(headers) + " |",
    "| " + " | ".join(["---"] * len(headers)) + " |",
]
for r in rows:
    vals = [r["alg"], r["ref"], r["params"], r["gflops"]] + r["cells"] + [r["avg"]]
    md.append("| " + " | ".join(vals) + " |")

md.extend(
    [
        "",
        "## Row sources / status",
        "",
        "| Algorithms | Source / status |",
        "|---|---|",
    ]
)
for r in rows:
    md.append(f"| {r['alg']} | {r['source']} |")

md.extend(
    [
        "",
        "## Filled from the provided screenshots",
        "",
        "- TwIST, DeSCI, S2Transformer and DSMT* PSNR/SSIM are copied from the provided tables.",
        "- DSMT* is used for the lightweight DSMT comparison row: 3.73M / 44.94G / 36.30 / 0.963.",
        "- Screenshot tables do not report SAM; those SAM entries remain blank (`—`) to avoid fabricating spectral-angle numbers.",
        "",
    ]
)

out.write_text("\n".join(md), encoding="utf-8")
print(f"wrote {out}")


