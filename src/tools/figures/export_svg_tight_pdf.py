from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(r"C:\Users\xwj\Desktop\study\Machine Learning\cassi重构\src")
FIG_DIR = ROOT / "paper" / "my work" / "paper" / "figures" / "final"
CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
TMP_DIR = Path(r"C:\Users\xwj\.codex\visualizations\2026\06\27\019f083c-0ec0-7ba0-a5ac-191cc55bb1d1\smile_svg_export")


def svg_size(svg_path: Path) -> tuple[int, int]:
    text = svg_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"<svg[^>]*\bwidth=\"([0-9.]+)\"[^>]*\bheight=\"([0-9.]+)\"", text)
    if not m:
        raise RuntimeError(f"Cannot find width/height in {svg_path}")
    return int(float(m.group(1))), int(float(m.group(2)))


def export_one(stem: str) -> None:
    svg = FIG_DIR / f"{stem}.svg"
    pdf = FIG_DIR / f"{stem}_tight.pdf"
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    tmp_svg = TMP_DIR / f"{stem}.svg"
    tmp_pdf = TMP_DIR / f"{stem}_tight.pdf"
    html = TMP_DIR / f"{stem}_tight_wrapper.html"
    shutil.copy2(svg, tmp_svg)
    width, height = svg_size(svg)
    uri = tmp_svg.resolve().as_uri()
    html.write_text(
        f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
@page {{ size: {width}px {height}px; margin: 0; }}
html, body {{
  margin: 0;
  padding: 0;
  width: {width}px;
  height: {height}px;
  overflow: hidden;
  background: white;
}}
img {{
  display: block;
  width: {width}px;
  height: {height}px;
}}
</style>
</head>
<body><img src="{uri}"></body>
</html>
""",
        encoding="utf-8",
    )
    if tmp_pdf.exists():
        tmp_pdf.unlink()
    subprocess.run(
        [
            str(CHROME),
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            "--disable-dev-shm-usage",
            f"--print-to-pdf={tmp_pdf}",
            html.resolve().as_uri(),
        ],
        check=True,
    )
    shutil.copy2(tmp_pdf, pdf)
    print(f"{stem}: {width}x{height} -> {pdf}")


def main() -> None:
    if not CHROME.exists():
        raise SystemExit(f"Chrome not found: {CHROME}")
    for stem in ["0CASSI", "2swap", "3swp"]:
        export_one(stem)


if __name__ == "__main__":
    main()

