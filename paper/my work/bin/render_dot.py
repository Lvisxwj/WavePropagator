"""
render_dot.py — 把 components/*.onnx.json 转换成 Graphviz .dot 文件。

用法
----
    python render_dot.py                  # 渲染全部到 dot_out/
    python render_dot.py swap lde         # 渲染部分

依赖
----
    pip install graphviz                  # Python 包（可选，仅用于直接生成 SVG）
    然后系统需安装 Graphviz CLI（含 `dot` 命令）：
        Windows  : choco install graphviz / winget install graphviz / 官网 msi
        macOS    : brew install graphviz
        Ubuntu   : sudo apt-get install graphviz

若不安装 Graphviz CLI，本脚本仍会输出 .dot 文本，可直接拖入 draw.io:
    File → Open from → Device，选择 .dot 文件；
    或粘贴到 https://dreampuf.github.io/GraphvizOnline 在线渲染。

每个 .json 输出一个 .dot；若 graphviz CLI 可用，额外输出同名 .svg。
"""

from __future__ import annotations
import json, sys, re, shutil, subprocess
from pathlib import Path


HERE = Path(__file__).parent
COMP = HERE / "components"
OUT  = HERE / "dot_out"

PART_BG = {"Part I": "#f3f2f7", "Part II": "#fff7e6", "Part III": "#e6f1ff"}

_DOT_HEADER = """digraph "{title}" {{
    graph [rankdir=TB, splines=spline, fontname="Helvetica", labelloc="t",
           label=<<B>{title}</B><BR/><FONT POINT-SIZE="10">{part} — formulas: {formulas}</FONT>>,
           bgcolor="{bg}"];
    node  [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#1a1a1a"];
    edge  [fontname="Helvetica", fontsize=9, color="#1a1a1a"];
"""


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def _node_line(nid, label, fill, shape="box"):
    fill = fill or "#ffffff"
    return (f'    {_safe(nid)} [label=<{label}>, fillcolor="{fill}", shape={shape}];')


def _label(op_type, sub_id=None, cond=None):
    head = f"<B>{op_type}</B>"
    sub = ""
    if sub_id and sub_id != op_type:
        sub += f'<BR/><FONT POINT-SIZE="9">{sub_id}</FONT>'
    if cond:
        sub += f'<BR/><FONT POINT-SIZE="8" COLOR="#9a9aa3"><I>if {cond}</I></FONT>'
    return head + sub


def _io_label(name, shape, role=None):
    shape_s = "×".join(map(str, shape)) if shape else ""
    role_s = (f'<BR/><FONT POINT-SIZE="8" COLOR="#7a4dba"><I>{role}</I></FONT>') if role else ""
    return f"<B>{name}</B><BR/><FONT POINT-SIZE='9'>{shape_s}</FONT>{role_s}"


def _render_graph(data: dict, nodes_key: str = "nodes") -> str:
    title = data.get("name", "graph")
    part  = data.get("part", "")
    bg    = data.get("background_color") or PART_BG.get(part, "#ffffff")
    formulas = "; ".join(data.get("formula_ref", []))

    lines = [_DOT_HEADER.format(title=title, part=part, bg=bg, formulas=formulas)]

    # input cluster
    if data.get("inputs"):
        lines.append('    subgraph cluster_inputs { label="Inputs"; rank=same; color="#7a4dba"; style="dashed";')
        for inp in data["inputs"]:
            nid = f"in_{inp['name']}"
            label = _io_label(inp["name"], inp.get("shape", []), inp.get("role"))
            lines.append(_node_line(nid, label, "#ffffff", shape="ellipse"))
        lines.append("    }")

    # nodes
    for n in data.get(nodes_key, []):
        label = _label(n.get("op_type", "?"), n["id"], n.get("condition"))
        lines.append(_node_line(n["id"], label, n.get("color")))

    # output cluster
    if data.get("outputs"):
        lines.append('    subgraph cluster_outputs { label="Outputs"; rank=same; color="#3a155c"; style="dashed";')
        for out in data["outputs"]:
            nid = f"out_{out['name']}"
            label = _io_label(out["name"], out.get("shape", []))
            lines.append(_node_line(nid, label, "#ffffff", shape="ellipse"))
        lines.append("    }")

    # edges
    name_to_id = {f"in_{inp['name']}": f"in_{inp['name']}" for inp in data.get("inputs", [])}
    name_to_id.update({out["name"]: f"out_{out['name']}" for out in data.get("outputs", [])})

    # tensor → node mapping
    tensor_src = {}
    for inp in data.get("inputs", []):
        tensor_src[inp["name"]] = f"in_{inp['name']}"
    for n in data.get(nodes_key, []):
        for o in n.get("output", []):
            tensor_src[o] = _safe(n["id"])

    for n in data.get(nodes_key, []):
        for src in n.get("input", []):
            if src in tensor_src:
                lines.append(f'    {tensor_src[src]} -> {_safe(n["id"])} [label="{src}"];')
        for dst in n.get("output", []):
            if dst in {o["name"] for o in data.get("outputs", [])}:
                lines.append(f'    {_safe(n["id"])} -> out_{_safe(dst)} [label="{dst}"];')

    # cross-part edges
    for e in data.get("edges_cross_part", []):
        src = _safe(e["from"].replace(".", "_"))
        dst = _safe(e["to"].replace(".", "_"))
        style = "dashed" if e.get("style") == "dashed" else "solid"
        color = e.get("color", "#a569bd")
        meaning = e.get("meaning", "")
        lines.append(
            f'    {src} -> {dst} [style={style}, color="{color}", '
            f'label=<<FONT POINT-SIZE="9"><I>{meaning}</I></FONT>>];'
        )

    lines.append("}")
    return "\n".join(lines)


def _render_variants(data: dict) -> str:
    parts = []
    for v in data.get("variants", []):
        sub = {**data, **v, "name": f"{data['name']} :: {v['name']}"}
        parts.append(_render_graph(sub))
    return "\n\n".join(parts)


def _render_stage_template(data: dict) -> str:
    tpl = data["stage_template"]
    sub = {
        **data,
        "inputs":  [{"name": n} for n in tpl.get("inputs", [])],
        "outputs": [{"name": n} for n in tpl.get("outputs", [])],
        "nodes":   tpl.get("nodes", []),
        "name":    f"{data['name']} (single stage)",
    }
    return _render_graph(sub)


def render_file(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "variants" in data:
        return _render_variants(data)
    if "stage_template" in data:
        return _render_stage_template(data)
    return _render_graph(data)


def _maybe_compile_svg(dot_path: Path):
    """If `dot` CLI is on PATH, also emit SVG next to .dot."""
    if not shutil.which("dot"):
        return False
    svg_path = dot_path.with_suffix(".svg")
    try:
        subprocess.run(
            ["dot", "-Tsvg", str(dot_path), "-o", str(svg_path)],
            check=True, capture_output=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"    [dot] failed for {dot_path.name}: {e.stderr.decode(errors='ignore')[:200]}")
        return False


def main(argv):
    OUT.mkdir(exist_ok=True)
    targets = argv[1:] if len(argv) > 1 else [p.stem.replace(".onnx", "") for p in COMP.glob("*.onnx.json")]
    has_dot = bool(shutil.which("dot"))
    if not has_dot:
        print("[info] Graphviz `dot` not found on PATH — .dot files will be emitted, SVG skipped.")
        print("       Install: https://graphviz.org/download/  or paste .dot text into draw.io/dreampuf.github.io/GraphvizOnline")
    rendered = 0
    for stem in targets:
        f = COMP / f"{stem}.onnx.json"
        if not f.exists():
            print(f"[skip] {f.name} not found")
            continue
        dot = render_file(f)
        out_dot = OUT / f"{stem}.dot"
        out_dot.write_text(dot, encoding="utf-8")
        svg_ok = _maybe_compile_svg(out_dot) if has_dot else False
        print(f"[ok]   {f.name} → {out_dot.relative_to(HERE)}" + (" + .svg" if svg_ok else ""))
        rendered += 1
    print(f"\nDone. {rendered} file(s) written to {OUT.relative_to(HERE)}/")


if __name__ == "__main__":
    main(sys.argv)
