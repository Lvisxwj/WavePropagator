"""
render_mermaid.py — 把 components/*.onnx.json 转换成 Mermaid flowchart。

用法
----
    python render_mermaid.py                 # 渲染所有组件到 mermaid_out/
    python render_mermaid.py swap            # 只渲染 swap.onnx.json
    python render_mermaid.py swap lde        # 渲染多个

每个 .json 输出一个 .mmd 文件，可直接粘贴到 https://mermaid.live 或
任何支持 Mermaid 的 Markdown 渲染器（VSCode / Typora / Obsidian）查看。
颜色、跨 Part 虚线、公式编号锚点都保留。
"""

from __future__ import annotations
import json, sys, re
from pathlib import Path


HERE = Path(__file__).parent
COMP = HERE / "components"
OUT  = HERE / "mermaid_out"

PART_BG = {"Part I": "#f3f2f7", "Part II": "#fff7e6", "Part III": "#e6f1ff"}


def _safe(name: str) -> str:
    """Mermaid node id 不能有空格/特殊字符。"""
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def _label(text: str) -> str:
    """Escape mermaid label content."""
    return text.replace('"', "&quot;").replace("|", "/").replace("\n", "<br/>")


def _emit_node(lines, nid, label, color):
    lines.append(f'    {_safe(nid)}["{_label(label)}"]')
    if color:
        lines.append(f"    style {_safe(nid)} fill:{color},stroke:#1a1a1a,color:#1a1a1a")


def _emit_io(lines, name, kind):
    """IO 节点：圆角矩形。"""
    nid = f"{kind}_{_safe(name)}"
    lines.append(f'    {nid}(("{_label(name)}"))')
    return nid


def _render_graph(data: dict, nodes_key: str = "nodes") -> str:
    """单图渲染（nodes_key 通常是 'nodes'，stage_template 时为另一个值）."""
    lines = ["flowchart TD"]
    title = data.get("name", "graph")
    part  = data.get("part", "")
    bg    = data.get("background_color") or PART_BG.get(part, "#ffffff")
    lines.append(f'    %% {title} ({part})')
    lines.append(f"    classDef partBox fill:{bg},stroke:#3a155c,stroke-width:2px")

    # subgraph 包裹
    sg_id = _safe(title)
    lines.append(f'    subgraph {sg_id}["{title} — {part}"]')
    lines.append("    direction TB")

    # inputs
    input_ids = {}
    for inp in data.get("inputs", []):
        name = inp["name"]
        shape = "×".join(map(str, inp.get("shape", [])))
        role = inp.get("role", "")
        label = f"{name}<br/><span style='font-size:10px'>{shape}</span>"
        if role:
            label += f"<br/><i>{role}</i>"
        nid = _emit_io(lines, name, "in")
        input_ids[name] = nid
        lines.append(f"    style {nid} fill:#ffffff,stroke:#7a4dba,stroke-width:1.5px")

    # outputs
    output_ids = {}
    for out in data.get("outputs", []):
        name = out["name"]
        shape = "×".join(map(str, out.get("shape", [])))
        label = f"{name}<br/><span style='font-size:10px'>{shape}</span>"
        nid = _emit_io(lines, name, "out")
        output_ids[name] = nid
        lines.append(f"    style {nid} fill:#ffffff,stroke:#3a155c,stroke-width:1.5px")

    # nodes
    name_to_id = {**input_ids, **output_ids}
    for n in data.get(nodes_key, []):
        nid = n["id"]
        op  = n.get("op_type", "?")
        label = f"<b>{op}</b><br/>{nid}"
        cond = n.get("condition")
        if cond:
            label += f"<br/><i>if {cond}</i>"
        _emit_node(lines, nid, label, n.get("color"))
        for o in n.get("output", []):
            name_to_id.setdefault(o, _safe(nid))

    # edges (data flow)
    for n in data.get(nodes_key, []):
        for src in n.get("input", []):
            if src in name_to_id:
                lines.append(f"    {name_to_id[src]} --> {_safe(n['id'])}")
        for dst in n.get("output", []):
            if dst in output_ids:
                lines.append(f"    {_safe(n['id'])} --> {output_ids[dst]}")

    lines.append("    end")
    lines.append(f"    class {sg_id} partBox")

    # cross-part edges (虚线)
    for e in data.get("edges_cross_part", []):
        src = _safe(e["from"].replace(".", "_"))
        dst = _safe(e["to"].replace(".", "_"))
        style = e.get("style", "solid")
        meaning = e.get("meaning", "")
        arrow = "-.->" if style == "dashed" else "==>"
        lines.append(f'    {src} {arrow}|"{_label(meaning)}"| {dst}')

    # formula footer
    refs = data.get("formula_ref", [])
    if refs:
        lines.append(f"    %% formulas: {'; '.join(refs)}")
    return "\n".join(lines)


def _render_variants(data: dict) -> str:
    """swap-AdaSpec 之类有 variants 字段的组件。"""
    out = [f'flowchart TD', f'    %% {data["name"]} variants']
    for v in data.get("variants", []):
        sub_data = {**data, **v, "name": f"{data['name']} :: {v['name']}"}
        out.append("")
        out.append(_render_graph(sub_data))
    return "\n".join(out)


def _render_stage_template(data: dict) -> str:
    """ahqs.onnx.json 用 stage_template 表示单 stage 内部图。"""
    tpl = data["stage_template"]
    sub_data = {
        **data,
        "inputs":  [{"name": n} for n in tpl.get("inputs", [])],
        "outputs": [{"name": n} for n in tpl.get("outputs", [])],
        "nodes":   tpl.get("nodes", []),
        "name":    f"{data['name']} (single stage)",
    }
    return _render_graph(sub_data)


def render_file(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "variants" in data:
        return _render_variants(data)
    if "stage_template" in data:
        return _render_stage_template(data)
    return _render_graph(data)


def main(argv):
    OUT.mkdir(exist_ok=True)
    targets = argv[1:] if len(argv) > 1 else [p.stem.replace(".onnx", "") for p in COMP.glob("*.onnx.json")]
    rendered = 0
    for stem in targets:
        f = COMP / f"{stem}.onnx.json"
        if not f.exists():
            print(f"[skip] {f.name} not found")
            continue
        mmd = render_file(f)
        out_path = OUT / f"{stem}.mmd"
        out_path.write_text(mmd, encoding="utf-8")
        print(f"[ok]   {f.name} → {out_path.relative_to(HERE)}")
        rendered += 1
    print(f"\nDone. {rendered} file(s) written to {OUT.relative_to(HERE)}/")
    print("Tips:")
    print("  - Paste *.mmd content into https://mermaid.live to preview.")
    print("  - VSCode users: install the 'Mermaid Preview' extension.")


if __name__ == "__main__":
    main(sys.argv)
