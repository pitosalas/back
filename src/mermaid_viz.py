#!/usr/bin/env python3
# mermaid_viz.py — Renders a CompGraph as a Mermaid flowchart HTML string
# Author: Pito Salas and Claude Code
# Open Source Under MIT license

from compgraph import CompGraph
from node import NodeType


TYPE_CLASS = {
    NodeType.INPUT: "input",
    NodeType.WEIGHT: "weight",
    NodeType.MULTIPLY: "op",
    NodeType.ADD: "op",
    NodeType.LOSS: "loss",
}

CLASS_DEFS = [
    "classDef input fill:#4682B4,stroke:#336699,color:white,font-size:15px",
    "classDef weight fill:#3CB371,stroke:#2d8a5e,color:white,font-size:15px",
    "classDef op fill:#e0e0e0,stroke:#999,color:#222,font-size:15px",
    "classDef loss fill:#FF6347,stroke:#cc4f3c,color:white,font-size:15px",
    "classDef hl stroke:#FFD700,stroke-width:5px",
]


def build_mermaid(g: CompGraph, show_gradients: bool, highlighted: str | None) -> str:
    lines = ["graph LR"]
    for defn in CLASS_DEFS:
        lines.append(f"    {defn}")
    for node_id in g.graph.nodes:
        attrs = g.graph.nodes[node_id]
        label = f"{node_id}<br/>{attrs['value']:.3f}"
        if show_gradients:
            label += f"<br/>grad: {attrs['gradient']:.3f}"
        cls = TYPE_CLASS[attrs["node_type"]]
        lines.append(f'    {node_id}["{label}"]:::{cls}')
    for from_id, to_id in g.graph.edges:
        lines.append(f"    {from_id} --> {to_id}")
    if highlighted:
        lines.append(f"    class {highlighted} hl")
    return "\n".join(lines)


def mermaid_html(g: CompGraph, show_gradients: bool, highlighted: str | None) -> str:
    diagram = build_mermaid(g, show_gradients, highlighted)
    import html as _html
    page = f"""<!DOCTYPE html>
<html>
<head>
  <script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
</head>
<body style="margin:0;background:transparent">
  <div class="mermaid">{diagram}</div>
  <script>
    mermaid.initialize({{startOnLoad:true}});
    window.addEventListener('load', function() {{
      parent.postMessage({{iframeHeight: document.body.scrollHeight + 20}}, '*');
    }});
  </script>
</body>
</html>"""
    escaped = _html.escape(page, quote=True)
    return (
        f'<iframe id="mermaid-frame" srcdoc="{escaped}" '
        f'style="width:100%;height:500px;border:none;" '
        f'onload="this.style.height=(this.contentWindow.document.body.scrollHeight+20)+\'px\'"></iframe>'
    )
