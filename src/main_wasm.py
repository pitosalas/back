#!/usr/bin/env python3
# main_wasm.py — Standalone single-file version for WASM/GitHub Pages deployment
# Author: Pito Salas and Claude Code
# Open Source Under MIT license
# All local modules inlined — edit the source files in src/, not this file.

import marimo

app = marimo.App()


@app.cell
def _():
    from dataclasses import dataclass
    from enum import Enum
    import html as _html
    import networkx as nx
    import marimo as mo

    # ── node / edge ───────────────────────────────────────────────────────────
    class NodeType(Enum):
        INPUT = "input"
        WEIGHT = "weight"
        MULTIPLY = "multiply"
        ADD = "add"
        LOSS = "loss"

    @dataclass
    class Node:
        id: str
        node_type: NodeType
        value: float

    @dataclass
    class Edge:
        from_id: str
        to_id: str

    # ── CompGraph ─────────────────────────────────────────────────────────────
    class CompGraph:
        def __init__(self):
            self.graph = nx.DiGraph()
            self.targets = {}

        def add_node(self, node):
            self.graph.add_node(node.id, node_type=node.node_type, value=node.value, gradient=0.0)

        def add_edge(self, edge):
            self.graph.add_edge(edge.from_id, edge.to_id, local_deriv=0.0)

        def set_target(self, node_id, target):
            self.targets[node_id] = target

        def get_value(self, node_id):
            return self.graph.nodes[node_id]["value"]

        def forward_pass_n(self, n):
            dispatch = {
                NodeType.MULTIPLY: self._mul,
                NodeType.ADD: self._add,
                NodeType.LOSS: self._loss,
            }
            count = 0
            for node_id in nx.topological_sort(self.graph):
                if count >= n:
                    break
                ntype = self.graph.nodes[node_id]["node_type"]
                if ntype in dispatch:
                    dispatch[ntype](node_id)
                    count += 1

        def computed_node_ids(self):
            computed = {NodeType.MULTIPLY, NodeType.ADD, NodeType.LOSS}
            return [n for n in nx.topological_sort(self.graph)
                    if self.graph.nodes[n]["node_type"] in computed]

        def _mul(self, node_id):
            preds = list(self.graph.predecessors(node_id))
            a, b = self.graph.nodes[preds[0]]["value"], self.graph.nodes[preds[1]]["value"]
            self.graph.nodes[node_id]["value"] = a * b
            self.graph[preds[0]][node_id]["local_deriv"] = b
            self.graph[preds[1]][node_id]["local_deriv"] = a

        def _add(self, node_id):
            preds = list(self.graph.predecessors(node_id))
            self.graph.nodes[node_id]["value"] = sum(self.graph.nodes[p]["value"] for p in preds)
            for p in preds:
                self.graph[p][node_id]["local_deriv"] = 1.0

        def _loss(self, node_id):
            pred_id = list(self.graph.predecessors(node_id))[0]
            pred_val = self.graph.nodes[pred_id]["value"]
            target = self.targets[node_id]
            self.graph.nodes[node_id]["value"] = (pred_val - target) ** 2
            self.graph[pred_id][node_id]["local_deriv"] = 2 * (pred_val - target)

    # ── examples ──────────────────────────────────────────────────────────────
    TURKEYS = [
        {"label": "Turkey 1", "height": 1.00, "length": 1.50, "target": 5000},
        {"label": "Turkey 2", "height": 0.75, "length": 1.25, "target": 3500},
        {"label": "Turkey 3", "height": 1.25, "length": 1.00, "target": 4500},
    ]

    def turkey_feather(height, length, w1, w2, target):
        g = CompGraph()
        g.add_node(Node("height", NodeType.INPUT, height))
        g.add_node(Node("length", NodeType.INPUT, length))
        g.add_node(Node("w1", NodeType.WEIGHT, w1))
        g.add_node(Node("w2", NodeType.WEIGHT, w2))
        g.add_node(Node("ht_term", NodeType.MULTIPLY, 0.0))
        g.add_node(Node("len_term", NodeType.MULTIPLY, 0.0))
        g.add_node(Node("prediction", NodeType.ADD, 0.0))
        g.add_node(Node("loss", NodeType.LOSS, 0.0))
        g.set_target("loss", target)
        g.add_edge(Edge("height", "ht_term"))
        g.add_edge(Edge("w1", "ht_term"))
        g.add_edge(Edge("length", "len_term"))
        g.add_edge(Edge("w2", "len_term"))
        g.add_edge(Edge("ht_term", "prediction"))
        g.add_edge(Edge("len_term", "prediction"))
        g.add_edge(Edge("prediction", "loss"))
        return g

    # ── steps ─────────────────────────────────────────────────────────────────
    def forward_step_label(g, node_id):
        ntype = g.graph.nodes[node_id]["node_type"]
        preds = list(g.graph.predecessors(node_id))
        result = g.get_value(node_id)
        if ntype == NodeType.MULTIPLY:
            a, b = preds[0], preds[1]
            return (f"**{node_id}** = {a} × {b} "
                    f"= {g.get_value(a):.2f} × {g.get_value(b):.2f} = **{result:.2f}**")
        if ntype == NodeType.ADD:
            terms = " + ".join(f"{g.get_value(p):.2f}" for p in preds)
            return f"**{node_id}** = {' + '.join(preds)} = {terms} = **{result:.2f}**"
        if ntype == NodeType.LOSS:
            pred_val = g.get_value(preds[0])
            target = g.targets[node_id]
            return (f"**{node_id}** = ({preds[0]} − target)² "
                    f"= ({pred_val:.2f} − {target:.2f})² = **{result:.2f}**")
        return f"**{node_id}** = {result:.2f}"

    # ── mermaid_viz ───────────────────────────────────────────────────────────
    _TYPE_CLASS = {
        NodeType.INPUT: "input", NodeType.WEIGHT: "weight",
        NodeType.MULTIPLY: "op", NodeType.ADD: "op", NodeType.LOSS: "loss",
    }
    _CLASS_DEFS = [
        "classDef input fill:#4682B4,stroke:#336699,color:white,font-size:15px",
        "classDef weight fill:#3CB371,stroke:#2d8a5e,color:white,font-size:15px",
        "classDef op fill:#e0e0e0,stroke:#999,color:#222,font-size:15px",
        "classDef loss fill:#FF6347,stroke:#cc4f3c,color:white,font-size:15px",
        "classDef hl stroke:#FFD700,stroke-width:5px",
    ]

    def mermaid_html(g, show_gradients, highlighted):
        lines = ["graph LR"] + [f"    {d}" for d in _CLASS_DEFS]
        for nid in g.graph.nodes:
            attrs = g.graph.nodes[nid]
            label = f"{nid}<br/>{attrs['value']:.3f}"
            lines.append(f'    {nid}["{label}"]:::{_TYPE_CLASS[attrs["node_type"]]}')
        for f, t in g.graph.edges:
            lines.append(f"    {f} --> {t}")
        if highlighted:
            lines.append(f"    class {highlighted} hl")
        diagram = "\n".join(lines)
        page = (f'<!DOCTYPE html><html><head>'
                f'<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>'
                f'</head><body style="margin:0;background:transparent">'
                f'<div class="mermaid">{diagram}</div>'
                f'<script>mermaid.initialize({{startOnLoad:true}});</script>'
                f'</body></html>')
        escaped = _html.escape(page, quote=True)
        return (f'<iframe srcdoc="{escaped}" style="width:100%;height:500px;border:none;" '
                f'onload="this.style.height=(this.contentWindow.document.body.scrollHeight+20)+\'px\'"></iframe>')

    # ── table_viz ─────────────────────────────────────────────────────────────
    GOLD = "#FFD700"
    _HEADER_BG = "#f0f0f0"
    _INPUT_NODES = ["height", "length", "w1", "w2"]
    COMPUTED_NODES = ["ht_term", "len_term", "prediction", "loss"]
    _ALL_NODES = _INPUT_NODES + COMPUTED_NODES
    _NODE_LABELS = {"ht_term": "height×w1", "len_term": "length×w2",
                    "prediction": "prediction", "loss": "loss"}

    def _cell(value, bg, bold):
        w = "font-weight:bold;" if bold else ""
        return f'<td style="text-align:center;padding:6px 10px;background:{bg};{w}">{value}</td>'

    def forward_pass_table(w1, w2, step):
        hdrs = ['<th style="padding:6px 10px;background:#ddd;text-align:left">Turkey</th>']
        for node in _ALL_NODES:
            ci = COMPUTED_NODES.index(node) + 1 if node in COMPUTED_NODES else None
            bg = GOLD if ci == step else _HEADER_BG
            hdrs.append(f'<th style="padding:6px 10px;background:{bg};text-align:center">{_NODE_LABELS.get(node, node)}</th>')
        rows = ["<tr>" + "".join(hdrs) + "</tr>"]
        for t in TURKEYS:
            g = turkey_feather(height=t["height"], length=t["length"], w1=w1, w2=w2, target=t["target"])
            g.forward_pass_n(step)
            cells = [f'<td style="padding:6px 10px;font-weight:bold;background:#f8f8f8">{t["label"]}</td>']
            for node in _ALL_NODES:
                ci = COMPUTED_NODES.index(node) + 1 if node in COMPUTED_NODES else None
                active = ci == step
                bg = GOLD if active else "white"
                if node in _INPUT_NODES:
                    cells.append(_cell(f"{g.get_value(node):g}", bg, False))
                elif ci is not None and ci <= step:
                    cells.append(_cell(f"{g.get_value(node):,.1f}", bg, active))
                else:
                    cells.append(_cell("—", bg, False))
            rows.append("<tr>" + "".join(cells) + "</tr>")
        loss_idx = COMPUTED_NODES.index("loss") + 1
        if step >= loss_idx:
            total = 0.0
            for t in TURKEYS:
                g = turkey_feather(height=t["height"], length=t["length"], w1=w1, w2=w2, target=t["target"])
                g.forward_pass_n(step)
                total += g.get_value("loss")
            n_sp = len(_ALL_NODES) - 1
            rows.append(
                "<tr>"
                '<td style="padding:6px 10px;font-weight:bold;background:#f0d0d0">Sum of squares<br/>'
                '<span style="font-weight:normal;font-size:12px">Σ(pred−target)²</span></td>'
                f'<td colspan="{n_sp}" style="padding:6px 10px;background:#fafafa"></td>'
                f'<td style="text-align:center;padding:6px 10px;font-weight:bold;background:#ffe0e0">{total:,.1f}</td>'
                "</tr>"
            )
        return f'<table style="border-collapse:collapse;font-size:14px;width:100%">{"".join(rows)}</table>'

    return (COMPUTED_NODES, forward_pass_table, forward_step_label, mermaid_html, mo, turkey_feather)


@app.cell
def _(mo):
    mo.md("""
# Backpropagation Explorer

Each cell below is one lesson. Work through them top to bottom.

| Cell | Lesson |
|------|--------|
| 2 | **The Model** — the turkey feather computation graph |
| 3 | **Computing Loss** — step through the computation one node at a time |
| 4 | **Changing a Weight** — use the sliders, watch loss respond |
""")
    return


@app.cell
def _(mo):
    mo.md("""
## The Model

In Chapters 3 and 4, the book uses a small dataset of three turkeys. Each turkey
has two measurements — **height** and **length** — and a known feather count
that the model should learn to predict:

| Turkey | Height (m) | Length (m) | Feathers (target) |
|--------|-----------|-----------|-------------------|
| 1      | 1.00      | 1.50      | 5,000             |
| 2      | 0.75      | 1.25      | 3,500             |
| 3      | 1.25      | 1.00      | 4,500             |

The model makes its prediction as a weighted sum:

> prediction = height × w1 + length × w2

We start with initial weights **w1 = 1000, w2 = 3000**.
""")
    return


@app.cell
def _(mermaid_html, mo, turkey_feather):
    _g = turkey_feather(height=1.0, length=1.5, w1=1000, w2=3000, target=5000)
    mo.Html(mermaid_html(_g, False, None))
    return


@app.cell
def _(mo):
    mo.md("""
## Computing Loss

Click **Next →** to compute each node for all three turkeys simultaneously.
The gold column shows what's being computed.
""")
    return


@app.cell
def _(mo):
    prev_btn = mo.ui.button(label="← Prev", value=0, on_click=lambda v: v + 1)
    next_btn = mo.ui.button(label="Next →", value=0, on_click=lambda v: v + 1)
    mo.hstack([prev_btn, next_btn], gap=1)
    return (next_btn, prev_btn)


@app.cell
def _(COMPUTED_NODES, forward_pass_table, forward_step_label, mo, next_btn, prev_btn, turkey_feather):
    _step = max(0, min(next_btn.value - prev_btn.value, len(COMPUTED_NODES)))
    if _step == 0:
        _explanation = mo.md("Click **Next →** to begin.")
    else:
        _node = COMPUTED_NODES[_step - 1]
        _g_ex = turkey_feather(height=1.0, length=1.5, w1=1000, w2=3000, target=5000)
        _g_ex.forward_pass_n(_step)
        _suffix = "  \n✓ Done." if _step == len(COMPUTED_NODES) else ""
        _explanation = mo.md(f"**Step {_step}/{len(COMPUTED_NODES)}:** {forward_step_label(_g_ex, _node)}{_suffix}")
    mo.vstack([mo.Html(forward_pass_table(1000, 3000, _step)), _explanation])
    return


@app.cell
def _(mo):
    mo.md("""
## Changing a Weight

Drag the sliders to change w1 and w2. Watch every prediction and loss update
instantly across all three turkeys. Try to minimise the sum of squares.

(Hint: the book's converged values are around **w1 = 2311, w2 = 1633**.)
""")
    return


@app.cell
def _(mo):
    w1_slider = mo.ui.slider(start=0, stop=5000, step=50, value=1000, label="w1")
    w2_slider = mo.ui.slider(start=0, stop=5000, step=50, value=3000, label="w2")
    mo.vstack([w1_slider, w2_slider])
    return (w1_slider, w2_slider)


@app.cell
def _(COMPUTED_NODES, forward_pass_table, mo, w1_slider, w2_slider):
    mo.Html(forward_pass_table(w1_slider.value, w2_slider.value, len(COMPUTED_NODES)))
    return


if __name__ == "__main__":
    app.run()
