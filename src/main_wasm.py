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
            return (f"**{node_id}** = ({preds[0]} − actual)² "
                    f"= ({pred_val:.2f} − {target:.2f})² = **{result:.2f}**")
        return f"**{node_id}** = {result:.2f}"

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
                '<span style="font-weight:normal;font-size:12px">Σ(pred−actual)²</span></td>'
                f'<td colspan="{n_sp}" style="padding:6px 10px;background:#fafafa"></td>'
                f'<td style="text-align:center;padding:6px 10px;font-weight:bold;background:#ffe0e0">{total:,.1f}</td>'
                "</tr>"
            )
        return f'<table style="border-collapse:collapse;font-size:14px;width:100%">{"".join(rows)}</table>'

    # ── chain_rule ────────────────────────────────────────────────────────────
    def chain_forward(x):
        a = x ** 2
        b = a ** 3
        return a, b

    def chain_derivs(x):
        a = x ** 2
        da_dx = 2 * x
        db_da = 3 * (a ** 2)
        return da_dx, db_da, da_dx * db_da

    def chain_html(x):
        a, b = chain_forward(x)
        da_dx, db_da, db_dx = chain_derivs(x)

        node_style = (
            "display:inline-block;padding:10px 18px;border-radius:8px;"
            "font-size:1.1em;font-weight:bold;text-align:center;"
        )
        input_style = node_style + "background:#cce5ff;border:2px solid #4a90d9;"
        op_style = node_style + "background:#f0f0f0;border:2px solid #aaa;font-size:0.9em;"
        arrow_style = (
            "display:inline-block;vertical-align:middle;"
            "text-align:center;margin:0 6px;"
        )

        def arrow(label):
            return (
                f'<span style="{arrow_style}">'
                f'<span style="font-size:0.8em;color:#666;">{label}</span><br>'
                f'<span style="font-size:1.4em;">→</span>'
                f'</span>'
            )

        diagram = (
            f'<div style="display:flex;align-items:center;gap:4px;flex-wrap:wrap;margin:16px 0;">'
            f'<span style="{input_style}">x = {x:g}</span>'
            f'{arrow(f"local deriv = {da_dx:g}")}'
            f'<span style="{op_style}">square<br><small>A = x²</small><br>A = {a:g}</span>'
            f'{arrow(f"local deriv = {db_da:g}")}'
            f'<span style="{op_style}">cube<br><small>B = A³</small><br>B = {b:g}</span>'
            f'</div>'
        )

        summary = (
            f'<p style="margin:8px 0;font-size:1em;">'
            f'<strong>Chain rule at x = {x:g}:</strong> '
            f'df/dx = (dA/dx) × (dB/dA) = {da_dx:g} × {db_da:g} = <strong>{db_dx:g}</strong>'
            f'</p>'
            f'<p style="margin:4px 0;color:#555;font-size:0.9em;">'
            f'A tiny nudge to x causes the output to change by {db_dx:g}× that amount.'
            f'</p>'
        )

        return f'<div style="font-family:sans-serif;">{diagram}{summary}</div>'

    return (COMPUTED_NODES, chain_html, forward_pass_table, forward_step_label, mo, turkey_feather)


@app.cell
def _(mo):
    mo.md("""
# Backpropagation Explorer

Each cell below is one lesson. Work through them top to bottom.

| Cell | Lesson |
|------|--------|
| 3 | **The Model** — the turkey feather computation graph |
| 4 | **Computing Loss** — step through the computation one node at a time |
| 5 | **Changing a Weight** — use the slider, watch loss respond |
| 6 | **The Chain Rule** — how gradients flow backwards through a graph |
| 7 | **The Backward Pass** — click Run to see gradients |
""")
    return


@app.cell
def _(mo):
    mo.md("""
## The Model

In Chapters 3 and 4, the book uses a small dataset of three turkeys. Each turkey
has two measurements — **height** and **length** — and a known feather count
that the model should learn to predict:

| Turkey | Height (m) | Length (m) | Feathers (actual) |
|--------|-----------|-----------|-------------------|
| 1      | 1.00      | 1.50      | 5,000             |
| 2      | 0.75      | 1.25      | 3,500             |
| 3      | 1.25      | 1.00      | 4,500             |

The model makes its prediction as a weighted sum:

> prediction = height × w1 + length × w2

We start with initial weights **w1 = 1000, w2 = 3000**. Here is the full
computation for Turkey 1, step by step:
""")
    return


@app.cell
def _(mo):
    _blue = "color:#1a6bb5;font-weight:bold;"
    _green = "color:#2a7a2a;font-weight:bold;"
    _eq = "padding:0 8px;color:#555;"
    _lbl = "text-align:right;padding-right:12px;color:#333;"

    _html = f"""
    <div style="font-family:monospace;font-size:1.05em;line-height:2.2;margin:16px 0;">
      <table style="border-collapse:collapse;">
        <tr>
          <td style="{_lbl}">prediction</td>
          <td style="{_eq}">=</td>
          <td><span style="{_blue}">height &times; w1</span></td>
          <td style="{_eq}">+</td>
          <td><span style="{_green}">length &times; w2</span></td>
        </tr>
        <tr>
          <td style="{_lbl}"></td>
          <td style="{_eq}">=</td>
          <td><span style="{_blue}">1.0 &times; 1000</span></td>
          <td style="{_eq}">+</td>
          <td><span style="{_green}">1.5 &times; 3000</span></td>
        </tr>
        <tr>
          <td style="{_lbl}"></td>
          <td style="{_eq}">=</td>
          <td style="{_blue}">1,000</td>
          <td style="{_eq}">+</td>
          <td style="{_green}">4,500</td>
        </tr>
        <tr>
          <td style="{_lbl}"></td>
          <td style="{_eq}">=</td>
          <td colspan="3"><strong>5,500</strong></td>
        </tr>
        <tr><td colspan="5" style="padding-top:12px;"></td></tr>
        <tr>
          <td style="{_lbl}">loss</td>
          <td style="{_eq}">=</td>
          <td colspan="3">(prediction &minus; actual)&sup2;</td>
        </tr>
        <tr>
          <td style="{_lbl}"></td>
          <td style="{_eq}">=</td>
          <td colspan="3">(5,500 &minus; 5,000)&sup2;</td>
        </tr>
        <tr>
          <td style="{_lbl}"></td>
          <td style="{_eq}">=</td>
          <td colspan="3">500&sup2; = <strong>250,000</strong></td>
        </tr>
      </table>
    </div>
    """
    mo.Html(_html)
    return


@app.cell
def _(mo):
    mo.md("""
## Computing Loss

The table below shows all three turkeys at once. Each column is a node in the
computation graph. Click **Next →** to compute the next node for all three
turkeys simultaneously. The gold column is the one being computed.

The first two computed columns are **height×w1** and **length×w2** — each input
multiplied by its weight. These are the two *terms* of the weighted sum. w1 and
w2 are the knobs the model will eventually learn to turn: a higher w1 means
height matters more, a higher w2 means length matters more.

**prediction** adds the two terms together — that's the model's guess at the
feather count. **loss** measures how wrong that guess is: (prediction − actual)².
Squaring makes all errors positive and penalizes large errors more than small ones.
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
        _explanation = mo.md("Click **Next →** to begin the forward pass.")
    else:
        _node = COMPUTED_NODES[_step - 1]
        _g_ex = turkey_feather(height=1.0, length=1.5, w1=1000, w2=3000, target=5000)
        _g_ex.forward_pass_n(_step)
        _suffix = "  \n✓ Forward pass complete." if _step == len(COMPUTED_NODES) else ""
        _explanation = mo.md(f"**Step {_step} of {len(COMPUTED_NODES)}:** {forward_step_label(_g_ex, _node)}{_suffix}")

    mo.vstack([mo.Html(forward_pass_table(1000, 3000, _step)), _explanation])
    return


@app.cell
def _(mo):
    mo.md("""
## Changing a Weight

So far w1 and w2 have been fixed at 1000 and 3000. But those were just a starting
guess — the whole point of training is to find better values.

Try dragging the **w1** slider below. w1 is the weight on **height**: a higher w1
means the model thinks tall turkeys have more feathers. Watch how every prediction
and every loss changes instantly across all three turkeys.

Your goal: find a value of w1 that makes the total loss as small as possible.
(Hint: the book's converged value is around 2311.)
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
    _table = forward_pass_table(w1_slider.value, w2_slider.value, len(COMPUTED_NODES))
    mo.Html(_table)
    return


@app.cell
def _(mo):
    mo.md("""
## The Chain Rule

Before we can run the backward pass on the turkey model, we need one idea: the
**chain rule**. It tells us how to find the derivative of a composed function —
one function fed into another.

Consider **f(x) = (x²)³**. We can think of it as two steps:

- **Step A:** square the input → A = x²
- **Step B:** cube the result → B = A³

Each step has a *local derivative* — how much its output changes when its input
nudges a tiny bit. At x = 3:

- dA/dx = 2x = **6** (squaring: derivative is 2×input)
- dB/dA = 3A² = **243** (cubing: derivative is 3×input²)

The chain rule says the overall derivative is just their product:
**df/dx = 6 × 243 = 1458**

Drag the slider below to see how the values and derivatives change with x.
""")
    return


@app.cell
def _(mo):
    x_slider = mo.ui.slider(start=1, stop=5, step=0.5, value=3, label="x")
    x_slider
    return (x_slider,)


@app.cell
def _(chain_html, mo, x_slider):
    mo.Html(chain_html(x_slider.value))
    return


@app.cell
def _(mo):
    mo.md("""
### From chain rule to partial derivatives

The turkey model is the same idea as (x²)³ — a chain of operations. Follow w1
through the computation: w1 feeds into **ht_term** (height × w1), which feeds
into **prediction** (ht_term + len_term), which feeds into **loss**
((prediction − actual)²). Three steps chained together.

A **partial derivative** asks: holding w2 fixed, how much does loss change if I
nudge w1 a tiny bit? We apply the chain rule backwards along that path,
multiplying local derivatives at each step.

At w1=1000, w2=3000, Turkey 1:

| Step | Local derivative | Value |
|------|-----------------|-------|
| loss w.r.t. prediction | 2 × (5500 − 5000) | 1000 |
| prediction w.r.t. ht_term | 1 (addition) | 1 |
| ht_term w.r.t. w1 | height | 1.0 |

Product for Turkey 1: 1000 × 1 × 1.0 = **1000**. Sum across all three turkeys
gives the partial derivative of total loss w.r.t. w1 = **1875**.

Since 1875 is positive, increasing w1 increases loss — so we decrease it.
With a learning rate of 0.01:

> new w1 = 1000 − 0.01 × 1875 = **981**

Loss goes down. Repeat thousands of times and w1 converges to ~2311. The next
lesson runs this process automatically for both weights at once — that's the
backward pass.
""")
    return


if __name__ == "__main__":
    app.run()
