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

        def get_gradient(self, node_id):
            return self.graph.nodes[node_id]["gradient"]

        def forward_pass(self):
            dispatch = {NodeType.MULTIPLY: self._mul, NodeType.ADD: self._add, NodeType.LOSS: self._loss}
            for node_id in nx.topological_sort(self.graph):
                ntype = self.graph.nodes[node_id]["node_type"]
                if ntype in dispatch:
                    dispatch[ntype](node_id)

        def backward_pass(self):
            for node_id in self.graph.nodes:
                self.graph.nodes[node_id]["gradient"] = 0.0
            loss_id = self._find_loss_node()
            self.graph.nodes[loss_id]["gradient"] = 1.0
            for node_id in reversed(list(nx.topological_sort(self.graph))):
                node_grad = self.graph.nodes[node_id]["gradient"]
                for pred_id in self.graph.predecessors(node_id):
                    local_deriv = self.graph[pred_id][node_id]["local_deriv"]
                    self.graph.nodes[pred_id]["gradient"] += node_grad * local_deriv

        def _find_loss_node(self):
            for node_id in self.graph.nodes:
                if self.graph.nodes[node_id]["node_type"] == NodeType.LOSS:
                    return node_id
            raise ValueError("No loss node found in graph")

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

    # ── Dataset + examples ────────────────────────────────────────────────────
    @dataclass
    class Dataset:
        name: str
        samples: list
        feat1_name: str
        feat2_name: str
        target_name: str
        w1_start: float
        w2_start: float
        w1_min: float
        w1_max: float
        w2_min: float
        w2_max: float
        w_step: float

    def build_graph(sample, dataset, w1, w2):
        f1, f2 = dataset.feat1_name, dataset.feat2_name
        g = CompGraph()
        g.add_node(Node(f1, NodeType.INPUT, sample[f1]))
        g.add_node(Node(f2, NodeType.INPUT, sample[f2]))
        g.add_node(Node("w1", NodeType.WEIGHT, w1))
        g.add_node(Node("w2", NodeType.WEIGHT, w2))
        g.add_node(Node("ht_term", NodeType.MULTIPLY, 0.0))
        g.add_node(Node("len_term", NodeType.MULTIPLY, 0.0))
        g.add_node(Node("prediction", NodeType.ADD, 0.0))
        g.add_node(Node("loss", NodeType.LOSS, 0.0))
        g.set_target("loss", sample["target"])
        g.add_edge(Edge(f1, "ht_term"))
        g.add_edge(Edge("w1", "ht_term"))
        g.add_edge(Edge(f2, "len_term"))
        g.add_edge(Edge("w2", "len_term"))
        g.add_edge(Edge("ht_term", "prediction"))
        g.add_edge(Edge("len_term", "prediction"))
        g.add_edge(Edge("prediction", "loss"))
        return g

    TURKEYS = [
        {"label": "Turkey 1", "height": 1.00, "length": 1.50, "target": 5000},
        {"label": "Turkey 2", "height": 0.75, "length": 1.25, "target": 3500},
        {"label": "Turkey 3", "height": 1.25, "length": 1.00, "target": 4500},
    ]

    TURKEY_DATASET = Dataset(
        name="Turkey Feathers",
        samples=TURKEYS,
        feat1_name="height",
        feat2_name="length",
        target_name="feathers",
        w1_start=1000.0,
        w2_start=3000.0,
        w1_min=0.0,
        w1_max=5000.0,
        w2_min=0.0,
        w2_max=5000.0,
        w_step=50.0,
    )

    CARS_DATASET = Dataset(
        name="Auto MPG",
        samples=[
            {"label": "Chevrolet Chevelle", "weight": 3.504, "horsepower": 1.30, "target": 18.0},
            {"label": "Buick Skylark 320",  "weight": 3.693, "horsepower": 1.65, "target": 15.0},
            {"label": "Datsun PL510",       "weight": 2.110, "horsepower": 0.95, "target": 28.0},
            {"label": "VW 1131 Deluxe",     "weight": 2.372, "horsepower": 0.75, "target": 30.0},
            {"label": "AMC Hornet",         "weight": 2.833, "horsepower": 0.90, "target": 26.0},
            {"label": "Pontiac Safari",     "weight": 4.746, "horsepower": 2.30, "target": 10.0},
            {"label": "Ford Galaxie 500",   "weight": 4.382, "horsepower": 1.98, "target": 14.0},
            {"label": "Toyota Corolla",     "weight": 2.130, "horsepower": 0.70, "target": 33.0},
            {"label": "Honda Civic",        "weight": 1.835, "horsepower": 0.65, "target": 31.0},
            {"label": "Dodge Challenger",   "weight": 3.609, "horsepower": 1.50, "target": 18.0},
        ],
        feat1_name="weight",
        feat2_name="horsepower",
        target_name="mpg",
        w1_start=5.0,
        w2_start=5.0,
        w1_min=-20.0,
        w1_max=20.0,
        w2_min=-20.0,
        w2_max=20.0,
        w_step=0.5,
    )

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
    COMPUTED_NODES = ["ht_term", "len_term", "prediction", "loss"]
    BACKWARD_NODES = ["loss", "prediction", "ht_term", "len_term", "w1", "w2"]

    def _cell(value, bg, bold):
        w = "font-weight:bold;" if bold else ""
        return f'<td style="text-align:center;padding:6px 10px;background:{bg};{w}">{value}</td>'

    def _header_row(step, dataset):
        f1, f2 = dataset.feat1_name, dataset.feat2_name
        all_nodes = [f1, f2, "w1", "w2"] + COMPUTED_NODES
        labels = {f1: f1, f2: f2, "w1": "w1", "w2": "w2",
                  "ht_term": f"{f1}×w1", "len_term": f"{f2}×w2",
                  "prediction": "prediction", "loss": "loss"}
        cells = ['<th style="padding:6px 10px;background:#ddd;text-align:left">Sample</th>']
        for node in all_nodes:
            ci = COMPUTED_NODES.index(node) + 1 if node in COMPUTED_NODES else None
            bg = GOLD if ci == step else _HEADER_BG
            cells.append(f'<th style="padding:6px 10px;background:{bg};text-align:center">{labels[node]}</th>')
        return "<tr>" + "".join(cells) + "</tr>"

    def _sample_row(t, w1, w2, step, dataset):
        f1, f2 = dataset.feat1_name, dataset.feat2_name
        input_nodes = [f1, f2, "w1", "w2"]
        g = build_graph(t, dataset, w1, w2)
        g.forward_pass_n(step)
        cells = [f'<td style="padding:6px 10px;font-weight:bold;background:#f8f8f8">{t["label"]}</td>']
        for node in input_nodes + COMPUTED_NODES:
            ci = COMPUTED_NODES.index(node) + 1 if node in COMPUTED_NODES else None
            is_active = ci == step
            bg = GOLD if is_active else "white"
            if node in input_nodes:
                cells.append(_cell(f"{g.get_value(node):g}", bg, False))
            elif ci is not None and ci <= step:
                cells.append(_cell(f"{g.get_value(node):,.2f}", bg, is_active))
            else:
                cells.append(_cell("—", bg, False))
        return "<tr>" + "".join(cells) + "</tr>"

    def _total_loss_row(w1, w2, step, dataset):
        if step < COMPUTED_NODES.index("loss") + 1:
            return ""
        total = 0.0
        for t in dataset.samples:
            g = build_graph(t, dataset, w1, w2)
            g.forward_pass_n(step)
            total += g.get_value("loss")
        label = '<td style="padding:6px 10px;font-weight:bold;background:#f0d0d0">Sum of squares<br/><span style="font-weight:normal;font-size:12px">Σ(pred−actual)²</span></td>'
        spacer = '<td colspan="7" style="padding:6px 10px;background:#fafafa"></td>'
        total_cell = f'<td style="text-align:center;padding:6px 10px;font-weight:bold;background:#ffe0e0">{total:,.2f}</td>'
        return "<tr>" + label + spacer + total_cell + "</tr>"

    def forward_pass_table(w1, w2, step, dataset):
        rows = [_header_row(step, dataset)]
        for t in dataset.samples:
            rows.append(_sample_row(t, w1, w2, step, dataset))
        total_row = _total_loss_row(w1, w2, step, dataset)
        if total_row:
            rows.append(total_row)
        return f'<table style="border-collapse:collapse;font-size:14px;width:100%">{"".join(rows)}</table>'

    def _back_header_row(step, dataset):
        f1, f2 = dataset.feat1_name, dataset.feat2_name
        back_labels = {"loss": "loss", "prediction": "prediction",
                       "ht_term": f"{f1}×w1", "len_term": f"{f2}×w2", "w1": "w1", "w2": "w2"}
        cells = [
            '<th style="padding:6px 10px;background:#ddd;text-align:left">Sample</th>',
            f'<th style="padding:6px 10px;background:{_HEADER_BG};text-align:center">actual</th>',
        ]
        for i, node in enumerate(BACKWARD_NODES):
            bg = GOLD if (i + 1) == step else _HEADER_BG
            cells.append(f'<th style="padding:6px 10px;background:{bg};text-align:center">{back_labels[node]}</th>')
        return "<tr>" + "".join(cells) + "</tr>"

    def _back_sample_row(t, w1, w2, step, dataset):
        g = build_graph(t, dataset, w1, w2)
        g.forward_pass()
        g.backward_pass()
        cells = [
            f'<td style="padding:6px 10px;font-weight:bold;background:#f8f8f8">{t["label"]}</td>',
            _cell(f"{t['target']:,g}", _HEADER_BG, False),
        ]
        for i, node in enumerate(BACKWARD_NODES):
            col_idx = i + 1
            is_active = col_idx == step
            bg = GOLD if is_active else "white"
            if col_idx <= step:
                cells.append(_cell(f"{g.get_gradient(node):,.2f}", bg, is_active))
            else:
                cells.append(_cell("—", bg, False))
        return "<tr>" + "".join(cells) + "</tr>"

    def _back_sum_row(w1, w2, step, dataset):
        w1_col = BACKWARD_NODES.index("w1") + 1
        w2_col = BACKWARD_NODES.index("w2") + 1
        if step < w1_col:
            return ""
        total_w1, total_w2 = 0.0, 0.0
        for t in dataset.samples:
            g = build_graph(t, dataset, w1, w2)
            g.forward_pass()
            g.backward_pass()
            total_w1 += g.get_gradient("w1")
            total_w2 += g.get_gradient("w2")
        label = '<td style="padding:6px 10px;font-weight:bold;background:#f0d0d0">Sum (all samples)</td>'
        spacer = f'<td colspan="{w1_col}" style="padding:6px 10px;background:#fafafa"></td>'
        w1_cell = f'<td style="text-align:center;padding:6px 10px;font-weight:bold;background:#ffe0e0">{total_w1:,.2f}</td>'
        w2_cell = (f'<td style="text-align:center;padding:6px 10px;font-weight:bold;background:#ffe0e0">{total_w2:,.2f}</td>'
                   if step >= w2_col else "")
        return "<tr>" + label + spacer + w1_cell + w2_cell + "</tr>"

    def backward_pass_table(w1, w2, step, dataset):
        rows = [_back_header_row(step, dataset)]
        for t in dataset.samples:
            rows.append(_back_sample_row(t, w1, w2, step, dataset))
        sum_row = _back_sum_row(w1, w2, step, dataset)
        if sum_row:
            rows.append(sum_row)
        return f'<table style="border-collapse:collapse;font-size:14px;width:100%">{"".join(rows)}</table>'

    def backward_step_labels(dataset):
        s0 = dataset.samples[0]
        g = build_graph(s0, dataset, dataset.w1_start, dataset.w2_start)
        g.forward_pass()
        pred = g.get_value("prediction")
        error = pred - s0["target"]
        pred_grad = 2 * error
        f1, f2 = dataset.feat1_name, dataset.feat2_name
        return [
            "**loss** gradient = 1. The backward pass asks: how much does each node affect the loss? "
            "The loss node *is* the loss, so ∂L/∂loss = 1 — a one-unit nudge to the loss node changes loss by exactly one unit. "
            "This seed value of 1 is what the rest of the backward pass multiplies through.",
            f"**prediction** gradient = loss_grad × 2(pred − actual). "
            f"loss = (pred − actual)², so its local derivative w.r.t. prediction is 2(pred − actual) — "
            f"the power rule: d/dx[x²] = 2x, where x = (pred − actual). "
            f"For {s0['label']}: 1 × 2 × ({pred:.2f} − {s0['target']:.2f}) = 1 × 2 × {error:.2f} = **{pred_grad:.2f}**. "
            f"Each sample differs because each has a different prediction error.",
            f"**ht_term** gradient = prediction_grad × 1. Addition passes the gradient through unchanged.",
            f"**len_term** gradient = prediction_grad × 1. Same rule — addition local derivative is always 1.",
            f"**w1** gradient = ht_term_grad × {f1}. The sum row shows the total across all samples.",
            f"**w2** gradient = len_term_grad × {f2}. Backward pass complete — both weight gradients ready.",
        ]

    # ── chain_rule ────────────────────────────────────────────────────────────
    def _chain_forward(x):
        a = x ** 2
        b = a ** 3
        return a, b

    def _chain_derivs(x):
        a = x ** 2
        da_dx = 2 * x
        db_da = 3 * (a ** 2)
        return da_dx, db_da, da_dx * db_da

    def chain_html(x):
        a, b = _chain_forward(x)
        da_dx, db_da, db_dx = _chain_derivs(x)

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
                f'<span style="font-size:0.8em;color:#000;">{label}</span><br>'
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
            f'<p style="margin:4px 0;color:#000;font-size:0.9em;">'
            f'A tiny nudge to x causes the output to change by {db_dx:g}× that amount.'
            f'</p>'
        )

        return f'<div style="font-family:sans-serif;">{diagram}{summary}</div>'

    return (BACKWARD_NODES, CARS_DATASET, COMPUTED_NODES, TURKEY_DATASET, backward_pass_table,
            backward_step_labels, build_graph, chain_html, forward_pass_table, forward_step_label, mo)


@app.cell
def _(mo):
    mo.md("""
# Backpropagation Explorer

Each section below is one lesson. Work through them top to bottom.

| Section | Lesson |
|---------|--------|
| 1 | **The Model** — the computation graph for the first sample |
| 2 | **Computing Loss** — step through the forward pass |
| 3 | **Changing a Weight** — use sliders, watch loss respond |
| 4 | **The Chain Rule** — how gradients flow backwards |
| 5 | **The Backward Pass** — automated gradient computation |

**Select a dataset to explore:**
""")
    return


@app.cell
def _(CARS_DATASET, TURKEY_DATASET, mo):
    dataset_selector = mo.ui.dropdown(
        {"Turkey Feathers": TURKEY_DATASET, "Auto MPG": CARS_DATASET},
        value="Turkey Feathers",
        label="Dataset",
    )
    dataset_selector
    return (dataset_selector,)


@app.cell
def _(dataset_selector):
    dataset = dataset_selector.value
    return (dataset,)


@app.cell
def _(dataset, mo):
    _s0 = dataset.samples[0]
    _f1, _f2 = dataset.feat1_name, dataset.feat2_name
    mo.md(f"""
## The Model

The **{dataset.name}** dataset has {len(dataset.samples)} samples. Each has two measurements —
**{_f1}** and **{_f2}** — and a known **{dataset.target_name}** the model should learn to predict.

The model makes its prediction as a weighted sum:

> prediction = {_f1} × w1 + {_f2} × w2

We start with initial weights **w1 = {dataset.w1_start:g}, w2 = {dataset.w2_start:g}**.
Here is the full computation for **{_s0['label']}**, step by step:
""")
    return


@app.cell
def _(build_graph, dataset, mo):
    _s0 = dataset.samples[0]
    _g = build_graph(_s0, dataset, dataset.w1_start, dataset.w2_start)
    _g.forward_pass()
    _f1, _f2 = dataset.feat1_name, dataset.feat2_name
    _f1v = _s0[_f1]
    _f2v = _s0[_f2]
    _ht = _g.get_value("ht_term")
    _lt = _g.get_value("len_term")
    _pred = _g.get_value("prediction")
    _target = _s0["target"]
    _err = _pred - _target
    _loss = _g.get_value("loss")

    _blue = "color:#1a6bb5;font-weight:bold;"
    _green = "color:#2a7a2a;font-weight:bold;"
    _eq = "padding:0 8px;color:#000;"
    _lbl = "text-align:right;padding-right:12px;color:#000;"

    _html = f"""
    <div style="font-family:monospace;font-size:1.05em;line-height:2.2;margin:16px 0;">
      <table style="border-collapse:collapse;">
        <tr>
          <td style="{_lbl}">prediction</td>
          <td style="{_eq}">=</td>
          <td><span style="{_blue}">{_f1} &times; w1</span></td>
          <td style="{_eq}">+</td>
          <td><span style="{_green}">{_f2} &times; w2</span></td>
        </tr>
        <tr>
          <td style="{_lbl}"></td>
          <td style="{_eq}">=</td>
          <td><span style="{_blue}">{_f1v:g} &times; {dataset.w1_start:g}</span></td>
          <td style="{_eq}">+</td>
          <td><span style="{_green}">{_f2v:g} &times; {dataset.w2_start:g}</span></td>
        </tr>
        <tr>
          <td style="{_lbl}"></td>
          <td style="{_eq}">=</td>
          <td style="{_blue}">{_ht:,.2f}</td>
          <td style="{_eq}">+</td>
          <td style="{_green}">{_lt:,.2f}</td>
        </tr>
        <tr>
          <td style="{_lbl}"></td>
          <td style="{_eq}">=</td>
          <td colspan="3"><strong>{_pred:,.2f}</strong></td>
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
          <td colspan="3">({_pred:,.2f} &minus; {_target:g})&sup2;</td>
        </tr>
        <tr>
          <td style="{_lbl}"></td>
          <td style="{_eq}">=</td>
          <td colspan="3">{_err:,.2f}&sup2; = <strong>{_loss:,.2f}</strong></td>
        </tr>
      </table>
    </div>
    """
    mo.Html(_html)
    return


@app.cell
def _(dataset, mo):
    _f1, _f2 = dataset.feat1_name, dataset.feat2_name
    mo.md(f"""
## Computing Loss

The table below shows all {len(dataset.samples)} samples at once. Each column is a node in the
computation graph. Click **Next →** to compute the next node for all samples
simultaneously. The gold column is the one being computed.

The first two computed columns are **{_f1}×w1** and **{_f2}×w2** — each input
multiplied by its weight. These are the two *terms* of the weighted sum. w1 and
w2 are the knobs the model will eventually learn to tune.

**prediction** adds the two terms together. **loss** measures how wrong that guess
is: (prediction − actual)². Squaring makes all errors positive and penalizes large
errors more than small ones.
""")
    return


@app.cell
def _(dataset, mo):
    prev_btn = mo.ui.button(label="← Prev", value=0, on_click=lambda v: v + 1)
    next_btn = mo.ui.button(label="Next →", value=0, on_click=lambda v: v + 1)
    mo.hstack([prev_btn, next_btn], gap=1)
    return (next_btn, prev_btn)


@app.cell
def _(COMPUTED_NODES, build_graph, dataset, forward_pass_table, forward_step_label, mo, next_btn, prev_btn):
    _step = max(0, min(next_btn.value - prev_btn.value, len(COMPUTED_NODES)))
    if _step == 0:
        _explanation = mo.md("Click **Next →** to begin the forward pass.")
    else:
        _node = COMPUTED_NODES[_step - 1]
        _g_ex = build_graph(dataset.samples[0], dataset, dataset.w1_start, dataset.w2_start)
        _g_ex.forward_pass_n(_step)
        _suffix = "  \n✓ Forward pass complete." if _step == len(COMPUTED_NODES) else ""
        _explanation = mo.md(f"**Step {_step} of {len(COMPUTED_NODES)}:** {forward_step_label(_g_ex, _node)}{_suffix}")
    mo.vstack([mo.Html(forward_pass_table(dataset.w1_start, dataset.w2_start, _step, dataset)), _explanation])
    return


@app.cell
def _(dataset, mo):
    _f1 = dataset.feat1_name
    mo.md(f"""
## Changing a Weight

So far w1 and w2 have been fixed at their starting values. But those are just an
initial guess — the whole point of training is to find better values.

Try dragging the **w1** slider below. w1 is the weight on **{_f1}**: a higher w1
means the model thinks {_f1} matters more. Watch how every prediction and every
loss changes instantly across all samples.

Your goal: find values of w1 and w2 that minimize the total loss.
""")
    return


@app.cell
def _(dataset, mo):
    w1_slider = mo.ui.slider(start=dataset.w1_min, stop=dataset.w1_max,
                              step=dataset.w_step, value=dataset.w1_start, label="w1")
    w2_slider = mo.ui.slider(start=dataset.w2_min, stop=dataset.w2_max,
                              step=dataset.w_step, value=dataset.w2_start, label="w2")
    mo.vstack([w1_slider, w2_slider])
    return (w1_slider, w2_slider)


@app.cell
def _(COMPUTED_NODES, dataset, forward_pass_table, mo, w1_slider, w2_slider):
    mo.Html(forward_pass_table(w1_slider.value, w2_slider.value, len(COMPUTED_NODES), dataset))
    return


@app.cell
def _(mo):
    mo.md("""
## The Chain Rule

Before we can run the backward pass, we need one idea: the **chain rule**. It
tells us how to find the derivative of a composed function — one function fed
into another.

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
def _(build_graph, dataset, mo):
    _s0 = dataset.samples[0]
    _f1, _f2 = dataset.feat1_name, dataset.feat2_name
    _g = build_graph(_s0, dataset, dataset.w1_start, dataset.w2_start)
    _g.forward_pass()
    _pred = _g.get_value("prediction")
    _target = _s0["target"]
    _loss_deriv = 2 * (_pred - _target)
    _f1_val = _s0[_f1]
    _chain_prod = _loss_deriv * 1 * _f1_val

    _total_w1 = 0.0
    for _s in dataset.samples:
        _gi = build_graph(_s, dataset, dataset.w1_start, dataset.w2_start)
        _gi.forward_pass()
        _gi.backward_pass()
        _total_w1 += _gi.get_gradient("w1")
    _new_w1 = dataset.w1_start - 0.01 * _total_w1
    _direction = ("positive — increasing w1 increases loss — so we decrease it"
                  if _total_w1 > 0 else
                  "negative — increasing w1 decreases loss — so we increase it")

    mo.md(f"""
### From chain rule to partial derivatives

The {dataset.name} model is the same idea as (x²)³ — a chain of operations. Follow w1
through the computation: w1 feeds into **ht_term** ({_f1} × w1), which feeds
into **prediction** (ht_term + len_term), which feeds into **loss**
((prediction − actual)²). Three steps chained together.

A **partial derivative** asks: holding w2 fixed, how much does loss change if I
nudge w1 a tiny bit? We apply the chain rule backwards along that path,
multiplying local derivatives at each step.

At w1={dataset.w1_start:g}, w2={dataset.w2_start:g}, {_s0['label']}:

| Step | Local derivative | Value |
|------|-----------------|-------|
| loss w.r.t. prediction | 2 × ({_pred:.2f} − {_target:g}) | {_loss_deriv:.2f} |
| prediction w.r.t. ht_term | 1 (addition) | 1 |
| ht_term w.r.t. w1 | {_f1} | {_f1_val:g} |

Product for {_s0['label']}: {_loss_deriv:.2f} × 1 × {_f1_val:g} = **{_chain_prod:.2f}**. Sum across all
{len(dataset.samples)} samples gives the partial derivative of total loss w.r.t. w1 = **{_total_w1:.2f}**.

{_total_w1:.2f} is {_direction}.
With a learning rate of 0.01:

> new w1 = {dataset.w1_start:g} − 0.01 × {_total_w1:.2f} = **{_new_w1:.2f}**

Loss goes down. Repeat thousands of times and the weights converge. The next
lesson runs this process automatically for both weights at once — that's the
backward pass.
""")
    return


@app.cell
def _(dataset, mo):
    _n = len(dataset.samples)
    mo.md(f"""
## The Backward Pass

In the last lesson we computed ∂L/∂w1 by hand — tracing one chain and
multiplying three local derivatives. That worked, but a real network has dozens
of weights. Doing it by hand for each one is impractical.

The **backward pass** automates the whole thing in one sweep. It starts at the
loss node (gradient = 1 by definition) and walks backwards: each node's gradient
equals its own gradient times the local derivative on each edge. Every weight's
gradient comes out in one sweep.

The table below steps through the backward pass for all {_n} {dataset.name} samples.
""")
    return


@app.cell
def _(dataset, mo):
    back_prev = mo.ui.button(label="← Prev", value=0, on_click=lambda v: v + 1)
    back_next = mo.ui.button(label="Next →", value=0, on_click=lambda v: v + 1)
    mo.hstack([back_prev, back_next], gap=1)
    return (back_next, back_prev)


@app.cell
def _(BACKWARD_NODES, back_next, back_prev, backward_pass_table, backward_step_labels, dataset, mo):
    _labels = backward_step_labels(dataset)
    _step = max(0, min(back_next.value - back_prev.value, len(BACKWARD_NODES)))
    if _step == 0:
        _explanation = mo.md("Click **Next →** to begin the backward pass.")
    else:
        _suffix = "  \n✓ Backward pass complete." if _step == len(BACKWARD_NODES) else ""
        _explanation = mo.md(f"**Step {_step} of {len(BACKWARD_NODES)}:** {_labels[_step - 1]}{_suffix}")
    mo.vstack([mo.Html(backward_pass_table(dataset.w1_start, dataset.w2_start, _step, dataset)), _explanation])
    return


@app.cell
def _(build_graph, dataset, mo):
    _total_w1, _total_w2 = 0.0, 0.0
    for _s in dataset.samples:
        _g = build_graph(_s, dataset, dataset.w1_start, dataset.w2_start)
        _g.forward_pass()
        _g.backward_pass()
        _total_w1 += _g.get_gradient("w1")
        _total_w2 += _g.get_gradient("w2")
    _new_w1 = dataset.w1_start - 0.01 * _total_w1
    _new_w2 = dataset.w2_start - 0.01 * _total_w2

    mo.md(f"""
### What do we do with the gradients?

Once the backward pass is complete, the sum row gives us the total gradient for
each weight across all {len(dataset.samples)} samples: **∂L/∂w1 = {_total_w1:.2f}** and **∂L/∂w2 = {_total_w2:.2f}**.

Applying the weight update rule with learning rate 0.01:

> new w1 = {dataset.w1_start:g} − 0.01 × {_total_w1:.2f} = **{_new_w1:.2f}**

> new w2 = {dataset.w2_start:g} − 0.01 × {_total_w2:.2f} = **{_new_w2:.2f}**

Repeat forward pass → backward pass → weight update thousands of times and the
weights converge. That's gradient descent.
""")
    return


if __name__ == "__main__":
    app.run()
