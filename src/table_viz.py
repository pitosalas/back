#!/usr/bin/env python3
# table_viz.py — Renders forward and backward pass for all samples as HTML tables
# Author: Pito Salas and Claude Code
# Open Source Under MIT license

from examples import build_graph, Dataset


GOLD = "#FFD700"
HEADER_BG = "#f0f0f0"
COMPUTED_NODES = ["ht_term", "len_term", "prediction", "loss"]
BACKWARD_NODES = ["loss", "prediction", "ht_term", "len_term", "w1", "w2"]


def _cell(value: str, bg: str, bold: bool) -> str:
    weight = "font-weight:bold;" if bold else ""
    return f'<td style="text-align:center;padding:6px 10px;background:{bg};{weight}">{value}</td>'


def _header_row(step: int, dataset: Dataset) -> str:
    f1, f2 = dataset.feat1_name, dataset.feat2_name
    all_nodes = [f1, f2, "w1", "w2"] + COMPUTED_NODES
    labels = {f1: f1, f2: f2, "w1": "w1", "w2": "w2",
              "ht_term": f"{f1}×w1", "len_term": f"{f2}×w2",
              "prediction": "prediction", "loss": "loss"}
    cells = ['<th style="padding:6px 10px;background:#ddd;text-align:left">Sample</th>']
    for node in all_nodes:
        ci = COMPUTED_NODES.index(node) + 1 if node in COMPUTED_NODES else None
        bg = GOLD if ci == step else HEADER_BG
        cells.append(f'<th style="padding:6px 10px;background:{bg};text-align:center">{labels[node]}</th>')
    return "<tr>" + "".join(cells) + "</tr>"


def _sample_row(t: dict, w1: float, w2: float, step: int, dataset: Dataset) -> str:
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


def _total_loss_row(w1: float, w2: float, step: int, dataset: Dataset) -> str:
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


def _back_header_row(step: int, dataset: Dataset) -> str:
    f1, f2 = dataset.feat1_name, dataset.feat2_name
    back_labels = {"loss": "loss", "prediction": "prediction",
                   "ht_term": f"{f1}×w1", "len_term": f"{f2}×w2", "w1": "w1", "w2": "w2"}
    cells = [
        '<th style="padding:6px 10px;background:#ddd;text-align:left">Sample</th>',
        f'<th style="padding:6px 10px;background:{HEADER_BG};text-align:center">actual</th>',
    ]
    for i, node in enumerate(BACKWARD_NODES):
        bg = GOLD if (i + 1) == step else HEADER_BG
        cells.append(f'<th style="padding:6px 10px;background:{bg};text-align:center">{back_labels[node]}</th>')
    return "<tr>" + "".join(cells) + "</tr>"


def _back_sample_row(t: dict, w1: float, w2: float, step: int, dataset: Dataset) -> str:
    g = build_graph(t, dataset, w1, w2)
    g.forward_pass()
    g.backward_pass()
    cells = [
        f'<td style="padding:6px 10px;font-weight:bold;background:#f8f8f8">{t["label"]}</td>',
        _cell(f"{t['target']:,g}", HEADER_BG, False),
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


def _back_sum_row(w1: float, w2: float, step: int, dataset: Dataset) -> str:
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


def backward_step_labels(dataset: Dataset) -> list:
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


def forward_pass_table(w1: float, w2: float, step: int, dataset: Dataset) -> str:
    rows = [_header_row(step, dataset)]
    for t in dataset.samples:
        rows.append(_sample_row(t, w1, w2, step, dataset))
    total_row = _total_loss_row(w1, w2, step, dataset)
    if total_row:
        rows.append(total_row)
    return f'<table style="border-collapse:collapse;font-size:14px;width:100%">{"".join(rows)}</table>'


def backward_pass_table(w1: float, w2: float, step: int, dataset: Dataset) -> str:
    rows = [_back_header_row(step, dataset)]
    for t in dataset.samples:
        rows.append(_back_sample_row(t, w1, w2, step, dataset))
    sum_row = _back_sum_row(w1, w2, step, dataset)
    if sum_row:
        rows.append(sum_row)
    return f'<table style="border-collapse:collapse;font-size:14px;width:100%">{"".join(rows)}</table>'
