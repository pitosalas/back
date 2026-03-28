#!/usr/bin/env python3
# table_viz.py — Renders forward pass for all turkeys as an HTML table
# Author: Pito Salas and Claude Code
# Open Source Under MIT license

from examples import turkey_feather, TURKEYS


GOLD = "#FFD700"
HEADER_BG = "#f0f0f0"
INPUT_NODES = ["height", "length", "w1", "w2"]
COMPUTED_NODES = ["ht_term", "len_term", "prediction", "loss"]
ALL_NODES = INPUT_NODES + COMPUTED_NODES

NODE_LABELS = {
    "ht_term": "height×w1",
    "len_term": "length×w2",
    "prediction": "prediction",
    "loss": "loss",
}


def _cell(value: str, bg: str, bold: bool) -> str:
    weight = "font-weight:bold;" if bold else ""
    return f'<td style="text-align:center;padding:6px 10px;background:{bg};{weight}">{value}</td>'


def _header_row(step: int) -> str:
    cells = ['<th style="padding:6px 10px;background:#ddd;text-align:left">Turkey</th>']
    for i, node in enumerate(ALL_NODES):
        computed_idx = COMPUTED_NODES.index(node) + 1 if node in COMPUTED_NODES else None
        bg = GOLD if computed_idx == step else HEADER_BG
        label = NODE_LABELS.get(node, node)
        cells.append(f'<th style="padding:6px 10px;background:{bg};text-align:center">{label}</th>')
    return "<tr>" + "".join(cells) + "</tr>"


def _turkey_row(t: dict, w1: float, w2: float, step: int) -> str:
    g = turkey_feather(height=t["height"], length=t["length"], w1=w1, w2=w2, target=t["target"])
    g.forward_pass_n(step)

    cells = [f'<td style="padding:6px 10px;font-weight:bold;background:#f8f8f8">{t["label"]}</td>']
    for node in ALL_NODES:
        computed_idx = COMPUTED_NODES.index(node) + 1 if node in COMPUTED_NODES else None
        is_active = computed_idx == step
        bg = GOLD if is_active else "white"
        if node in INPUT_NODES:
            val = f"{g.get_value(node):g}"
            cells.append(_cell(val, bg, False))
        elif computed_idx is not None and computed_idx <= step:
            val = f"{g.get_value(node):,.1f}"
            cells.append(_cell(val, bg, is_active))
        else:
            cells.append(_cell("—", bg, False))
    return "<tr>" + "".join(cells) + "</tr>"


def _total_loss_row(w1: float, w2: float, step: int) -> str:
    loss_idx = COMPUTED_NODES.index("loss") + 1
    if step < loss_idx:
        return ""
    total = 0.0
    for t in TURKEYS:
        g = turkey_feather(height=t["height"], length=t["length"], w1=w1, w2=w2, target=t["target"])
        g.forward_pass_n(step)
        total += g.get_value("loss")
    n_spacer = len(ALL_NODES) - 1
    label = '<td style="padding:6px 10px;font-weight:bold;background:#f0d0d0">Sum of squares<br/><span style="font-weight:normal;font-size:12px">Σ(pred−target)²</span></td>'
    spacer = f'<td colspan="{n_spacer}" style="padding:6px 10px;background:#fafafa"></td>'
    total_cell = f'<td style="text-align:center;padding:6px 10px;font-weight:bold;background:#ffe0e0">{total:,.1f}</td>'
    return "<tr>" + label + spacer + total_cell + "</tr>"


BACKWARD_NODES = ["loss", "prediction", "ht_term", "len_term", "w1", "w2"]

BACKWARD_LABELS = {
    "loss": "loss",
    "prediction": "prediction",
    "ht_term": "height×w1",
    "len_term": "length×w2",
    "w1": "w1",
    "w2": "w2",
}

BACKWARD_STEP_LABELS = [
    "**loss** gradient = 1. The backward pass asks: how much does each node affect the loss? "
        "The loss node *is* the loss, so ∂L/∂loss = 1 — a one-unit nudge to the loss node changes loss by exactly one unit. "
        "This seed value of 1 is what the rest of the backward pass multiplies through.",
    "**prediction** gradient = loss_grad × 2(pred − actual). "
        "loss = (pred − actual)², so its local derivative w.r.t. prediction is 2(pred − actual) — "
        "the power rule: d/dx[x²] = 2x, where x = (pred − actual). "
        "For Turkey 1: 1 × 2 × (5500 − 5000) = 1 × 2 × 500 = **1000**. "
        "Each turkey differs because each has a different prediction error.",
    "**ht_term** gradient = prediction_grad × 1. Addition passes the gradient through unchanged.",
    "**len_term** gradient = prediction_grad × 1. Same rule — addition local derivative is always 1.",
    "**w1** gradient = ht_term_grad × height. The sum row shows the total across all three turkeys.",
    "**w2** gradient = len_term_grad × length. Backward pass complete — both weight gradients ready.",
]


def _back_header_row(step: int) -> str:
    cells = [
        '<th style="padding:6px 10px;background:#ddd;text-align:left">Turkey</th>',
        f'<th style="padding:6px 10px;background:{HEADER_BG};text-align:center">actual</th>',
    ]
    for i, node in enumerate(BACKWARD_NODES):
        bg = GOLD if (i + 1) == step else HEADER_BG
        cells.append(f'<th style="padding:6px 10px;background:{bg};text-align:center">{BACKWARD_LABELS[node]}</th>')
    return "<tr>" + "".join(cells) + "</tr>"


def _back_turkey_row(t: dict, w1: float, w2: float, step: int) -> str:
    g = turkey_feather(height=t["height"], length=t["length"], w1=w1, w2=w2, target=t["target"])
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
            cells.append(_cell(f"{g.get_gradient(node):,.1f}", bg, is_active))
        else:
            cells.append(_cell("—", bg, False))
    return "<tr>" + "".join(cells) + "</tr>"


def _back_sum_row(w1: float, w2: float, step: int) -> str:
    w1_col = BACKWARD_NODES.index("w1") + 1
    w2_col = BACKWARD_NODES.index("w2") + 1
    if step < w1_col:
        return ""
    total_w1, total_w2 = 0.0, 0.0
    for t in TURKEYS:
        g = turkey_feather(height=t["height"], length=t["length"], w1=w1, w2=w2, target=t["target"])
        g.forward_pass()
        g.backward_pass()
        total_w1 += g.get_gradient("w1")
        total_w2 += g.get_gradient("w2")
    label = '<td style="padding:6px 10px;font-weight:bold;background:#f0d0d0">Sum (all turkeys)</td>'
    # +1 for the extra actual column
    spacer = f'<td colspan="{w1_col}" style="padding:6px 10px;background:#fafafa"></td>'
    w1_cell = f'<td style="text-align:center;padding:6px 10px;font-weight:bold;background:#ffe0e0">{total_w1:,.1f}</td>'
    w2_cell = (f'<td style="text-align:center;padding:6px 10px;font-weight:bold;background:#ffe0e0">{total_w2:,.1f}</td>'
               if step >= w2_col else "")
    return "<tr>" + label + spacer + w1_cell + w2_cell + "</tr>"


def backward_pass_table(w1: float, w2: float, step: int) -> str:
    rows = [_back_header_row(step)]
    for t in TURKEYS:
        rows.append(_back_turkey_row(t, w1, w2, step))
    sum_row = _back_sum_row(w1, w2, step)
    if sum_row:
        rows.append(sum_row)
    inner = "\n".join(rows)
    return f'<table style="border-collapse:collapse;font-size:14px;width:100%">{inner}</table>'


def forward_pass_table(w1: float, w2: float, step: int) -> str:
    rows = [_header_row(step)]
    for t in TURKEYS:
        rows.append(_turkey_row(t, w1, w2, step))
    total_row = _total_loss_row(w1, w2, step)
    if total_row:
        rows.append(total_row)
    inner = "\n".join(rows)
    return (
        f'<table style="border-collapse:collapse;font-size:14px;width:100%">'
        f'{inner}</table>'
    )
