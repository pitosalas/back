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
