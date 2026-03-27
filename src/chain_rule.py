#!/usr/bin/env python3
# chain_rule.py — Forward computation and local derivatives for f(x) = (x²)³
# Author: Pito Salas and Claude Code
# Open Source Under MIT license


def chain_forward(x: float) -> tuple[float, float]:
    """Returns (a, b) where a = x², b = a³."""
    a = x ** 2
    b = a ** 3
    return a, b


def chain_derivs(x: float) -> tuple[float, float, float]:
    """Returns (da_dx, db_da, db_dx) — local and overall derivatives."""
    a = x ** 2
    da_dx = 2 * x
    db_da = 3 * (a ** 2)
    return da_dx, db_da, da_dx * db_da


def chain_html(x: float) -> str:
    """Returns an HTML string visualizing the chain rule for f(x) = (x²)³."""
    a, b = chain_forward(x)
    da_dx, db_da, db_dx = chain_derivs(x)

    node_style = (
        "display:inline-block;padding:10px 18px;border-radius:8px;"
        "font-size:1.1em;font-weight:bold;text-align:center;"
    )
    input_style = node_style + "background:#cce5ff;border:2px solid #4a90d9;"
    op_style = node_style + "background:#f0f0f0;border:2px solid #aaa;font-size:0.9em;"
    output_style = node_style + "background:#d4edda;border:2px solid #28a745;"
    arrow_style = (
        "display:inline-block;vertical-align:middle;"
        "text-align:center;margin:0 6px;"
    )

    def arrow(label: str) -> str:
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
        f'<span style="{output_style}">output<br>f(x) = {b:g}</span>'
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
