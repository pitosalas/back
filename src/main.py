#!/usr/bin/env python3
# main.py — Marimo notebook entry point for the backpropagation explorer
# Author: Pito Salas and Claude Code
# Open Source Under MIT license

import marimo

app = marimo.App()


@app.cell
def _():
    import marimo as mo
    from mermaid_viz import mermaid_html
    from examples import turkey_feather
    from steps import forward_step_label
    from table_viz import forward_pass_table, COMPUTED_NODES
    return (COMPUTED_NODES, forward_pass_table, forward_step_label, mermaid_html, mo, turkey_feather)


@app.cell
def _(mo):
    mo.md("""
# Backpropagation Explorer

Each cell below is one lesson. Work through them top to bottom.

| Cell | Lesson |
|------|--------|
| 3 | **The Model** — the turkey feather computation graph |
| 4 | **The Forward Pass** — step through the computation one node at a time |
| 5 | **Changing a Weight** — use the slider, watch loss respond |
| 6 | **The Backward Pass** — click Run to see gradients |
| 7 | **Both Weights** — find the minimum |
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

We start with initial weights **w1 = 1000, w2 = 3000**. The graph below shows
the computation structure — inputs (blue) and weights (green) flow into multiply
nodes, which feed the prediction (add), which feeds the loss.
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
## The Forward Pass

The table below shows all three turkeys at once. Each column is a node in the
computation graph. Click **Next →** to compute the next node for all three
turkeys simultaneously. The gold column is the one being computed.

The first two computed columns are **height×w1** and **length×w2** — each input
multiplied by its weight. These are the two *terms* of the weighted sum. w1 and
w2 are the knobs the model will eventually learn to turn: a higher w1 means
height matters more, a higher w2 means length matters more.

**prediction** adds the two terms together — that's the model's guess at the
feather count. **loss** measures how wrong that guess is: (prediction − target)².
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


if __name__ == "__main__":
    app.run()
