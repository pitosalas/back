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
    from chain_rule import chain_forward, chain_derivs, chain_html
    return (COMPUTED_NODES, chain_derivs, chain_forward, chain_html, forward_pass_table, forward_step_label, mermaid_html, mo, turkey_feather)


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

| Turkey | Height (m) | Length (m) | Feathers (target) |
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
    _val = "padding-left:12px;"

    _html = f"""
    <div style="font-family:monospace;font-size:1.05em;line-height:2.2;margin:16px 0;">
      <table style="border-collapse:collapse;">
        <tr>
          <td style="{_lbl}">prediction</td>
          <td style="{_eq}">=</td>
          <td><span style="{_blue}">height &times; w1</span></td>
          <td style="{_eq}">+</td>
          <td><span style="{_green}">length &times; w2</span></td>
          <td></td>
        </tr>
        <tr>
          <td style="{_lbl}"></td>
          <td style="{_eq}">=</td>
          <td><span style="{_blue}">1.0 &times; 1000</span></td>
          <td style="{_eq}">+</td>
          <td><span style="{_green}">1.5 &times; 3000</span></td>
          <td></td>
        </tr>
        <tr>
          <td style="{_lbl}"></td>
          <td style="{_eq}">=</td>
          <td style="{_blue}">1,000</td>
          <td style="{_eq}">+</td>
          <td style="{_green}">4,500</td>
          <td></td>
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
