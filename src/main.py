#!/usr/bin/env python3
# main.py — Marimo notebook entry point for the backpropagation explorer
# Author: Pito Salas and Claude Code
# Open Source Under MIT license

import marimo

app = marimo.App()


@app.cell
def _():
    import marimo as mo
    from examples import TURKEY_DATASET, CARS_DATASET, build_graph
    from steps import forward_step_label
    from table_viz import forward_pass_table, backward_pass_table, backward_step_labels, COMPUTED_NODES, BACKWARD_NODES
    from chain_rule import chain_html
    return (BACKWARD_NODES, CARS_DATASET, COMPUTED_NODES, TURKEY_DATASET, backward_pass_table, backward_step_labels, build_graph, chain_html, forward_pass_table, forward_step_label, mo)


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
