# Checkpoint

## Current state (session 3)

Four lessons are working and deployed at https://pitosalas.github.io/back/

### What's built and working

**Lesson 1 — The Model**
- Annotated step-by-step equation for Turkey 1 (replaced Mermaid graph)
- Height term in blue, length term in green
- Shows full computation: prediction = 1000 + 4500 = 5500, loss = 500² = 250,000
- Prose explains dataset and weighted sum formula
- Uses "actual" instead of "target" throughout

**Lesson 2 — Computing Loss**
- HTML table showing all 3 turkeys as rows, computation nodes as columns
- Next/Prev buttons step through forward pass one node at a time for all turkeys
- Gold column highlights the active computation
- `—` for uncomputed cells
- Step explanation below the table (uses forward_step_label)
- Sum-of-squares total row appears once loss column is reached

**Lesson 3 — Changing a Weight**
- w1 slider (0–5000, step 50, default 1000) and w2 slider (default 3000)
- Table rerenders reactively — no button needed
- Sum-of-squares visible at all times
- Prose hints at converged values (w1≈2311, w2≈1633)

**Lesson 4 — The Chain Rule**
- Prose introduces chain rule via f(x) = (x²)³ example
- Interactive slider for x (1–5, step 0.5, default 3)
- Visual display: nodes with forward values and local derivatives, chain rule product
- "From chain rule to partial derivatives" section:
  - Shows the w1 → ht_term → prediction → loss chain
  - Table of local derivatives with actual numbers at w1=1000, w2=3000
  - Shows partial derivative of total loss w.r.t. w1 = 1875
  - Shows one gradient descent step: new w1 = 1000 − 0.01×1875 = 981
  - Teases the backward pass lesson

### Source file structure

```
src/
  main.py         ← development entry point (imports local modules)
  main_wasm.py    ← deployment build (all modules inlined for Pyodide)
  compgraph.py    ← CompGraph with forward_pass, forward_pass_n, backward_pass,
                     computed_node_ids
  node.py         ← Node dataclass, NodeType enum
  edge.py         ← Edge dataclass
  examples.py     ← TURKEYS list (3 dicts), turkey_feather() factory
  steps.py        ← forward_step_label(g, node_id) → markdown string
  chain_rule.py   ← chain_forward(x), chain_derivs(x), chain_html(x)
  mermaid_viz.py  ← mermaid_html(g, show_gradients, highlighted) → HTML string
                     renders via iframe; kept for F12 backward pass lesson
  table_viz.py    ← forward_pass_table(w1, w2, step) → HTML string
                     COMPUTED_NODES = ["ht_term", "len_term", "prediction", "loss"]
  visualizer.py   ← matplotlib draw_graph(g, show_gradients, highlighted)
                     kept intact but not used in current lessons
```

### Turkey dataset
- Turkey 1: height=1.0, length=1.5, actual=5000
- Turkey 2: height=0.75, length=1.25, actual=3500
- Turkey 3: height=1.25, length=1.0, actual=4500
- Formula: prediction = height × w1 + length × w2
- Starting weights: w1=1000, w2=3000
- Book converged: w1≈2311, w2≈1633; optimal: w1=2347, w2=1604

### Node naming
- Input nodes: height, length
- Weight nodes: w1, w2
- Multiply nodes: ht_term (height×w1), len_term (length×w2)
- Add node: prediction
- Loss node: loss

### Key Marimo patterns learned
- Cell output = last bare expression (NOT assigned to variable, NOT in return)
- return (x, y) exports named variables to other cells
- UI elements: assign to variable, use as bare expression to display, return in tuple
- mo.ui.button(value=0, on_click=lambda v: v+1) for counter
- Two buttons (next/prev): step = max(0, min(next.value - prev.value, max_step))
- mo.vstack([...]) to combine multiple elements as one cell output
- mo.Html(...) for raw HTML
- WASM export: local Python imports don't work — must inline everything into one file

### WASM deployment
- Export: `uv run marimo export html-wasm src/main_wasm.py -o docs --mode run -f`
- GitHub Pages: repo → Settings → Pages → branch: main, folder: /docs
- Live URL: https://pitosalas.github.io/back/
- Pain point: main_wasm.py must be manually kept in sync with module files

### Tests (27 passing)
- test_compgraph.py — forward pass values, backward pass gradients, zero at minimum
- test_visualizer.py — draw_graph returns Figure, highlighted node, step labels, node order
- test_mermaid_viz.py — build_mermaid contains nodes/edges/highlight, iframe wrapper
- test_table_viz.py — table contains labels/headers/dashes, step computation, w1 changes
- test_chain_rule.py — forward values, local derivatives, chain rule product, html output

### Next lesson to build
- F12: The Backward Pass — run backward_pass on turkey graph, show gradients on
  Mermaid diagram, connect gradient values to weight update rule

### Design decisions
- Ditched matplotlib for graph display (text too small) → switched to Mermaid
- Ditched Mermaid for The Model lesson (not a real network) → switched to annotated equation
- Ditched per-turkey graph step-through → switched to 3-row table (more compact)
- Kept visualizer.py and mermaid_viz.py intact for potential use in F12
- table_viz and mermaid_viz are independent modules — easy to swap
- Use "actual" not "target" for known feather counts (clearer to readers)
- No default parameters anywhere (coding rules)
- All functions ≤ 50 lines, files ≤ 300 lines
