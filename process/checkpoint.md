# Checkpoint

## Current state (session 2)

Three lessons are working and deployed at https://pitosalas.github.io/back/

### What's built and working

**Lesson 1 — The Model**
- Mermaid diagram of turkey feather graph rendered via iframe (to avoid WASM script issues)
- Forward pass runs on load so actual values are shown (not 0s)
- Prose explains the dataset and the weighted sum formula

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
  mermaid_viz.py  ← mermaid_html(g, show_gradients, highlighted) → HTML string
                     renders via iframe to work in both local and WASM contexts
  table_viz.py    ← forward_pass_table(w1, w2, step) → HTML string
                     COMPUTED_NODES = ["ht_term", "len_term", "prediction", "loss"]
  visualizer.py   ← matplotlib draw_graph(g, show_gradients, highlighted)
                     kept intact but not used in current lessons
```

### Turkey dataset
- Turkey 1: height=1.0, length=1.5, target=5000
- Turkey 2: height=0.75, length=1.25, target=3500
- Turkey 3: height=1.25, length=1.0, target=4500
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

### Tests (20 passing)
- test_compgraph.py — forward pass values, backward pass gradients, zero at minimum
- test_visualizer.py — draw_graph returns Figure, highlighted node, step labels, node order
- test_mermaid_viz.py — build_mermaid contains nodes/edges/highlight, iframe wrapper
- test_table_viz.py — table contains labels/headers/dashes, step computation, w1 changes

### Next lessons to build
- F11: The Backward Pass — run it, see gradients appear on the graph
- F12: Both Weights — two sliders, minimize total loss, connect to the idea of gradient descent

### Design decisions
- Ditched matplotlib for graph display (text too small) → switched to Mermaid
- Ditched per-turkey graph step-through → switched to 3-row table (more compact, shows all turkeys at once)
- Kept visualizer.py intact so graph approach can be revived
- table_viz and mermaid_viz are independent modules — easy to swap
- No default parameters anywhere (coding rules)
- All functions ≤ 50 lines, files ≤ 300 lines
