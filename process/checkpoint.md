# Checkpoint

## Where we left off

Completed F01, F02, F03. About to start F04 (first marimo sub-lesson).

## Features done

### F01 — project environment setup ✓
- uv project initialized, pyproject.toml configured
- Dependencies: marimo, matplotlib, networkx, numpy, pytest
- pytest configured with pythonpath = ["src"] in pyproject.toml
- Minimal marimo placeholder at src/main.py
- `uv run marimo run src/main.py` launches at http://localhost:2718
- `uv run pytest` runs all tests

### F02 — computation graph data model ✓
- src/node.py — Node dataclass + NodeType enum (INPUT, WEIGHT, MULTIPLY, ADD, LOSS)
- src/edge.py — Edge dataclass
- src/compgraph.py — CompGraph backed by NetworkX DiGraph
  - forward_pass: topological order, dispatch by NodeType
  - backward_pass: reverse topological, chain rule, sums gradients at multi-input nodes
  - set_target(node_id, target) for loss nodes
  - set_weight, get_value, get_gradient helpers
- src/examples.py — turkey_feather(length, width, w_len, w_wid, target) factory
- 3 tests passing

### F03 — graph visualization ✓
- src/visualizer.py — draw_graph(g, show_gradients) returns matplotlib Figure
  - layout via nx.topological_generations + nx.multipartite_layout (left to right)
  - node colors: INPUT=steelblue, WEIGHT=mediumseagreen, MULTIPLY/ADD=lightgray, LOSS=tomato
  - labels: node id, value, and optionally "grad: X.XXX"
  - directed edges with arrows
  - legend in lower right
- 2 tests passing

## Next feature: F04

First marimo sub-lesson. The notebook structure is a TOC at top linking to sequential
sections. F04 = sub-lesson 1: display the static turkey feather graph with prose
introducing the model, no controls yet.

The pattern for each sub-lesson in marimo:
  - prose cell (markdown)
  - visualization cell (calls draw_graph, marimo displays the figure)
  - interaction cell (slider or button — appears only when narrative is ready)

## Key design decisions made

- Flat src/ layout (not src/back/ nesting)
- NetworkX DiGraph as backing store for CompGraph
- draw_graph returns matplotlib Figure — no marimo dependency in visualizer
- Node names use domain language: length, width, w_len, w_wid, len_term, wid_term, prediction, loss
- Gradient labels use "grad: X.XXX" not ∇ symbol
- src/main.py is the marimo entry point; all logic in separate modules

## Test command
```bash
uv run pytest        # 6 tests, all passing
uv run marimo run src/main.py   # launches at http://localhost:2718
```
