# back — Backpropagation Explorer

An interactive Marimo notebook for readers who have just finished Chapters 3–4 of
[How They Think](https://ericsilberstein1.github.io/how-they-think-book/chapter-4.html)
by Eric Silberstein. Rather than re-explaining backpropagation, it lets you explore
it hands-on — stepping through computations, manipulating weights, and watching loss
respond in real time across all three turkeys from the book.

**Live demo:** https://pitosalas.github.io/back/

## What it covers

The notebook is structured as sequential lessons:

1. **The Model** — the turkey feather computation graph with actual computed values
2. **Computing Loss** — step through the forward pass one node at a time for all three turkeys simultaneously, with Next/Prev buttons
3. **Changing a Weight** — w1 and w2 sliders; every prediction and the sum-of-squares total loss update instantly

Planned lessons (not yet built):
4. **The Backward Pass** — run it, see gradients
5. **Both Weights** — find the minimum by driving total loss to zero

## The turkey dataset

From Table 3.1 in the book (fictional data):

| Turkey | Height (m) | Length (m) | Feathers (target) |
|--------|-----------|-----------|-------------------|
| 1      | 1.00      | 1.50      | 5,000             |
| 2      | 0.75      | 1.25      | 3,500             |
| 3      | 1.25      | 1.00      | 4,500             |

Starting weights: w1 = 1000, w2 = 3000. Book's converged values: w1 ≈ 2311, w2 ≈ 1633.

## Tech stack

- [Marimo](https://marimo.io) — reactive notebook
- [NetworkX](https://networkx.org) — computation graph and topological sort
- [Mermaid](https://mermaid.js.org) — graph diagrams rendered in the browser
- [uv](https://docs.astral.sh/uv/) — package management

## Run locally

```bash
git clone https://github.com/pitosalas/back
cd back
uv sync
uv run marimo run src/main.py
```

Then open http://localhost:2718 in your browser.

## Tests

```bash
uv run pytest
```

## Project structure

```
back/
├── src/
│   ├── main.py        — marimo notebook (development)
│   ├── main_wasm.py   — single-file build for GitHub Pages (all modules inlined)
│   ├── compgraph.py   — CompGraph: forward pass, backward pass, topological sort
│   ├── node.py        — Node dataclass + NodeType enum
│   ├── edge.py        — Edge dataclass
│   ├── examples.py    — TURKEYS dataset + turkey_feather() factory
│   ├── steps.py       — plain-English forward pass step labels
│   ├── mermaid_viz.py — Mermaid diagram renderer via iframe
│   ├── table_viz.py   — HTML table showing all 3 turkeys + sum-of-squares row
│   └── visualizer.py  — matplotlib graph renderer (kept for reference)
├── docs/              — GitHub Pages output (generated, do not edit)
├── tests/
│   ├── test_compgraph.py
│   ├── test_visualizer.py
│   ├── test_mermaid_viz.py
│   └── test_table_viz.py
└── process/           — feature and task tracking
```

## Deployment

The GitHub Pages build uses `src/main_wasm.py` — a standalone single-file version
with all local modules inlined, so Pyodide (the browser Python runtime) can run it
without needing filesystem access. To rebuild after changes:

```bash
uv run marimo export html-wasm src/main_wasm.py -o docs --mode run -f
git add docs/ && git commit -m "Rebuild WASM" && git push
```

**Important:** when making functional changes, update both `src/main.py` (and the
relevant module files) AND inline the same changes into `src/main_wasm.py`.

## License

MIT — see [LICENSE](LICENSE)
