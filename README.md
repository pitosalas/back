# back

An interactive Marimo notebook for readers who have just finished Chapter 4 of [How They Think](https://ericsilberstein1.github.io/how-they-think-book/chapter-4.html) by Eric Silberstein. Rather than re-explaining backpropagation, it lets you explore it hands-on — manipulating weights, running forward and backward passes, and watching gradients flow through a computation graph in real time.

## What it covers

The notebook is structured as sequential sub-lessons with a table of contents:

1. **The Model** — the turkey feather computation graph from the book, static and labeled
2. **The Forward Pass** — step-by-step evaluation with a button
3. **Changing a Weight** — one slider, watch loss respond
4. **The Backward Pass** — run it, see gradients appear on the graph
5. **Both Weights** — find the minimum by driving loss to zero
6. **A Slightly Bigger Model** — multiple paths, gradients sum
7. **A Full Layer** — 3 inputs, 2 hidden nodes, 1 output

Each section adds one new concept. Controls appear only when the narrative is ready for them.

## Tech stack

- [Marimo](https://marimo.io) — reactive notebook (cells update automatically when inputs change)
- [NetworkX](https://networkx.org) — computation graph structure and topological sort
- [matplotlib](https://matplotlib.org) — graph visualization
- [uv](https://docs.astral.sh/uv/) — package management

## Installation

```bash
git clone <repo url>
cd back
uv sync
```

## Usage

```bash
uv run marimo run src/main.py
```

Then open http://localhost:2718 in your browser.

## Development

```bash
uv run pytest
```

## Project structure

```
back/
├── src/
│   ├── main.py        — marimo notebook entry point
│   ├── compgraph.py   — CompGraph class (forward + backward pass)
│   ├── node.py        — Node dataclass + NodeType enum
│   ├── edge.py        — Edge dataclass
│   ├── examples.py    — turkey_feather() and other graph factories
│   └── visualizer.py  — draws a CompGraph as a matplotlib figure
├── tests/
│   ├── test_compgraph.py
│   └── test_visualizer.py
└── process/           — feature and task tracking
```

## License

MIT — see [LICENSE](LICENSE)
