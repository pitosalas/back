# Tasks for Feature F04

## T01 — add sys.path setup cell
**Status**: done
**Description**: Add a Marimo cell at the top of main.py that appends src/ to sys.path so imports work when run via `uv run marimo run src/main.py`.

## T02 — add imports cell
**Status**: done
**Description**: Add a Marimo cell that imports marimo as mo, and imports draw_graph from visualizer and turkey_feather from examples.

## T03 — add prose cell
**Status**: done
**Description**: Add a Marimo cell that returns mo.md(...) with a short intro: title "The Model", 2-3 sentences explaining the turkey feather graph — inputs (length, width), weights (w_len, w_wid), multiply nodes, prediction (add), and loss.

## T04 — add graph cell
**Status**: done
**Description**: Add a Marimo cell that builds a turkey_feather graph with representative values (length=5.0, width=3.0, w_len=0.5, w_wid=0.4, target=4.0), calls draw_graph(g, False), and returns mo.pyplot(fig).

## T05 — write tests
**Status**: done
**Description**: No unit tests needed for main.py (Marimo UI cells are not testable with pytest). Verify manually by running `uv run marimo run src/main.py` and confirming the prose and graph appear correctly.
