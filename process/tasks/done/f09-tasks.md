# Tasks for Feature F09

## T01 — create table_viz.py
**Status**: done
**Description**: Write src/table_viz.py with forward_pass_table(turkeys, w1, w2, step) that returns an HTML string. Columns are all nodes in topological order. Rows are the three turkeys. Input/weight columns always show their values. Computed columns show the value if step has reached them, "—" otherwise. The active column (current step) is highlighted gold. Takes turkeys (list of dicts), w1, w2, step (int).

## T02 — update main.py forward pass lesson
**Status**: done
**Description**: Replace the turkey radio + per-turkey graph cells with: a Next button, the table cell (reacts to Next), and a step explanation line. Keep the static model graph above unchanged.

## T03 — write tests
**Status**: done
**Description**: Test that forward_pass_table returns an HTML string containing all turkey labels and node ids. Test that uncomputed cells show "—" and computed cells show numeric values at the correct step.
