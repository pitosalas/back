# Feature description for feature F03
## F03 — computation graph visualization
**Priority**: High
**Done:** no
**Tests Written:** no
**Test Passing:** no
**Description**: Create src/visualizer.py that draws a CompGraph as a matplotlib figure. Nodes are colored by type (input=blue, weight=green, operation=gray, loss=red), labeled with id and current value. Edges are directed arrows. Layout uses NetworkX topological_generations to assign layers left-to-right automatically. An optional mode shows gradient values on nodes alongside forward values. Returns a matplotlib Figure — no marimo dependency. This module is the visual engine used by all sub-lessons.
