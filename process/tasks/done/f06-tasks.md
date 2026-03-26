# Tasks for Feature F06

## T01 — add highlighted node support to visualizer
**Status**: done
**Description**: Add an optional `highlighted` parameter (a node id string, or None) to draw_graph(). When set, draw that node with a bright gold border and slightly larger circle so it stands out. All other nodes render as before.

## T02 — define forward pass steps
**Status**: done
**Description**: In examples.py or a new steps.py, define a list of steps for the turkey feather forward pass. Each step is a dict with: `node` (the node id being computed), `label` (plain-English explanation, e.g. "len_term = length × w_len = 5.0 × 0.5 = 2.5"). Steps are in topological order: len_term, wid_term, prediction, loss.

## T03 — update forward pass lesson in main.py
**Status**: done
**Description**: Replace the current Run button forward pass cell with a step-by-step experience: show the equation first, then a Next button that advances through steps. Each step runs the forward pass up to that node, redraws the graph with that node highlighted, and shows the step's explanation text below the graph.

## T04 — increase graph figure size and font sizes
**Status**: done
**Description**: In visualizer.py, increase figsize to (14, 7), increase node label fontsize to 10, and increase legend fontsize to 10. Makes the graph legible without zooming.

## T05 — write tests
**Status**: done
**Description**: Add tests for draw_graph with a highlighted node: verify it returns a Figure without raising, for both a valid node id and None. Add tests for the steps list: verify correct number of steps, correct node ids, and that labels are non-empty strings.
