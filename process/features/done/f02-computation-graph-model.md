# Feature description for feature F02
## F02 — computation graph data model
**Priority**: High
**Done:** yes
**Tests Written:** yes
**Test Passing:** yes
**Description**: Define the core data structures that represent a computation graph. All source files live in src/, including main.py. Graph structure and traversal use NetworkX DiGraph.A graph consists of nodes and edges. Each node has a type (input, weight, operation, loss), a current value, and a gradient. Each edge has a direction and carries a local derivative used during backprop. The graph must support forward pass evaluation (compute all node values from inputs and weights) and backward pass evaluation (compute all gradients from loss back to weights). This feature has no UI and no marimo dependency — it is pure data and logic, fully testable in isolation via pytest. The turkey feather model from the book must be expressible as an instance of this graph.
