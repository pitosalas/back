# Tasks for Feature F02

## T01 — define Node dataclass
**Status**: done
**Description**: Create src/node.py with a Node dataclass. Fields: id (str), node_type (enum: input | weight | operation | loss), value (float), gradient (float).

## T02 — define Edge dataclass
**Status**: done
**Description**: Create src/edge.py with an Edge dataclass. Fields: from_id (str), to_id (str), local_deriv (float).

## T03 — define CompGraph class
**Status**: done
**Description**: Create src/compgraph.py with a CompGraph class backed by a NetworkX DiGraph. Supports add_node and add_edge methods. Node and edge attributes stored in the NetworkX graph.

## T04 — implement forward pass
**Status**: done
**Description**: Add forward_pass method to CompGraph. Use NetworkX topological sort to evaluate nodes in order, computing each operation node's value from its inputs. Support multiply and add operations initially.

## T05 — implement backward pass
**Status**: done
**Description**: Add backward_pass method to CompGraph. Starting from the loss node (gradient=1), walk nodes in reverse topological order propagating gradients via chain rule. Use NetworkX to find predecessors and sum gradients at nodes with multiple incoming edges.

## T06 — encode turkey feather example
**Status**: done
**Description**: Create src/examples.py with a turkey_feather() factory function that returns a CompGraph representing the book's model (2 inputs, 2 weights, multiply ops, add, loss). This is the canonical test fixture.

## T07 — write tests
**Status**: done
**Description**: Write pytest tests in tests/test_compgraph.py. Import directly from src.compgraph and src.examples — no marimo dependency. Cover: forward pass produces correct values, backward pass produces correct gradients matching the book, gradients are zero at the known minimum.
