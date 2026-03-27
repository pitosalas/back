# Tasks for Feature F11

## T01 — design the (x²)³ computation graph
**Status**: not done
**Description**: Define the two-node chain graph in Python (or as a simple data structure): node A squares the input, node B cubes the result. Decide whether to reuse CompGraph or represent it inline. Keep it minimal — this graph is never trained, just illustrated.

## T02 — add prose cell explaining chain rule
**Status**: not done
**Description**: Add a markdown cell introducing the chain rule concept. Use the (x²)³ example: show the two steps, explain local derivative at each node, and that the overall derivative is the product of local derivatives going right to left.

## T03 — add interactive forward pass cell
**Status**: not done
**Description**: Add a mo.ui.number or slider for x (default 3). Show the forward pass values at each node (x²=9, (x²)³=729) reactively. Label each node with its local derivative at the current value.

## T04 — add prose cell on partial derivatives
**Status**: not done
**Description**: Add a markdown cell explaining partial derivatives: holding w2 fixed and nudging w1 — what happens to loss? Connect this to the chain rule: the gradient flows back through the graph to each weight.

## T05 — write tests
**Status**: not done
**Description**: Test that the (x²)³ forward values are correct for x=3 (A=9, B=729). Test that local derivatives are correct (dA/dx=6, dB/dA=243, overall=1458).
