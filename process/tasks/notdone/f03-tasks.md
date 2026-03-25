# Tasks for Feature F03

## T01 — assign layout positions
**Status**: not done
**Description**: In src/visualizer.py, write a _layout(g) function that uses nx.topological_generations to assign a layer index to each node, then calls nx.multipartite_layout to produce x/y positions. Returns a dict of {node_id: (x, y)}.

## T02 — define node colors by type
**Status**: not done
**Description**: Write a _node_color(node_type) function that returns a color string per NodeType: INPUT=steelblue, WEIGHT=mediumseagreen, MULTIPLY/ADD=lightgray, LOSS=tomato.

## T03 — draw nodes
**Status**: not done
**Description**: Write a _draw_nodes(ax, g, pos, show_gradients) function. Draws each node as a circle using nx.draw_networkx_nodes. Labels show node id and value. If show_gradients is True, also show gradient below the value.

## T04 — draw edges
**Status**: not done
**Description**: Write a _draw_edges(ax, g, pos) function. Draws directed edges with arrows using nx.draw_networkx_edges with arrowstyle='->' .

## T05 — assemble draw_graph function
**Status**: not done
**Description**: Write the public draw_graph(g, show_gradients) function. Creates a matplotlib Figure and Axes, calls _layout, _draw_nodes, _draw_edges, sets axis off, returns the Figure.

## T06 — write tests
**Status**: not done
**Description**: Write tests in tests/test_visualizer.py. Verify draw_graph returns a matplotlib Figure without raising, for both show_gradients=True and show_gradients=False, using the turkey_feather fixture after forward and backward pass.
