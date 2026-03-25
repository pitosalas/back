#!/usr/bin/env python3
# visualizer.py — Draws a CompGraph as a matplotlib figure
# Author: Pito Salas and Claude Code
# Open Source Under MIT license

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from node import NodeType
from compgraph import CompGraph


NODE_COLORS = {
    NodeType.INPUT: "steelblue",
    NodeType.WEIGHT: "mediumseagreen",
    NodeType.MULTIPLY: "lightgray",
    NodeType.ADD: "lightgray",
    NodeType.LOSS: "tomato",
}


def draw_graph(g: CompGraph, show_gradients: bool) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    pos = _layout(g)
    _draw_edges(ax, g, pos)
    _draw_nodes(ax, g, pos, show_gradients)
    _draw_legend(ax)
    ax.axis("off")
    fig.tight_layout()
    return fig


def _layout(g: CompGraph) -> dict:
    for layer, nodes in enumerate(nx.topological_generations(g.graph)):
        for node_id in nodes:
            g.graph.nodes[node_id]["layer"] = layer
    return nx.multipartite_layout(g.graph, subset_key="layer", align="vertical")


def _node_color(node_type: NodeType) -> str:
    return NODE_COLORS[node_type]


def _draw_nodes(ax: plt.Axes, g: CompGraph, pos: dict, show_gradients: bool):
    for node_id, (x, y) in pos.items():
        attrs = g.graph.nodes[node_id]
        color = _node_color(attrs["node_type"])
        circle = mpatches.Circle((x, y), radius=0.08, color=color, zorder=3)
        ax.add_patch(circle)
        label = f"{node_id}\n{attrs['value']:.3f}"
        if show_gradients:
            label += f"\ngrad: {attrs['gradient']:.3f}"
        ax.text(x, y, label, ha="center", va="center", fontsize=7, zorder=4)


def _draw_edges(ax: plt.Axes, g: CompGraph, pos: dict):
    nx.draw_networkx_edges(
        g.graph,
        pos,
        ax=ax,
        arrows=True,
        arrowstyle="->",
        arrowsize=15,
        edge_color="dimgray",
        node_size=1200,
    )


def _draw_legend(ax: plt.Axes):
    patches = [
        mpatches.Patch(color="steelblue", label="input"),
        mpatches.Patch(color="mediumseagreen", label="weight"),
        mpatches.Patch(color="lightgray", label="operation"),
        mpatches.Patch(color="tomato", label="loss"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=8)
