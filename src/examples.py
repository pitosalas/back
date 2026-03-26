#!/usr/bin/env python3
# examples.py — Factory functions for example computation graphs
# Author: Pito Salas and Claude Code
# Open Source Under MIT license

from compgraph import CompGraph
from node import Node, NodeType
from edge import Edge


TURKEYS = [
    {"label": "Turkey 1", "height": 1.00, "length": 1.50, "target": 5000},
    {"label": "Turkey 2", "height": 0.75, "length": 1.25, "target": 3500},
    {"label": "Turkey 3", "height": 1.25, "length": 1.00, "target": 4500},
]


def turkey_feather(height: float, length: float, w1: float, w2: float, target: float) -> CompGraph:
    """Two-input, two-weight model from Chapter 3/4 of How They Think.
    prediction = height × w1 + length × w2
    Turkey 1: height=1.0, length=1.5, w1=1000, w2=3000, target=5000
    Turkey 2: height=0.75, length=1.25, w1=1000, w2=3000, target=3500
    Turkey 3: height=1.25, length=1.0, w1=1000, w2=3000, target=4500
    """
    g = CompGraph()
    g.add_node(Node("height", NodeType.INPUT, height))
    g.add_node(Node("length", NodeType.INPUT, length))
    g.add_node(Node("w1", NodeType.WEIGHT, w1))
    g.add_node(Node("w2", NodeType.WEIGHT, w2))
    g.add_node(Node("ht_term", NodeType.MULTIPLY, 0.0))
    g.add_node(Node("len_term", NodeType.MULTIPLY, 0.0))
    g.add_node(Node("prediction", NodeType.ADD, 0.0))
    g.add_node(Node("loss", NodeType.LOSS, 0.0))
    g.set_target("loss", target)
    g.add_edge(Edge("height", "ht_term"))
    g.add_edge(Edge("w1", "ht_term"))
    g.add_edge(Edge("length", "len_term"))
    g.add_edge(Edge("w2", "len_term"))
    g.add_edge(Edge("ht_term", "prediction"))
    g.add_edge(Edge("len_term", "prediction"))
    g.add_edge(Edge("prediction", "loss"))
    return g
