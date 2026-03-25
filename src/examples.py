#!/usr/bin/env python3
# examples.py — Factory functions for example computation graphs
# Author: Pito Salas and Claude Code
# Open Source Under MIT license

from compgraph import CompGraph
from node import Node, NodeType
from edge import Edge


def turkey_feather(length: float, width: float, w_len: float, w_wid: float, target: float) -> CompGraph:
    """Two-input, two-weight model from Chapter 4 of How They Think."""
    g = CompGraph()
    g.add_node(Node("length", NodeType.INPUT, length))
    g.add_node(Node("width", NodeType.INPUT, width))
    g.add_node(Node("w_len", NodeType.WEIGHT, w_len))
    g.add_node(Node("w_wid", NodeType.WEIGHT, w_wid))
    g.add_node(Node("len_term", NodeType.MULTIPLY, 0.0))
    g.add_node(Node("wid_term", NodeType.MULTIPLY, 0.0))
    g.add_node(Node("prediction", NodeType.ADD, 0.0))
    g.add_node(Node("loss", NodeType.LOSS, 0.0))
    g.set_target("loss", target)
    g.add_edge(Edge("length", "len_term"))
    g.add_edge(Edge("w_len", "len_term"))
    g.add_edge(Edge("width", "wid_term"))
    g.add_edge(Edge("w_wid", "wid_term"))
    g.add_edge(Edge("len_term", "prediction"))
    g.add_edge(Edge("wid_term", "prediction"))
    g.add_edge(Edge("prediction", "loss"))
    return g
