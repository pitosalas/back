#!/usr/bin/env python3
# examples.py — Factory functions for example computation graphs
# Author: Pito Salas and Claude Code
# Open Source Under MIT license

from compgraph import CompGraph
from node import Node, NodeType
from edge import Edge


def turkey_feather(x1: float, x2: float, w1: float, w2: float, target: float) -> CompGraph:
    """Two-input, two-weight model from Chapter 4 of How They Think."""
    g = CompGraph()
    g.add_node(Node("x1", NodeType.INPUT, x1))
    g.add_node(Node("x2", NodeType.INPUT, x2))
    g.add_node(Node("w1", NodeType.WEIGHT, w1))
    g.add_node(Node("w2", NodeType.WEIGHT, w2))
    g.add_node(Node("mul1", NodeType.MULTIPLY, 0.0))
    g.add_node(Node("mul2", NodeType.MULTIPLY, 0.0))
    g.add_node(Node("pred", NodeType.ADD, 0.0))
    g.add_node(Node("loss", NodeType.LOSS, 0.0))
    g.set_target("loss", target)
    g.add_edge(Edge("x1", "mul1"))
    g.add_edge(Edge("w1", "mul1"))
    g.add_edge(Edge("x2", "mul2"))
    g.add_edge(Edge("w2", "mul2"))
    g.add_edge(Edge("mul1", "pred"))
    g.add_edge(Edge("mul2", "pred"))
    g.add_edge(Edge("pred", "loss"))
    return g
