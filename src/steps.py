#!/usr/bin/env python3
# steps.py — Step labels for forward pass walkthrough
# Author: Pito Salas and Claude Code
# Open Source Under MIT license

from compgraph import CompGraph
from node import NodeType


def forward_step_label(g: CompGraph, node_id: str) -> str:
    """Return a plain-English explanation of what this node just computed."""
    node_type = g.graph.nodes[node_id]["node_type"]
    preds = list(g.graph.predecessors(node_id))
    result = g.get_value(node_id)

    if node_type == NodeType.MULTIPLY:
        a, b = preds[0], preds[1]
        return (
            f"**{node_id}** = {a} × {b} "
            f"= {g.get_value(a):.2f} × {g.get_value(b):.2f} "
            f"= **{result:.2f}**"
        )
    if node_type == NodeType.ADD:
        terms = " + ".join(f"{g.get_value(p):.2f}" for p in preds)
        names = " + ".join(preds)
        return f"**{node_id}** = {names} = {terms} = **{result:.2f}**"
    if node_type == NodeType.LOSS:
        pred_id = preds[0]
        pred_val = g.get_value(pred_id)
        target = g.targets[node_id]
        return (
            f"**{node_id}** = ({pred_id} − target)² "
            f"= ({pred_val:.2f} − {target:.2f})² "
            f"= **{result:.2f}**"
        )
    return f"**{node_id}** = {result:.2f}"
