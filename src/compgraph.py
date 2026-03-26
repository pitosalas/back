#!/usr/bin/env python3
# compgraph.py — Computation graph with forward and backward pass
# Author: Pito Salas and Claude Code
# Open Source Under MIT license

import networkx as nx
from node import Node, NodeType
from edge import Edge


class CompGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.targets = {}

    def add_node(self, node: Node):
        self.graph.add_node(node.id, node_type=node.node_type, value=node.value, gradient=0.0)

    def add_edge(self, edge: Edge):
        self.graph.add_edge(edge.from_id, edge.to_id, local_deriv=0.0)

    def set_target(self, node_id: str, target: float):
        self.targets[node_id] = target

    def set_weight(self, node_id: str, value: float):
        self.graph.nodes[node_id]["value"] = value

    def get_value(self, node_id: str) -> float:
        return self.graph.nodes[node_id]["value"]

    def get_gradient(self, node_id: str) -> float:
        return self.graph.nodes[node_id]["gradient"]

    def forward_pass(self):
        dispatch = {
            NodeType.MULTIPLY: self._compute_multiply,
            NodeType.ADD: self._compute_add,
            NodeType.LOSS: self._compute_loss,
        }
        for node_id in nx.topological_sort(self.graph):
            node_type = self.graph.nodes[node_id]["node_type"]
            if node_type in dispatch:
                dispatch[node_type](node_id)

    def forward_pass_n(self, n: int):
        """Run the forward pass for the first n computed nodes only."""
        dispatch = {
            NodeType.MULTIPLY: self._compute_multiply,
            NodeType.ADD: self._compute_add,
            NodeType.LOSS: self._compute_loss,
        }
        count = 0
        for node_id in nx.topological_sort(self.graph):
            if count >= n:
                break
            node_type = self.graph.nodes[node_id]["node_type"]
            if node_type in dispatch:
                dispatch[node_type](node_id)
                count += 1

    def computed_node_ids(self) -> list[str]:
        """Return node ids that are computed (not INPUT or WEIGHT), in topological order."""
        computed = {NodeType.MULTIPLY, NodeType.ADD, NodeType.LOSS}
        return [
            node_id for node_id in nx.topological_sort(self.graph)
            if self.graph.nodes[node_id]["node_type"] in computed
        ]

    def backward_pass(self):
        for node_id in self.graph.nodes:
            self.graph.nodes[node_id]["gradient"] = 0.0
        loss_id = self._find_loss_node()
        self.graph.nodes[loss_id]["gradient"] = 1.0
        for node_id in reversed(list(nx.topological_sort(self.graph))):
            node_grad = self.graph.nodes[node_id]["gradient"]
            for pred_id in self.graph.predecessors(node_id):
                local_deriv = self.graph[pred_id][node_id]["local_deriv"]
                self.graph.nodes[pred_id]["gradient"] += node_grad * local_deriv

    def _find_loss_node(self) -> str:
        for node_id in self.graph.nodes:
            if self.graph.nodes[node_id]["node_type"] == NodeType.LOSS:
                return node_id
        raise ValueError("No loss node found in graph")

    def _compute_multiply(self, node_id: str):
        preds = list(self.graph.predecessors(node_id))
        a_val = self.graph.nodes[preds[0]]["value"]
        b_val = self.graph.nodes[preds[1]]["value"]
        self.graph.nodes[node_id]["value"] = a_val * b_val
        self.graph[preds[0]][node_id]["local_deriv"] = b_val
        self.graph[preds[1]][node_id]["local_deriv"] = a_val

    def _compute_add(self, node_id: str):
        preds = list(self.graph.predecessors(node_id))
        self.graph.nodes[node_id]["value"] = sum(self.graph.nodes[p]["value"] for p in preds)
        for p in preds:
            self.graph[p][node_id]["local_deriv"] = 1.0

    def _compute_loss(self, node_id: str):
        pred_id = list(self.graph.predecessors(node_id))[0]
        pred_val = self.graph.nodes[pred_id]["value"]
        target = self.targets[node_id]
        self.graph.nodes[node_id]["value"] = (pred_val - target) ** 2
        self.graph[pred_id][node_id]["local_deriv"] = 2 * (pred_val - target)
