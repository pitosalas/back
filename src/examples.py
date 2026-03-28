#!/usr/bin/env python3
# examples.py — Dataset abstraction and computation graph factory
# Author: Pito Salas and Claude Code
# Open Source Under MIT license

from dataclasses import dataclass
from compgraph import CompGraph
from node import Node, NodeType
from edge import Edge


@dataclass
class Dataset:
    name: str
    samples: list
    feat1_name: str
    feat2_name: str
    target_name: str
    w1_start: float
    w2_start: float
    w1_min: float
    w1_max: float
    w2_min: float
    w2_max: float
    w_step: float


def build_graph(sample: dict, dataset: Dataset, w1: float, w2: float) -> CompGraph:
    f1, f2 = dataset.feat1_name, dataset.feat2_name
    g = CompGraph()
    g.add_node(Node(f1, NodeType.INPUT, sample[f1]))
    g.add_node(Node(f2, NodeType.INPUT, sample[f2]))
    g.add_node(Node("w1", NodeType.WEIGHT, w1))
    g.add_node(Node("w2", NodeType.WEIGHT, w2))
    g.add_node(Node("ht_term", NodeType.MULTIPLY, 0.0))
    g.add_node(Node("len_term", NodeType.MULTIPLY, 0.0))
    g.add_node(Node("prediction", NodeType.ADD, 0.0))
    g.add_node(Node("loss", NodeType.LOSS, 0.0))
    g.set_target("loss", sample["target"])
    g.add_edge(Edge(f1, "ht_term"))
    g.add_edge(Edge("w1", "ht_term"))
    g.add_edge(Edge(f2, "len_term"))
    g.add_edge(Edge("w2", "len_term"))
    g.add_edge(Edge("ht_term", "prediction"))
    g.add_edge(Edge("len_term", "prediction"))
    g.add_edge(Edge("prediction", "loss"))
    return g


TURKEYS = [
    {"label": "Turkey 1", "height": 1.00, "length": 1.50, "target": 5000},
    {"label": "Turkey 2", "height": 0.75, "length": 1.25, "target": 3500},
    {"label": "Turkey 3", "height": 1.25, "length": 1.00, "target": 4500},
]

TURKEY_DATASET = Dataset(
    name="Turkey Feathers",
    samples=TURKEYS,
    feat1_name="height",
    feat2_name="length",
    target_name="feathers",
    w1_start=1000.0,
    w2_start=3000.0,
    w1_min=0.0,
    w1_max=5000.0,
    w2_min=0.0,
    w2_max=5000.0,
    w_step=50.0,
)

CARS_DATASET = Dataset(
    name="Auto MPG",
    samples=[
        {"label": "Chevrolet Chevelle", "weight": 3.504, "horsepower": 1.30, "target": 18.0},
        {"label": "Buick Skylark 320",  "weight": 3.693, "horsepower": 1.65, "target": 15.0},
        {"label": "Datsun PL510",       "weight": 2.110, "horsepower": 0.95, "target": 28.0},
        {"label": "VW 1131 Deluxe",     "weight": 2.372, "horsepower": 0.75, "target": 30.0},
        {"label": "AMC Hornet",         "weight": 2.833, "horsepower": 0.90, "target": 26.0},
        {"label": "Pontiac Safari",     "weight": 4.746, "horsepower": 2.30, "target": 10.0},
        {"label": "Ford Galaxie 500",   "weight": 4.382, "horsepower": 1.98, "target": 14.0},
        {"label": "Toyota Corolla",     "weight": 2.130, "horsepower": 0.70, "target": 33.0},
        {"label": "Honda Civic",        "weight": 1.835, "horsepower": 0.65, "target": 31.0},
        {"label": "Dodge Challenger",   "weight": 3.609, "horsepower": 1.50, "target": 18.0},
    ],
    feat1_name="weight",
    feat2_name="horsepower",
    target_name="mpg",
    w1_start=5.0,
    w2_start=5.0,
    w1_min=-20.0,
    w1_max=20.0,
    w2_min=-20.0,
    w2_max=20.0,
    w_step=0.5,
)


def turkey_feather(height: float, length: float, w1: float, w2: float, target: float) -> CompGraph:
    """Two-input, two-weight model from Chapter 3/4 of How They Think (kept for tests)."""
    sample = {"height": height, "length": length, "target": target, "label": ""}
    return build_graph(sample, TURKEY_DATASET, w1, w2)
