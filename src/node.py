#!/usr/bin/env python3
# node.py — Node dataclass and NodeType enum for computation graphs
# Author: Pito Salas and Claude Code
# Open Source Under MIT license

from dataclasses import dataclass
from enum import Enum


class NodeType(Enum):
    INPUT = "input"
    WEIGHT = "weight"
    MULTIPLY = "multiply"
    ADD = "add"
    LOSS = "loss"


@dataclass
class Node:
    id: str
    node_type: NodeType
    value: float
