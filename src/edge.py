#!/usr/bin/env python3
# edge.py — Edge dataclass for computation graphs
# Author: Pito Salas and Claude Code
# Open Source Under MIT license

from dataclasses import dataclass


@dataclass
class Edge:
    from_id: str
    to_id: str
