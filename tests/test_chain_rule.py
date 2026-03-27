#!/usr/bin/env python3
# test_chain_rule.py — Tests for chain_rule.py
# Author: Pito Salas and Claude Code
# Open Source Under MIT license

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from chain_rule import chain_forward, chain_derivs, chain_html


def test_forward_at_x3():
    a, b = chain_forward(3)
    assert a == 9
    assert b == 729


def test_forward_at_x2():
    a, b = chain_forward(2)
    assert a == 4
    assert b == 64


def test_derivs_at_x3():
    da_dx, db_da, db_dx = chain_derivs(3)
    assert da_dx == 6
    assert db_da == 243
    assert db_dx == 1458


def test_derivs_at_x2():
    da_dx, db_da, db_dx = chain_derivs(2)
    assert da_dx == 4
    assert db_da == 48
    assert db_dx == 192


def test_chain_rule_product():
    """Overall derivative equals the product of local derivatives."""
    for x in [1, 2, 3, 4, 5]:
        da_dx, db_da, db_dx = chain_derivs(x)
        assert db_dx == da_dx * db_da


def test_html_contains_values():
    html = chain_html(3)
    assert "729" in html
    assert "243" in html
    assert "1458" in html
    assert "x = 3" in html


def test_html_is_string():
    html = chain_html(3)
    assert isinstance(html, str)
    assert "<div" in html
