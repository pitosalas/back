from examples import turkey_feather
from mermaid_viz import build_mermaid, mermaid_html


def test_build_mermaid_contains_nodes():
    g = turkey_feather(height=1.0, length=1.5, w1=1000, w2=3000, target=5000)
    result = build_mermaid(g, False, None)
    for node_id in ["height", "length", "w1", "w2", "ht_term", "len_term", "prediction", "loss"]:
        assert node_id in result


def test_build_mermaid_contains_edges():
    g = turkey_feather(height=1.0, length=1.5, w1=1000, w2=3000, target=5000)
    result = build_mermaid(g, False, None)
    assert "height --> ht_term" in result
    assert "prediction --> loss" in result


def test_build_mermaid_highlight():
    g = turkey_feather(height=1.0, length=1.5, w1=1000, w2=3000, target=5000)
    result = build_mermaid(g, False, "ht_term")
    assert "class ht_term hl" in result


def test_mermaid_html_wraps_in_iframe():
    g = turkey_feather(height=1.0, length=1.5, w1=1000, w2=3000, target=5000)
    result = mermaid_html(g, False, None)
    assert "<iframe" in result
    assert "mermaid" in result
    assert "graph LR" in result


def test_build_mermaid_shows_gradients():
    g = turkey_feather(height=1.0, length=1.5, w1=1000, w2=3000, target=5000)
    g.forward_pass()
    g.backward_pass()
    result = build_mermaid(g, True, None)
    assert "grad:" in result
