import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from examples import turkey_feather
from visualizer import draw_graph
from steps import forward_step_label


def test_draw_graph_no_highlight():
    g = turkey_feather(height=1.0, length=1.5, w1=1000, w2=3000, target=5000)
    g.forward_pass()
    fig = draw_graph(g, False, None)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_draw_graph_with_gradients():
    g = turkey_feather(height=1.0, length=1.5, w1=1000, w2=3000, target=5000)
    g.forward_pass()
    g.backward_pass()
    fig = draw_graph(g, True, None)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_draw_graph_with_highlight():
    g = turkey_feather(height=1.0, length=1.5, w1=1000, w2=3000, target=5000)
    g.forward_pass()
    fig = draw_graph(g, False, "ht_term")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_forward_step_labels_all_nodes():
    g = turkey_feather(height=1.0, length=1.5, w1=1000, w2=3000, target=5000)
    g.forward_pass()
    for node_id in g.computed_node_ids():
        label = forward_step_label(g, node_id)
        assert isinstance(label, str)
        assert len(label) > 0
        assert node_id in label


def test_computed_node_ids_order():
    g = turkey_feather(height=1.0, length=1.5, w1=1000, w2=3000, target=5000)
    ids = g.computed_node_ids()
    assert ids == ["ht_term", "len_term", "prediction", "loss"]
