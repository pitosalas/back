import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from examples import turkey_feather
from visualizer import draw_graph


def test_draw_graph_returns_figure():
    g = turkey_feather(length=2.0, width=3.0, w_len=0.5, w_wid=0.4, target=2.0)
    g.forward_pass()
    fig = draw_graph(g, show_gradients=False)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_draw_graph_with_gradients():
    g = turkey_feather(length=2.0, width=3.0, w_len=0.5, w_wid=0.4, target=2.0)
    g.forward_pass()
    g.backward_pass()
    fig = draw_graph(g, show_gradients=True)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
