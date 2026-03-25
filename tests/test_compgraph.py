from examples import turkey_feather


def test_forward_pass_values():
    g = turkey_feather(length=2.0, width=3.0, w_len=0.5, w_wid=0.4, target=2.0)
    g.forward_pass()
    assert abs(g.get_value("len_term") - 1.0) < 1e-9
    assert abs(g.get_value("wid_term") - 1.2) < 1e-9
    assert abs(g.get_value("prediction") - 2.2) < 1e-9
    assert abs(g.get_value("loss") - 0.04) < 1e-9


def test_backward_pass_gradients():
    g = turkey_feather(length=2.0, width=3.0, w_len=0.5, w_wid=0.4, target=2.0)
    g.forward_pass()
    g.backward_pass()
    assert abs(g.get_gradient("w_len") - 0.8) < 1e-9
    assert abs(g.get_gradient("w_wid") - 1.2) < 1e-9
    assert abs(g.get_gradient("prediction") - 0.4) < 1e-9


def test_gradients_zero_at_minimum():
    # w_len=1.0, w_wid=0.0 gives prediction = 2*1.0 + 3*0.0 = 2.0 = target
    g = turkey_feather(length=2.0, width=3.0, w_len=1.0, w_wid=0.0, target=2.0)
    g.forward_pass()
    g.backward_pass()
    assert abs(g.get_gradient("w_len")) < 1e-9
    assert abs(g.get_gradient("w_wid")) < 1e-9
