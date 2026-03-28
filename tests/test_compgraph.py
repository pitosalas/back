from examples import turkey_feather, TURKEYS


def test_forward_pass_values():
    g = turkey_feather(height=2.0, length=3.0, w1=0.5, w2=0.4, target=2.0)
    g.forward_pass()
    assert abs(g.get_value("ht_term") - 1.0) < 1e-9
    assert abs(g.get_value("len_term") - 1.2) < 1e-9
    assert abs(g.get_value("prediction") - 2.2) < 1e-9
    assert abs(g.get_value("loss") - 0.04) < 1e-9


def test_backward_pass_gradients():
    g = turkey_feather(height=2.0, length=3.0, w1=0.5, w2=0.4, target=2.0)
    g.forward_pass()
    g.backward_pass()
    assert abs(g.get_gradient("w1") - 0.8) < 1e-9
    assert abs(g.get_gradient("w2") - 1.2) < 1e-9
    assert abs(g.get_gradient("prediction") - 0.4) < 1e-9


def test_gradients_zero_at_minimum():
    # height=2, length=3, w1=1, w2=0 gives prediction = 2*1 + 3*0 = 2.0 = target
    g = turkey_feather(height=2.0, length=3.0, w1=1.0, w2=0.0, target=2.0)
    g.forward_pass()
    g.backward_pass()
    assert abs(g.get_gradient("w1")) < 1e-9
    assert abs(g.get_gradient("w2")) < 1e-9


def test_backward_pass_turkey1_starting_weights():
    # Turkey 1: height=1.0, length=1.5, target=5000, w1=1000, w2=3000
    # prediction=5500, loss=(500)^2=250000
    # grad_w1 = 2*(5500-5000)*1*1.0 = 1000
    # grad_w2 = 2*(5500-5000)*1*1.5 = 1500
    g = turkey_feather(height=1.0, length=1.5, w1=1000, w2=3000, target=5000)
    g.forward_pass()
    g.backward_pass()
    assert abs(g.get_gradient("w1") - 1000) < 1e-6
    assert abs(g.get_gradient("w2") - 1500) < 1e-6


def test_backward_pass_summed_gradients_all_turkeys():
    # Summed gradients across all 3 turkeys at w1=1000, w2=3000
    # should give grad_w1=1875, grad_w2=3500 (matches F11 chain rule lesson)
    total_w1 = 0.0
    total_w2 = 0.0
    for t in TURKEYS:
        g = turkey_feather(height=t["height"], length=t["length"], w1=1000, w2=3000, target=t["target"])
        g.forward_pass()
        g.backward_pass()
        total_w1 += g.get_gradient("w1")
        total_w2 += g.get_gradient("w2")
    assert abs(total_w1 - 1875) < 1e-6
    assert abs(total_w2 - 3500) < 1e-6
