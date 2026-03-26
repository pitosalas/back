from table_viz import forward_pass_table, COMPUTED_NODES


def test_table_contains_turkey_labels():
    result = forward_pass_table(1000, 3000, 0)
    for label in ["Turkey 1", "Turkey 2", "Turkey 3"]:
        assert label in result


def test_table_contains_node_headers():
    result = forward_pass_table(1000, 3000, 0)
    for label in ["height", "length", "w1", "w2", "height×w1", "length×w2", "prediction", "loss"]:
        assert label in result


def test_uncomputed_cells_show_dash():
    result = forward_pass_table(1000, 3000, 0)
    assert "—" in result


def test_step1_computes_ht_term():
    result = forward_pass_table(1000, 3000, 1)
    # Turkey 1: height=1.0 * w1=1000 = 1000.0
    assert "1,000.0" in result
    # Turkey 2: height=0.75 * w1=1000 = 750.0
    assert "750.0" in result


def test_full_pass_no_dashes():
    result = forward_pass_table(1000, 3000, len(COMPUTED_NODES))
    assert "—" not in result


def test_w1_zero_ht_term_is_zero():
    # w1=0 means height*w1=0 for all turkeys
    result = forward_pass_table(0, 3000, len(COMPUTED_NODES))
    # ht_term column should show 0.0 for all rows
    assert result.count("0.0") >= 3


def test_different_w1_changes_table():
    # changing w1 should produce different output
    result_1000 = forward_pass_table(1000, 3000, len(COMPUTED_NODES))
    result_2000 = forward_pass_table(2000, 3000, len(COMPUTED_NODES))
    assert result_1000 != result_2000
