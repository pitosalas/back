from table_viz import forward_pass_table, COMPUTED_NODES
from examples import TURKEY_DATASET


def test_table_contains_turkey_labels():
    result = forward_pass_table(1000, 3000, 0, TURKEY_DATASET)
    for label in ["Turkey 1", "Turkey 2", "Turkey 3"]:
        assert label in result


def test_table_contains_node_headers():
    result = forward_pass_table(1000, 3000, 0, TURKEY_DATASET)
    for label in ["height", "length", "w1", "w2", "height×w1", "length×w2", "prediction", "loss"]:
        assert label in result


def test_uncomputed_cells_show_dash():
    result = forward_pass_table(1000, 3000, 0, TURKEY_DATASET)
    assert "—" in result


def test_step1_computes_ht_term():
    result = forward_pass_table(1000, 3000, 1, TURKEY_DATASET)
    # Turkey 1: height=1.0 * w1=1000 = 1000.00
    assert "1,000.00" in result
    # Turkey 2: height=0.75 * w1=1000 = 750.00
    assert "750.00" in result


def test_full_pass_no_dashes():
    result = forward_pass_table(1000, 3000, len(COMPUTED_NODES), TURKEY_DATASET)
    assert "—" not in result


def test_w1_zero_ht_term_is_zero():
    result = forward_pass_table(0, 3000, len(COMPUTED_NODES), TURKEY_DATASET)
    assert result.count("0.00") >= 3


def test_different_w1_changes_table():
    result_1000 = forward_pass_table(1000, 3000, len(COMPUTED_NODES), TURKEY_DATASET)
    result_2000 = forward_pass_table(2000, 3000, len(COMPUTED_NODES), TURKEY_DATASET)
    assert result_1000 != result_2000
