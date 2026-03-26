# Tasks for Feature F08

## T01 — define turkey dataset in examples.py
**Status**: not done
**Description**: Add a TURKEYS constant — a list of three dicts with keys height, length, target and label ("Turkey 1" etc). Used by the UI and graph cells.

## T02 — add turkey radio selector cell
**Status**: not done
**Description**: Add a mo.ui.radio cell with options ["Turkey 1", "Turkey 2", "Turkey 3"]. Export it so the graph cell can react to it.

## T03 — update forward pass graph cell to use selected turkey
**Status**: not done
**Description**: The step-by-step graph cell reads the selected turkey from the radio button and builds the graph with that turkey's data.

## T04 — add persistent loss scoreboard
**Status**: not done
**Description**: Add a cell below the graph that always runs all three turkeys' full forward pass with the current weights and shows a small table: Turkey | Prediction | Loss. This stays visible regardless of which turkey is selected or what step the reader is on.

## T05 — write tests
**Status**: not done
**Description**: Test that TURKEYS has 3 entries with correct values. Test that the scoreboard computes correct losses for all three turkeys given known weights.
