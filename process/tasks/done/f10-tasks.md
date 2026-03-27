# Tasks for Feature F10

## T01 — make forward_pass_table accept variable w1/w2
**Status**: done
**Description**: forward_pass_table already takes w1 and w2 as arguments — verify it works correctly for arbitrary weight values (not just 1000/3000). Already done by design; just confirm with a test.

## T02 — add prose cell
**Status**: done
**Description**: Add a markdown cell explaining the lesson: dragging the slider changes w1, which changes all predictions and losses. The reader's goal is to get a feel for how weight changes affect the model before learning how to do it automatically.

## T03 — add w1 slider cell
**Status**: done
**Description**: Add a mo.ui.slider for w1 with range 0–5000, step 100, starting at 1000. Label it "w1". Export it.

## T04 — add reactive table cell
**Status**: done
**Description**: Add a cell that reads w1 from the slider, runs full forward pass for all three turkeys, and shows the completed table (step=4, all columns filled). No Next/Prev buttons.

## T05 — write tests
**Status**: done
**Description**: Test forward_pass_table with w1=0 gives prediction=length*w2 for each turkey. Test with w1=2347 (optimal) gives low loss values.
