# Tasks for Feature F13

## T01 — define Dataset abstraction in examples.py
**Status**: not done
**Description**: Create a Dataset dataclass with fields: name, samples (list of dicts with
feature1, feature2, target, label), feature1_name, feature2_name, target_name, w1_start,
w2_start, w1_min, w1_max, w2_min, w2_max, node_labels (dict mapping ht_term/len_term display
strings). Migrate TURKEYS into a TURKEY_DATASET instance. Add CARS_DATASET using the 10-sample
Auto MPG data with inputs normalized (weight÷1000, horsepower÷100). Keep turkey_feather()
factory; add a generic graph factory that takes a Dataset sample plus w1/w2.

## T02 — refactor table_viz.py to accept Dataset
**Status**: not done
**Description**: Change forward_pass_table(w1, w2, step, dataset) and
backward_pass_table(w1, w2, step, dataset) to use dataset.samples and dataset.node_labels
instead of hardcoded TURKEYS and BACKWARD_LABELS. Move BACKWARD_STEP_LABELS generation
into a function that takes dataset so the prediction step label shows sample[0] values.

## T03 — refactor steps.py to accept Dataset
**Status**: not done
**Description**: Update forward_step_label(g, node_id, dataset) to use dataset.feature names
for display rather than hardcoded "height"/"length" strings.

## T04 — refactor Lesson 1 (The Model) to be dynamic
**Status**: not done
**Description**: Replace the hardcoded Turkey 1 annotated equation HTML with a dynamic
version that reads feature names, sample[0] values, starting weights, prediction, and loss
from the selected dataset. Same visual style, different values.

## T05 — refactor Lesson 2 (Computing Loss) prose to be dynamic
**Status**: not done
**Description**: Replace hardcoded feature name references ("height", "length") in the
Computing Loss markdown cell with dataset.feature1_name and dataset.feature2_name.

## T06 — refactor Lesson 3 (Changing a Weight) sliders and prose to be dynamic
**Status**: not done
**Description**: Drive slider start, stop, and value from dataset fields. Remove the
turkey-specific converged-value hint or make it dataset-specific. Slider labels should
show dataset.feature1_name and dataset.feature2_name.

## T07 — refactor Lesson 4 (Chain Rule partial derivatives) to be dynamic
**Status**: not done
**Description**: The "From chain rule to partial derivatives" section has a hardcoded table
of local derivatives and product for Turkey 1. Generate these values dynamically from
dataset.samples[0] and starting weights so they update when the dataset changes.

## T08 — add dataset selector cell to main.py
**Status**: not done
**Description**: Add a new cell near the top that displays a mo.ui.dropdown or radio
buttons for dataset selection. Returns the selected Dataset object. All downstream cells
that use dataset-specific values take it as a parameter.

## T09 — write tests
**Status**: not done
**Description**: Add tests for: CARS_DATASET graph factory produces correct forward pass
values; backward_pass on cars sample[0] produces expected gradients; forward_pass_table
and backward_pass_table render correctly for both datasets (check sample labels and node
labels in output HTML).

