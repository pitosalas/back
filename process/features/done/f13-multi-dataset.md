# Feature description for feature F13
## F13 — Multi-dataset support with selector
**Priority**: High
**Done:** yes
**Tests Written:** yes
**Test Passing:** yes
**Description**: Refactor the app to support multiple datasets. Add a Dataset abstraction
that encapsulates samples, feature names, target name, starting weights, and slider ranges.
Implement two datasets: the existing turkey feather model and a new 10-sample Auto MPG
dataset (weight, horsepower → mpg) with normalized inputs (weight÷1000, horsepower÷100).
Add a dataset selector at the top of the notebook; all lessons update reactively when the
user switches. Lesson prose that references specific values (Turkey 1, 5500, etc.) must
be generated dynamically from the selected dataset's first sample.
