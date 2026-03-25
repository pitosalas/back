# General
* You shall not write any code for the tool withot a corresponding task
* You shall not write any task description without a corresponding feature
* You shall not write a feature which is not consistent with the spec
* You shall not close off a feature unless there is a complete suite of tests and they run

# features
* new features get a feature id like f01-create-process-folder
* each feature has a markdown file in the process/features/done or /notdonw folder
* inside features folder there is a file called template.md which shows the format

# tasks
* Before any design is done or code is written, a set of tasks must be developed
* All the tasks for a feature can be found in a file in the tasks/done or tasks/notdone folders
* inside tasks folder there is a file called template.md which shows the format
* Every feature must include a task for writing tests; this task must always be proposed along with the other tasks
* When the last task for a feature is marked done, move the task file from tasks/notdone/ to tasks/done/
* Then update the feature file: set Done to yes, and Tests Written and Test Passing to yes if applicable
* Then move the feature file from features/notdone/ to features/done/

# bootstrap
* When asked to bootstrap a new project, create the following structure:
  - `LICENSE` — copy from `.j3/LICENSE.template`, fill in year and author
  - `README.md` — copy from `.j3/README.md.template`, fill in app name
  - `.gitignore` — copy from `.j3/.gitignore.template` as-is
  - `CLAUDE.md` — copy from `.j3/CLAUDE.md.template`, fill in app name
  - `process/spec.md` — blank spec file for the target app
  - `process/features/notdone/` — folder for pending features
  - `process/features/done/` — folder for completed features
  - `process/features/template.md` — copy from j3 template
  - `process/tasks/notdone/` — folder for pending tasks
  - `process/tasks/done/` — folder for completed tasks
  - `process/tasks/template.md` — copy from j3 template
* After scaffolding, prompt the user to fill in `process/spec.md` before defining any features

# github
* Whenever asked you will create a new commit with a good message
* Whenever asked you will push to github




