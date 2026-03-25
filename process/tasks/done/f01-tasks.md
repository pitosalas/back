# Tasks for Feature F01

## T01 — initialize uv project
**Status**: not done
**Description**: Run `uv init` in the project root to create pyproject.toml. Set project name to "back", Python version to latest stable.

## T02 — add dependencies
**Status**: not done
**Description**: Use `uv add` to install: marimo, matplotlib, numpy. Use `uv add --dev` for pytest.

## T03 — verify marimo launches
**Status**: not done
**Description**: Create a minimal placeholder marimo file at back/main.py with a single cell that prints "hello". Run `uv run marimo run back/main.py` and confirm it starts without errors.

## T04 — verify pytest runs
**Status**: not done
**Description**: Create a minimal tests/ folder with a trivial test (assert 1 == 1). Run `uv run pytest` and confirm it passes. This proves the test harness is wired up correctly.
