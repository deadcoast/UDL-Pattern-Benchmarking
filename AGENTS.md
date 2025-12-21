# Repository Guidelines

## Project Structure & Module Organization
- `udl_rating_framework/` holds the UDL Rating Framework core, metrics, CLI, and integration tooling.
- `models/`, `tasks/`, and `utils/` contain CTM research models, task training code, and shared helpers.
- `tests/` is the primary test suite; `tests/unit/` is a small unit subset.
- `docs/`, `examples/`, `demo_visualizations/`, and `deployment/` provide documentation, notebooks, dashboards, and API/Docker/K8s assets.
- Generated artifacts typically land in `output/`, `htmlcov/`, and `demo_visualizations/` exports.

## Build, Test, and Development Commands
- Install dependencies with `uv sync` (add `--extra dev --extra docs` for tooling).
- Run the CLI with `udl-rating --help` or `udl-rating rate ./my_udls --recursive --output report.json`.
- Run tests with `uv run pytest` or narrow scope with `uv run pytest tests/unit/`.
- Coverage report: `uv run pytest --cov=udl_rating_framework --cov-report=html` (outputs `htmlcov/`).
- Format/lint/type-check: `uv run black udl_rating_framework/`, `uv run flake8 udl_rating_framework/`, `uv run mypy udl_rating_framework/`.
- Train CTM tasks via module entry points, e.g. `python -m tasks.image_classification.train`.

## Coding Style & Naming Conventions
- Python 3.10+; 4-space indentation; keep code Black-compatible (line length 88).
- Type hints are required for new functions (mypy enforces `disallow_untyped_defs`).
- Use Google-style docstrings for Python APIs (see `docs/STYLE_GUIDE.md`).
- Tests follow `test_*.py` filenames and `Test*` classes (pytest config in `pyproject.toml` and `pytest.ini`).

## Testing Guidelines
- Pytest + Hypothesis; markers include `unit`, `property`, `integration`, and `performance`.
- Prefer targeted runs with `-k` (e.g., `-k "property"`), or directory scoping.
- Test docstrings should explain behavior and include property/requirement references when applicable (see `tests/README.md`).

## Commit & Pull Request Guidelines
- Recent history uses Conventional Commits (e.g., `feat(validation): ...`, `test(api-integration): ...`, `docs: ...`, `chore: ...`, `ci: ...`).
- No PR template is present; include a concise summary, link issues if applicable, and list tests run.
- Add screenshots or short clips for visualization/UI changes.
- Optional git hooks and CI quality checks are described in `docs/udl/integration_guide.md`.

## Configuration & Deployment Notes
- Use `config_example.yaml` as a starting point; avoid committing secrets.
- Deployment assets live in `deployment/` (Docker/K8s/API entry at `deployment/api/main.py`).
