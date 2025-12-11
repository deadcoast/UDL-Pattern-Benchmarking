---
inclusion: always
---

# Python Environment Requirements

## UV Usage

**CRITICAL**: Always use `uv` for all Python commands in this project.

### Required Commands:
- Use `uv run python` instead of `python`
- Use `uv add` instead of `pip install`
- Use `uv run pytest` instead of `pytest`

### Examples:
```bash
# CORRECT
uv run pytest tests/
uv run python -c "import sys; print(sys.executable)"
uv add package

# WRONG - DO NOT USE
python -m pytest tests/
.venv/bin/python -m pytest tests/
python -c "import sys; print(sys.executable)"
pip install package
```

### Rationale:
- Prevents conflicts with system Python
- Ensures consistent dependency versions
- Avoids terminal crashes and environment issues
- Maintains project isolation
- Uses modern Python package management

**This requirement applies to ALL Python execution in this project.**