---
inclusion: always
---

# Python Environment Requirements

## Virtual Environment Usage

**CRITICAL**: Always use the project's virtual environment for all Python commands.

### Required Commands:
- Use `.venv/bin/python` instead of `python`
- Use `.venv/bin/pip` instead of `pip`
- Use `.venv/bin/pytest` instead of `pytest`

### Examples:
```bash
# CORRECT
.venv/bin/python -m pytest tests/
.venv/bin/python -c "import sys; print(sys.executable)"
.venv/bin/pip install package

# WRONG - DO NOT USE
python -m pytest tests/
python -c "import sys; print(sys.executable)"
pip install package
```

### Rationale:
- Prevents conflicts with system Python
- Ensures consistent dependency versions
- Avoids terminal crashes and environment issues
- Maintains project isolation

**This requirement applies to ALL Python execution in this project.**