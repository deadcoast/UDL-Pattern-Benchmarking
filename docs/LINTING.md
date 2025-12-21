# Linting & Type Checks

## Consistent Lint Report (Ruff)
```bash
ruff check --output-format=concise --statistics --target-version=py310 udl_rating_framework tests deployment scripts
```

## Formatting Check (Ruff)
```bash
ruff format --check --target-version=py310 udl_rating_framework tests deployment scripts
```

## Optional: Project Dev Tooling (requires dev extras)
```bash
uv run flake8 --format="%(path)s:%(row)d:%(col)d: %(code)s %(text)s" --statistics udl_rating_framework tests deployment scripts
uv run mypy --show-error-codes --no-color-output udl_rating_framework tests deployment
uv run black --check udl_rating_framework tests deployment scripts
```
