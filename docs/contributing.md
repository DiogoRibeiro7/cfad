# Contributing

## Development Setup

1. Clone the repository and create a virtual environment.
2. Install editable dependencies:

   ```bash
   pip install -e .[dev]
   ```

3. Build Cython extensions for performance-sensitive testing:

   ```bash
   python setup.py build_ext --inplace
   ```

## Quality Checks

Run formatting, linting, and tests before submitting changes:

```bash
ruff check .
black .
python -m pytest tests/ --cov=cfad --cov-fail-under=80
```

## Documentation Workflow

Install the documentation dependencies:

```bash
pip install -e .[docs]
```

Serve the docs locally with:

```bash
mkdocs serve
```

Build the static site with:

```bash
mkdocs build --strict
```

If you change public APIs or docstrings, update the relevant Markdown pages in
`docs/` and include usage examples when behavior changes.
