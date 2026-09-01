# Contributing

CFAD is a research-software repository. Contributions should keep the code,
documentation, benchmarks, and scientific claims aligned.

## Development Setup

1. Clone the repository and create a virtual environment.
2. Install editable development dependencies:

   ```bash
   python -m pip install "setuptools>=68" wheel "numpy>=1.24" "Cython>=3.0" "packaging>=24.2"
   python -m pip install -e ".[dev]" --no-build-isolation
   ```

3. Build optional Cython extensions for performance-sensitive testing:

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

The CI coverage threshold may differ from the local target during release
engineering. Prefer adding focused tests for changed behavior rather than only
adjusting thresholds.

## Documentation Workflow

Install documentation dependencies:

```bash
python -m pip install -e ".[docs]" --no-build-isolation
```

Serve the docs locally:

```bash
mkdocs serve
```

Build the static site strictly:

```bash
mkdocs build --strict
```

The published site is deployed from `main` by the `Docs` GitHub Actions
workflow. Pull requests that touch docs or MkDocs configuration should pass the
same strict build.

## Documentation Standards

When changing public behavior:

- update the relevant Markdown page under `docs/`;
- add or update docstrings for public functions and classes;
- include a runnable example when the workflow is user-facing;
- keep validation claims tied to benchmark evidence;
- avoid reviving the retired empirical contour-residue interpretation.

## Benchmark and Evidence Practices

Benchmark records under `benchmarks/` are part of the scientific evidence trail.
Do not overwrite frozen failed records after inspecting results. If a method
changes, add a new record with its own protocol, runner, seed plan, and
interpretation.

For validation changes, include:

- the data-generating processes or market data sources;
- null and alternative definitions;
- calibration rules fixed before seeing results;
- baseline comparators;
- false-alarm, power, delay, and specificity metrics where applicable.

## Release Notes

Update `CHANGELOG.md` for user-facing changes, release-engineering changes, and
scientific-interpretation changes. Be explicit when a release changes only
infrastructure and leaves the statistical method unchanged.
