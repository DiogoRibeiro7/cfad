# Installation

## Install from PyPI

```bash
pip install cfad
```

## Build Cython Extensions From Source

If you are working from a source checkout and want maximum performance, build
the Cython extensions in place:

```bash
python setup.py build_ext --inplace
```

## Optional Dependency Groups

Use the optional extras below depending on your workflow.

| Extra | Purpose |
|---|---|
| `[notebooks]` | Jupyter notebooks and plotting tools for experiments. |
| `[docs]` | MkDocs, Material for MkDocs, and API documentation tooling. |
| `[dev]` | Testing, linting, and local development tooling. |

## Verify Installation

```python
from cfad import detect

print(detect.__doc__)
```
