Contributing
============

Development Setup
-----------------

1. Clone the repository and create a virtual environment.
2. Install editable dependencies:

   .. code-block:: bash

      pip install -e .[dev]

3. Build Cython extensions for performance-sensitive testing:

   .. code-block:: bash

      python setup.py build_ext --inplace

Quality Checks
--------------

Run formatting, linting, and tests before submitting changes:

.. code-block:: bash

   ruff check .
   black .
   python -m pytest tests/ --cov=cfad --cov-fail-under=80

Documentation Workflow
----------------------

Build docs locally with:

.. code-block:: bash

   sphinx-build -b html docs/source docs/build/html

If you change public APIs or docstrings, update relevant pages in
``docs/source/`` and include usage examples when behavior changes.
