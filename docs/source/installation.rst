Installation
============

Install from PyPI
-----------------

.. code-block:: bash

   pip install cfad

Build Cython Extensions From Source
-----------------------------------

If you are working from a source checkout and want maximum performance, build
the Cython extensions in place:

.. code-block:: bash

   python setup.py build_ext --inplace

Optional Dependency Groups
--------------------------

Use the optional extras below depending on your workflow.

.. list-table::
   :header-rows: 1

   * - Extra
     - Purpose
   * - ``[notebooks]``
     - Jupyter notebooks and plotting tools for experiments.
   * - ``[docs]``
     - Sphinx, themes, and extensions for documentation builds.
   * - ``[dev]``
     - Testing, linting, and local development tooling.

Verify Installation
-------------------

.. code-block:: python

   from cfad import detect
   print(detect.__doc__)
