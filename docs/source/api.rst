API Reference
=============

High-Level API
--------------

.. automodule:: cfad.api
   :members:
   :show-inheritance:

Detection Objects
-----------------

.. autoclass:: cfad.detection.RollingDetector
   :members:
   :show-inheritance:

.. autoclass:: cfad.detection.AnomalyReport
   :members:
   :show-inheritance:

ECF and Scoring Functions
-------------------------

.. autofunction:: cfad.empirical_cf.ecf_at

.. autofunction:: cfad.empirical_cf.rolling_ecf

.. autofunction:: cfad.contour.ecf_residue_scores

.. autofunction:: cfad.contour.rectangular_contour

.. autofunction:: cfad.residue_score.normalise_scores

.. autofunction:: cfad.residue_score.rolling_pvalue

.. autofunction:: cfad.residue_score.threshold_by_fpr

Model Classes
-------------

.. autoclass:: cfad.models.gaussian.GaussianCF
   :members:
   :show-inheritance:

.. autoclass:: cfad.models.nig.NIGCF
   :members:
   :show-inheritance:

.. autoclass:: cfad.models.cgmy.CGMYCF
   :members:
   :show-inheritance:

.. autoclass:: cfad.models.levy_stable.LevyStableCF
   :members:
   :show-inheritance:
