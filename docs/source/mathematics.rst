Mathematics
===========

Empirical Characteristic Function
---------------------------------

Let :math:`r_1, \dots, r_n` be log-returns. The empirical characteristic
function (ECF) is

.. math::

   \hat{\varphi}_n(\xi) = \frac{1}{n}\sum_{j=1}^{n} e^{i \xi r_j}.

Under standard regularity conditions, :math:`\hat{\varphi}_n(\xi)` converges
pointwise to the true characteristic function as :math:`n \to \infty`.
ECF-based testing is classical in statistics (Epps and Pulley, 1983).

Residue-Theorem Detector
------------------------

For each rolling window, ``cfad`` evaluates a contour score

.. math::

   S_t = \left| \oint_C \hat{\varphi}_n(\xi)\, d\xi \right|.

The Residue Theorem implies

.. math::

   \oint_C f(z)\,dz = 2\pi i \sum_k \operatorname{Res}(f, z_k),

where the sum runs over singularities enclosed by :math:`C`.

If the characteristic function is entire on and inside :math:`C`, Cauchy's
theorem implies the sum is empty and therefore :math:`S_t = 0` (up to numerical
integration error in finite samples).

Analyticity Strip for NIG
-------------------------

For the Normal Inverse Gaussian characteristic function, analyticity is limited
to the strip

.. math::

   \left\{ \xi \in \mathbb{C} : |\operatorname{Im}\xi| < \alpha - |\beta| \right\}.

This strip determines contour heights that remain in an analytic region versus
heights that can intersect non-analytic structure.

Rectangular Contour
-------------------

The detector uses a rectangular contour with corners
:math:`\xi_{\min} \pm i\eta` and :math:`\xi_{\max} \pm i\eta`, traversed
counterclockwise. Numerically, the integral is approximated along the four
segments:

.. math::

   C = C_{\text{bottom}} \cup C_{\text{right}} \cup C_{\text{top}} \cup C_{\text{left}}.

This contour keeps integration close to the real line while probing
complex-plane structure through the imaginary offset :math:`\eta`.

Sequential CUSUM Layer
----------------------

The score stream :math:`\{S_t\}` is passed to a two-sided CUSUM monitor. For
the positive branch:

.. math::

   S_t^+ = \max\!\left(0, S_{t-1}^+ + \frac{S_t - \mu_0}{\sigma_0} - k\right),

with alarm threshold :math:`h`. This converts window-level structural evidence
into sequential alarms.

Reference
---------

Epps, T. W., and Pulley, L. B. (1983). *A test for normality based on the
empirical characteristic function*. Biometrika, 70(3), 723-726.
