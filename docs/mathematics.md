# Mathematics

## Empirical Characteristic Function

Let \(r_1, \dots, r_n\) be returns in one rolling window. The empirical
characteristic function (ECF) is

\[
\widehat{\varphi}_n(\xi)
= \frac{1}{n}\sum_{j=1}^{n} e^{i \xi r_j},
\qquad \xi \in \mathbb{R}.
\]

Under standard conditions the ECF converges pointwise to the population
characteristic function. ECF-based goodness-of-fit methods are classical in
statistics, including Epps and Pulley (1983).

## Gaussian Shape Reference

For the same window, estimate the sample mean and sample standard deviation,
\(\widehat\mu_n\) and \(\widehat\sigma_n\). The fitted Gaussian characteristic
function is

\[
\varphi_G(\xi)
= \exp\!\left(
    i\widehat\mu_n\xi
    -\frac{1}{2}\widehat\sigma_n^2\xi^2
  \right).
\]

CFAD measures the discrepancy between the ECF and this fitted Gaussian reference
on a finite real-frequency interval:

\[
D_n =
\left[
\frac{1}{\xi_{\max}-\xi_{\min}}
\int_{\xi_{\min}}^{\xi_{\max}}
\left|
\widehat\varphi_n(\xi)-\varphi_G(\xi)
\right|^2 d\xi
\right]^{1/2}.
\]

Because location and scale are fitted inside every window, \(D_n\) is aimed
primarily at higher-order distributional shape changes such as tail or skewness
changes. It is not invariant to every possible finite-sample effect, so its
operating characteristics must be established empirically for the application.

## Why This Is Not an Empirical Residue Test

For a finite sample, the ECF can be extended to complex \(z\) as

\[
\widehat\varphi_n(z)
= \frac{1}{n}\sum_{j=1}^{n} e^{iz r_j}.
\]

Each term is entire in \(z\), and a finite sum of entire functions is entire.
Therefore, for any closed contour \(C\), Cauchy's theorem gives

\[
\oint_C \widehat\varphi_n(z)\,dz = 0.
\]

A finite-sample ECF contour integral therefore cannot identify branch cuts or
poles of the population characteristic function. Earlier CFAD development text
used that interpretation; it is superseded by the real-frequency discrepancy
score above.

## Parametric Complex-Plane Diagnostics

The package retains a general closed-contour integration helper for parametric
functions that are actually evaluated at complex arguments. Such calculations
may be useful for studying a specified parametric characteristic function, but
they are separate from the empirical anomaly score and should not be interpreted
as evidence extracted from a finite-sample ECF residue.

## Sequential CUSUM Layer

Let \(D_t\) denote the rolling score sequence. Estimate in-control moments
\(\mu_0\) and \(\sigma_0\) from a prespecified calibration prefix and
standardize

\[
z_t = \frac{D_t-\mu_0}{\sigma_0}.
\]

CFAD then applies a two-sided Page-CUSUM:

\[
S_t^+ = \max(0, S_{t-1}^+ + z_t-k),
\]

\[
S_t^- = \max(0, S_{t-1}^- - z_t-k).
\]

An alarm is emitted when either branch exceeds a decision threshold \(h\).
Because \(z_t\) is standardized, the Page reference value \(k\) is dimensionless.

## References

Epps, T. W., and Pulley, L. B. (1983). *A test for normality based on the
empirical characteristic function*. Biometrika, 70(3), 723-726.

Page, E. S. (1954). *Continuous Inspection Schemes*. Biometrika, 41(1/2),
100-115.
