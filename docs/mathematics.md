# Mathematics

This page records the statistical definition implemented by CFAD. The key point
is that the empirical anomaly score is a real-frequency ECF discrepancy, not a
closed-contour residue statistic.

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

For the same window, CFAD estimates the sample mean and sample standard
deviation, \(\widehat\mu_n\) and \(\widehat\sigma_n\). The fitted Gaussian
characteristic function is

\[
\varphi_G(\xi)
= \exp\!\left(
    i\widehat\mu_n\xi
    -\frac{1}{2}\widehat\sigma_n^2\xi^2
  \right).
\]

The detector then measures the finite-grid approximation to the normalized
real-frequency discrepancy

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
changes. This targeting is imperfect in finite samples, which is why validation
against null laws and negative controls is part of the repository design.

## Frequency Standardization

Characteristic-function distances depend on the frequency grid. Larger
frequencies can encode finer distributional structure, but finite-window ECF
estimates are noisier there. The project therefore treats the frequency range as
a statistical tuning parameter and provides sensitivity helpers for it.

The current detector operates on the real axis. The relevant range is
`xi_range` or `xi_min`/`xi_max`; the deprecated `height` argument is ignored by
the public high-level API.

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

An alarm is emitted when either branch exceeds a decision threshold \(h\). After
an alarm, both branches are reset. Because \(z_t\) is standardized, the Page
reference value \(k\) is dimensionless.

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

## Practical Consequences

The mathematics imply several practical rules:

- report the frequency grid and rolling-window size with any score;
- calibrate CUSUM on data that is defensibly in-control;
- compare against simpler baselines such as skewness and kurtosis distances;
- evaluate false alarms and power separately;
- avoid singularity or branch-cut language for empirical ECF scores.

## References

Epps, T. W., and Pulley, L. B. (1983). *A test for normality based on the
empirical characteristic function*. Biometrika, 70(3), 723-726.

Page, E. S. (1954). *Continuous Inspection Schemes*. Biometrika, 41(1/2),
100-115.
