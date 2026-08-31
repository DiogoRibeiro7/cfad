---
title: 'Omnibus versus Targeted Statistics for Finite-Window Distributional Change Detection'
authors:
  - name: Diogo Ribeiro
    orcid: 0009-0001-2022-7072
    affiliation: 1
affiliations:
  - name: Faculty of Media Arts and Design, Technical University of Porto, Portugal
    index: 1
bibliography: references.bib
---

# Abstract

Distributional change detectors often trade interpretability for generality. Omnibus discrepancies based on empirical characteristic functions, energy statistics, kernel embeddings, or optimal transport can respond to broad classes of alternatives, whereas targeted summaries such as skewness and kurtosis may be substantially more efficient when the relevant change direction is known. This study asks when the additional generality of an omnibus statistic is justified in finite monitoring windows. Motivated by a sequence of preregistered validation failures in a characteristic-function anomaly detector, we compare targeted moment discrepancies with empirical-characteristic-function, energy-distance, maximum-mean-discrepancy, and Wasserstein scores under controlled changes in tails, skewness, multimodality, and contamination. All primary windows are standardized locally so the estimand is distributional shape rather than location or scale. The study evaluates discrimination, null-law transportability, specificity, worst-case performance, and computational cost across multiple window sizes and in-control distributions. The aim is not to identify a universally best detector, but to quantify the finite-sample cost of omnibus sensitivity and establish when broad distributional scores earn their additional complexity.

# 1. Introduction

Methods for change detection can be divided loosely into targeted procedures, which monitor selected features of a distribution, and omnibus procedures, which attempt to detect broader departures. The distinction is important in finite windows. A statistic that is sensitive to every distributional difference need not be the most statistically efficient statistic for a particular departure when only a few dozen observations are available.

Empirical characteristic functions provide one natural omnibus representation because characteristic functions exist for every probability distribution and uniquely determine the law under standard conditions. Related approaches based on energy statistics, kernel maximum mean discrepancy, and optimal transport likewise compare distributions without reducing them to a small set of moments. Their generality is attractive, but it creates a practical question that is often left implicit: how much finite-sample power is lost by spreading sensitivity across many possible alternatives?

This paper studies that question directly. The motivation came from the validation of the CFAD research software, where a corrected characteristic-function score proved scientifically coherent but failed two frozen validation programmes. Sequential calibration under a Gaussian reference did not transfer adequately to a stationary heavy-tailed null, and a subsequent empirical-reference redesign improved null robustness without outperforming simpler moment-based scores on the registered shape alternatives. Rather than tuning the detector further, those negative results motivate a broader methodological comparison.

The contribution of this paper is therefore not a new characteristic-function change-point detector. It is a preregistered finite-window comparison of targeted and omnibus distributional discrepancies under a common experimental design.

# 2. Background

## 2.1 Targeted moment summaries

Skewness and excess kurtosis target particular departures from symmetry and Gaussian tail behaviour. They are low-dimensional and interpretable, but cannot represent arbitrary changes in distributional shape.

## 2.2 Empirical characteristic-function distance

For standardized observations $z_1,\ldots,z_n$, the empirical characteristic function is

$$
\widehat\varphi_n(\xi)=\frac{1}{n}\sum_{j=1}^n e^{i\xi z_j}.
$$

Given a frozen in-control reference ECF $\widehat\varphi_0$, the registered CF score is

$$
D_{\mathrm{ECF}}
=
\left[
\frac{1}{\xi_{\max}-\xi_{\min}}
\int_{\xi_{\min}}^{\xi_{\max}}
\left|\widehat\varphi_n(\xi)-\widehat\varphi_0(\xi)\right|^2d\xi
\right]^{1/2}.
$$

## 2.3 Energy distance

Energy distance compares distributions through expectations of Euclidean distances and yields zero only when the distributions agree under the usual moment conditions. In one dimension it offers an omnibus sample-level discrepancy without a kernel bandwidth or frequency grid.

## 2.4 Maximum mean discrepancy

Kernel maximum mean discrepancy compares mean embeddings in a reproducing-kernel Hilbert space. The registered implementation uses a Gaussian kernel with bandwidth fixed exclusively from the reference block.

## 2.5 Wasserstein distance

The one-dimensional Wasserstein-1 distance compares empirical quantile functions directly and provides an interpretable transport metric between standardized samples.

# 3. Motivation from frozen CFAD validation

The CFAD validation sequence provides the empirical motivation but is not pooled into the new confirmatory experiment. V2 demonstrated that a Gaussian-calibrated sequential characteristic-function score could achieve acceptable Gaussian false-alarm control while remaining poorly calibrated under a stable heavy-tailed null. V3 removed the sequential layer, standardized the frequency scale, and replaced the Gaussian reference with an empirical in-control ECF. That redesign repaired null-law robustness and location/scale specificity, but the primary ECF score still failed the prespecified shape-discrimination screen and did not outperform the excess-kurtosis comparator.

These results suggest that two distinct questions must be separated: whether an omnibus score is well specified, and whether its extra generality is worth its finite-sample variance. The present study addresses the second question directly.

# 4. Registered comparative design

The complete frozen design is recorded in `paper/validation_study_protocol.md`. Primary monitoring windows are standardized by their own sample mean and sample standard deviation. The comparison includes targeted skewness and kurtosis scores, a joint moment score, empirical-reference ECF distance, energy distance, Gaussian-kernel MMD, and Wasserstein-1 distance.

The primary alternatives cover heavier and lighter tails, positive and negative skewness, symmetric bimodality, and contamination. Monitoring-window sizes are 30, 60, and 120 observations, with 2,000 Monte Carlo replicates per registered cell. Gaussian, Student-t, and skew-normal in-control laws test null-law transportability.

# 5. Results

To be populated only from the frozen comparative benchmark. No exploratory result is admissible as a confirmatory result unless it belongs to the registered design.

# 6. Robustness across in-control laws

To be populated from the registered null-law comparison.

# 7. Computational trade-offs

To report score computation time per window alongside discrimination performance. Computational efficiency is secondary to statistical validity but matters when two methods are otherwise comparable.

# 8. Discussion

The central interpretation will distinguish generality from efficiency. An omnibus discrepancy may be preferable when the relevant alternative is genuinely unknown and its worst-case performance remains strong. A targeted summary may be preferable when it captures the scientifically relevant change with materially greater finite-window discrimination and similar null robustness.

No conclusion will claim universal superiority of either class. The registered experiment is deliberately finite-sample and alternative-family specific.

# 9. Implications for sequential monitoring

A sequential alarm rule compounds the statistical properties of its underlying score. The CFAD validation sequence showed why score validation should precede sequential calibration. Accordingly, no new CUSUM or other sequential layer is part of the present confirmatory study. Sequential design is justified only after a score demonstrates adequate discrimination, robustness, and specificity on its own.

# 10. Reproducibility and negative results

The repository retains the failed CFAD v2 and v3 evidence records, the present preregistration, and the future comparative benchmark outputs. Failed criteria will remain visible rather than being removed through post-result tuning.
