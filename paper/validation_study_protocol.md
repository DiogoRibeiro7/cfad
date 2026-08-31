# Comparative finite-window validation study

## Status

Frozen design before any new comparative benchmark results.

## Research question

When monitoring short standardized windows for distributional-shape change, when does an omnibus distributional discrepancy earn enough finite-sample discrimination to justify its extra generality over targeted moment summaries?

The study is motivated by the corrected CFAD validation sequence. V2 showed that sequential calibration under a Gaussian reference did not transfer to a stable heavy-tailed null. V3 showed that standardized empirical-reference ECF scoring substantially improved null-law robustness and location/scale specificity, but did not outperform targeted kurtosis/skewness summaries on the prespecified shape alternatives.

The methodological paper therefore treats ECF scoring as one member of a broader class rather than presuming it is the preferred detector.

## Method families

All primary methods operate on windows standardized by their own sample mean and sample standard deviation so the primary estimand is distributional shape rather than location or scale.

### Targeted summaries

1. absolute excess-kurtosis difference from the frozen reference block;
2. absolute skewness difference from the frozen reference block;
3. a two-dimensional moment score using the Euclidean norm of standardized skewness and excess-kurtosis deviations.

### Omnibus discrepancies

1. standardized empirical characteristic-function L2 distance to a frozen empirical reference ECF;
2. energy distance between the reference sample and monitoring window;
3. Gaussian-kernel maximum mean discrepancy, with bandwidth fixed from the reference block by the median pairwise-distance rule and never retuned after monitoring begins;
4. one-dimensional Wasserstein-1 distance between standardized empirical distributions.

The existing standardized-Gaussian ECF score remains an ablation, not a primary competitor.

## Data-generating families

Each replicate contains an in-control reference block followed by one monitoring window. Reference and monitoring sample sizes are fixed before results are inspected.

Primary sample sizes:

- reference block: 300 observations;
- monitoring window: 30, 60, and 120 observations.

Primary in-control laws:

- Gaussian;
- Student-t with 5 degrees of freedom, variance standardized to one;
- standardized skew-normal with shape parameter 4.

Primary alternatives preserve mean zero and variance one after generation and isolate shape changes:

- heavier tails: Gaussian to Student-t with df 4;
- lighter tails: Student-t df 5 to Gaussian;
- positive skew: Gaussian to standardized skew-normal shape 8;
- negative skew: Gaussian to standardized skew-normal shape -8;
- symmetric bimodality: balanced two-component Gaussian mixture standardized to unit variance;
- contamination: Gaussian with 5% observations drawn from a zero-mean high-variance component.

Negative controls:

- pure mean shift of 1.5 standard deviations before within-window standardization;
- pure variance multiplication by 1.75 before within-window standardization.

## Monte Carlo design

- 2,000 replicates per reference-law × alternative × window-size cell;
- common random-number pairing across methods within every replicate;
- seed blocks disjoint from all CFAD v1-v3 experiments;
- no data-dependent method deletion, bandwidth retuning, frequency-grid retuning, or alternative redefinition after results are generated.

## Primary estimands

For each method, reference law, alternative, and window size:

1. ROC AUC distinguishing null monitoring windows from changed monitoring windows;
2. median score under the null and alternative;
3. null-law transportability, summarized by the ratio of null medians and pairwise KS statistics across reference laws;
4. specificity to pure location and scale changes after standardization.

Secondary estimands:

- AUC loss relative to the best method in each cell;
- average rank across alternatives;
- worst-case AUC across the six shape alternatives;
- computational cost per scored window.

## Confirmatory comparisons

The primary comparison is not whether any omnibus method beats every targeted statistic on every alternative. That would reward a deliberately unrealistic universal-dominance claim.

Instead, the study asks whether any omnibus method simultaneously achieves:

1. worst-case shape AUC at least 0.70 at window size 60;
2. mean shape AUC within 0.02 of the best targeted method averaged across the six alternatives;
3. no reference-law null median ratio above 2.0;
4. no reference-law KS statistic above 0.35;
5. mean- and variance-shift AUCs between 0.40 and 0.60 after standardization.

A targeted method is considered preferable when it exceeds an omnibus method by at least 0.05 AUC on its matched alternative without materially worse null transportability.

## Interpretation rules

- Failure of ECF scoring is evidence about the tested finite-window formulation, not a claim that characteristic-function methods are generally inferior.
- Success of an omnibus method means robust finite-window performance across the registered alternative family, not universal optimality.
- The sequential CUSUM layer is outside this study. No sequential detector will be redesigned from these results unless a score first demonstrates adequate score-level performance.
- Negative results remain part of the paper and repository.

## Paper structure

1. Introduction: omnibus versus targeted change statistics in finite windows.
2. Background: empirical characteristic functions, energy distance, MMD, Wasserstein distance, skewness and kurtosis.
3. Motivation from CFAD v1-v3 validation failures.
4. Registered comparative design.
5. Results by alternative family and window size.
6. Robustness across in-control laws.
7. Computational trade-offs.
8. Discussion: generality, finite-sample variance and the cost of omnibus sensitivity.
9. Implications for sequential detector construction.

## Existing evidence boundary

The already-completed CFAD v2 and v3 experiments are historical motivation and must not be pooled into the new confirmatory Monte Carlo cells. Their seeds and results remain immutable evidence records.
