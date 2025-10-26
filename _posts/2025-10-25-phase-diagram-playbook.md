---
layout: distill
title: "A Shared Coordinate System for Whole-Brain Dynamics"
tags: [computational neuroscience, shared latent representations, tutorial, subject alignment, neural state-spaces, energy landscapes, intersubject comparability, interpretable descriptors, brain dynamics, statistical physics]
giscus_comments: true
date: 2025-10-25
featured: false

authors:
  - name: "Julian Kędys"
    url: "https://www.linkedin.com/in/julian-kedys-a332222a6/"
    affiliations:
      - name: "Department of Digital Medicine, Poznań Supercomputing and Networking Center (PSNC), Polish Academy of Sciences"
        url: "https://www.psnc.pl"
  - name: "Cezary Mazurek"
    url: "http://pl.linkedin.com/in/cezarymazurek"
    affiliations:
      - name: "Department of Digital Medicine, Poznań Supercomputing and Networking Center (PSNC), Polish Academy of Sciences"
        url: "https://www.psnc.pl"
---

# Contents:
### Introduction
### Motivation and contribution
### Pipeline overview
  -  1) ELA-secure preprocessing (brief)
  -  2) Population-universal shared latent space
  -  3) Binarisation of latent time series
  -  4) Pairwise maximum-entropy (Ising) fitting
  -  5) Energy-Landscape Analysis (ELA): descriptors and kinetics
  -  6) Phase-Diagram Analysis (PDA): multi-observable placement

### Robustness, uncertainty and diagnostics
### Limitations and scope
### Outlook

---

## Overview

We present a modular, modality-agnostic workflow that turns heterogeneous whole-brain time series into cohort-comparable, interpretable coordinates on a shared phase diagram, together with energy-landscape descriptors such as attractors, barriers, and kinetics.

Key steps:
1) population-universal latent spaces (SRM, MCCA, Group PCA or Group ICA with consensus and automated dimensionality selection)
2) per-latent binarisation to the +/-1 format
3) PMEM or Ising fitting (exact for small N, pseudo-likelihood, or variational Bayes
4) energy-landscape analysis: minima, disconnectivity, barriers, occupancies, kinetics
5) phase-diagram analysis: multi-observable placement on a shared reference surface with uncertainty
   
Outputs include uncertainty, quality control, and interactive visuals. Methods are user-tweakable, reliable, and reproducible.

---

## Motivation and contribution

The UniReps community prioritises representational alignment and comparability across subjects, datasets, and models. This post demonstrates:

- a subject-alignment front-end that produces shared latent representations with stable semantics across a population
- a stitched, physics-grounded back-end (PMEM to ELA to PDA) that yields mechanistically interpretable descriptors and shared coordinates
- a robustness-first toolkit that includes consensus alignment, automated selection, uncertainty quantification, diagnostics, and review-ready artefacts

---

## Pipeline overview

<div class="fake-img l-page">
  <p>PIPELINE_OVERVIEW_PLACEHOLDER</p>
</div>

1. Preprocess and engineer: ELA-secure detrending, conservative handling of missing data, safe within-subject run concatenation, per-region standardisation.
2. Population-aware alignment and dimensionality reduction: shared latents via SRM, MCCA, or Group PCA or Group ICA with consensus; automatic dimensionality selection.
3. Binarisation: per-latent threshold (median or mean), yielding +/-1 time series.
4. PMEM or Ising fitting: exact (small N), safeguarded pseudo-likelihood, or variational Bayes.
5. Energy-landscape analysis: attractors, barriers, disconnectivity graph, occupancies, kinetics.
6. Phase-diagram analysis: multi-observable placement on a shared reference surface with confidence intervals and near-criticality indices.

---

## 1) ELA-secure preprocessing (brief overview)

Aim: remove spikes, outliers, and slow global drifts while **preserving the on/off switching structure** that drives binarisation, PMEM fitting, and ELA/PDA. The procedure is modality-agnostic, non-invasive, and parameterised to be reproducible.

* Adaptive per-region parameters are computed from simple statistics and Welch spectra, then re-adapted after each step if requested (robust mode).
* Despiking uses derivative-based detection with local, percentile-scaled replacements in short contextual windows; consecutive spikes are handled as blocks.
* Outlier removal is IQR-based with the same local, percentile-scaled replacement; an optional second pass is available.
* Population-aware detrending uses a cohort-optimised LOESS trend (fraction selected by stationarity and autocorrelation reduction). The global trend is estimated on the mean signal and scaled per region, which corrects drift without flattening transitions.
* Optional steps: light smoothing, bandpass filtering, breaking long flat runs, temporal standardisation, and spatial normalisation.
* Outputs include per-step amplitude deltas and residual checks; we also report the concordance between pre- and post-detrending binary states to ensure switching patterns are retained.


<figcaption>Example region: Striatum dorsal region (R). Multi panel view of the preprocessing pipeline applied to one regional time series: Original Signal → After Despiking (derivative based spike detection with local percentile scaled replacement) → After Outlier Removal (IQR based with local context) → After Universal Detrending (global LOESS model optimised on the cohort) → After Spatial Normalization (min max to [0,1]). The sequence removes spikes and outliers, corrects slow global drift, and rescales amplitudes while preserving the switching structure required for later binarisation and PMEM, ELA, PDA.</figcaption> 

<figcaption>Automatic adaptation and step audit for the same region. The printout reports initial region statistics (median, IQR, std, noise estimate, SNR, main frequency), the adapted parameters chosen for this instance (filter band and order, despike and outlier thresholds and windows, polynomial order and windows for detrending and smoothing), and per step summaries such as max amplitude change. It records that the universal LOESS detrending was used and shows the selected LOESS fraction. These diagnostics make the preprocessing reproducible and provide quality control before alignment and binarisation.</figcaption> ::contentReference[oaicite:0]{index=0}


<figcaption>Regional examples: dentate gyrus (R) and thalamus (R). Left panels show original versus detrended regional signals after applying the universal global-trend model. Right panels show the resulting binary state rasters using per-region median thresholds; the title reports binary-state concordance between original and detrended signals. These visuals quantify how detrending shifts baselines yet preserves the switching patterns used for PMEM, ELA and downstream binarisation.</figcaption>

<figcaption>Cohort-wide evidence for a global trend. Top: absolute linear-trend correlations (abs(r)) for each recording with a decision threshold at 0.3. Bottom: relative trend magnitudes with a threshold at 0.1. The fact that many recordings exceed these thresholds motivates a universal detrending step before alignment and binarisation.</figcaption>

<figcaption>Detrending model selection on a representative dataset. Top: the global signal overlaid with linear, quadratic and LOESS trends; the preferred method is highlighted as the best. Bottom: residuals after each detrending method. The chosen model is the one that yields residuals that are most stationary and least autocorrelated; here LOESS is selected and its fraction is later optimised across datasets.</figcaption>

<figcaption>Summary table of trend metrics for all recordings. For each dataframe we report r value, p value, relative trend magnitude and whether residuals are stationary after linear, quadratic or LOESS detrending. This audit supports the universal-method choice and sets LOESS parameters used in the preprocessing pipeline.</figcaption>
::contentReference[oaicite:0]{index=0}



{% details Click to expand: practical notes %}

* Replacement never injects artificial structure: values are drawn from local percentiles and clamped to local ranges; neighbors can be skipped to avoid bleeding flagged samples.
* Block handling groups consecutive indices to avoid fragmentation; a de-blocking fix prevents long identical segments after replacements.
* Universal LOESS fraction is chosen across the cohort to balance residual stationarity, autocorrelation reduction, and variance explained; region-wise application only scales that global trend.
* All steps are seedable and config-driven; logs capture chosen parameters, max amplitude changes, and QC metrics for auditability.

#### Remark:

For ELA, methods must enhance quality without compromising temporal structure. Safe, beneficial steps:

* Despiking: generally beneficial; percentile/MAD-based, no hardcoded amplitude thresholds.
* Outlier removal: single or repeated; statistics-driven safeguards prevent over-removal.
* Detrending: crucial for our data; prefer the universal LOESS approach for cross-subject/session/region comparability. Region-wise detrending should be used only with domain context; the cohort-wide analysis supports global detrending, with expert review welcomed.
* Flat-blocks breaking: apply if replacements create long identical runs.
* Smoothing: only very mild, primarily cosmetic; avoid aggressive windows that could distort binarisation.

Bandpass filtering can be applied if not already applied to the data - many recording devices perform it automatically. Temporal standardisation and spatial normalisation are not required for ELA itself but are retained for general use; spatial normalisation is applied for the imputation pipeline to align signs and amplitudes. LOESS may shift some series below zero; this is expected and accounted for. All parameterisations are chosen to remain strictly ELA-compatible.

  {% enddetails %}

---

## 2) Population-universal shared latent space

Goal: obtain shared, ELA-secure latent components whose semantics are stable across subjects and runs, so downstream binarisation → PMEM → ELA/PDA is directly comparable and computationally tractable.

#### What’s new/strong here (one glance):

* Pick-and-mix reductions: (Group PCA, Group ICA, SRM, MCCA) with robust tweaks (varimax, multi-restart ICA, PCA-initialised SRM, whitened MCCA).
* A population-aware, multi-metric objective (structural + temporal + method-specific) → auto-selection of dimensionality and hyperparameters.
* Alignment toolkit (orthogonal Procrustes with column-wise sign fixes, Hungarian matching, and a neuro-Procrustes consensus).
* Synergy-weighted consensus across methods with dominance checks, stability (RV) analysis, and per-component contribution audits.
* Efficient, reproducible compute (SVD on averaged covariances, fast downsampling for temporal metrics, parallel grid search, seeded randomness).

<div class="fake-img l-page">
  <p>SHARED_COMPONENTS_PLACEHOLDER (e.g., heatmaps of method-specific and consensus components)</p>
</div>

---

### 2.1 Methods (population-aware, ELA-secure)

All methods preserve temporal ordering (only linear projections or orthogonal transforms over channels), and we quantify temporal faithfulness (trustworthiness/continuity, autocorrelation preservation).

* Group PCA (variance capture + varimax): eigenvectors of the average subject covariance; optional varimax keeps components sparse/interpretable without breaking orthogonality.
*Extras:* elbow detection, subject-wise EVR, reconstruction error, and post-rotation low-variance pruning.

* Group ICA (shared independent subspaces): subject-wise PCA → concatenation → FastICA with multi-restart selection (best negentropy proxy via kurtosis deviation), independence verification (mean abs(corr), kNN mutual information, squared-corr checks), sign harmonisation of subject contributions before averaging.

* SRM (shared timecourses with subject-specific maps): iterative orthogonal subject mappings (W_i) and shared response (S), PCA-based initialisation, Hungarian-matched alignment of mappings across subjects, orthogonality diagnostics, shared variance quantification.

* MCCA (maximising cross-subject correlation): per-subject whitening → SVD on concatenated features (SUMCOR-style) → iterative refinement of subject loadings (a_i). We report cross-subject alignment, canonical correlations to shared response, orthogonality in whitened space, and shared variance in native space.

> Why four methods? They trade off interpretability, independence, correlation sharing, and variance capture. We score them with the same unified, ELA-aware metric suite and fuse them, yielding a stable, population-universal latent space.

---

### 2.2 Alignment & sign consistency (critical for comparability)

* Orthogonal Procrustes with column-wise sign checks (flip a component if it anticorrelates with the reference).
* Hungarian matching aligns component order across methods/subjects by maximal absolute correlations.
* Neuro-Procrustes consensus: iterative, order-robust alignment across methods (SVD-based reference) with final sign harmonisation.
* Optional biological sign protocol (baseline-anchored flips) to stabilise polarities across datasets.

<div class="fake-img">
  <p>ALIGNMENT_FLOW_PLACEHOLDER (schematic of Procrustes → sign fix → Hungarian → consensus)</p>
</div>

---

### 2.3 Metrics & selection (multi-objective, normalised to [0,1])

Structural fidelity (RSA-style): correlation between original region-by-region correlation structure and that reconstructed from latents; sign preservation weighted by abs(corr); Procrustes disparity; Kruskal stress on inter-region distances.

Temporal faithfulness: trustworthiness/continuity of neighbour relations, temporal neighbourhood hits, autocorrelation preservation at task-relevant lags.

Method-specific:

* PCA: mean subject EVR, reconstruction error.
* ICA: independence (mean abs(corr)↓, MI↓), kurtosis (≠3), sparsity.
* SRM: shared alignment (Hungarian-matched), orthogonality, shared variance.
* MCCA: cross-subject alignment, shared variance, canonical correlations, orthogonality (whitened).

Normalisation uses principled maps (e.g., $$x \mapsto \tfrac{1}{1+x}$$ for “smaller-is-better”, linear $$[-1,1]\to[0,1]$$ for correlations).
Composite score: weighted average (user- or default weights), then auto-optimise (k) and hyperparameters via seeded, parallel grid search.

{% details Click to expand: metric formulas %}

Kruskal stress (upper-triangle, variance-matched):
$$
\operatorname{Stress}=\sqrt{\frac{\sum_{i<j}\left(d_{ij}-\hat d_{ij}\right)^2}{\sum_{i<j} d_{ij}^2}}
$$

Procrustes (disparity surrogate): for $$A,B\in\mathbb{R}^{n\times k}$$, orthogonal $$R^\star=\arg\min_{R\in\mathbb{O}(k)}\lVert A-BR\rVert_F$$, with $$R^\star=UV^\top$$ from $$\mathrm{SVD}(B^\top A)=U\Sigma V^\top$$.

RV stability (consensus vs method):
$$
\mathrm{RV}(A,B)=\frac{\operatorname{tr}(A^\top B)}{\sqrt{\operatorname{tr}(A^\top A)\operatorname{tr}(B^\top B)}}
$$

Composite (normalised metrics $$\tilde m\in[0,1]$$):
$$
J=\frac{\sum_m w_m,\tilde m}{\sum_m w_m}
$$

{% enddetails %}

---

### 2.4 Consensus across methods (synergy-weighted, stability-checked)

We build a weighted consensus matrix after cross-method alignment:
1. Align each method’s components (Hungarian or neuro-Procrustes).
2. Synergy weights per method:

$$
\tilde w_m \propto \big(\mathrm{composite}_{\mathrm{specific},m}\big)^{\alpha}\,
\big(\mathrm{composite}_{\mathrm{common},m}\big)^{1-\alpha},\qquad
w_m=\frac{\tilde w_m}{\sum_j \tilde w_j}.
$$
3. Weighted sum of aligned components → column L2 re-normalisation.
4. Validation: RSA metrics, Procrustes disparity, variance explained, RV stability, reconstruction consistency with each method, and a dominance index (Gini/CV/TV-distance) to ensure no method overwhelms the fusion.

{% details Click to expand: consensus diagnostics & thresholds %}

* Component alignment quality: mean inter-method abs(corr) per component, with permutation and bootstrap thresholds, plus a data-driven clustering criterion for bimodal quality profiles.
* Subject consistency: mean cross-subject abs(corr) of per-component timecourses, with phase-randomised surrogate thresholds (preserve autocorrelation).
* Dominance (balance of synergy weights): we report normalised Gini, CV, TVD, and min/max ratio (all mapped to [0,1]).
{% enddetails %}

<div class="fake-img">
  <p>CONSENSUS_DIAGNOSTICS_PLACEHOLDER (barplots: component alignment & subject consistency with thresholds)</p>
</div>

---

### 2.5 Mathematical cores (rigorous but compact)

SRM (orthogonal, correlation-maximising form). For subjects $X_i\in\mathbb{R}^{T_i\times d}$ find $W_i\in\mathbb{R}^{d\times k}$ with $W_i^\top W_i=I$ and shared response $S\in\mathbb{R}^{T\times k}$:

{% raw %}
$$
\min_{\{W_i\},\, S}\sum_i \lVert X_i W_i - S \rVert_F^2
\;\Longleftrightarrow\;
\max_{\{W_i\},\, S}\sum_i \operatorname{tr}\!\big(S^\top X_i W_i\big),
\qquad W_i^\top W_i = I.
$$
{% endraw %}

Updates: $S\leftarrow \tfrac{1}{n}\sum_i X_i W_i$ (column-wise z-scored),
$W_i \leftarrow U V^\top$ where $U\Sigma V^\top = \operatorname{SVD}(X_i^\top S)$.

MCCA (SUMCOR-style, whitened). Let $X_i$ be centred and whitened to $\tilde X_i$. Find $a_i\in\mathbb{R}^{d\times k}$ maximising total cross-correlation:

{% raw %}
$$
\max_{\{a_i\}}\; \sum_{i<j} \operatorname{tr}\!\big((\tilde X_i a_i)^\top(\tilde X_j a_j)\big),
\qquad a_i^\top a_i = I.
$$
{% endraw %}

Solution via SVD on the concatenation
{% raw %}
$$
\operatorname{SVD}\!\big([\,\tilde X_1\;\tilde X_2\;\cdots\;\tilde X_n\,]\big),
$$
{% endraw %}
then map back with subject-specific unwhitening.

Group PCA (population-aware). With subject covariances $\Sigma_i$, use $\bar\Sigma=\tfrac{1}{n}\sum_i \Sigma_i$; take top-$k$ eigenvectors and optionally apply varimax rotation $R^\star$ maximising

{% raw %}
$$
\sum_j\!\left(\sum_p L_{pj}^4 - \frac{\gamma}{p}\left(\sum_p L_{pj}^2\right)^2\right)
$$
{% endraw %}

for loadings $L = \mathrm{evecs}\, R$.

Group ICA (robust). Subject PCA → concatenation → FastICA. We restart with different seeds, pick the run with the highest negentropy proxy (mean $\lvert\mathrm{kurtosis}-3\rvert$), and verify independence (low mean $\lvert\mathrm{corr}\rvert$, low kNN-MI, low squared-corr).


---


### 2.6 Practicalities (efficiency, reproducibility)

* Efficient linear algebra: SVD on averaged covariances (Group PCA), batched downsampling for temporal metrics, parallel grid search with seeded restarts.
* Diagnostics at each stage: subject EVR, reconstruction error, independence checks, alignment scores, consensus stability (RV), method dominance, and retention rationale for kept/dropped components.
* ELA-secure by design: all transformations are linear in space (no temporal warping), detrending was handled upstream, and temporal metrics explicitly guard the switching dynamics used by ELA.

<div class="fake-img">
  <p>SCREE_AND_VARIANCE_PLACEHOLDER (scree, per-component & cumulative variance; consensus variance explained)</p>
</div>

---

### 2.7 Outputs (what to expect)

* Method-specific models (components/loadings), aligned across methods.
* Consensus model (columns = population-universal latents), with weights and stability report.
* Per-subject projections (time × components) ready for binarisation → ELA/PDA.
* QC bundle: metric table, thresholded alignment & consistency plots, dominance index, and retained-component justifications.

---

{% details Click to expand: implementation highlights & safeguards %}

* RSA-style structure preservation reconstructs regions from components (per-region least squares) to compare correlation matrices before/after, reporting Pearson on vectorised upper triangles, Frobenius/mean diffs, and weighted sign preservation.
* Temporal metrics (trustworthiness/continuity, neighbourhood hit) use downsampled representations for speed without losing neighbourhood signal; autocorr preservation checked at task-relevant lags.
* Robust sign handling: column-wise correlation checks after Procrustes; optional baseline-anchored flipping (percentile-based) prevents biologically implausible polarity swaps.
* Dominance index offers four normalised variants (Gini/CV/TVD/min-max) to ensure balanced consensus.
* Permutation/phase-randomisation thresholds provide statistical guardrails for declaring alignment/consistency “good enough.”
  {% enddetails %}

---

### References (core methods and metrics)

* SRM: Chen, R. S., Chen, P.-H., et al. A Reduced-Dimension fMRI Shared Response Model. NeurIPS, 2015.
* MCCA: Kettenring, J. R. Canonical analysis of several sets of variables. Biometrika, 1971.
* CCA: Hotelling, H. Relations Between Two Sets of Variates. Biometrika, 1936.
* Orthogonal Procrustes: Schönemann, P. H. A generalized solution of the orthogonal Procrustes problem. Psychometrika, 1966.
* Hungarian algorithm: Kuhn, H. W. The Hungarian Method for the Assignment Problem. Naval Research Logistics Quarterly, 1955.
* PCA/Varimax: Jolliffe, I. T. Principal Component Analysis; Kaiser, H. F. The Varimax Criterion. Psychometrika, 1958.
* ICA / FastICA: Hyvärinen, A. and Oja, E. Independent Component Analysis: Algorithms and Applications. Neural Networks, 2000.
* Trustworthiness/Continuity: Venna, J. and Kaski, S. Neighborhood Preservation in Nonlinear Projection Methods. 2001.
* Kruskal stress: Kruskal, J. B. Multidimensional scaling by optimizing goodness of fit to a nonmetric hypothesis. Psychometrika, 1964.
* RSA: Kriegeskorte, N. et al. Representational Similarity Analysis. Frontiers in Systems Neuroscience, 2008.
* RV coefficient: Robert, P. and Escoufier, Y. A Unifying Tool for Linear Multivariate Statistical Methods: The RV-Coefficient. 1976.

---

<div class="fake-img l-page">
  <p>PIPELINE_SUMMARY_PLACEHOLDER (overview from preprocessing → pick-and-mix methods → alignment → consensus → QC)</p>
</div>
::contentReference[oaicite:0]{index=0}

---


## 3) Binarisation of latent time series

After alignment and dimensionality reduction, each latent x_i(t) is thresholded per-latent using median or mean and mapped to s_i(t) in the set {-1,+1}.

This respects component-specific baselines, keeps PMEM tractable, and standardises inputs for inter-subject comparability.

---

## 4) Pairwise maximum-entropy (Ising) fitting

We model binary latents with the pairwise maximum-entropy distribution

$$
P(\mathbf{s}) \propto \exp\left(\sum_i h_i s_i + \sum_{i<j} J_{ij} s_i s_j\right).
$$

The model matches first and second moments while keeping assumptions minimal.

Inference routes:
- Exact for small N by state enumeration for gold-standard checks
- Pseudo-likelihood with safeguarded line-search, symmetric updates, L2 regularisation, and relative-norm stopping
- Variational Bayes with Gaussian priors on parameters, posterior precisions, ELBO trace, and optional Monte Carlo moment estimates for larger N

Fit quality and sanity checks:
- Moment matching for means and correlations, information metrics such as mutual information explained and KL reduction vs independence
- Optional MCMC validation of observables with R-hat and effective sample size diagnostics and optional heat-capacity estimates

---

## 5) Energy-Landscape Analysis (ELA): descriptors and kinetics

From h and J we derive:
- Local minima or basins, disconnectivity graph, and barrier matrix
- Occupancies, dwell times, transition structure including multi-step transitions, and basin entropy
- Kinetic descriptors including mean first-passage times, committor fields, relaxation spectrum, and Kemeny constant

<div class="l-page">
  <iframe src="{{ '/assets/plotly/ela_barriers.html' | relative_url }}" frameborder="0" scrolling="no" height="480px" width="100%" style="border: 1px dashed grey;"></iframe>
</div>

---

## 6) Phase-Diagram Analysis (PDA): multi-observable placement

Each subject is placed on a shared reference phase surface parameterised by mu and sigma via a variance-balanced discrepancy over multiple observables, for example

$$
\{m,\; q,\; \chi_{\mathrm{SG}},\; \chi_{\mathrm{Uni}},\; C\}.
$$

We compute the optimum, bootstrap confidence ellipses, and distance-to-criticality, enabling direct cross-subject comparison that is not tied to a single healthy baseline.

<div class="l-page">
  <iframe src="{{ '/assets/plotly/pda_surface.html' | relative_url }}" frameborder="0" scrolling="no" height="520px" width="100%" style="border: 1px dashed grey;"></iframe>
</div>

---

## Robustness, uncertainty and diagnostics

- Uncertainty via variational-Bayes posteriors for h and J; bootstrap intervals for mu and sigma and for near-criticality; block bootstrap for autocorrelation
- Convergence checks including MCMC R-hat and effective sample size where applicable, pseudo-likelihood relative-norm stopping, and ELBO improvements for variational Bayes
- Quality-control gates including permutation or null thresholds for spurious minima, component-stability filters, and sensitivity to number of latents and alignment choice
- Ablations: pooled versus subgroup reference surfaces; method toggles among SRM, MCCA, and ICA; median versus mean thresholds

---

## Interactive figures

- Phase placements with confidence intervals
  <div class="l-page">
    <iframe src="{{ '/assets/plotly/pda_points.html' | relative_url }}" frameborder="0" scrolling="no" height="420px" width="100%" style="border: 1px dashed grey;"></iframe>
  </div>

- Basin visits over time (stripe plot)
  <div class="l-page">
    <iframe src="{{ '/assets/plotly/ela_minima_timecourse.html' | relative_url }}" frameborder="0" scrolling="no" height="260px" width="100%" style="border: 1px dashed grey;"></iframe>
  </div>

- Barrier distribution and kinetics
  <div class="l-page">
    <iframe src="{{ '/assets/plotly/ela_kinetics.html' | relative_url }}" frameborder="0" scrolling="no" height="420px" width="100%" style="border: 1px dashed grey;"></iframe>
  </div>

---

## Reproducibility and artefacts

We operate config-first with deterministic seeds and machine-parsable outputs.

{% highlight yaml %}
# configs/pipeline.yaml
seed: 123
preprocess:
  detrend: conservative
  concat_runs: true
  impute: short_gaps_only
alignment:
  methods: [SRM, MCCA, GroupPCA, GroupICA]
  consensus: true
  select_dim: auto
binarise:
  threshold: median
ising:
  mode: PL            # {EXACT|PL|VB}
  l2_h: 1e-5
  l2_J: 1e-4
  pl_tol: 1e-6        # relative-norm stopping
  vb:
    prior_prec_h: 6.0
    prior_prec_J: 30.0
ela:
  minima_search: exhaustive
  kinetics: true
pda:
  observables: [m, q, chiSG, chiUni, C]
  bootstrap: true
  ci_level: 0.95
reports:
  phase_report: true
  landscape_report: true
{% endhighlight %}

Repository layout for this post:
- post file: _posts/2025-10-25-a-shared-coordinate-system.md
- static images: assets/img/a-shared-coordinate-system/…
- plotly HTML: assets/plotly/…
- BibTeX: assets/bibliography/a-shared-coordinate-system.bib

---

## Limitations and scope

- Binarisation coarsens signals, but enables interpretable PMEM fitting and stable cross-subject comparability
- Latent selection is influential; consensus and metric-guided selection mitigate but semantics remain partly model-dependent
- The choice of reference surface affects PDA; we quantify sensitivity and expose cost bowls for transparency
- Designed for resting or task neural time series; extension to other binarisable dynamical systems is often straightforward

---

## Outlook

Population-universal latents combined with physics-grounded descriptors provide a shared language for multi-subject brain dynamics that is portable across modalities, tasks, and species, and a bridge to mechanistic interpretation and clinical translation. Planned extensions include multi-modal fusion, alignment-aware causal probes, and targeted clinical studies.

---

## Appendix: Mathematical details

### PMEM objective via pseudo-likelihood

Maximise pseudo-likelihood with L2 regularisation:

$$
\mathcal{L}_{\mathrm{PL}}(\mathbf{h},\mathbf{J})
=
\sum_{t,i}\log P\left(s_i^{(t)}\mid \mathbf{s}_{\setminus i}^{(t)}\right)
-
\frac{\lambda_h}{2}\lVert \mathbf{h}\rVert_2^2
-
\frac{\lambda_J}{2}\lVert \mathbf{J}\rVert_F^2.
$$

### Variational Bayes sketch

Gaussian priors on parameters; update posterior precision via empirical feature covariances; track the evidence lower bound; optional Monte Carlo moment approximation for larger N.

### Observables for PDA

$$
\begin{aligned}
m &= \frac{1}{N}\sum_i \langle s_i\rangle, \\
q &= \frac{1}{N}\sum_i \langle s_i\rangle^2, \\
\chi_{\mathrm{Uni}} &= \frac{1}{N}\sum_{i\neq j}\mathrm{Cov}(s_i,s_j), \\
\chi_{\mathrm{SG}} &= \frac{1}{N}\sum_k \lambda_k^2, \\
C &= \frac{1}{N}\left(\langle E^2\rangle - \langle E\rangle^2\right).
\end{aligned}
$$

---

### Acknowledgements

We thank colleagues at PSNC for discussions and infrastructure support.

---

### References

Rendered from the bibliography file.
