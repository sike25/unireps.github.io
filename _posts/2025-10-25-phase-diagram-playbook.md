---
layout: distill
title: "Shared Coordinates for Cross-Subject Brain Dynamics: Universal Latents and Directly Comparable Phase Diagrams"
categories: [shared latent representations, tutorial, subject alignment, neural state-spaces, energy landscapes, intersubject comparability, state transitions, interpretable descriptors, brain dynamics]
giscus_comments: true
date: 2025-10-25
featured: false

authors:
  - name: Julian Kędys
    url: "https://www.linkedin.com/in/julian-kedys-a332222a6/"
    affiliations:
      name: Department of Digital Medicine, Poznań Supercomputing and Networking Center (PSNC), Polish Academy of Sciences
  - name: Cezary Mazurek
    url: "http://pl.linkedin.com/in/cezarymazurek"

bibliography: 2025-10-25-phase-diagram-playbook.bib

toc:
  - name: "Overview"
  - name: "Introduction"
    subsections:
      - name: "1. Big‑picture overview - why shared latents + PMEM → ELA → PDA?"
      - name: "2. Where the methods come from (intuitive recap)"
      - name: "3. Why shared latents are necessary (and useful)"
  - name: "Motivation and contribution"
  - name: "Pipeline overview"
    subsections:
      - name:  "1) ELA-secure preprocessing (brief)"
      - name:  "2) Population-universal shared latent space"
      - name:  "3) Binarisation of latent time series"
      - name:  "4) Pairwise maximum-entropy (Ising) fitting"
      - name:  "5) Energy-Landscape Analysis (ELA): descriptors and kinetics"
      - name:  "6) Phase-Diagram Analysis (PDA): multi-observable placement"
  - name: "Robustness, uncertainty and diagnostics"
  - name: "Limitations and scope"
  - name: "Outlook"


---

## Overview (TL;DR)

We present a modular, modality-agnostic workflow that turns heterogeneous whole-brain time series into cohort-comparable, interpretable coordinates on a shared phase diagram, together with energy-landscape descriptors such as attractors, barriers, and kinetics. 

Key steps:
1) population-universal latent spaces (Shared Response Model - SRM, Multiset Canonical Correlation Analysis - MCCA, Group PCA or Group ICA with consensus and automated dimensionality selection)
2) per-latent binarisation to the +/-1 format
3) Pairwise Maximum Entropy Model (PMEM) or Ising fitting: exact for small N, pseudo-likelihood, or variational Bayes
4) energy landscape analysis (ELA): minima, disconnectivity, barriers, occupancies, kinetics
5) phase diagram analysis (PDA): novel multi-observable placement on a shared reference surface with uncertainty
   
Outputs include uncertainty, quality control, and interactive visuals. Methods are user-tweakable, reliable, and reproducible.


---

## Introduction

#### 1. Big‑picture overview - why shared latents + PMEM → ELA → PDA?
Modern whole‑brain recordings are heterogeneous across subjects, sessions, tasks and modalities. If we analyse each participant in their own idiosyncratic space, descriptors of “brain state” are not directly comparable. Our pipeline solves this in two moves:
1. Population‑universal shared latents. We align subjects into a common, low‑dimensional space (SRM / MCCA / Group PCA‑ICA with consensus). Variables have stable meaning across participants and runs, so everything downstream is comparable and reproducible.
2. Physics‑grounded descriptors. On the binarised latents we fit a pairwise maximum‑entropy model (PMEM/Ising), then read out two complementary summaries:
   * Energy‑Landscape Analysis (ELA) - an attractor‑and‑barrier view of the fitted Ising energy. It yields minima/basins, disconnectivity graphs, barrier spectra, and kinetic descriptors (basin dwell times, mean first-passage times (MFPTs), committors, relaxation times). This is the mechanistic, state‑space view.
   * Phase‑Diagram Analysis (PDA) - a macroscopic view that places each subject on the **($$\mu$$, $$\sigma$$)** plane of a disordered (via parametric perturbations) Ising model (SK‑like). In broad outline, it uses multiple observables at once to locate individuals relative to critical boundaries, providing cohort‑comparable coordinates and uncertainty.

#### 2. Where the methods come from (intuitive recap)
* **PMEM/Ising:** Among all binary models that match the empirical means and pairwise correlations, PMEM has maximum entropy. It is equivalent to the zero‑temperature Ising family used throughout statistical physics. Minimal assumptions; parameters are interpretable as fields $$h_i$$ and couplings $$J_{ij}$$.
* **ELA:** Treat the fitted Ising as an energy landscape: 
  $$
  E(\mathbf{s}) = -\sum_i h_i s_i - \tfrac{1}{2}\sum_{i\neq j} J_{ij} s_i s_j
  $$, 
  
  over binary states $$\mathbf{s}\in\{-1,+1\}^N$$.

  Local minima = attractors; energy differences = barriers; transition graphs + Markov kinetics = how the brain moves between them.
* **PDA:** In spin‑glass models, the distribution of couplings matters. If the off‑diagonal $$J_{ij}$$ have mean $$\mu$$ and standard deviation $$\sigma$$ with $$h_i\approx 0$$, the system sits in regions (paramagnetic / spin‑glass / ferromagnetic) that govern ordering, glassiness and susceptibility. PDA maps each subject onto this phase surface so cohorts can be compared at a glance.
  
#### 3. Why shared latents are necessary (and useful)
* **Stable semantics:** Each latent represents the same population‑level pattern across participants, which makes ELA basins and PDA coordinates directly comparable.
* **Tractability:** PMEM scales as $$\mathcal{O}(N^2)$$ parameters; a well‑chosen latent space puts us in the sweet spot between information retention and robust estimation.
* **Downstream identifiability:** Binarisation → PMEM → ELA/PDA relies on on/off switching. Our alignment preserves this structure and gives us comparable switching rasters across the cohort.
* **Utility‑forward:** with one aligned space we can publish shared phase diagrams and landscape reports that are re‑usable across datasets and modalities, enabling baseline‑independent comparisons and cross‑study synthesis.

**Select practical advantages of our framework:**

The workflow is largely standalone and implemented locally, from scratch - allowing researchers/analysts to adapt its workings to their exact needs - with numerical-stability and computational-efficiency improvements (relative to analogous implementations in the domain), novel algorithmic extensions (i.a. for multi-subject dimnensionality reduction, comparative phase diagram analysis of heterogenous brain dynamics), informative metrics and rich visualisations, and emphasis on **a)** automated parameter optimisation - not requiring domain expertise or significant prior experience with the pipeline from the user, **b)** data-driven model selection, **c)** data-modality universality and independece of heuristics/meta-knowledge throughout the entire design process, and **d)** availability of alternative methods and hyperparameters for key processing/modelling stages, so as to best fit the needs of the user 

**Limitations worth remembering:** 
- Binarisation coarsens signals (whereas more granular discretisation becomes computationally prohibitve almost instantly for real-life data/problems)
- Results depend on the selection of binarisation thresholds, dimensionality-reduction models and target counts of obtained latent features
- For exact modelling methods, the set of possible states doubles in size with every additional feature/node/brain region
- PDA assumes $$h\approx 0$$ and an SK‑like (Sherrington-Kirkpatrick) parametrisation
- ELA explores single‑spin‑flip pathways, which is a limited and simplified assumption

To counteract the influence of initial choice of (hyper)parameters, we quantify uncertainty in the workflow, track convergence wherever applicable, offer truly data-driven and bias-free optimisation of pipeline parameters, and expose diagnostics so these choices remain transparent and testable.

---

## Motivation and contribution

Reflecting the UniReps community's interest in representational alignment and comparability across subjects, datasets, and models, this post demonstrates:

- a subject-alignment front-end that produces shared latent representations with stable semantics across a population, offering several alternative approaches
- a stitched, physics-grounded back-end (PMEM to ELA to PDA) that yields mechanistically interpretable descriptors and shared phase-diagram coordinates derived with our original methodology
- a robustness-first toolkit that includes custom-built consensus alignment, automated parameter selection, uncertainty quantification, diagnostics, and review-ready artefacts

---

## Pipeline overview

<div class="fake-img l-page">
  <p>PIPELINE_OVERVIEW_PLACEHOLDER</p>
</div>

1. Preprocess and engineer: ELA-secure detrending, conservative handling of missing data, safe within-subject run concatenation, per-region standardisation.
2. Population-aware alignment and dimensionality reduction: shared latents via SRM, MCCA, or Group PCA or Group ICA with consensus; automatic dimensionality selection.
3. Binarisation: per-latent threshold (usually median or mean, unless domain-expertise justifies, e.g., percentile-based thresholding), yielding +/-1 time series.
4. PMEM or Ising fitting: exact (small N), safeguarded pseudo-likelihood, or variational Bayes - each enriched with its adequate set of solution-stability and significance/quality assessments.
5. Energy-landscape analysis: attractors, barriers, disconnectivity graph, occupancies, kinetics, and many more descriptors providing mechanistic, biologically meaningful insight into brain dynamics, as well as facilitating direct and intuitive comparisons between subjects/cohorts.
6. Phase-diagram analysis: multi-observable placement on a shared reference surface with our custom cost function, reports on confidence intervals, and near-criticality indices.

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

Bandpass filtering can be performed if not already applied to the data - many recording devices perform it automatically. Temporal standardisation and spatial normalisation are not required for ELA itself but are retained for general use; spatial normalisation is applied for the imputation pipeline to align signs and amplitudes. LOESS may shift some series below zero; this is expected and accounted for. All parameterisations are chosen to remain strictly ELA-compatible.

  {% enddetails %}

---

## 2) Population-universal shared latent space

Goal: obtain shared, ELA-secure latent components whose semantics are stable across subjects and runs, so downstream binarisation → PMEM → ELA/PDA is directly comparable and computationally tractable.

#### Novelty factor at a glance:

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

* Group ICA (shared independent subspaces): subject-wise PCA → concatenation → FastICA with multi-restart selection (best negentropy proxy via kurtosis deviation), independence verification (mean $$\lvert\mathrm{corr}\rvert$$, kNN mutual information, squared-corr checks), sign harmonisation of subject contributions before averaging.

* SRM (shared timecourses with subject-specific maps): iterative orthogonal subject mappings $$(W_i)$$ and shared response $$(S)$$, PCA-based initialisation, Hungarian-matched alignment of mappings across subjects, orthogonality diagnostics, shared variance quantification.

* MCCA (maximising cross-subject correlation): per-subject whitening → SVD on concatenated features (SUMCOR-style) → iterative refinement of subject loadings $$(a_i)$$. We report cross-subject alignment, canonical correlations to shared response, orthogonality in whitened space, and shared variance in native space.

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

Structural fidelity (RSA-style): correlation between original region-by-region correlation structure and that reconstructed from latents; sign preservation weighted by $$\lvert\mathrm{corr}\rvert$$; Procrustes disparity; Kruskal stress on inter-region distances.

Temporal faithfulness: trustworthiness/continuity of neighbour relations, temporal neighbourhood hits, autocorrelation preservation at task-relevant lags.

Method-specific:

* PCA: mean subject EVR, reconstruction error.
* ICA: independence (mean $$\lvert\mathrm{corr}\rvert$$↓, Mutual Information (MI)↓), kurtosis (≠3), sparsity.
* SRM: shared alignment (Hungarian-matched), orthogonality, shared variance.
* MCCA: cross-subject alignment, shared variance, canonical correlations, orthogonality (whitened).

Normalisation uses principled maps (e.g., $$x \mapsto \tfrac{1}{1+x}$$ for “smaller-is-better”, linear $$[-1,1]\to[0,1]$$ for correlations).
Composite score: weighted average (user- or default weights), then auto-optimise $$(k)$$ and hyperparameters via seeded, parallel grid search.

{% details Click to expand: metric formulas %}

Kruskal stress (upper-triangle, variance-matched):

{% raw %}
$$
\mathrm{Stress}_1
=\min_{b>0}\sqrt{\frac{\sum_{i<j}\big(d_{ij}-b\hat d_{ij}\big)^2}
{\sum_{i<j} d_{ij}^2}},\qquad
b=\frac{\sum_{i<j} d_{ij},\hat d_{ij}}{\sum_{i<j} \hat d_{ij}^2}
$$
{% endraw %}


Procrustes (disparity surrogate):

$$
R^\star=\arg\min_{R\in O(k)}||A-BR||_F,\qquad
  with \quad B^\top A = U\Sigma V^\top, \quad R^\star = UV^\top,
  \quad A,B\in\mathbb{R}^{n\times k}
$$


RV stability (consensus vs method):
$$
\mathrm{RV}(A,B)=\frac{\operatorname{tr}(A^\top B)}{\sqrt{\operatorname{tr}(A^\top A)\operatorname{tr}(B^\top B)}}
\quad$$ 

(illustrative RV formulation for comparing matrices of the same size: recording sites X number of observations)

Composite (normalised metrics $$\tilde m\in[0,1]$$):
$$
J=\frac{\sum_m w_m\tilde m}{\sum_m w_m}
$$

{% enddetails %}

---

### 2.4 Consensus across methods (synergy-weighted, stability-checked)

We build a weighted consensus matrix after cross-method alignment:
1. Align each method’s components (Hungarian or neuro-Procrustes).
2. Synergy weights per method:
$$\tilde w_m \propto \big(\mathrm{composite}_{\mathrm{specific},m}\big)^{\alpha}\,
\big(\mathrm{composite}_{\mathrm{common},m}\big)^{1-\alpha},\qquad
w_m=\frac{\tilde w_m}{\sum_j \tilde w_j}$$


1. Weighted sum of aligned components → column L2 re-normalisation.
2. Validation: RSA metrics, Procrustes disparity, variance explained, RV stability, reconstruction consistency with each method, and a dominance index (Gini/CV/TV-distance) to ensure no method overwhelms the fusion.

{% details Click to expand: consensus diagnostics & thresholds %}

* Component alignment quality: mean inter-method $$\lvert\mathrm{corr}\rvert$$ per component, with permutation and bootstrap thresholds, plus a data-driven clustering criterion for bimodal quality profiles.
* Subject consistency: mean cross-subject $$\lvert\mathrm{corr}\rvert$$ of per-component timecourses, with phase-randomised surrogate thresholds (preserve autocorrelation).
* Dominance (balance of synergy weights): we report normalised Gini, CV, TVD, and min/max ratio (all mapped to [0,1]).
{% enddetails %}

<div class="fake-img">
  <p>CONSENSUS_DIAGNOSTICS_PLACEHOLDER (barplots: component alignment & subject consistency with thresholds)</p>
</div>

---

{% details Click to expand: Mathematical underpinnings %}


### 2.5 Mathematical cores (rigorous but compact)

**SRM (orthogonal, correlation‑maximising form)**

For subjects $$X_i \in \mathbb{R}^{T \times d}$$, find $$W_i \in \mathbb{R}^{d \times k}$$ with $$W_i^\top W_i = I$$ and a shared response $$S \in \mathbb{R}^{T \times k}$$:

$$
\min_{\{W_i\},\, S}\; \sum_i \lVert X_i W_i - S \rVert_F^2
\;\Longleftrightarrow\;
\max_{\{W_i\},\, S}\; \sum_i \operatorname{tr}\!\big(S^\top X_i W_i\big)
\quad \text{s.t. } W_i^\top W_i = I .
$$

Updates:

$$
S \leftarrow \frac{1}{n} \sum_i X_i W_i ,
$$

then z‑score $$S$$ column‑wise.

$$
W_i \leftarrow U V^\top, \qquad \text{where } U \Sigma V^\top = \operatorname{SVD}\!\big(X_i^\top S\big).
$$

*The above formulation reflects our shared T assumption - since all the subjects from the dataset used for developing the pipeline had the exact same numebr of frames in their time series.

---

**MCCA (SUMCOR‑style, whitened)**

Let $$X_i$$ be centred and whitened to $$\tilde{X}_i$$. Find $$a_i \in \mathbb{R}^{d \times k}$$ maximising total cross‑correlation:

$$
\max_{\{a_i\}} \; \sum_{i<j} \operatorname{tr}\!\big((\tilde{X}_i a_i)^\top (\tilde{X}_j a_j)\big)
\quad \text{s.t. } a_i^\top a_i = I .
$$

Solution via SVD on the concatenation:

$$
\operatorname{SVD}\!\left(\big[\, \tilde{X}_1 \ \ \tilde{X}_2 \ \ \cdots \ \ \tilde{X}_n \,\big]\right),
$$

then map back with subject‑specific unwhitening.

---



**Group PCA (population‑aware)**

With subject covariances $$\Sigma_i \in \mathbb{R}^{d \times d}$$, form the mean covariance
$$
\bar{\Sigma}=\frac{1}{n}\sum_{i=1}^{n}\Sigma_i .
$$


Eigen‑decompose the mean covariance
$$
\bar{\Sigma} = Q \Lambda Q^\top,
$$
where the columns of $$Q=[q_1,\dots,q_d]$$ are orthonormal eigenvectors and $$\Lambda=\operatorname{diag}(\lambda_1\ge\dots\ge\lambda_d)$$ .


Select the top‑$$k$$ eigenvectors
$$
E := [q_1\ \cdots\ q_k] \in \mathbb{R}^{d \times k}.
$$

(Optional) apply an orthogonal rotation $$R\in\mathbb{R}^{k\times k}$$ (e.g., varimax) to improve sparsity/interpretability. 

The rotated loadings are
$$
L = ER, \text{  with } R^\top R = I.
$$

A standard varimax objective (orthomax with parameter $$\gamma\in[0,1]$$) is:
$$
R^\star = \arg\max_{R^\top R=I} \sum_{j=1}^k \left( \sum_{p=1}^d L_{pj}^4 - \frac{\gamma}{d}\left(\sum_{p=1}^d L_{pj}^2\right)^2 \right),
\quad \text{for loadings } L = E R .
$$

Per‑subject time‑series data $$X_i \in \mathbb{R}^{T_i \times d}$$ are projected to latent timecourses via:

$$
Z_i = X_iL \in \mathbb{R}^{T_i \times k},
$$

or, without rotation: $$Z_i = X_iE$$.


**Notes:**
* Using the average covariance $$\bar{\Sigma}$$ ensures the components reflect population structure.
* Rotation is optional and keeps components orthogonal (since $$R$$ is orthogonal). If we prefer unrotated principal components, use $$L=E$$.

--- 

**Group ICA (robust)** 

Subject PCA → concatenation → FastICA. Restart with multiple seeds; select the run with the largest negentropy proxy (mean $$\lvert\mathrm{kurtosis} - 3\rvert$$), and verify independence (low mean $$\lvert\operatorname{corr}\rvert$$, low kNN‑MI, low mean squared correlation).

{% enddetails %}

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

After alignment and dimensionality reduction, each latent $$x_i(t)$$ is thresholded per-latent using median or mean and mapped to $$s_i(t) \in \{-1,+1\}$$.

This respects component-specific baselines, keeps PMEM tractable, and standardises inputs for inter-subject comparability.

---

## 4) Pairwise maximum-entropy (Ising) fitting
We model each time‑point’s binary latent vector $$\mathbf{s}\in\{-1,+1\}^N$$ with the pairwise maximum‑entropy (PMEM/Ising) distribution:
$$
P(\mathbf{s}) \propto
\exp\Big(\sum_{i} h_i s_i + \sum_{i<j} J_{ij} s_i s_j\Big),
\qquad
E(\mathbf{s}) = -\sum_{i} h_i s_i - \tfrac12 \sum_{i\ne j} J_{ij} s_i s_j
$$

PMEM matches the empirical first and second moments with minimal assumptions while remaining expressive for mesoscopic neural populations (as well as on the scale of entire networks of brain regions, and extending naturally to dynamical systems beyond neuroscience).

### Inference routes (complementary, scale‑aware):
* **Exact (small $$N$$)**: Enumerate all $$2^N$$ states to obtain the exact log‑likelihood and moments for gold‑standard checks
* **Pseudo‑likelihood (PL)**: Optimise the sum of node‑wise logistic conditionals with L2 penalties and a safeguarded Armijo line‑search; enforce symmetry $$J_{ij}=J_{ji}, J_{ii}=0$$ and use a relative‑norm stopping rule (scale‑free, robust).

  **Local field:**
  
  $$
  f_i^{(t)} = h_i + \sum_{j\neq i} J_{ij}s_j^{(t)} .
  $$

  **Node-wise conditional and log-conditional:**
  
  $$
  p\left(s_i^{(t)}\mid \mathbf s_{\setminus i}^{(t)}\right)
  = \frac{\exp\big(s_i^{(t)} f_i^{(t)}\big)}{2\cosh f_i^{(t)}},
  \qquad
  \log p\left(s_i^{(t)}\mid \mathbf s_{\setminus i}^{(t)}\right)
  = s_i^{(t)} f_i^{(t)}-\log\big(2\cosh f_i^{(t)}\big).
  $$
  
  **PL objective with L2 penalties (optimise over $$(h)$$ and the upper-triangle $$\{J_{ij}\}_{i<j}$$ ):**
  

  $$
  \mathcal L_{\mathrm{PL}}(h,J)
  = \overline{ \sum_{i=1}^N \Big[s_i f_i - \log\big(2\cosh f_i\big)\Big] }
  - \frac{\lambda_h}{2}||h||_2^2
  - \frac{\lambda_J}{2}\sum_{i<j} J_{ij}^2 ,
  \qquad J_{ij}=J_{ji}, \qquad J_{ii}=0 .
  $$



  ---
  {% details Click to expand: Gradients %}
  
  $$\nabla_{h_i}\mathcal L_{\mathrm{PL}}
  = \overline{s_i - \tanh f_i,}-\lambda_h h_i$$

  $$\nabla_{J_{ij}}\mathcal L_{\mathrm{PL}}
  = \overline{2s_i s_j - s_j \tanh f_i - s_i \tanh f_j,}
  -\lambda_J J_{ij},
  \qquad i<j$$

  (Bars denote averages over time; gradients include L2 terms)

  {% enddetails %}
  ---
  



* **Variational Bayes (VB):** Gaussian prior on $$(h,J)$$ with separate precisions for fields/couplings; a quadratic bound on the log‑partition yields closed‑form majorise–minimise updates:
  $$
  \Sigma^{-1}=\Lambda_0 + TC_\eta,\qquad
  \mu=\theta_0+\Sigma,T\big(\bar{\Phi}-m_\eta\big)
  $$ , \qquad where $$m_\eta=\mathbb{E}{p\eta}[\Phi]$$ and $$C_\eta=\operatorname{Cov}{p\eta}(\Phi)$$ are model moments/curvature at the current anchor $$\eta=\mu$$.
 
  Credible intervals come from $$\Sigma$$; optional Gamma hyper‑priors give ARD‑style shrinkage.

### Fit quality & sanity

We report (i) moment matching (means and pairwise correlations), (ii) multi‑information explained and KL‑reduction vs. independence, and (iii) empirical‑vs‑Boltzmann pattern agreement. 

For Monte‑Carlo checks we use multi‑chain sampling with **Ř**/effective sample size (ESS) diagnostics and estimate observables (magnetisation $$m$$, Edwards–Anderson $$q$$, spin‑glass and uniform susceptibilities $$\chi_{\mathrm{SG}},\chi_{\mathrm{Uni}}$$, specific heat $$C$$).

### Implementation highlights

PL uses mean‑field initialisation, symmetric updates, Armijo backtracking and a relative gradient test; VB stores posterior precisions and ELBO traces for convergence auditing. (See robustness notes in Section Robustness, uncertainty and diagnostics.)

---

## 5) Energy-Landscape Analysis (ELA): descriptors and kinetics

Once $$(h,J)$$ are fitted, the Ising energy

$$
E(\mathbf{s})=-\sum_i h_i s_i-\tfrac{1}{2}\sum_{i\neq j}J_{ij},s_i s_j
$$

induces a rugged energy landscape over $$\{-1,+1\}^N$$. 

We compute:
* **Attractors and basins:** Local minima are states whose single‑spin flips all increase energy. Every state is assigned to a basin by steepest‑descent (or best‑improving) paths. We summarise basins by occupancies and a disconnectivity graph
* **Barriers and disconnectivity:** The barrier between basins $$(\alpha,\alpha')$$ is the minimum, over all paths connecting them, of the maximum energy along the path; the disconnectivity graph visualises these heights. Denoting the (symmetrised) minimal saddle by $$\overline{E}_{\alpha\alpha'}$$, 

$$
\overline{E}_{\alpha\alpha'} = \min_{\gamma: \alpha\to\alpha'}\ \max_{\mathbf{s}\in\gamma} E(\mathbf{s}).
$$


* We also estimate a depth threshold (null‑model percentile) to guard against spurious minima.


**Kinetics (Markov view):** Build a single-spin-flip Metropolis chain with proposal “flip one spin uniformly” and transition probability:

$$
p(\mathbf{s}\to\mathbf{s}^{(i)}) = \frac{1}{N}\min\{1,\ \exp\big(E(\mathbf{s})-E(\mathbf{s}^{(i)})\big)\},
$$

where $$\mathbf{s}^{(i)}$$ is $$\mathbf{s}$$ with spin $$i$$ flipped. This yields a $$2^N\times 2^N$$ transition matrix $$P$$ (or a restriction to the visited subgraph).

**From $$P$$ we derive:**

* Stationary distribution $$\pi$$, dwell-time distributions, and basin occupancy.
* Mean first-passage times (MFPT): from a set $$A$$ to $$B$$ via the fundamental matrix $$Z=(I-Q)^{-1}$$ of the transient block.
* Committors $$q_{AB}$$ solving $$(I-Q)q=b$$, where $$b$$ collects transitions into $$B$$.
* Relaxation spectrum (mixing time-scales): non-unit eigenvalues $$\lambda_i(P)$$ with $$\tau_i=-1/\log\|\lambda_i\|$$, and the Kemeny constant (mean mixing time) $$K=\sum_{i\ge2}\frac{1}{1-\lambda_i}$$.

**Read-outs:** (i) attractor maps (patterns + labels), (ii) disconnectivity graphs, (iii) barrier distributions, (iv) transition/reachability matrices (one-step and multi-step), and (v) kinetic summaries (MFPT heatmaps, committor fields, relaxation spectra, Kemeny constants). These quantify stability, switching propensity, and heterogeneity of access between states.


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

Below are the core mathematical underpinnings of the pseudo‑likelihood and variational‑Bayes alternatives to the exact‑likelihood pairwise maximum‑entropy model (PMEM / Ising), as well as of the relevant auxilliary methods (e.g., for VB, the ARD/shrinkage or convergence diagnostics).

---

## Notation (common to PL and VB)

- Binary state $$\mathbf{s}\in\{-1,+1\}^N$$ with samples $$\{\mathbf{s}^{(t)}\}_{t=1}^T$$.
- Parameters $$\theta = \big[h_1,\ldots,h_N,\ \{J_{ij}\}_{i<j}\big]^\top \in \mathbb{R}^{P}$$, where $$P=N+N(N-1)/2$$.
- Feature map (sufficient statistics) $$\Phi(\mathbf{s}) = \big[s_1,\ldots,s_N,\ \{s_i s_j\}_{i<j}\big]^\top \in \mathbb{R}^{P}$$.
- Ising / PMEM:
  $$p_\theta(\mathbf{s}) \propto \exp\!\big(\theta^\top \Phi(\mathbf{s})\big), \qquad
  A(\theta)=\log Z(\theta)=\log \sum_{\mathbf{s}}\exp\!\big(\theta^\top \Phi(\mathbf{s})\big).$$
- Empirical feature mean:
  $$\bar{\Phi}=\frac{1}{T}\sum_{t=1}^T \Phi\!\big(\mathbf{s}^{(t)}\big).$$
- Model moments at parameter $$\theta$$:
  $$m_\theta=\mathbb{E}_{p_\theta}[\Phi], \qquad
  C_\theta=\operatorname{Cov}_{p_\theta}(\Phi)=\nabla^2 A(\theta).$$
  *(Here $$C_\theta$$ is the Fisher information.)*

---

## 1) Pseudo‑likelihood (PL) PMEM

**Node‑wise conditional** (logistic form). For
$$f_i(\mathbf{s}_{\setminus i}) = h_i + \sum_{j\neq i} J_{ij}\, s_j,$$
we have
$$\log p\!\big(s_i \mid \mathbf{s}_{\setminus i}\big)
= s_i f_i - \log\!\big(2\cosh f_i\big), \qquad
p\!\big(s_i \mid \mathbf{s}_{\setminus i}\big)=\frac{\exp(s_i f_i)}{2\cosh f_i}.$$

**Objective with L2** (ridge on $$h$$ and off‑diagonal $$J$$):
$$\mathcal{L}_{\mathrm{PL}}(h,J)
=\sum_{t=1}^T\sum_{i=1}^N \log p\!\big(s_i^{(t)}\mid \mathbf{s}_{\setminus i}^{(t)}\big)
-\frac{\lambda_h}{2}\lVert h\rVert_2^2
-\frac{\lambda_J}{2}\lVert J\rVert_F^2,$$
with $$J_{ii}=0,\quad J_{ij}=J_{ji}.$$

**Gradients** (for L‑BFGS/CG):
$$\frac{\partial \mathcal{L}_{\mathrm{PL}}}{\partial h_i}
=\sum_{t}\!\big[s_i^{(t)}-\tanh f_i^{(t)}\big]-\lambda_h h_i,$$
$$\frac{\partial \mathcal{L}_{\mathrm{PL}}}{\partial J_{ij}}
=\sum_{t}\!\big[s_i^{(t)}s_j^{(t)}-s_j^{(t)}\tanh f_i^{(t)}-s_i^{(t)}\tanh f_j^{(t)}\big]-\lambda_J J_{ij}.$$

**Practical symmetry step.** Fit $$N$$ independent logistic regressions, then **symmetrise** $$J$$:
$$J_{ij}\leftarrow \tfrac{1}{2}\big(\hat\beta^{(i)}_j+\hat\beta^{(j)}_i\big),\qquad J_{ii}=0.$$

---

## 2) Variational Bayes (VB) PMEM — Gaussian posterior over $$\theta$$

### (a) Prior and variational family
Stack $$\theta=[h;\, J_{i<j}]$$. Use zero‑mean Gaussian prior with separate precisions for fields and couplings:
$$p(\theta)=\mathcal{N}\!\big(\theta\mid \theta_0,\ \Lambda_0^{-1}\big),\qquad
\theta_0=\mathbf{0},\qquad
\Lambda_0=\operatorname{diag}\!\big(\tau_h \mathbf{I}_N,\ \tau_J \mathbf{I}_{P-N}\big).$$
Variational posterior:
$$q(\theta)=\mathcal{N}\!\big(\theta\mid \mu,\ \Sigma\big).$$

### (b) Quadratic bound on the log‑partition
By convexity of $$A(\theta)$$, for any anchor point $$\eta$$ (majorisation parameter),
$$A(\theta)\ \le\ A(\eta) + m_\eta^\top(\theta-\eta)
+\tfrac{1}{2}(\theta-\eta)^\top C_\eta (\theta-\eta).$$
This turns the intractable $$\mathbb{E}_q[A(\theta)]$$ into a tractable quadratic form.

### (c) ELBO under the quadratic bound
Let $$\mathcal{F}(\mu,\Sigma\mid\eta)$$ denote the bound on the ELBO:
$$\begin{aligned}
\mathcal{F}(\mu,\Sigma\mid\eta)
&= T\Big[\mu^\top \bar{\Phi} - A(\eta) - m_\eta^\top(\mu-\eta)
-\tfrac{1}{2}\operatorname{tr}(C_\eta \Sigma)
-\tfrac{1}{2}(\mu-\eta)^\top C_\eta (\mu-\eta)\Big] \\
&\quad -\tfrac{1}{2}\Big[(\mu-\theta_0)^\top \Lambda_0 (\mu-\theta_0) + \operatorname{tr}(\Lambda_0\Sigma)\Big]
+\tfrac{1}{2}\log\!\det(2\pi e\,\Sigma).
\end{aligned}$$

### (d) Stationary‑point updates (closed form)
Maximising $$\mathcal{F}$$ w.r.t. $$\mu,\Sigma$$ with $$\eta$$ fixed gives
$$\boxed{\ \ \Sigma^{-1} = \Lambda_0 + T\, C_\eta\ \ } \qquad
\boxed{\ \ \mu = \theta_0 + \Sigma\, T\big(\bar{\Phi} - m_\eta\big)\ \ }.$$
A convenient step form (with $$\theta_0=\mathbf{0}$$) is
$$\mu \leftarrow \eta + \Sigma\, T\big(\bar{\Phi}-m_\eta\big).$$
Then set $$\eta \leftarrow \mu$$ and repeat (majorise‑minimise) until convergence.

### (e) Moments needed in (d)
- $$m_\eta=\mathbb{E}_{p_\eta}[\Phi] = \big[\langle s_i\rangle_\eta,\ \langle s_i s_j\rangle_\eta\big].$$
- $$C_\eta=\operatorname{Cov}_{p_\eta}(\Phi)$$ (Fisher matrix).

They can be obtained **exactly** by state enumeration for small $$N$$, or **approximately** for larger $$N$$ via Monte Carlo:
$$\hat m_\eta=\frac{1}{M}\sum_{m=1}^M \Phi(\mathbf{s}^{[m]}),\qquad
\hat C_\eta=\frac{1}{M-1}\sum_{m=1}^M \big(\Phi(\mathbf{s}^{[m]})-\hat m_\eta\big)\big(\Phi(\mathbf{s}^{[m]})-\hat m_\eta\big)^\top,$$
where $$\mathbf{s}^{[m]} \sim p_\eta$$ (e.g., Gibbs/Metropolis).

### (f) Optional ARD / shrinkage on precisions
With Gamma hyperpriors $$\tau_h\sim\operatorname{Gamma}(a_h,b_h)$$ and $$\tau_J\sim\operatorname{Gamma}(a_J,b_J)$$ (type‑II ML / evidence updates),
$$\tau_h \leftarrow \frac{N/2 + a_h - 1}{\tfrac{1}{2}\big(\lVert \mu_h\rVert_2^2 + \operatorname{tr}\Sigma_{hh}\big) + b_h},\qquad
\tau_J \leftarrow \frac{(P-N)/2 + a_J - 1}{\tfrac{1}{2}\big(\lVert \mu_J\rVert_2^2 + \operatorname{tr}\Sigma_{JJ}\big) + b_J}.$$

### (g) Convergence & diagnostics
- Relative change: $$\max\!\Big(\tfrac{\lVert \mu^{(t)}-\mu^{(t-1)}\rVert}{\lVert \mu^{(t-1)}\rVert},\ \tfrac{\lVert \Sigma^{(t)}-\Sigma^{(t-1)}\rVert_F}{\lVert \Sigma^{(t-1)}\rVert_F}\Big) < \varepsilon$$.
- Monotone ascent of $$\mathcal{F}(\mu,\Sigma\mid \eta)$$ (re‑compute with $$\eta=\mu$$).
- Posterior standard deviations from $$\Sigma$$ provide credible intervals for $$h$$ and $$J$$.

### (h) What each core formula captures (VB intuition)
- $$\bar{\Phi}$$ — the **empirical** first/second moments of the data.
- $$m_\eta$$ — the **model** moments at the current parameter anchor; mismatch $$\bar{\Phi}-m_\eta$$ drives the mean update.
- $$C_\eta$$ — the curvature (Fisher) of the log‑partition; it **tempers** the update and sets the posterior covariance.
- $$\Sigma^{-1} = \Lambda_0 + T C_\eta$$ — precision adds **prior precision** and **data precision** (information additivity).
- $$\mu = \theta_0 + \Sigma T(\bar{\Phi}-m_\eta)$$ — mean moves in the direction that reduces **moment mismatch**.
- Hyper‑precisions $$\tau_h, \tau_J$$ — control shrinkage of fields vs couplings (can be learned).

### (i) Minimal algorithm (majorise‑minimise VB)
1. Initialise $$\mu$$ (e.g., PL‑MAP), choose $$\Lambda_0$$ (or $$\tau_h,\tau_J$$).
2. Repeat:
   - Compute $$m_\eta, C_\eta$$ at $$\eta=\mu$$ (exact/MCMC/mean‑field).
   - Update $$\Sigma^{-1}=\Lambda_0+T C_\eta$$.
   - Update $$\mu=\theta_0+\Sigma T(\bar{\Phi}-m_\eta)$$ (or step form above).
   - Optionally update $$\tau_h,\tau_J$$; check ELBO and relative change.
3. Output $$q(\theta)=\mathcal{N}(\mu,\Sigma)$$ and credible intervals.

---

### Acknowledgements

We thank colleagues at PSNC for discussions and infrastructure support.

---

### References

Rendered from the bibliography file.