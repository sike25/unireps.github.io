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
  - name: "Results in a word"
  - name: "Robustness, uncertainty and diagnostics"
  - name: "Reproducibility and artefacts"
  - name: "Limitations and scope"
  - name: "Outlook"


---

## Overview (TL;DR)

We present a modular, modality-agnostic workflow that turns heterogeneous whole-brain time series into cohort-comparable, interpretable coordinates on a shared phase diagram, together with energy-landscape descriptors such as attractors, barriers, and kinetics. 

Key steps:
1) population-universal latent spaces (Shared Response Model - SRM, Multiset Canonical Correlation Analysis - MCCA, Group PCA or Group ICA with consensus and automated dimensionality selection) <d-cite key="NIPS2015_b3967a0e,60b14841-3c7e-3751-88f2-15308f78bf55,decheveigne2019mcca,correa2010mcca,SMITH2014738,calhoun2001groupica"></d-cite>
2) per-latent binarisation to the +/-1 format
3) Pairwise Maximum Entropy Model (PMEM) or Ising fitting: exact for small N, pseudo-likelihood, or variational Bayes <d-cite key="jaynes1957maxent,schneidman2006nature,besag1975pl,ravikumar2010ising,opper2001advancedmf"></d-cite>
4) energy landscape analysis (ELA): minima, disconnectivity, barriers, occupancies, kinetics <d-cite key="watanabe2014ela,becker1997disconnectivity,wales2006jpcb"></d-cite>
5) phase diagram analysis (PDA): novel multi-observable placement on a shared reference surface with uncertainty <d-cite key="edwards1975ea,sherrington1975sk,ezaki2020critical"></d-cite>
   
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

Comparing whole-brain dynamics across individuals is hard without a common reference that preserves interpretability and quantifies uncertainty. This challenge becomes even more apparent in studies of complex brain processes spanning cognition, sensory integration, and perception; when the data are limited; or when comparing brain dynamics driven by systematically different sub-types of various neurodevelopmental, psychiatric, or neurodegenerative conditions. Aiming to address these challenges and reflecting the UniReps community's interest in representational alignment and comparability across subjects, datasets, and models, this post demonstrates:

- a subject-alignment front-end that produces shared latent representations with stable semantics across a population, offering several alternative approaches
- a stitched, physics-grounded back-end (PMEM to ELA to PDA) that yields mechanistically interpretable descriptors and shared phase-diagram coordinates derived with our original methodology
- a robustness-first toolkit that includes custom-built consensus alignment, automated parameter selection, uncertainty quantification, diagnostics, and review-ready artefacts

---

## Pipeline overview

1. Preprocess and engineer: ELA-secure detrending, conservative handling of missing data, safe within-subject run concatenation, per-region standardisation.
2. Population-aware alignment and dimensionality reduction: shared latents via SRM, MCCA, or Group PCA or Group ICA with consensus; automatic dimensionality selection.
3. Binarisation: per-latent threshold (usually median or mean, unless domain-expertise justifies, e.g., percentile-based thresholding), yielding +/-1 time series.
4. PMEM or Ising fitting: exact (small N), safeguarded pseudo-likelihood, or variational Bayes - each enriched with its adequate set of solution-stability and significance/quality assessments.
5. Energy-landscape analysis: attractors, barriers, disconnectivity graph, occupancies, kinetics, and many more descriptors providing mechanistic, biologically meaningful insight into brain dynamics, as well as facilitating direct and intuitive comparisons between subjects/cohorts.
6. Phase-diagram analysis: multi-observable placement on a shared reference surface with our custom cost function, reports on confidence intervals, and near-criticality indices.

**Data summary** (used for the development and testing of the pipeline; details not central to current discussion in themselves):

Resting-state fUS from N=8 mice (7 Cre-lox ASD models spanning 4 subtypes; 1 control with no symptomatic manifestation modelled); 54 bilateral regions (27 L/R pairs; whole-brain collection) - unified across all the subjects; two runs per mouse (1494 frames for each recording session; TR ≈ 0.6 s); runs concatenated per subject.

---

## 1) ELA-secure preprocessing (brief overview)

Aim: remove spikes, outliers, and slow global drifts while **preserving the on/off switching structure** that drives binarisation, PMEM fitting, and ELA/PDA. The procedure is modality-agnostic, non-invasive, and parameterised to be reproducible.

* Adaptive per-region parameters are computed from simple statistics and Welch spectra, then re-adapted after each step if requested (robust mode). <d-cite key="welch1967psd"></d-cite>
* Despiking uses derivative-based detection with local, percentile-scaled replacements in short contextual windows; consecutive spikes are handled as blocks.
* Outlier removal is IQR-based with the same local, percentile-scaled replacement; an optional second pass is available.
* Population-aware detrending uses a cohort-optimised LOESS trend (fraction selected by stationarity and autocorrelation reduction)<d-cite key="cleveland1979lowess"></d-cite>. The global trend is estimated on the mean signal and scaled per region, which corrects drift without flattening transitions.
* Optional steps: light smoothing, bandpass filtering, breaking long flat runs, temporal standardisation, and spatial normalisation.
* Outputs include per-step amplitude deltas and residual checks; we also report the concordance between pre- and post-detrending binary states to ensure switching patterns are retained.

<figure class="l-page">
  <img src="{{ '/assets/img/2025-10-25-phase-diagram-playbook/Screenshot%202025-10-26%20111942.png' | relative_url }}"
       alt="Five-step preprocessing pipeline panels for one region">
  <figcaption style="color:#f5f5f5;background:rgba(0,0,0,.45);padding:.6rem .8rem;border-radius:8px;">
    Five-step preprocessing for a representative region (Striatum dorsal, R): Original → after despiking
    (derivative-based detection with local replacement) → after outlier removal (IQR with local context) →
    after universal LOESS detrending (global trend removed, transitions preserved) → after spatial
    normalisation to [0, 1].
  </figcaption>
</figure>

<figure class="l-page">
  <img src="{{ '/assets/img/2025-10-25-phase-diagram-playbook/Screenshot%202025-10-26%20111451.png' | relative_url }}"
       alt="Bars showing global linear-trend strength and magnitude across recordings">
  <figcaption style="color:#f5f5f5;background:rgba(0,0,0,.45);padding:.6rem .8rem;border-radius:8px;">
    Cohort-wide evidence for a global trend. Top: absolute linear-trend correlation \(|r|\) for each
    recording with a decision threshold (red dashed line). Bottom: relative trend magnitude with its
    threshold. Many runs exceed both criteria, motivating a universal detrending step.
  </figcaption>
</figure>

<figure class="l-page">
  <img src="{{ '/assets/img/2025-10-25-phase-diagram-playbook/Screenshot%202025-10-26%20111514.png' | relative_url }}"
       alt="Global signal with linear, quadratic and LOESS trends; residuals comparison">
  <figcaption style="color:#f5f5f5;background:rgba(0,0,0,.45);padding:.6rem .8rem;border-radius:8px;">
    Model selection for detrending on an exemplar run. Top: global signal with linear, quadratic, and LOESS
    fits (LOESS selected). Bottom: residuals after each method. LOESS yields the most stationary, least
    autocorrelated residuals, hence chosen for the universal detrending step.
  </figcaption>
</figure>

<figure class="l-page">
  <img src="{{ '/assets/img/2025-10-25-phase-diagram-playbook/Screenshot%202025-10-26%20111548.png' | relative_url }}"
       alt="Example regions: raw vs detrended signals and binary-state rasters">
  <figcaption style="color:#f5f5f5;background:rgba(0,0,0,.45);padding:.6rem .8rem;border-radius:8px;">
    Region-wise example showing that global detrending preserves the on/off switching used downstream.
    Left: raw vs detrended signals for dentate gyrus (top) and thalamus (bottom).
    Right: binary rasters before/after median thresholding; “Concordance” in the panel titles reports the
    fraction of timepoints whose binary state is unchanged by detrending.
  </figcaption>
</figure>

<figure class="l-page">
  <img src="{{ '/assets/img/2025-10-25-phase-diagram-playbook/Screenshot%202025-10-26%20111524.png' | relative_url }}"
       alt="Table of global-trend metrics and post-detrending stationarity flags for all recordings">
  <figcaption style="color:#f5f5f5;background:rgba(0,0,0,.45);padding:.6rem .8rem;border-radius:8px;">
    Audit table across all recordings: linear-trend correlation and \(p\)-value, relative trend magnitude,
    and stationarity of residuals after linear, quadratic, and LOESS detrending. Most datasets achieve
    stationarity only with LOESS, supporting the universal-detrending choice.
  </figcaption>
</figure>





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
* Synergy-weighted consensus across methods with dominance checks, stability (RV) analysis, and per-component contribution audits <d-cite key="robert1976rv"></d-cite>.
* Efficient, reproducible compute (SVD on averaged covariances, fast downsampling for temporal metrics, parallel grid search, seeded randomness).


---
### 2.1 Methods (population-aware, ELA-secure)

All methods preserve temporal ordering (only linear projections or orthogonal transforms over channels), and we quantify temporal faithfulness (trustworthiness/continuity, autocorrelation preservation) <d-cite key="venna2006localmds"></d-cite>.

* Group PCA (variance capture + varimax): eigenvectors of the average subject covariance; optional varimax keeps components sparse/interpretable without breaking orthogonality <d-cite key="SMITH2014738,kaiser1958varimax"></d-cite>.
*Extras:* elbow detection, subject-wise EVR, reconstruction error, and post-rotation low-variance pruning.

* Group ICA (shared independent subspaces): subject-wise PCA → concatenation → FastICA <d-cite key="hyvarinen1999fastica"></d-cite> with multi-restart selection (best negentropy proxy via kurtosis deviation), independence verification (mean $$\lvert\mathrm{corr}\rvert$$, kNN mutual information <d-cite key="kraskov2004mi"></d-cite>, squared-corr checks), sign harmonisation of subject contributions before averaging.

* SRM (shared timecourses with subject-specific maps): iterative orthogonal subject mappings $$(W_i)$$ and shared response $$(S)$$, PCA-based initialisation, Hungarian-matched alignment of mappings across subjects, orthogonality diagnostics, shared variance quantification <d-cite key="NIPS2015_b3967a0e"></d-cite>.

* MCCA (maximising cross-subject correlation): per-subject whitening → SVD on concatenated features (SUMCOR-style) → iterative refinement of subject loadings $$(a_i)$$ <d-cite key="60b14841-3c7e-3751-88f2-15308f78bf55,decheveigne2019mcca,correa2010mcca"></d-cite>. We report cross-subject alignment, canonical correlations to shared response, orthogonality in whitened space, and shared variance in native space.

<!-- Group PCA scree -->
<figure class="l-page">
  <img src="{{ '/assets/img/2025-10-25-phase-diagram-playbook/Screenshot%202025-10-26%20180713.png' | relative_url }}"
       alt="Group PCA scree: individual and cumulative explained variance with elbow and totals">
  <figcaption style="color:#f5f5f5;background:rgba(0,0,0,.45);padding:.6rem .8rem;border-radius:8px;">
    <strong>Group PCA scree.</strong> Light-blue bars show individual explained-variance ratio (EVR) per
    component; the black line is cumulative EVR. The dotted vertical line marks the elbow (here at the 1st
    component), and the title reports total variance explained (65.6%). This plot guides the initial range
    for dimensionality selection before our multi-metric model choice.
  </figcaption>
</figure>

<!-- SRM: consensus mapping + within-subject component correlations -->
<figure class="l-page">
  <div style="display:flex; gap:1rem; flex-wrap:wrap;">
    <img style="flex:1 1 360px; max-width:49%;"
         src="{{ '/assets/img/2025-10-25-phase-diagram-playbook/Screenshot%202025-10-26%20181009.png' | relative_url }}"
         alt="SRM consensus mapping W: region-by-component loadings heatmap">
    <img style="flex:1 1 360px; max-width:49%;"
         src="{{ '/assets/img/2025-10-25-phase-diagram-playbook/Screenshot%202025-10-26%20181039.png' | relative_url }}"
         alt="SRM example subject: reduced correlation between latent components">
  </div>
  <figcaption style="color:#f5f5f5;background:rgba(0,0,0,.45);padding:.6rem .8rem;border-radius:8px;">
    <strong>SRM latent space.</strong> <em>Left:</em> consensus SRM mapping \(W\) (regions × components).
    Colours indicate signed loadings (arbitrary overall sign but harmonised across subjects), highlighting
    stable spatial patterns shared across the cohort. <em>Right:</em> example subject’s correlation matrix
    between SRM latent time series. Light (near-white) off-diagonals indicate low cross-component correlation,
    i.e. little superfluous overlap—evidence of efficient representation and dimensionality reduction.
  </figcaption>
</figure>

<!-- MCCA: within-subject component correlations + subject projection matrix -->
<figure class="l-page">
  <div style="display:flex; gap:1rem; flex-wrap:wrap;">
    <img style="flex:1 1 360px; max-width:49%;"
         src="{{ '/assets/img/2025-10-25-phase-diagram-playbook/Screenshot%202025-10-26%20181248.png' | relative_url }}"
         alt="MCCA example subject: reduced correlation between latent components">
    <img style="flex:1 1 360px; max-width:49%;"
         src="{{ '/assets/img/2025-10-25-phase-diagram-playbook/Screenshot%202025-10-26%20181218.png' | relative_url }}"
         alt="MCCA subject projection matrix a: region-by-component loadings">
  </div>
  <figcaption style="color:#f5f5f5;background:rgba(0,0,0,.45);padding:.6rem .8rem;border-radius:8px;">
    <strong>MCCA latent space.</strong> <em>Left:</em> example subject’s correlation matrix between MCCA
    components—again, light off-diagonals reflect low redundancy across latents. <em>Right:</em> subject-specific
    MCCA projection matrix \(a\) (regions × components), showing how each brain region contributes to each
    shared component; structured bands point to interpretable, population-aligned patterns.
  </figcaption>
</figure>


> Why four methods? They trade off interpretability, independence, correlation sharing, and variance capture. We score them with the same unified, ELA-aware metric suite and fuse them, yielding a stable, population-universal latent space.

---

### 2.2 Alignment & sign consistency (critical for comparability)

* Orthogonal Procrustes with column-wise sign checks (flip a component if it anticorrelates with the reference) <d-cite key="schonemann1966procrustes"></d-cite>.
* Hungarian matching aligns component order across methods/subjects by maximal absolute correlations <d-cite key="kuhn1955hungarian"></d-cite>.
* Neuro-Procrustes consensus: iterative, order-robust alignment across methods (SVD-based reference) with final sign harmonisation.
* Optional biological sign protocol (baseline-anchored flips) to stabilise polarities across datasets.
---

### 2.3 Metrics & selection (multi-objective, normalised to [0,1])

Structural fidelity (RSA-style): correlation between original region-by-region correlation structure and that reconstructed from latents; sign preservation weighted by $$\lvert\mathrm{corr}\rvert$$; Procrustes disparity; Kruskal stress <d-cite key="kruskal1964nmds1"></d-cite> on inter-region distances.

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


<figure class="l-page">
  <img loading="lazy" alt="Bar chart of cross-subject component consistency with statistical thresholds"
       src="{{ '/assets/img/2025-10-25-phase-diagram-playbook/Screenshot%202025-10-26%20181523.png' | relative_url }}">
  <figcaption style="color:#f5f5f5;background:rgba(0,0,0,.45);padding:.6rem .8rem;border-radius:8px;">
    <strong>Cross-subject consistency of shared components.</strong> Each bar is the mean absolute correlation of the same
    component across subjects (higher = more reproducible). Dashed/dotted lines show permutation and 95% CI thresholds. 
    Components above both guides generalise well across mice and are kept for the consensus latent space.
  </figcaption>
</figure>

<figure class="l-page">
  <img loading="lazy" alt="Bar chart of cross-method alignment quality with thresholds"
       src="{{ '/assets/img/2025-10-25-phase-diagram-playbook/Screenshot%202025-10-26%20181355.png' | relative_url }}">
  <figcaption style="color:#f5f5f5;background:rgba(0,0,0,.45);padding:.6rem .8rem;border-radius:8px;">
    <strong>Alignment quality across methods.</strong> Bars show how well each consensus component matches its counterparts
    from Group PCA, Group ICA, SRM and MCCA after Procrustes + Hungarian alignment (mean |corr|). Higher values and crossing
    the dashed/dotted thresholds indicate robust cross-method agreement, supporting the fused consensus basis.
  </figcaption>
</figure>


<figure class="l-page">
  <img loading="lazy" alt="Stacked bars showing method contributions to each consensus component"
       src="{{ '/assets/img/2025-10-25-phase-diagram-playbook/Screenshot%202025-10-26%20181438.png' | relative_url }}">
  <figcaption style="color:#f5f5f5;background:rgba(0,0,0,.45);padding:.6rem .8rem;border-radius:8px;">
    <strong>Method contributions to the consensus components.</strong> Stacked bars report the weighted influence of each
    method (Group PCA / Group ICA / SRM / MCCA) on every consensus component. Balanced contributions mean the final space is
    not dominated by a single method and retains structure that is consistent across approaches.
  </figcaption>
</figure>


<figure class="l-page">
  <img loading="lazy" alt="Heatmap of Group PCA transformation matrix after alignment"
       src="{{ '/assets/img/2025-10-25-phase-diagram-playbook/Screenshot%202025-10-26%20183547.png' | relative_url }}">
  <figcaption style="color:#f5f5f5;background:rgba(0,0,0,.45);padding:.6rem .8rem;border-radius:8px;">
    <strong>Example aligned loadings: Group PCA.</strong> Brain regions × components matrix (unit-normalised) after
    cross-subject/method alignment. Warmer/colder cells mark stronger positive/negative loadings; light colours near zero
    indicate sparse, non-overlapping contributions—useful for efficient representation and reduced redundancy.
  </figcaption>
</figure>


<figure class="l-page">
  <img loading="lazy" alt="Triangular heatmap of cross-method component correlations after alignment"
       src="{{ '/assets/img/2025-10-25-phase-diagram-playbook/Screenshot%202025-10-26%20183612.png' | relative_url }}">
  <figcaption style="color:#f5f5f5;background:rgba(0,0,0,.45);padding:.6rem .8rem;border-radius:8px;">
    <strong>Cross-method component correlations (post-alignment).</strong> Each block compares components across the four
    methods. A sharp diagonal with light off-diagonals shows one-to-one matches and little superfluous overlap, validating
    that different methods recover consistent latent directions.
  </figcaption>
</figure>


<figure class="l-page">
  <img loading="lazy" alt="Heatmap of the final consensus transformation matrix"
       src="{{ '/assets/img/2025-10-25-phase-diagram-playbook/Screenshot%202025-09-08%20083629.png' | relative_url }}">
  <figcaption style="color:#f5f5f5;background:rgba(0,0,0,.45);padding:.6rem .8rem;border-radius:8px;">
    <strong>Final consensus transformation.</strong> Regions × components loadings that define the population-universal
    latent space used for all subjects. The mapping is sign-harmonised and unit-normalised so that latents have consistent
    semantics across mice and runs.
  </figcaption>
</figure>


<figure class="l-page">
  <img loading="lazy" alt="Correlation matrix of consensus component time series for one subject"
       src="{{ '/assets/img/2025-10-25-phase-diagram-playbook/Screenshot%202025-10-26%20183651.png' | relative_url }}">
  <figcaption style="color:#f5f5f5;background:rgba(0,0,0,.45);padding:.6rem .8rem;border-radius:8px;">
    <strong>Consensus latent space — within-subject orthogonality.</strong> Correlation matrix of consensus component
    time courses for an example mouse. Near-zero off-diagonals (light colours) indicate low redundancy between latents,
    which aids binarisation and stabilises the downstream PMEM/ELA/PDA steps.
  </figcaption>
</figure>


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


## 3) Binarisation of latent time series

After alignment and dimensionality reduction, each latent $$x_i(t)$$ is thresholded per-latent using median or mean and mapped to $$s_i(t) \in \{-1,+1\}$$.

This respects component-specific baselines, keeps PMEM tractable, and standardises inputs for inter-subject comparability.

<!-- Binarisation: proportion of +1 states over time -->
<figure class="l-page">
  <img src="{{ '/assets/img/2025-10-25-phase-diagram-playbook/Screenshot%202025-10-26%20183815.png' | relative_url }}"
       alt="Proportion of +1 (active) binary states across latents over time with linear trend, min and max markers">
  <figcaption style="color:#f5f5f5;background:rgba(0,0,0,.45);padding:.6rem .8rem;border-radius:8px;">
    <strong>Binarisation sanity check: activity fraction over time.</strong>
    The blue trace shows, at each time step, the proportion of latents in the +1 state after per-latent
    median thresholding. The grey dashed line marks 0.5; orange dotted lines show the observed range
    (min ≈ 0.20, max ≈ 0.80). The red dashed line is the fitted linear trend
    (+0.052 percentage points per 100 steps; total change ≈ 1.57 pp). Near-stationary behaviour centred
    around 0.5 indicates balanced on/off usage and supports downstream PMEM fitting; large drifts would
    flag thresholding issues or residual global trends.
  </figcaption>
</figure>


---

## 4) Pairwise maximum-entropy (Ising) fitting
We model each time‑point’s binary latent vector $$\mathbf{s}\in\{-1,+1\}^N$$ with the pairwise maximum‑entropy (PMEM/Ising) distribution:
$$
P(\mathbf{s}) \propto
\exp\Big(\sum_{i} h_i s_i + \sum_{i<j} J_{ij} s_i s_j\Big),
\qquad
E(\mathbf{s}) = -\sum_{i} h_i s_i - \tfrac12 \sum_{i\ne j} J_{ij} s_i s_j
$$

PMEM matches the empirical first and second moments with minimal assumptions while remaining expressive for mesoscopic neural populations (as well as on the scale of entire networks of brain regions, and extending naturally to dynamical systems beyond neuroscience) <d-cite key="jaynes1957maxent,schneidman2006nature"></d-cite>.

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
  \begin{aligned}
  \Sigma^{-1} &= \Lambda_0 + T\,C_\eta,\\
  \mu &= \theta_0 + \Sigma\,T\big(\bar{\Phi} - m_\eta\big),
  \end{aligned}
  $$

  where
  $$
  m_\eta = \mathbb{E}_{p_\eta}[\Phi],
  \qquad
  C_\eta = \operatorname{Cov}_{p_\eta}(\Phi),
  $$
  are the model moments/curvature evaluated at the current anchor $$\eta=\mu$$.

 
  Credible intervals come from $$\Sigma$$; optional Gamma hyper‑priors give ARD‑style shrinkage.

  Here are pasting-ready figures + captions with visible text colour.

---

Got it — here are the same two figures with **light, readable captions** (inline styles), in plain HTML for copy–paste.

---


<!-- VB Ising / uncertainty diagnostics -->

<figure class="l-page">
  <img
    src="{{ '/assets/img/2025-10-25-phase-diagram-playbook/Screenshot%202025-09-09%20234938.png' | relative_url }}"
    alt="Variational Bayes Ising diagnostics: uncertainty vs magnitude, posterior sd(J) maps, ELBO trajectory, field uncertainties, ARD precision spectrum, data–model probability agreement, CV histogram, fit-quality indices">
  <figcaption style="color:#f5f7ff; text-shadow:0 1px 2px rgba(0,0,0,.55);">
    <strong>Variational-Bayes PMEM: uncertainty and fit checks.</strong>
    Panels summarise one VB run:
    (i) posterior uncertainty vs coupling magnitude;
    (ii, v) heatmaps of posterior s.d. for couplings (diagonal masked; log-scale variant);
    (iii) ELBO trajectory decreasing across iterations (convergence);
    (iv) histogram of field uncertainties;
    (vi) ARD precision spectrum indicating data-driven shrinkage;
    (vii) model-vs-empirical state probabilities (log–log; closer to the diagonal is better);
    (viii) histogram of coupling coefficient-of-variation \( \mathrm{CV}=\sigma/|\mu| \);
    (ix) two fit-quality indices (moment-matching accuracy). Together these quantify parameter credibility and goodness-of-fit before ELA/PDA.
  </figcaption>
</figure>

<!-- Energy–probability diagnostic -->

<figure class="l-page">
  <img
    src="{{ '/assets/img/2025-10-25-phase-diagram-playbook/Screenshot%202025-09-09%20234218.png' | relative_url }}"
    alt="Energy–probability diagnostic: empirical probability vs shifted energy with basin colours, histograms and slope fit">
  <figcaption style="color:#f5f7ff; text-shadow:0 1px 2px rgba(0,0,0,.55);">
    <strong>Energy–probability diagnostic.</strong>
    Empirical pattern probabilities \( P_{\mathrm{emp}}(\sigma) \) vs shifted energies \( E(\sigma)-E_{\min} \).
    An approximately linear trend in log-probability vs energy (dashed fit; slope and \( R^2 \) shown) is consistent with Boltzmann structure.
    Points are coloured by basin; the circled marker denotes the global minimum. Marginal histograms summarise energy and count distributions.
    Deviations at the extremes flag rare states and help identify outliers before landscape and phase-diagram analyses.
  </figcaption>
</figure>
::contentReference[oaicite:0]{index=0}


### Fit quality & sanity

We report (i) moment matching (means and pairwise correlations), (ii) multi‑information explained and KL‑reduction vs. independence, and (iii) empirical‑vs‑Boltzmann pattern agreement. 

For Monte‑Carlo checks we use multi‑chain sampling<d-cite key="metropolis1953mcmc"></d-cite> with **Ř**/effective sample size (ESS) <d-cite key="vehtari2021rhat"></d-cite> diagnostics and estimate observables (magnetisation $$m$$, Edwards–Anderson $$q$$, spin‑glass and uniform susceptibilities $$\chi_{\mathrm{SG}},\chi_{\mathrm{Uni}}$$, specific heat $$C$$).

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
* **Attractors and basins:** Local minima are states whose single‑spin flips all increase energy. Every state is assigned to a basin by steepest‑descent (or best‑improving) paths. We summarise basins by occupancies and a disconnectivity graph <d-cite key="becker1997disconnectivity,wales2006jpcb"></d-cite>.
* **Barriers and disconnectivity:** The barrier between basins $$(\alpha,\alpha')$$ is the minimum, over all paths connecting them, of the maximum energy along the path; the disconnectivity graph visualises these heights. Denoting the (symmetrised) minimal saddle by $$\overline{E}_{\alpha\alpha'}$$, 

$$
\overline{E}_{\alpha\alpha'} = \min_{\gamma: \alpha\to\alpha'}\ \max_{\mathbf{s}\in\gamma} E(\mathbf{s}).
$$


* We also estimate a depth threshold (null‑model percentile) to guard against spurious minima.


**Kinetics (Markov view):** Build a single-spin-flip Metropolis chain with proposal “flip one spin uniformly” <d-cite key="metropolis1953mcmc"></d-cite> and transition probability:

$$
p(\mathbf{s}\to\mathbf{s}^{(i)}) = \frac{1}{N}\min\{1,\ \exp\big(E(\mathbf{s})-E(\mathbf{s}^{(i)})\big)\},
$$

where $$\mathbf{s}^{(i)}$$ is $$\mathbf{s}$$ with spin $$i$$ flipped. This yields a $$2^N\times 2^N$$ transition matrix $$P$$ (or a restriction to the visited subgraph).

**From $$P$$ we derive:**

* Stationary distribution $$\pi$$, dwell-time distributions, and basin occupancy.
* Mean first-passage times (MFPT): from a set $$A$$ to $$B$$ via the fundamental matrix $$Z=(I-Q)^{-1}$$ of the transient block <d-cite key="kemeny1960finite"></d-cite>.
* Committors $$q_{AB}$$ solving $$(I-Q)q=b$$, where $$b$$ collects transitions into $$B$$ <d-cite key="e2006tpt"></d-cite>.
* Relaxation spectrum (mixing time-scales): non-unit eigenvalues $$\lambda_i(P)$$ with $$\tau_i=-1/\log\|\lambda_i\|$$, and the Kemeny constant (mean mixing time) $$K=\sum_{i\ge2}\frac{1}{1-\lambda_i}$$.

**Read-outs:** (i) attractor maps (patterns + labels), (ii) disconnectivity graphs, (iii) barrier distributions, (iv) transition/reachability matrices (one-step and multi-step), and (v) kinetic summaries (MFPT heatmaps, committor fields, relaxation spectra, Kemeny constants). These quantify stability, switching propensity, and heterogeneity of access between states.

Crucially, these mechanistic and interpretable descriptors and metrics provide an additional high-level framework for comparing brain dynamics across different individuals, or even cohorts with systematically divergent patterns of neural activity - a discrete and more intuitive alternative to classic means for unifying/juxtaposing representations in computational systems. 

<!-- Energy Landscape Analysis (ELA) – composite panel --> <figure class="l-page"> <img src="{{ '/assets/img/2025-10-25-phase-diagram-playbook/Screenshot%202025-10-22%20225317.png' | relative_url }}" alt="Energy Landscape Analysis (ELA) panel with attractor patterns, 3D energy surface with basins and paths, transition matrices, disconnectivity graph, basin visit counts and basin sizes"> <figcaption style="color:#f5f7ff; text-shadow:0 1px 2px rgba(0,0,0,.55);"> <strong>Energy-Landscape Analysis (ELA) — summary descriptors.</strong> The composite figure illustrates the standard read-outs used downstream of the fitted Ising model: <br> <em>(A)</em> <strong>Local-minimum patterns</strong> (binary states for each attractor); <em>(B)</em> <strong>3-D energy surface</strong> with labelled minima (white dots) and most-probable transition paths (white arrows); <em>(C)</em> <strong>Direct transition counts</strong> between minima (Metropolis single-flip kernel); <em>(D)</em> <strong>Disconnectivity graph</strong> showing barrier heights that separate basins; <em>(E)</em> <strong>Basin visit frequencies</strong> (empirical occupancy); <em>(F)</em> <strong>Basin sizes</strong> (number of micro-states per basin in state-space); <em>(G)</em> <strong>Direct/indirect transition counts</strong> summarising multi-step reachability. Deeper basins and higher barriers indicate more stable, harder-to-leave states; denser transition lanes point to preferred switching routes. </figcaption> </figure>


---
## 6) Phase‑Diagram Analysis (PDA): multi‑observable placement

**Goal:** 
Place every subject on a *shared* Sherrington–Kirkpatrick‑like $$(\mu,\sigma)$$ phase surface using *multiple* observables at once, with uncertainty, so that cohorts become directly comparable without needing a fixed “healthy baseline”. PDA sits downstream of our shared‑latent → binarisation → Ising (PMEM) fit, and is designed to be robust, auditable, and reproducible from end to end. <d-cite key="edwards1975ea,sherrington1975sk,ezaki2020critical"></d-cite>

---

### 6.1  What PDA estimates (observables and phase coordinates)

For a fitted Ising model on binary latents $$\mathbf{s}\in\{-1,+1\}^N$$ with fields $$h_i$$ and couplings $$J_{ij}$$, PDA uses macroscopic observables:

- **Magnetisation**  
  $$m=\frac{1}{N}\sum_i \langle s_i\rangle$$

- **Edwards–Anderson order**  
  $$q=\frac{1}{N}\sum_i \langle s_i\rangle^2$$

- **Spin‑glass susceptibility** (using the eigenvalues $$\{\lambda_k\}$$ of the spin covariance)  
  $$\chi_{\mathrm{SG}}=\frac{1}{N}\sum_{k=1}^{N}\lambda_k^2$$

- **Uniform susceptibility**  
  $$\chi_{\mathrm{Uni}}=\frac{1}{N}\sum_{i\neq j}\mathrm{Cov}(s_i,s_j)$$

- **Specific heat** (from energy variance)  
  $$C=\frac{\langle E^2\rangle-\langle E\rangle^2}{N},\qquad
  E(\mathbf{s})=-\sum_i h_i s_i-\frac{1}{2}\sum_{i\neq j}J_{ij}s_is_j$$

In practice, we compute these from Monte‑Carlo samples of the fitted model (with automatic convergence checks). For quick diagnostics, the first four are also obtainable directly from the binarised data.

---

### 6.2  The reference phase surface $$(\mu,\sigma)\mapsto\{m,q,\chi_{\mathrm{SG}},\chi_{\mathrm{Uni}},C\}$$

We construct a *reference* coupling matrix $$J_{\mathrm{ref}}$$ (either *pooled* across the cohort or *control*‑only), then generate a dense grid over target $$(\mu,\sigma)$$ by *affinely transforming the off‑diagonal entries* of $$J_{\mathrm{ref}}$$. Let $$\mu_{\text{old}}$$ and $$\sigma_{\text{old}}$$ be the mean and std of off‑diagonal entries of $$J_{\mathrm{ref}}$$. For a target $$(\mu,\sigma)$$:

$$
J_{ij}^{(\mu,\sigma)}=
\begin{cases}
\big(J_{ij}-\mu_{\text{old}}\big)\dfrac{\sigma}{\sigma_{\text{old}}+\varepsilon}+\mu, & i\neq j,\\[6pt]
0, & i=j~,
\end{cases}
$$

with all **fields zeroed** $$h_i\equiv 0$$ (diagonal remains 0) to recover the canonical spin‑glass phase structure. For each grid point we Monte‑Carlo sample the observables above and cache five surfaces $$\{\mathrm{m},\mathrm{q},\chi_{\mathrm{SG}},\chi_{\mathrm{Uni}},C\}$$.

**Working vs display grids:** We build a high‑resolution *working* grid used for optimisation/placement and optionally a wider *display* grid for visuals. We automatically pick $$(\mu,\sigma)$$ limits by running quick PL fits per subject to estimate native $$(\hat\mu,\hat\sigma)$$, then expand by a safety factor; we refuse overly coarse grids (e.g., if $$\Delta\mu$$ or $$\Delta\sigma>0.01$$).

**MC convergence safeguards:** We run multiple chains with increasing sweep budgets until *all* observables reach $$\hat R<1.05$$ and an effective sample size threshold, or a maximum sweep cap is hit (in which case a warning is issued).

---

### 6.3  Multi‑observable placement (cost projection with balanced weights)

Given a subject’s *observed* $$(m_o,q_o,\chi^{o}_{\mathrm{SG}},\chi^{o}_{\mathrm{Uni}})$$ from their binarised latents, we **project** onto the reference surface by minimising a *variance‑balanced* squared error:

$$
\underset{\mu,\sigma}{\arg\min}\;\sum_{k\in\{\mathrm{m},\mathrm{q},\chi_{\mathrm{SG}},\chi_{\mathrm{Uni}}\}}
w_k\big(O_k^{\text{obs}}-\widehat{O}_k(\mu,\sigma)\big)^2,
$$

where $$\widehat{O}_k(\mu,\sigma)$$ is obtained by regular‑grid interpolation of the precomputed surfaces and weights $$w_k$$ default to the inverse *range* of each surface (less sensitive than $$1/\mathrm{var}$$). Optimisation uses L‑BFGS‑B on the working grid bounds; we also provide an **iso‑curve** fallback (intersect level sets of $$\chi_{\mathrm{SG}}$$ and $$\chi_{\mathrm{Uni}}$$ with a simple border‑safety check) and a brute‑force grid fallback.

We return $$(\hat\mu,\hat\sigma)$$, the final cost value, and the method used (“cost_minimisation”, “iso_curves”, or “fallback_grid”).

---

### 6.4  Uncertainty, robustness, and diagnostics

**Bootstrap CIs:** For each subject we run a *circular block bootstrap* along time, refit the Ising per resample, and recompute $$(\mu,\sigma)$$ or $$(m,q,\chi_{\mathrm{SG}},\chi_{\mathrm{Uni}})$$ as needed to report 95% intervals. Specific heat $$C$$ can be bootstrapped likewise via short MC runs per resample.

**Control shrinkage (optional):** When a control is available, we stabilise its estimate by bootstrapping its $$J$$, pooling across the cohort, and forming a convex combination $$J^{\text{ctrl,shrunk}}=(1-\lambda)J^{\text{ctrl}}+\lambda J^{\text{pooled}}$$; we then summarise $$(\mu_c,\sigma_c)$$ from the shrunk $$J$$.

**Cost bowls:** Around each optimum we display the *local cost landscape* (2‑D filled contour and 3‑D surface) to judge identifiability; narrow, well‑curved bowls suggest precise placement.

**Group‑level tests:** Small helpers allow groupwise comparisons in $$\sigma$$ (e.g., Welch ANOVA and permutation tests) using the bootstrapped distributions.

**Critical structure:** We plot *critical contours* (e.g., a fixed fraction of the maximum of an observable) on the display surfaces; a simple near‑criticality index is the minimal Euclidean distance from $$(\hat\mu,\hat\sigma)$$ to the chosen contour.

---

### 6.5  Practical recipe (what the code actually does)

1) **Prepare data**: shared latents → binarise per latent to $$\pm1$$.  
2) **Rough bounds**: quick PL fits to get subject‑wise $$(\hat\mu,\hat\sigma)$$; expand to working ranges; verify resolution is fine enough.  
3) **Build reference surface**: choose `reference_mode ∈ {pooled, control}`, set $$h\equiv 0$$, sweep $$(\mu,\sigma)$$ by affine off‑diagonal transforms of $$J_{\mathrm{ref}}$$, run multi‑chain MC with convergence checks, and cache $$\{\mathrm{m},\mathrm{q},\chi_{\mathrm{SG}},\chi_{\mathrm{Uni}},C\}$$ surfaces.  
4) **Place subjects**: minimise the balanced multi‑observable cost (or use the iso‑curve / grid fallbacks with border guard).  
5) **Uncertainty**: bootstrap along time to report CIs; optionally shrink the control to summarise $$(\mu_c,\sigma_c)$$.  
6) **Visuals**: 2‑D contour panels and interactive 3‑D surfaces, with group‑colours and critical lines; “cost bowls” per subject for identifiability.

---

### 6.6  Mathematical and implementation notes (exact to this workflow)

- **Affine mapping of $$J$$ to $$(\mu,\sigma)$$** modifies *only* off‑diagonals and preserves $$J_{ii}=0$$. This avoids artefactual self‑coupling and keeps the spin‑glass structure intact. Fields $$h$$ are explicitly zeroed for the surface.  
- **Observables from data vs model.** Fast “data‑side” $$m,q,\chi$$ provide immediate checks, while MC‑side estimates (including $$C$$) are used to build the surface; both routes are available.  
- **Balanced weights** $$w_k$$ default to inverse *range* (normalises sensitivity without over‑penalising heavy‑tailed metrics). Equal weights are also supported.  
- **Convergence gating** uses multi‑chain Gelman-Rubin $$\hat R$$ and a conservative ESS check with geometric back‑off of sweeps up to a cap; we report if the cap is reached.  
- **Border guard** prevents pathological “corner locking” when intersecting $$\chi$$‑iso‑curves on coarse grids.

---

### 6.7  Interpretation tips

- **$$\mu$$** (mean coupling) captures net ordering tendency; **$$\sigma$$** (coupling dispersion) captures disorder/glassiness. Movement along $$\sigma$$ at constant $$\mu$$ corresponds to increasing heterogeneity at fixed mean interaction strength.  
- **High $$\chi_{\mathrm{SG}}$$** with muted $$m$$ signals a spin‑glass‑like regime (multiple competing basins), while **high $$m$$ with low $$\chi_{\mathrm{SG}}$$** indicates ferromagnetic‑like ordering.  
- **Specific heat $$C$$** often peaks near phase boundaries and can indicate broad susceptibility to perturbations in this coarse‑grained description.

---

### 6.8  Reproducibility knobs (defaults)

- MC: chains = 12, start sweeps = 8k, burn‑in = 1k, cap = 128k, $$\hat R$$ tol = 1.05.  
- Working grid: $$140\times140$$ by default; guard: $$\max(\Delta\mu,\Delta\sigma)\le 0.01$$.  
- Objective: weights = “balanced” (inverse range), optimiser = L‑BFGS‑B within bounds.  
- Bootstrap: circular blocks, user‑set size; control shrinkage parameter $$\lambda\in[0,1]$$.

---

### 6.9  Illustrative pseudocode (orientation only — code is not released here)

> The snippet below mirrors the sequence used to generate the figures; it is descriptive rather than an API.

```python
# 1) Build the reference phase surfaces (pooled or control)
phase = build_phase_surfaces(
    binaries_pm1 = BINARIES_PM1_OR_01,         # dict: subject → DataFrame (±1)
    reference    = {"mode": "pooled", "subject": None},  # or {"mode": "control","subject": "ID"}
    grid_size    = 150,                         # working grid resolution
    mc_settings  = {"chains": 12, "burn_in": 1000, "start_sweeps": 8000}
)

# 2) Place subjects via multi-observable cost (balanced weights by default)
positions = place_subjects_on_surface(
    binaries_pm1 = BINARIES_PM1_OR_01,
    phase        = phase,
    method       = "cost",              # or "iso_curves"
    weights_mode = "balanced"
)

# 3) Inspect local identifiability (“cost bowl”) for one subject
w = objective_weights(phase, mode="balanced")
plot_cost_landscape(
    subject_id = "SUBJECT_ID",
    positions  = positions,
    phase      = phase,
    window     = 0.06,
    steps      = 200,
    weights    = w
)

```

---

### 6.10  Caveats specific to PDA

- PDA assumes an SK‑like parameterisation (coupling distribution matters); it analyses **macrostates** and does not replace ELA’s mechanistic, basin‑level descriptors.  
- The choice of reference surface (pooled vs control) can shift placements; we therefore expose the cost bowls and allow both options to be reported.  
- Grid resolution and MC budgets matter near sharp boundaries; guards and diagnostics make this explicit.

<div class="l-page">
  <iframe src="{{ '/assets/plotly/pda_surface.html' | relative_url }}" frameborder="0" scrolling="no" height="520px" width="100%" style="border: 1px dashed grey;"></iframe>
</div>

---

## Results in a word
On resting-state functional ultrasound (fUS) recordings (mesoscopic, whole-brain), we observe low placement residuals $$(10^{-6}–10^{-4})$$, tight bootstrap confidence regions, convergent models, and stable ordering under pooled vs subgroup phase references; example estimates span **σ ≈ 0.15–0.32** and **μ ≈ −0.01 to +0.03**. The outputs - susceptibility to perturbations, ordering vs glassiness, transition propensity - form compact, biologically meaningful fingerprints. 

For experimentalists, this is a mechanistic dashboard (what
states exist, how deep, how likely to switch); for theorists, it anchors subjects in a physics-grounded phase
space with interpretable axes. 

Overall, the multi-observable placement and the combination of shared embeddings with ELA/PDA provide reliability-minded, comparable, and interpretable read-outs that support discovery, phenotyping, and model-based hypothesis generation across cohorts, data modalities, tasks, and species.

---

## Robustness, uncertainty and diagnostics

- Uncertainty via variational-Bayes posteriors for h and J; bootstrap intervals for mu and sigma and for near-criticality; block bootstrap for autocorrelation <d-cite key="politis1992cbb,politis2004blocklength"></d-cite>
- Convergence checks including MCMC R-hat and effective sample size <d-cite key="vehtari2021rhat"></d-cite> where applicable, pseudo-likelihood relative-norm stopping, and ELBO improvements for variational Bayes
- Quality-control gates including permutation or null thresholds for spurious minima, component-stability filters, and sensitivity to number of latents and alignment choice
- Ablations: pooled versus subgroup reference surfaces; method toggles among SRM, MCCA, and ICA; median versus mean thresholds

---
## Reproducibility and artefacts

We report key settings and diagnostics to make the computational provenance clear, even though the code is not released with this post. Runs are seed-controlled with machine-parsable configuration files; convergence (e.g., R-hat, ESS) and grid resolution checks are documented in the figure captions and text.

**Example configuration stub (indicative):**

```yaml
# example-config.yaml
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
  ```

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

{% details Click to expand the entire Appendix %}

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

{% enddetails %}

---