---
layout: distill
title: Separatrix Locator
description: Finding Separatrices with Deep squashed Koopman Eigenfunctions
tags: distill formatting
giscus_comments: true
date: 2025-09-28
featured: true

authors:
  - name: Kabir V. Dabholkar
    url: "https://kabirdabholkar.github.io"
    affiliations:
      name: Faculty of Mathematics, Technion
  - name: Omri Barak
    url: "https://barak.net.technion.ac.il"
    affiliations:
      name: Rappaport Faculty of Medicine and Network Biology Research Laboratories, Technion

bibliography: 2025-09-29-Separatrix-Locator.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: "Introduction: Beyond Fixed Points"
    subsections:
    - name: "Setting"
  - name: "The Sandwich of Bistability"
  - name: "Enter Deep Neural Networks"
    subsections:
    - name: "Degeneracies and how to fight them"
  - name: "Relation to Koopman theory"
  - name: "Tips and tricks"
    subsections:
    - name: "Architecture"
    - name: "Scaling x"


# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---



## Introduction: Beyond Fixed Points
Many natural and artificial systems — from neural circuits making decisions to ecosystems switching between healthy and diseased states — are modeled as **multistable dynamical systems**. Their behavior is governed by multiple **attractors** in state space, each corresponding to a stable mode of activity. Understanding these systems often boils down to understanding their **geometry**: where are the stable states, and how are the different basins of attraction separated?

For the last decade, a workhorse of neural circuit analysis has been **fixed point analysis**. By finding points where the flow vanishes and linearizing around them, one can uncover local motifs underlying computation: line attractors, saddle structures, rotational channels, and so on. This has yielded powerful insights into how trained RNNs implement cognitive tasks.

But fixed points are only half the story.

When a system receives a perturbation — for example, a sensory input or an optogenetic pulse — the key question is often not *where* it started, but *which side of the separatrix it ends up on*. The **separatrix** is the boundary in state space separating different basins of attraction. Crossing it means switching decisions, memories, or ecological states. Failing to cross means staying put. For high-dimensional systems, these boundaries are typically **nonlinear, curved hypersurfaces**, invisible to fixed points and local linearisations.

> **What if we could learn a single smooth scalar function whose zero level set *is* the separatrix?**  
> This is the central idea behind **(squashed) Koopman eigenfunctions**.


<!-- Motivation: many dynamical systems have multiple attractors. decisions, memories, ecological states.

Fixed point analysis is powerful but local.

Separatrices (basin boundaries) are critical for predicting effects of perturbations (e.g., optogenetics).

Teaser figure: 2D bistable system with flow, fixed points, kinetic energy vs separatrix vs squashed KEF zero contour.

“What if we could learn a single scalar function whose zero level set is the separatrix?” -->

### Setting
We consider autonomous dynamical flows of the form:

$$\begin{equation}
\dot{\boldsymbol{x}} = f(\boldsymbol{x})
\label{eq:ODE}
\end{equation}$$ 

governing the state  $$\boldsymbol{x} \in \mathbb R^N$$,
where $$\dot \square$$ is shorthand for the time derivative $$\frac{d}{dt}\square$$ and $$f: \mathbb R^N \to \mathbb R^N$$ defines the dynamical flow.

> **The Goal:**  
> Find a smooth scalar function $$\psi:\mathbb{R}^N\to\mathbb{R}$$ that encodes the distance from the separatix, i.e., $$\psi(\boldsymbol x)=0$$ for $$x\in\text{separatix}$$ and grows as it moves away from the separatrix.
{: .goal-box }


## The Sandwich of Bistability
Any bistable system can decomposed as follows: it will have two attractors, their respective basins of attraction and the separatrix between them. This is like a cheese sandwich: the attractors are slices of bread, and the separatrix is the slice of cheese between them. We can call this the **Sandwich of Bistability**. In general, this sandwich could be arbitrarily oriented in $$\mathbb R^N$$ and even nonlinearly warped. 

<div style="text-align: center;">
  <img src="/blog/assets/img/2025-09-29-Separatrix-Locator/sandwich_of_bistability.png" alt="Sandwich of Bistability" width="500" />
</div>

*The Sandwich of Bistability: Two attractors (bread slices) separated by a separatrix (cheese slice).*


With our scalar function $$\psi:\mathbb{R}^N\to\mathbb{R}$$ we would like to perform a special type of dimensionality reduction: we only care to identify our location along the attractor -- separatrix -- attractor axis, i.e., along the depth of sandwich. 

One way to achieve this is to have this scalar observable $$\psi(\boldsymbol x)$$ *imitate* the bistable dynamics along this axis. Thus we pick a simple example of bistable dynamics in 1D:

$$
\begin{equation}
\dot \psi = \lambda (\psi-\psi^3)
\label{eq:sKEFsimple}
\end{equation}
$$

with $$\lambda>0$$, dropping the $$\boldsymbol x$$ notation for a moment for clarity.  This system has fixed point attractors at $$\pm 1$$ and an unstable fixed point (a separatrix) at $$0$$ -- a 1D Sandwich of Bistability.

Now we want to couple the $$\psi$$ dynamics with the $$\boldsymbol x$$ dynamics so we bring back the $$\boldsymbol x$$ dependence. Specifically as $$\boldsymbol x(t)$$ evolves in time according to $$\eqref{eq:ODE}$$:

$$\begin{equation}
\frac{d}{dt}\bigg(\psi\big(\boldsymbol{x}(t)\big)\bigg) = \lambda\bigg[\psi\big(\boldsymbol{x}(t)\big) - \psi\big(\boldsymbol{x}(t)\big)^3\bigg].
\label{eq:sKEF}
\end{equation}$$

This means that if we were to *observe* the value of $$\psi(\boldsymbol x)$$ as $$\boldsymbol x$$ evolved in time, that value would evolve according $$\eqref{eq:sKEFsimple}$$.

By the chain rule, the left hand side of equation $$\eqref{eq:sKEF}$$ is:

$$\begin{equation}
\frac{d}{dt}\bigg(\psi\big(\boldsymbol{x}(t)\big)\bigg) = \nabla_{\boldsymbol{x}}\psi\big(\boldsymbol{x}(t)\big) \cdot \dot{\boldsymbol{x}}(t)
\label{eq:chainrule}
\end{equation}$$

Substituting $$\eqref{eq:ODE}$$ and equating with the right hand side of $$\eqref{eq:sKEF}$$:

$$\begin{equation}
\nabla_{\boldsymbol{x}}\psi(\boldsymbol{x}) \cdot f(\boldsymbol{x}) = \lambda[\psi(\boldsymbol{x}) - \psi(\boldsymbol{x})^3]
\label{eq:sKEFPDE}
\end{equation}$$

This is a first order nonlinear partial differential equation (PDE) for $$\psi(\boldsymbol{x})$$. If we can find a function $$\psi$$ that satisfies this PDE, then its zero level set will be the separatrix we seek, right? Not quite, unfortunately $$\eqref{eq:sKEFPDE}$$ also admits several unhelpful solutions. We address these in [Degeneracies and how to fight them](#degeneracies-and-how-to-fight-them).

But the first challenge is to solve this PDE for high-dimensional nonlinear system. This is where deep neural networks come in...

## Enter Deep Neural Networks 
### Degeneracies and how to fight them


## Relation to Koopman theory


## Tips and tricks

Sometimes training fails. Here are some things to know. We have not run systematic tests on all of these: it's more from trial and error. So take them with a grain of salt.

### Architecture
Choices that improve training convergence:
- DNNs with `tanh` over `ReLU`,
- adding residual layers helps
- deeper and wider networks improved convergence, at the cost of compute.
- larger batch size helps

### Scaling $$\boldsymbol x$$

Mind the saturation of `tanh`. Normalise or whitten $$\boldsymbol{x}$$ before feeding to the DNN so that it doesn't saturate the activation function. We include this as a fixed scalar parameter in the forward pass definition so that it's part of $$\psi(\boldsymbol x)$$ and not a separate pre-processing step.

### Ensuring $$\vert\psi\vert<1$$ at initialisation and that $$\langle\psi\rangle\approx0$$.