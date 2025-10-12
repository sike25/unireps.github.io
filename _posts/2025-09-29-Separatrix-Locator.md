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
  .equation-with-plot {
    position: relative;
    display: inline-block;
    cursor: help;
  }
  .equation-with-plot .plot-tooltip {
    visibility: hidden;
    opacity: 0;
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    background: white;
    border: 2px solid #333;
    border-radius: 8px;
    padding: 0;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    z-index: 1000;
    transition: opacity 0.3s, visibility 0.3s;
    margin-bottom: 10px;
    width: 420px;
    height: 320px;
  }
  .equation-with-plot:hover .plot-tooltip {
    visibility: visible;
    opacity: 1;
  }
  .plot-tooltip iframe {
    display: block;
    width: 100%;
    height: 100%;
    border: none;
    border-radius: 8px;
  }
---



## Introduction: Fixed Points and beyond
Many natural and artificial systems — from neural circuits making decisions to ecosystems switching between healthy and diseased states — are modeled as **multistable dynamical systems**. Their behavior is governed by multiple **attractors** in state space, each corresponding to a stable mode of activity. Understanding these systems often boils down to understanding their **geometry**: where are the stable states, and how are the different basins of attraction separated?

For the last decade, a workhorse of neural circuit analysis has been **fixed point analysis**. By finding points where the flow vanishes and linearizing around them, one can uncover local motifs underlying computation: line attractors, saddle structures, rotational channels, and so on. This has yielded powerful insights into how trained RNNs implement cognitive tasks.

### Finding fixed points
First consider a bistable dynamical system in 2 dimensions. Below is a phase-portrait of such a system, with two stable fixed points and one unstable fixed point. Click on plot to realise trajectories of the dynamics.

<div class="l-body" style="text-align: center; margin: 2rem 0;">
  <iframe src="/blog/assets/html/clickable_phase_portrait_simple.html" 
          scrolling="no"
          style="width: 100%; height: 600px; border: none; border-radius: 8px; overflow: hidden;">
  </iframe>
</div>

Trajectories converge to either one of the two fixed points. This naturally provides an algorithm to find the stable fixed points: just run evolve the dynamics from many initial conditions.

Notice, however, that it is challenging to plot a trajectory that precisely intercepts the unstable fixed point due to it's repulsivity. This motivates developing a principled way to find such points. One solution is to define a specific scalar function of the dynamics whose only minima are given by all the fixed points. One such function is the kinetic energy $$q(\boldsymbol x)=\frac{1}{2}\Vert f(\boldsymbol x)\Vert^2$$ <d-cite key="sussillo_opening_2013,golub_fixedpointfinder_2018"></d-cite>. By differentiating this function, one can perform gradient descent to find these minima. The interactive plot below realises such trajectories.

<div class="l-body" style="text-align: center; margin: 2rem 0;">
  <iframe src="/blog/assets/html/gradient_descent_phase_portrait.html" 
          scrolling="no"
          style="width: 100%; height: 600px; border: none; border-radius: 8px; overflow: hidden;">
  </iframe>
</div>

Now we can find both stable *and unstable* fixed points. Linearising around the fixed points provides an interpretable approximation of the dynamics in the neighborhood of those points. Several works adopt this approach of fixed point finding to reverse-engineer either task-trained or data-trained RNNs <d-cite key="carnevale_dynamic_2015,maheswaranathan_reverse_2019,maheswaranathan_universality_2019,finkelstein_attractor_2021,mante_context-dependent_2013,liu_encoding_2024,driscoll_flexible_2024,chaisangmongkon_computing_2017,jaffe_modelling_2023,pagan_individual_2025,wang_flexible_2018"></d-cite>.

But fixed points are only half the story.

When a system receives a perturbation — for example, a sensory input or an optogenetic pulse — the key question is often not *where* it started, but *which side of the separatrix it ends up on*. The **separatrix** is the boundary in state space separating different basins of attraction. Crossing it means switching decisions, memories, or ecological states. Failing to cross means staying put. For high-dimensional systems, these boundaries are typically **nonlinear, curved hypersurfaces**, invisible to fixed points and local linearisations around them.

> **What if we could learn a single smooth scalar function whose zero level set *is* the separatrix?** 

Below is an example of such a function that we constructed for this simple system (click on it to run trajectories). 

<div class="l-body" style="text-align: center; margin: 2rem 0;">
  <iframe src="/blog/assets/html/absolute_value_gradient_descent.html" 
          scrolling="no"
          style="width: 100%; height: 600px; border: none; border-radius: 8px; overflow: hidden;">
  </iframe>
</div>

We provide a method to approximate such functions using deep neural network to find separatrices in multistable dynamical systems embedded in high-dimensions.

<!-- > This is the central idea behind **(squashed) Koopman eigenfunctions**. -->


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
{: .goal-box #goal}


## The Sandwich of Bistability
Any bistable system can decomposed as follows: it will have two attractors, their respective basins of attraction and the separatrix between them. This is like a cheese sandwich: the attractors are slices of bread, and the separatrix is the slice of cheese between them. We can call this the **Sandwich of Bistability**. In general, this sandwich could be arbitrarily oriented in $$\mathbb R^N$$ and even nonlinearly warped. 

<div style="text-align: center;">
  <img src="/blog/assets/img/2025-09-29-Separatrix-Locator/sandwich_of_bistability.png" alt="Sandwich of Bistability" width="500" />
  <div style="max-width: 500px; margin: 0.5rem auto; text-align: center;">
    <em>The Sandwich of Bistability: Two attractors and their basins of attraction (bread slices) separated by a separatrix (cheese slice). We only care about mapping the coordinates along bistable axis.</em>
  </div>
</div>


With our scalar function $$\psi:\mathbb{R}^N\to\mathbb{R}$$ we would like to perform a special type of dimensionality reduction: we only care to identify our location along the attractor -- separatrix -- attractor axis, i.e., along the depth of sandwich. 

One way to achieve this is to have this scalar observable $$\psi(\boldsymbol x)$$ *imitate* the bistable dynamics along this axis. Thus we pick a simple example of bistable dynamics in 1D (hover your cursor over it to see the plot):

<div class="equation-with-plot">

$$
\begin{equation}
\dot \psi = \lambda (\psi-\psi^3)
\label{eq:sKEFsimple}
\end{equation}
$$

<div class="plot-tooltip">
  <iframe src="/blog/assets/html/bistable_1d_plot.html" scrolling="no"></iframe>
</div>

</div>

with $$\lambda>0$$, dropping the $$\boldsymbol x$$ notation for a moment for clarity.  This system has fixed point attractors at $$\pm 1$$ and an unstable fixed point (a separatrix) at $$0$$ -- a 1D Sandwich of Bistability.

<div style="text-align: center;">
  <img src="/blog/assets/img/2025-09-29-Separatrix-Locator/mapping_illustration_horizontal.png" alt="Mapping" width="500" />
  <div style="max-width: 500px; margin: 0.5rem auto; text-align: center;">
    <em>Mapping the high-D state to a 1D bistable system.</em>
  </div>
</div>

Now we want to couple the $$\psi$$ dynamics with the $$\boldsymbol x$$ dynamics so we bring back the $$\boldsymbol x$$ dependence. Specifically as $$\boldsymbol x(t)$$ evolves in time according to $$\eqref{eq:ODE}$$:

$$\begin{equation}
\frac{d}{dt}\bigg(\psi\big(\boldsymbol{x}(t)\big)\bigg) = \lambda\bigg[\psi\big(\boldsymbol{x}(t)\big) - \psi\big(\boldsymbol{x}(t)\big)^3\bigg].
\label{eq:sKEF}
\end{equation}$$

This means that if we were to *observe* the value of $$\psi(\boldsymbol x)$$ as $$\boldsymbol x$$ evolved in time, that value would evolve according $$\eqref{eq:sKEFsimple}$$.

It seems that finding solutions to $$\eqref{eq:sKEF}$$ could be the key to constructing a *separatrix locator*. The value $$\psi(\boldsymbol x)$$ would be the coordinate of $$\boldsymbol x$$ along the bistable axis. This value would be $$0$$ when $$\boldsymbol x$$ is anywhere on the separatrix, exactly our stated [goal](#goal).

## (squashed) Koopman Eigenfunctions 

At this stage, it's worth noticing that the left hand side of $$\eqref{eq:sKEF}$$ is actually a known object called the [Lie derivative](https://en.wikipedia.org/wiki/Lie_derivative) of $$\psi$$ along the flow given by $$f$$, and also known as the infinitesimal generator of the [Koopman operator](https://en.wikipedia.org/wiki/Composition_operator), (See <d-cite key="brunton_notes_2019"></d-cite>).

To make this link explicit, we first define the propogator function $$F_\tau(x(t)):=x(t+\tau)$$ where $x(t)$ is any solution to $$\eqref{eq:ODE}$$. The Koopman operator $$\mathcal K_\tau$$ is defined as 

$$\mathcal K_\tau g = g \circ F_\tau$$

where $$g$$ is any<d-footnote>$$g$$ must belong to a Hilbert space, meaning that it must come with inner product (and it's associated norm), e.g., $$\langle f,g\rangle:=\int_{\mathbb R^N}f(\boldsymbol x)g(\boldsymbol x)d\boldsymbol x$$ thus requiring that the function be square integrable.</d-footnote> scalar function of the state-space $$\mathbb R^N$$. Its infinitesimal generator $$\mathcal K$$ (dropping the subscript) is essentially a time-derivative:

$$\mathcal Kg = \lim_{\tau\to0} \frac{\mathcal K_\tau g - g}{\tau} =\lim_{\tau\to0} \frac{g\circ F_\tau - g}{\tau} = \frac{d}{d\tau} g \circ F_\tau\bigg\vert_{\tau=0}.$$

The last version if evaluated on a trajectory $$x(t)$$ is the left hand side of $$\eqref{eq:sKEF}$$, allowing us to re-writing it compactly as 

$$\begin{equation}
\mathcal K\psi = \lambda (\psi-\psi^3).
\label{eq:sKEF_compact}
\end{equation}$$

Notice that this almost an eigenfunction equation, if we drop the cubic term: 

$$\begin{equation}
\mathcal K\phi = \lambda \phi.
\label{eq:KEF}
\end{equation}$$

Infact, the two problems are closely related. We can show that solutions to $$\eqref{eq:KEF}$$ can be transformed into solutions of $$\eqref{eq:sKEF_compact}$$ and vice versa.

<!-- By the chain rule, the left hand side of equation $$\eqref{eq:sKEF}$$ is: -->

<!-- $$\begin{equation}
\frac{d}{dt}\bigg(\psi\big(\boldsymbol{x}(t)\big)\bigg) = \nabla_{\boldsymbol{x}}\psi\big(\boldsymbol{x}(t)\big) \cdot \dot{\boldsymbol{x}}(t)
\label{eq:chainrule}
\end{equation}$$  -->

<!-- Substituting $$\eqref{eq:ODE}$$ and equating with the right hand side of $$\eqref{eq:sKEF}$$:

$$\begin{equation}
\nabla_{\boldsymbol{x}}\psi(\boldsymbol{x}) \cdot f(\boldsymbol{x}) = \lambda[\psi(\boldsymbol{x}) - \psi(\boldsymbol{x})^3]
\label{eq:sKEFPDE}
\end{equation}$$ -->

<!-- This is a first order nonlinear partial differential equation (PDE) for $$\psi(\boldsymbol{x})$$. It seems that finding solutions to  -->

<!-- If we can find a function $$\psi$$ that satisfies this PDE, then its zero level set will be the separatrix we seek, right? Not quite, unfortunately $$\eqref{eq:sKEFPDE}$$ also admits several unhelpful solutions. We address these in [Degeneracies and how to fight them](#degeneracies-and-how-to-fight-them). Solutions to  -->

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