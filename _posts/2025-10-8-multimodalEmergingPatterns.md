---
layout: distill
title: Understanding Adversarial Vulnerabilities and Emergent Patterns in Multimodal RL
description: Using a simplified multimodal RL agent to explore adversarial vulnerabilities that emerge when using different modalities.
tags: Adversarial Robustness, Reinforcement Learning, Multimodal models
giscus_comments: true
date: 2025-10-08
featured: true

authors:
  - name: Shayan Jalalipour
    url: "shayan2@pdx.edu"
    affiliations:
      name: Portland State University
  - name: Danielle Justo
    affiliations:
      name: Portland State University
  - name: Banafsheh Rekabdar
    affiliations:
      name: Portland State University

# Put a bib file for this post under assets/bibliography/
bibliography: 2025-10-08-multimodalEmergingPatterns.bib

toc:
  - name: Abstract
  - name: Introduction
  - name: Background
    subsections:
      - name: Related Works
      - name: Adversarial Attacks
      - name: Adversarial Defenses
      - name: Soft Actor-Critic Models
  - name: Methodology
    subsections:
      - name: Agent
      - name: Environment
      - name: Adversarial Attack
      - name: Adversarial Defenses
      - name: Adversarial Evaluation
  - name: Results
    subsections:
      - name: Baseline Performance
      - name: Attacking an Un-Defended Model
      - name: Attacking a Defended Model
  - name: Conclusions
  - name: Reproducibility Details
---


## Introduction

Traditional deep learning models are often built around the use of a single input type or modality such as RGB pixel values for images <d-cite key="resnet2015"></d-cite>, or tokenized text for language generation <d-cite key="Radford2018ImprovingLU"></d-cite>. However, recent advances in machine learning have begun to take advantage of the use of multiple modalities <d-cite key="jiao-multi-modal-survey"></d-cite>. This may take on the form of merging modalities for model use: e.g., merging many sensors for an autonomous vehicle <d-cite key="panduru2025exploring"></d-cite>, converting one modality to another (such as image-to-text, text-to-speech, etc.), or providing multiple modalities such as image and language prompting for both large language models <d-cite key="openai2024gpt4technicalreport"></d-cite> and image generation tasks <d-cite key="rombach2022highresolutionimagesynthesislatent"></d-cite>. Multi-modal models have many diverse uses, and their development is an actively researched field.

As multi-modal models grow in popularity, so too does the need to assess their vulnerability to adversarial manipulation. Like all deep learning systems, their reliability, particularly in high-risk applications hinges on robustness against such attacks. Understanding how adversaries can exploit multi-modal systems, and how emerging dynamics between modalities give rise to both robustness and new vulnerabilities, is essential. While traditional models are known to be susceptible to input perturbations and data poisoning, current research offers limited insight into how these threats play out in multi-modal contexts.

Additionally, adversarial attacks on live on-policy models, particularly reinforcement learning agents, pose unique challenges. Unlike static classifiers these agents interact with dynamic environments, requiring tailored attack and defense strategies. The lack of labeled data in their training regimes further complicates implementation and design.

Exploring the effects of adversarial attacks on multi-modal models is quite important, as there are many critical and high-risk uses for RL policies such as in the fields of robotics and autonomous vehicles. In which a malfunctioning policy has potential to cause great bodily harm or significant damage to expensive equipment.

As such, our work seeks to understand the interactions between adversarial attacks, defenses, and various combinations of modalities on a baseline multi-modal RL agent. We show relevant baseline data, and compare and contrast empirical results to reveal how different modalities effect model outputs when attacked both together or separately. We also show results that indicate what influence a single modality may have on model output compared to others, as well as how different modalities behave when left un-defended from an adversary, and how the behavior of modalities may change when defended.

To generate necessary datasets, train baseline models, and evaluate combinations of attacks and defenses on various modalities, we develop and release a pipeline extending open-source code provided by authors of DDiffPG <d-cite key="li2024learningmultimodalbehaviorsscratch"></d-cite>.

Our contributions:
- Provide a testbed for adversarial evaluation of a multi-modal RL agent.
- Empirically characterize how attacking one or both modalities changes behavior.
- Show that defenses introduce emergent patterns across modalities, sometimes improving robustness and sometimes destabilizing training-time adapted policies.


### Background & Related Works
Due to the popularity and prevalence of large language models, much research has gone into how adversarial attacks effect language models as multi-modal models that work on modalities such as text-to-image and image-to-text, with attacks focusing on various image and prompt manipulations <d-cite key="wu2025dissectingadversarialrobustnessmultimodal,wang2025manipulatingmultimodalagentscrossmodal"></d-cite>. However decision-making agents that operate on a combination of sensor inputs have had limited literature, primarily focusing on autonomous vehicles <d-cite key="chi_autonomous_survey2024,roheda2021multimodal"></d-cite>.

As embodied agents begin to gain complexity and popularity, it is equally important to better understand how mixing modalities may alter model security. Learning how to further develop robust multi-modal models is vital for improving trust in agents with high-risk environments and tasks of ever increasing complexity.

### Adversarial Attacks
Adversarial attacks are a branch of machine learning focused on manipulating model behavior in unintended or harmful ways. In particular, adversarial attacks are aimed at a "victim" model often designed with the flaws of a particular type of model or architecture in mind. These attacks typically involve introducing carefully designed "perturbations" to the input, which are intended to mislead or alter the model's outputs. Depending on the attacker's level of access to the internals of the model, attacks are classified as "white box" (full access, such as weights or gradient values), "gray box" (partial access, such as particular values or weights) or "black box" (no access at all, only inputs and outputs can be discerned around the black box). While perturbing inputs is the most common form of adversarial attack, other methods such as dataset "Poisoning" exist which alter training data to induce some desired behavior from the victim model.

### Adversarial Defenses
Defenses against adversarial attacks have three main approaches: Augmenting training processes with adversarial samples <d-cite key="zizzo2021certified,tramer_adaptive_2020,kuzina_defending_2022"></d-cite>, detection of adversarial attacks, perturbations or anomalous data <d-cite key="roth_odds_2019,fidel_when_2020,guo_detecting_2019,GolchinAnomolyDetection"></d-cite>, and removing, filtering, or disrupting adversarial perturbations from the input data <d-cite key="zhang2021defense,nie_diffusion_2022,yoon_adversarial_2021"></d-cite>. For the purposes of our experiments, we utilize 3 methods for defense: Disruption, via gaussian noise; Detection, via neural network classifier and traditional clustering methods; Filtering, via Variational Auto-Encoder (VAE).

### Soft Actor-Critic Models
Soft Actor-Critic (SAC) <d-cite key="haarnoja2018softactorcritic"></d-cite> models are a Reinforcement Learning (RL) algorithm designed to solve unsupervised tasks such as embodying the Ant-agent in our "Ant-Maze" task. It uses an "Actor", a neural network acting as the agent policy, and "Critic" (or multiple critics) network as a value estimator. The key difference between a "Soft" and traditional Actor-Critic is its use of an entropy term in its objective function, and the use of a stochastic policy with the intention of increasing training stability by encouraging exploration to avoid sub-optimal convergence.

## Methodology

### Agent
To create a baseline agent, we train a Soft Actor-Critic (SAC) agent <d-cite key="haarnoja2018softactorcritic"></d-cite> to embody the MuJoCo Ant <d-cite key="todorov2012mujoco,towers2024gymnasium"></d-cite>, a quadruped controlled by 8 rotors (with one positioned at each joint) with 4 limbs comprised of 2 "Links" each conjoined by a joint rotor.

The agent's observation space includes a velocity modality with linear and angular velocity using meters and radians per second, respectively. This includes velocities for all limbs and joints, including each limb link. Observations also include an angular modality in radians, tracking the angle between each link, the ant's torso orientation, as well as the angles of the limbs from the torso. Additionally, the model is supplied with a z-coordinate torso reading. A position in meters representing the torso's height from the ground. Importantly, all observations are formally unbounded with a range of (-Inf, Inf).

The detailed observation space is outlined in the table below:

| Num | Observation | Name | Joint | Unit |
|-----|-------------|------|-------|------|
| 0 | z-coordinate of the torso (centre) | torso | free | position (m) |
| 1 | x-orientation of the torso (centre) | torso | free | angle (rad) |
| 2 | y-orientation of the torso (centre) | torso | free | angle (rad) |
| 3 | z-orientation of the torso (centre) | torso | free | angle (rad) |
| 4 | w-orientation of the torso (centre) | torso | free | angle (rad) |
| 5 | angle between torso and front left link on front left | hip_1 (front_left_leg) | hinge | angle (rad) |
| 6 | angle between the two links on the front left | ankle_1 (front_left_leg) | hinge | angle (rad) |
| 7 | angle between torso and front right link on front right | hip_2 (front_right_leg) | hinge | angle (rad) |
| 8 | angle between the two links on the front right | ankle_2 (front_right_leg) | hinge | angle (rad) |
| 9 | angle between torso and back left link on back left | hip_3 (back_left_leg) | hinge | angle (rad) |
| 10 | angle between the two links on the back left | ankle_3 (back_left_leg) | hinge | angle (rad) |
| 11 | angle between torso and back right link on back right | hip_4 (right_back_leg) | hinge | angle (rad) |
| 12 | angle between the two links on the back right | ankle_4 (right_back_leg) | hinge | angle (rad) |
| 13 | x-coordinate velocity of the torso | torso | free | velocity (m/s) |
| 14 | y-coordinate velocity of the torso | torso | free | velocity (m/s) |
| 15 | z-coordinate velocity of the torso | torso | free | velocity (m/s) |
| 16 | x-coordinate angular velocity of the torso | torso | free | angular velocity (rad/s) |
| 17 | y-coordinate angular velocity of the torso | torso | free | angular velocity (rad/s) |
| 18 | z-coordinate angular velocity of the torso | torso | free | angular velocity (rad/s) |
| 19 | angular velocity of the angle between torso and front left link | hip_1 (front_left_leg) | hinge | angle (rad) |
| 20 | angular velocity of the angle between front left links | ankle_1 (front_left_leg) | hinge | angle (rad) |
| 21 | angular velocity of the angle between torso and front right link | hip_2 (front_right_leg) | hinge | angle (rad) |
| 22 | angular velocity of the angle between front right links | ankle_2 (front_right_leg) | hinge | angle (rad) |
| 23 | angular velocity of the angle between torso and back left link | hip_3 (back_left_leg) | hinge | angle (rad) |
| 24 | angular velocity of the angle between back left links | ankle_3 (back_left_leg) | hinge | angle (rad) |
| 25 | angular velocity of the angle between torso and back right link | hip_4 (right_back_leg) | hinge | angle (rad) |
| 26 | angular velocity of the angle between back right links | ankle_4 (right_back_leg) | hinge | angle (rad) |

*Table: Observations space for the MuJoCo Ant-Maze task. Ranges for all values are -Inf to +Inf.*

The action space consists of 8 torque values applied to the rotors:

| Num | Action | Name | Joint | Type (Unit) |
|-----|--------|------|-------|-------------|
| 0 | Torque applied on the rotor between the torso and back right hip | hip_4 (right_back_leg) | hinge | torque (N m) |
| 1 | Torque applied on the rotor between the back right two links | angle_4 (right_back_leg) | hinge | torque (N m) |
| 2 | Torque applied on the rotor between the torso and front left hip | hip_1 (front_left_leg) | hinge | torque (N m) |
| 3 | Torque applied on the rotor between the front left two links | angle_1 (front_left_leg) | hinge | torque (N m) |
| 4 | Torque applied on the rotor between the torso and front right hip | hip_2 (front_right_leg) | hinge | torque (N m) |
| 5 | Torque applied on the rotor between the front right two links | angle_2 (front_right_leg) | hinge | torque (N m) |
| 6 | Torque applied on the rotor between the torso and back left hip | hip_3 (back_leg) | hinge | torque (N m) |
| 7 | Torque applied on the rotor between the back left two links | angle_3 (back_leg) | hinge | torque (N m) |

*Table: MuJoCo Ant agent action space. All values range between [-1, 1].*

Below is an illustration of the body and rotor layout used in our experiments:

<div class="l-body-outset">
  <div class="row">
    <div class="col-sm-6">
      <figure>
        <img src="{{ '/assets/img/2025-10-08-multimodalEmergingPatterns/example_ant.png' | relative_url }}" alt="Example Mujoco Ant body." style="width: 100%; max-width: 300px;" />
        <figcaption>Example MuJoCo Ant body <d-cite key="towers2024gymnasium"></d-cite>.</figcaption>
      </figure>
    </div>
    <div class="col-sm-6">
      <figure>
        <img src="{{ '/assets/img/2025-10-08-multimodalEmergingPatterns/ant_joints.png' | relative_url }}" alt="Rotor layout for Ant." style="width: 100%; max-width: 300px;" />
        <figcaption>Rotor layout for Ant <d-cite key="towers2024gymnasium"></d-cite>.</figcaption>
      </figure>
    </div>
  </div>
</div>

### Environment
In addition to training the agent to embody its quadruped ant, we train it for the AI Gymnasium task "Ant Maze" <d-cite key="towers2024gymnasium"></d-cite>. This task requires the agent to learn how to walk with its body then utilize its learned movements to maneuver around obstacles to reach a predetermined destination. This can be seen in pathing and exploration heat map figures throughout this paper.

### Adversarial Attack
We use the Fast Gradient Sign Method (FGSM) <d-cite key="goodfellow2015explaining"></d-cite> as our baseline attack to conduct all of our experiments. This is a white box attack that uses the sign vector of the gradient from the victim model to perturb images. Standard FGSM is:

$$
\mathbf{x}_{\text{adv}} = \mathbf{x} + \epsilon\,\mathrm{sign}\big(\nabla_{\mathbf{x}} J(\theta,\mathbf{x},y)\big)
$$

Where $\mathbf{x}$ is the original input, $y$ is the true label, $\epsilon$ is a small scalar controlling the perturbation size, $J(\theta,\mathbf{x},y)$ is the loss function, $\nabla_{\mathbf{x}} J(\theta,\mathbf{x},y)$ is the gradient of the loss with respect to the input, $\mathrm{sign}(\cdot)$ is the element-wise sign function, and $\mathbf{x}_{\text{adv}}$ is the resulting adversarial example.

It is important to note that two key changes were made in comparison to FGSM's source material. First, FGSM is typically employed as an attack against image classifiers and perturbations are often shown in the form of pixel values. To adapt the attack to our use case we apply similar perturbations in the vector form matching our observation space. This means perturbations represent values relative to which modality is being targeted (such as radians or meters per second).

Second, FGSM is typically designed for attacking classification tasks where a label is present, hence the equation's use of a loss function given model parameters, inputs, and a label. However we must restructure the equation to use a different loss function matching that of the RL model, which is unsupervised (and has no labels). For RL without labels, we instead use the critic's Q-value to define a targeted degradation objective:

$$
\mathbf{x}_{\text{adv}} = \mathbf{x} + \epsilon\,\mathrm{sign}\big(\nabla_{\mathbf{x}} Q_{\theta}(\mathbf{x}, a)\big)
$$

This change uses the Q function as per the "Critic" in our "Actor-Critic" model, rather than a direct loss based on class labels. This allows FGSM to be applied directly to the active policy, leveraging the Q-Network's valuation of the agent's action to generate adversarial perturbations.

### Adversarial Defenses
As a baseline we employ a basic disruption method, using a scaled gaussian filter to disrupt perturbations with noise. Second, we test simple baseline adversarial detection methods by training a classifier to identify perturbed and un-perturbed "Benign" observations. Additionally we compare this detection method to traditional clustering algorithms K-Means and Gaussian Mixture Model (GMM) to detect anomalous data in the form of perturbations. We also test a purification method in the form of a defense VAE, a generative auto-encoder model that attempts to re-generate inputs as their benign equivalent.

Neural-Network-based and clustering-based adversarial detection methods are trained on a dataset gathered during the SAC training and evaluation process. While the SAC agent trains on 3 million steps, we allow the first half of training (first 1.5M steps) to proceed as a warm-up period. This helps prevent collecting faulty data before the agent can properly utilize its limbs, thus potentially resulting in too many strange or uncommon sensor readings. We allow training to continue uninterrupted for the final 1.5M steps, collecting and storing the agent's observation into a benign observation dataset. We then generate and store FGSM perturbed observations in an adversarial dataset without giving them to the agent, again preventing any bias towards strange or unreasonable sensor readings in the final dataset. Once collected, the dataset results in two 1.5M observation subsets, each observation paired with its adversarially perturbed counterpart.

#### Gaussian Noise Defense
A Gaussian Noise filter was used as our baseline defense. As described in the equation below as a naive defense method, we sample additional perturbations from a normal gaussian distribution, then scale them down using scaling factor $\epsilon$:

$$
\mathbf{x_{def}} = x + \epsilon \cdot \mathbf{n}, \quad \text{where } \mathbf{n} \sim \mathcal{N}(0, \mathbf{I})
$$

The intuition behind this defense is that small perturbations crafted with the intention of manipulating model behavior are both specific as well as sensitive to change. Because of this, applying a noise filter can throw off perturbations and prevent successful attacks, opting for simple noise-induced performance degradation. While this defense may not result in the highest performance, it is by far the least sophisticated and computationally inexpensive baseline.

#### Defense VAE
We base this defense method on prior work on creating a defense Variational-Auto-Encoder (Defense-VAE) <d-cite key="li2019defensevaefastaccuratedefense,shayanNDVAE"></d-cite>. It operates by training a VAE on a dataset containing pairs of benign and adversarial examples (such as the dataset we collect and describe at the top of this section). The encoder learns to encode both benign and adversarial samples, while the decoder reconstructs input using only the benign samples as labels for its loss. The end result is such that the encode-decode process learns a mapping from both benign and adversarial input spaces to a strictly benign output space.

This methodology was adapted from image-based defense to a single dimensional input matching that of the observation vector. To achieve this, given the observations are substantially smaller in size, we create a downsized VAE that utilizes 4 fully connected layers with RELU activation in place of a deep convolutional neural network. Because the modalities are not time series-based nor positionally related we avoid the use of 1-D convolutions as well.

#### Adversarial Detection
The focus of these defense methods is to identify the presence of an attack. Our tests focus on accuracy of attack detection, and do not effect the model during its run time. This is primarily because detection is a separate process from handling a detected attack, which may include something like allowing the model to fail gracefully, reset its sensors, or some other combination of circumventive measures. However the effectiveness of these measures are their own research question and thus avoided. Rather, we track the accuracy and F1-score of various detection methods.

We test multiple baseline detection models trained to classify the presence of the attack. Our collected observation dataset is treated as a labeled dataset in which both benign and adversarial observations are labeled respectively and shuffled together for classification. Our tests include Support-Vector-Machines (SVM), K-Nearest Neighbors, and Neural Network classifiers, as well as more naive clustering methods K-means and Gaussian-Mixture-Models (GMM) to compare more complex classifiers against a more simple signal separation algorithm.

Accuracy of the classifiers was determined by binary prediction accuracy (benign and adversarial classifications). Clustering methods are fit to separate data points into 2 clusters, each representing a potential class in which the more "favorable" label for accuracy is assigned to its respective class (due to the unsupervised nature of clustering algorithms).

### Adversarial Evaluation
To test the effects of adversarial attacks on our baseline agent across its modalities, we use the following process:

1. The agent is trained normally on benign inputs.
2. Every evaluation episode, the attack is applied to perturb the input: First across both modalities simultaneously, then individually by modifying only the target modality values in the agent's observation vector.
3. We compare benign training performance and adversarial evaluation to identify effects of the attack on the agent.

To develop a thorough understanding of model performance when defended, each defense method is applied in two configurations: In the first configuration, the model trains as normal, and the defense method is applied during evaluation, prior to the attacked data reaching the model. In the second configuration, the model also trains on benign inputs that have first been given to the defense method. The model is then attacked as usual. This helps compare the effects of model tuning on each defense.

## Results

### Baseline Performance
We train an SAC agent on the Ant Maze task for 3 Million steps to prepare our baseline agent such that it consistently solves the task of navigating around a simple obstacle. The agent is trained with a memory size of 1e6, a $\tau$ (soft update coefficient) of 0.05, and a $\gamma$ (discount factor) of 0.99.

The successful training progress of the SAC agent is shown in the figures below. We can see its training reward plateau after 3M steps, as well as the heatmap of exploration the agent takes finding an adequate route around the obstacle. Furthermore, evaluation performance reaches its maximum at the end of training, as well as the agent's pathing history showing its learned navigation from the start point to the goal.

Without interference, we can consider this agent to have solved its task, at least for the simple single-obstacle arrangement it was trained to complete.

<div class="l-page-outset">
  <div class="row">
    <div class="col-sm-6">
      <figure>
        <img src="{{ '/assets/img/2025-10-08-multimodalEmergingPatterns/baseline_train.png' | relative_url }}" alt="SAC training reward." style="width: 100%; max-width: 400px;" />
        <figcaption>Training reward progression (3M steps).</figcaption>
      </figure>
    </div>
    <div class="col-sm-6">
      <figure>
        <img src="{{ '/assets/img/2025-10-08-multimodalEmergingPatterns/baseline_explore.png' | relative_url }}" alt="Exploration heatmap." style="width: 100%; max-width: 400px;" />
        <figcaption>Exploration heatmap during learning.</figcaption>
      </figure>
    </div>
  </div>
</div>

<div class="l-page-outset">
  <div class="row">
    <div class="col-sm-6">
      <figure>
        <img src="{{ '/assets/img/2025-10-08-multimodalEmergingPatterns/baseline_eval.png' | relative_url }}" alt="Evaluation reward." style="width: 100%; max-width: 400px;" />
        <figcaption>Evaluation performance plateaus after training.</figcaption>
      </figure>
    </div>
    <div class="col-sm-6">
      <figure>
        <img src="{{ '/assets/img/2025-10-08-multimodalEmergingPatterns/baseline_paths.png' | relative_url }}" alt="Paths to goal." style="width: 100%; max-width: 400px;" />
        <figcaption>Typical paths from start to goal after training.</figcaption>
      </figure>
    </div>
  </div>
</div>

### Attacking an Un-Defended Model
Using the modified FGSM attack (with scaling factor 0.005) from the equation above, we target the agent's observation vector with perturbations during the agent's evaluation phase, throughout the training process. The intention of the attack is to maximize the loss according the "Critic" value function of the SAC algorithm. The adversarial perturbations effectively attempt to have the agent take actions that result in the lowest value state.

#### Attacking Both Modalities
We apply the attack on the entire input observation attacking both modalities, as can be seen in the figure below. Successful attacks can be seen by the sudden drops in reward (on the purple line indicating the attacked agent), and in the event the attack fails we see the performance return to its typical benign performance (as indicated by the red line). Importantly, we can also use this information to note the frequency with which an attack is successful via the width and frequency of these sharp dips in reward.

To better visualize the effects of the attack, the path comparison figures juxtapose example steps in which the FGSM attack succeeds and fails as compared to their benign counterparts. The top pair of paths show a failed FGSM attack, that at best switches the direction with which the model navigates around the obstacle to its goal. However when the FGSM attack does succeed, we see the rate at which the agent gets lost increases to the extent that it often fails before leaving the initial starting area.

#### Attacking Individual Modalities
We next apply the FGSM attack to the model's two modalities individually. FGSM is computed and applied only to the portion of the observation vector representing each chosen modality.

The resulting effects on agent performance can be seen in the individual modality figure. We can see how both velocity (green line) and angle (blue line) modalities are effected differently in comparison to the attack on both modalities (purple line). It is important to note that velocity only represents about a quarter of the observation space.

As such, we see the effects of the attack are proportional to how much of the observation a given modality represents. We can observe a greater frequency of peaks in reward when velocity is attacked, compared to angles or both together. This shows us a greater frequency of attack failure, and can be seen repeated to a smaller degree comparing the larger angular modality to attacking both. By order of modality size we see a nested behavior in which attacks on the smallest modality approximately upper bound performance compared to attacks on progressively larger portions of the observation.

This allows us to draw an intuitive conclusion with empirical evidence: Effectiveness of attacking a modality is limited by how much of a model's input space is represented by that modality.

<figure class="l-page">
  <img src="{{ '/assets/img/2025-10-08-multimodalEmergingPatterns/fgsm_eval.png' | relative_url }}" alt="FGSM evaluation performance." style="width: 100%; max-width: 600px;" />
  <figcaption>Benign (red) vs adversarial (purple) evaluation. Sharp reward drops indicate successful attacks.</figcaption>
</figure>

<div class="l-page-outset">
  <div class="row">
    <div class="col-sm-6">
      <figure>
        <img src="{{ '/assets/img/2025-10-08-multimodalEmergingPatterns/fgsm_path_success.png' | relative_url }}" alt="FGSM success example." style="width: 100%; max-width: 350px;" />
        <figcaption>FGSM attack success: agent gets lost early.</figcaption>
      </figure>
    </div>
    <div class="col-sm-6">
      <figure>
        <img src="{{ '/assets/img/2025-10-08-multimodalEmergingPatterns/fgsm_path_fail.png' | relative_url }}" alt="FGSM failure example." style="width: 100%; max-width: 350px;" />
        <figcaption>FGSM attack failure: minimal route alteration.</figcaption>
      </figure>
</div>
</div>
</div>

Modality-specific FGSM shows that attack effectiveness scales with the attacked fraction of the observation space.

<figure class="l-page">
  <img src="{{ '/assets/img/2025-10-08-multimodalEmergingPatterns/individual_modality_fgsm.png' | relative_url }}" alt="Modality-specific FGSM results." style="width: 100%; max-width: 600px;" />
  <figcaption>Performance under FGSM for both modalities (purple), angles-only (blue), and velocity-only (green).</figcaption>
</figure>

### Attacking a Defended Model
After observing the most basic interaction of an adversarial attack against a multimodal model, we show the effects and observability of such attacks when various preventative measures are taken.

#### Gaussian Noise Defense
We employ a Gaussian noise filter as a baseline defense for the agent that attempts to disrupt attack perturbations by subtly altering all values in the observation with random noise perturbations. We sample noise with a mean of 0 and standard deviation 1, using a scaling factor of 0.005.

As per the noise defense figures we can see that even a small amount of noise can disrupt the attack and result in a higher reward during adversarial evaluation. Additionally, training the model on noise helps reduce the effects of noise on agent performance, resulting in more frequent reward peaks and lower attack success frequency. We test the model both against noise only in evaluation as well as preemptively training it on noisy values. Regardless of whether the agent is trained on noisy data (orange line) or only encountering noise during evaluation (red line), both scenarios perform better than undefended perturbations (purple line).

When testing the gaussian noise to attacks against specific modalities, we begin to see an interesting change in behavior. Attacks on the velocity modality replicated almost identical results to the multimodal noise defense, however we can observe an interesting change when the angular modality is targeted specifically.

Rather than follow the same pattern, training on noise appears to destabilize model performance more than it helps. We see in the angular noise defense figure that the best performing version of adding noise as a defense is when it is only included in evaluation, with the model never encountering noise during its training. It is unclear if this is due to this modality being the predominant data in the observation space for the agent, or if there are some other influencing factors. This was observed as consistent behavior across multiple runs.

<div class="l-page-outset">
  <div class="row">
    <div class="col-sm-6">
      <figure>
        <img src="{{ '/assets/img/2025-10-08-multimodalEmergingPatterns/noise_defense_fgsm.png' | relative_url }}" alt="Noise defense under multimodal FGSM." style="width: 100%; max-width: 400px;" />
        <figcaption>Defense only (red), trained with noise (orange), undefended (purple).</figcaption>
      </figure>
    </div>
    <div class="col-sm-6">
      <figure>
        <img src="{{ '/assets/img/2025-10-08-multimodalEmergingPatterns/angular_noise_defense_fgsm.png' | relative_url }}" alt="Noise defense for angle-only attacks." style="width: 100%; max-width: 400px;" />
        <figcaption>Angle-only attacks: defense at eval (purple) outperforms training-through-noise (green) and baseline (blue).</figcaption>
      </figure>
    </div>
  </div>
</div>

#### Defense-VAE
A small fully connected Defense-VAE improves evaluation rewards under FGSM when used only at evaluation. Training the agent on VAE-filtered inputs prevented solving the task. Under modality-specific attacks, the VAE generalized worst to single-modality perturbations, working best when both modalities were attacked.

The VAE defense shows interesting behavior when applied to different attack scenarios. When both modalities are attacked simultaneously, the VAE provides substantial protection, as evidenced by the blue line in the VAE defense figures showing improved performance compared to the undefended purple line. However, when individual modalities are targeted, the VAE's effectiveness diminishes significantly.

This suggests that the VAE's training on paired benign-adversarial examples may have learned to recognize and correct perturbations that affect the entire observation space, but struggles to generalize to more targeted, modality-specific attacks. The defense appears to be most effective when the perturbation pattern matches the training distribution, which primarily consisted of full-observation attacks.

<div class="l-page-outset">
  <div class="row">
    <div class="col-sm-6">
      <figure>
        <img src="{{ '/assets/img/2025-10-08-multimodalEmergingPatterns/vae_both_modalities.png' | relative_url }}" alt="VAE defense under multimodal FGSM." style="width: 100%; max-width: 400px;" />
        <figcaption>VAE defense (blue) vs undefended adversarial inputs (purple) at $\epsilon=0.007$.</figcaption>
      </figure>
    </div>
    <div class="col-sm-6">
      <figure>
        <img src="{{ '/assets/img/2025-10-08-multimodalEmergingPatterns/vae_individual_modalities.png' | relative_url }}" alt="VAE defense under modality-specific FGSM." style="width: 100%; max-width: 400px;" />
        <figcaption>VAE under both-modalities (blue), velocity-only (red), angle-only (orange) attacks.</figcaption>
      </figure>
    </div>
  </div>
</div>

#### Detection Results (Summary)
Across SVM, KNN, and a small NN detector, detection accuracy improved with higher $\epsilon$. Angular perturbations were more detectable at $\epsilon=0.007$, while velocity was more detectable at $\epsilon=0.015$. Clustering baselines (KMeans, GMM) were near chance on average.

The clustering model metrics reveal more inconsistent findings. We can see from K-means and GMM, that using the angular perturbations at both magnitudes of FGSM allowed for better clustering accuracy than using the velocity perturbation data alone. Overall, adversarial detection is most accurate when both modalities are attacked, however perturbations to the angles of MuJoCo ant's joints are more easily detected than velocity perturbations.

Critically, we also observe a difference between FGSM attack detectability depending on scaling factor epsilon. This indicates high sensitivity of scaling factor's effects on attack detection. Both result in approximately the same attack success rate, but one is substantially more detectable.

The detailed results for all detection methods are shown in the table below:

| Detection Method | Multi-modal | | | Velocity | | | Angular | | |
|------------------|-------------|---|---|----------|---|---|---------|---|---|
| | Accuracy | F1 Score | Epsilon | Accuracy | F1 Score | Epsilon | Accuracy | F1 Score | Epsilon |
| SVM | 0.6027 | 0.6936 | 0.007 | 0.4993 | 0.6227 | 0.007 | 0.5606 | **0.6596** | 0.007 |
| SVM | 0.7188 | 0.7608 | 0.015 | 0.6070 | 0.7012 | 0.015 | 0.5048 | 0.3782 | 0.015 |
| KNN | 0.5909 | 0.5897 | 0.007 | 0.5164 | 0.4718 | 0.007 | 0.5391 | 0.5412 | 0.007 |
| KNN | 0.6606 | 0.6522 | 0.015 | 0.5850 | 0.5769 | 0.015 | 0.5152 | 0.3962 | 0.015 |
| NN | 0.7349 | 0.725 | 0.007 | 0.5494 | 0.6437 | 0.007 | 0.7104 | 0.7372 | 0.007 |
| NN | **0.9892** | **0.9892** | 0.015 | **0.8766** | **0.8810** | 0.015 | **0.8211** | 0.8264 | 0.015 |
| GMM | 0.4989 | N/A | 0.007 | 0.4984 | N/A | 0.007 | 0.4996 | N/A | 0.007 |
| GMM | 0.5094 | N/A | 0.015 | 0.4976 | N/A | 0.015 | 0.5021 | N/A | 0.015 |
| Kmeans | 0.5033 | N/A | 0.007 | 0.4998 | N/A | 0.007 | 0.5026 | N/A | 0.007 |
| Kmeans | 0.5416 | N/A | 0.015 | 0.4602 | N/A | 0.015 | 0.4855 | N/A | 0.015 |

*Table: Classifier performance across modalities with FGSM epsilon (scaling) values of 0.007 and 0.015.*

## Conclusions
Through the use of our testing methodology, we show that a multi-modal agent can be vulnerable to adversarial attacks on any of its modalities. Each can differently affect how the model behaves. We also observe that robustness of a particular modality is proportional to how much of the model's input space it represents, despite uncertainty around a specific modality's inherent robustness. We further provide evidence that the behaviors of different modalities can change the way each responds to its applied attacks and defenses. This includes attacks on modalities negatively effecting defenses that initially appear robust when both modalities are targeted (such as the use of our VAE defense). We also contribute a useful adversarial test-bed for a simple baseline multi-modal agent, as well as extending it as dataset creation tool for further testing.

Multi-modal RL agents are vulnerable to modality-specific and combined attacks. Attack effectiveness scales with the attacked proportion of the observation space. Defenses introduce nontrivial modality interactions: simple noise helps generally, but training-through-noise can destabilize specific modalities; VAE purification is most effective when both modalities are attacked and struggles to generalize to single-modality attacks. These results highlight emergent cross-modality patterns and motivate modality-aware defenses.

While the baselines used here are likely too simple to be fully representative of behaviors found in more complex multi-modal models, the focus for this work is to create a foundation for understanding the simplest form of this emerging complex task. We hope this to instigate and enable future work on better understanding and mitigating adversarial risks in the rapidly developing use of multi-modal systems.

## Reproducibility Details
Experiments used SAC with 3M steps, replay size 1e6, $\tau=0.05$, $\gamma=0.99$, evaluation every 100 steps, and Adam learning rates as in our code. Defense-VAE used fully connected enc/dec layers (256-128-64 latent-64-128-256) with latent size 24, trained for 50 epochs. Classifier details and additional hyperparameters are available in the appendix of the paper and our codebase.

### SAC Hyperparameters

| Parameter | Value |
|-----------|-------|
| Horizon Length | 1 |
| Memory Size | $1 \times 10^6$ |
| Batch Size | 4096 |
| $N$-step | 1 |
| $\tau$ (Target smoothing coefficient) | 0.05 |
| $\gamma$ (Discount factor) | 0.99 |
| Warm-up Steps | 32 |
| Critic Class | Double Q-Network |
| Evaluation Frequency | 100 |
| Learning Rate (Alpha) | 0.005 |
| Update Times per Step | 8 |

### VAE Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | $1 \times 10^{-3}$ |
| Batch Size | 32 |
| Training Epochs | 50 |
| Latent Dimension | 24 |
| Encoder Layers | 5 (Observation, 256, 128, 64, Latent Dimension) |
| Decoder Layers | 5 (Latent Dimension, 64, 128, 256, Observation) |

### Neural Network Hyperparameters

| Parameter | Value |
|-----------|-------|
| Input Dimension | 28 |
| Output Dimension | 1 |
| Hidden Layers | 5 (256, 256, 128, 32, 1) |
| Learning Rate | 0.0001 |
| Training Epochs | 60 |

### Support Vector Machine (SVM) Hyperparameters

| Parameter | Value |
|-----------|-------|
| $C$ | 1000 |
| Gamma | 1 |
| Degree | 3 |
| Decision Function | One-vs-One (ovo) |

### GMM Hyperparameters

| Parameter | Value |
|-----------|-------|
| Number of Components | 2 |
| Initialization Method | k-means++ |
| Covariance Type | full |
| Convergence Tolerance ($\texttt{tol}$) | 0.001 |
| Regularization of Covariance ($\texttt{reg\_covar}$) | $1 \times 10^{-6}$ |
| Max Iterations | 100 |
| Number of Initializations ($\texttt{n\_init}$) | 1 |

### K-Means
K-Means clustering algorithm as implemented by Scikit-learn, with a k value of 2 clusters.

<d-appendix>
  <d-footnote>
    We release a lightweight pipeline adapted from DDiffPG <d-cite key="li2024learningmultimodalbehaviorsscratch"></d-cite> to reproduce training, dataset generation, attacks, and defenses.
  </d-footnote>
</d-appendix>
