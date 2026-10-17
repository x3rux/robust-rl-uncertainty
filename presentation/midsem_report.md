# Robust RL under Environmental Uncertainty and Adversarial Disturbances

## Mid-Semester Report: How Fragile Is a Clean-Trained Agent?
**Author:** Ayush Chauhan (`2023B5A71089P`) — Department of Computer Science

---

## 1. Introduction and Motivation

Deep RL agents are usually trained and evaluated in the same simulator with fixed physics,
perfect sensors and instantaneous actuation. Any deployed system breaks these assumptions:
sensors are noisy, commands arrive late, payloads change the mass of the robot and the
ground can be slippery. Worse, an adversary may manipulate the agent's inputs on purpose.

The overall question of this project is:

> *Can an RL agent keep performing well when the test environment differs from the training
> environment, and which kind of shift is the most damaging?*

Before designing or comparing robust training methods, we first need a careful measurement
of **how and when a standard agent fails**. That is the goal of this mid-semester phase:

1. Train standard (non-robust) Soft Actor-Critic (SAC) agents to a strong level on two
   continuous-control tasks with contrasting mechanics (`Walker2d-v5` and `HalfCheetah-v5`).
2. Stress-test them under five disturbance types spanning sensors, actuators, and physics,
   increasing the intensity step by step.
3. Find the **breaking point** of each agent under each disturbance and compare disturbance
   types, random vs. adversarial perturbations, and environments.

These results serve as the reference ("baseline fragility") against which robust training
strategies will be compared in the second half of the project.

## 2. Background and Related Work

**Robust MDPs and distributionally robust RL.** A robust MDP replaces the single transition
kernel $P$ with an uncertainty set $\mathcal{U}$ and optimises the worst case,
$\max_\pi \min_{P\in\mathcal{U}} J(\pi, P)$. Liang et al. (ICML 2024) show how to learn such
a policy from a *single trajectory* without a simulator. They use the dual form of an
f-divergence-constrained inner problem and estimate it with multi-timescale stochastic
approximation (DRQ). This is the natural defence against **dynamics shift**.

**Corrupted data.** Ye et al. (ICML 2024) study model-based RL when an adversary corrupts the
transitions used for learning. They down-weight samples according to an uncertainty
(total-variation) information ratio, and prove that the regret grows only additively in the
total corruption budget $C$. This principle is relevant against **reward and transition
corruption during training**.

**Adversarial observation perturbations.** In the state-adversarial MDP view, an adversary
perturbs the agent's observation within a norm ball while the true state is unchanged. Zhou et
al. (ICML 2025) show that training against a single deterministic worst-case attacker overfits
to that attack. They propose a *stochastic* (maximum-entropy) adversary, ATSA, so that the
learned robustness generalises across perturbations.

**Benchmarks.** Robust Gymnasium (Gu et al., ICLR 2025) unifies disruptions on observations,
actions, rewards and environment parameters over 60+ tasks. It reports that no existing method
is robust to all disruption types. Our disturbance taxonomy follows the same axes
(observation / action / environment).

## 3. Problem Formulation

Let $M_{\text{train}} = (\mathcal{S},\mathcal{A},P,R,\gamma)$ be the clean MDP and
$\pi$ a policy trained on it. A *disturbance* is a pair $(k,\epsilon)$ of type $k$ and
intensity $\epsilon$. It changes the interaction loop in one of three places:

| Where | Type $k$ | Definition at test time | Intensity unit |
|---|---|---|---|
| Agent input | sensor noise (`obs_noise`) | $\tilde s_t = s_t + \epsilon\,\sigma_s \odot \xi_t,\ \xi_t\sim\mathcal N(0,I)$ | $\epsilon$ in units of per-dimension state std $\sigma_s$ |
| Agent input | sensor attack (`obs_adv`) | $\tilde s_t = s_t+\delta_t,\ \delta_t = \arg\min_{\lvert\delta_i\rvert\le\epsilon\sigma_{s,i}} Q(s_t,\pi(s_t+\delta))$ | same as above |
| Actuator | action delay (`act_delay`) | executed action at $t$ is $a_{t-\epsilon}$ | control steps |
| Physics | body mass (`mass`) | all masses and inertias $\times\,\epsilon$ | multiplier |
| Physics | ground friction (`friction`) | sliding friction $\times\,\epsilon$ | multiplier |

Sensor perturbations do **not** change the true dynamics. Only the agent's belief about the
state changes. Actuator delay and physics perturbations change the *effective* transition kernel.

**Metrics.** For agent $\pi$ (one training seed), let $J_0$ be its mean clean return.

- *Normalised return* $\bar J(k,\epsilon) = J(k,\epsilon)/J_0$. A value of 1 means no
  degradation.
- *Degradation onset* $\epsilon_{80}$: the smallest intensity, moving away from the nominal
  value, at which $\bar J < 0.8$.
- *Breaking point* $\epsilon_{50}$: the smallest intensity at which $\bar J < 0.5$, i.e. the agent
  has lost half of its performance.
- *Fall rate*: the fraction of episodes that end early because the robot fell (`Walker2d-v5`).
- *Policy sensitivity* $\lVert\pi(\tilde s)-\pi(s)\rVert_2$: how much a corrupted observation
  changes the chosen action, i.e. how "confused" the agent is.

## 4. Experimental Setup

**Environments.** Two contrasting MuJoCo locomotion tasks from Gymnasium 1.x (v5), both having
a 17-dimensional continuous state space and a 6-dimensional continuous action space:
- **Walker2d-v5** (bipedal upright walking, 17-D state / 6-D action): requires active upright
  balance; episodes terminate early if the torso falls. Control step $\Delta t = 8\text{ ms}$.
- **HalfCheetah-v5** (planar running, 17-D state / 6-D action): mechanically stable and cannot
  fall; episodes always run for 1,000 steps. Control step $\Delta t = 50\text{ ms}$.

**Agent.** Soft Actor-Critic (Stable-Baselines3 2.x) with standard MuJoCo hyper-parameters:

| Hyper-parameter | Value |
|---|---|
| Networks (actor / twin critics) | MLP 256-256, ReLU |
| Learning rate | 3e-4 (Adam) |
| Batch size / replay buffer | 256 / 1M |
| Discount $\gamma$ / Polyak $\tau$ | 0.99 / 0.005 |
| Entropy coefficient | automatically tuned |
| Warm-up (random actions) | 10,000 steps |
| Training budget | 300,000 environment steps |
| Seeds | 3 per environment (0, 1, 2) |

**Evaluation protocol.**
- Each trained agent is evaluated with its deterministic (mean) action for **10 episodes**
  per (disturbance, intensity) pair: 43 conditions per agent, 2,580 episodes in total.
- The same 10 environment reset seeds are used for every condition (common random numbers),
  so differences between conditions come from the disturbance and not from different
  initial states.
- Observation-noise scales use the per-dimension state std measured on 5 clean rollouts of
  the same agent.

**Adversarial attack.** We use a white-box PGD attack on the agent's own critic. It finds a
bounded $\delta$ that minimises $\min(Q_1,Q_2)(s,\pi(s+\delta))$, evaluated at the true state
$s$. The attack runs 10 iterations with step size $0.25\epsilon$, a random start inside the
box, and projection onto $\lvert\delta_i\rvert\le\epsilon\sigma_{s,i}$ after each step.

## 5. Results

### 5.1 Clean baseline performance

![Learning curves](figures/learning_curves.png)

*Figure 1: Clean evaluation return during training (mean ± std over 3 seeds, 5 episodes per
evaluation point).*

| Environment | Clean return (mean ± std over seeds) | Clean fall rate |
|---|---|---|
| HalfCheetah-v5 | 8246 ± 765 | n/a (cannot fall) |
| Walker2d-v5 | 3788 ± 288 | 6.7% |

### 5.2 Degradation under increasing disturbance

![Degradation curves](figures/degradation_all.png)

*Figure 2: Normalised return vs. disturbance intensity for all five disturbance types. Shaded
bands show ±1 std over 3 seeds. The red line is the 50% failure threshold.*

![Absolute returns](figures/degradation_absolute.png)

*Figure 3: The same experiment in absolute return. The dotted line is each agent's clean
baseline.*

**Breaking points** ($\epsilon_{80}$ = degradation onset, $\epsilon_{50}$ = failure;
"never" = not reached within the tested range):

| Disturbance | Walker2d $\epsilon_{80}$ / $\epsilon_{50}$ | HalfCheetah $\epsilon_{80}$ / $\epsilon_{50}$ |
|---|---|---|
| Sensor noise σ (`obs_noise`) | 0.24 / 0.40 | 0.16 / 0.41 |
| Sensor attack ε (`obs_adv`) | 0.09 / 0.16 | 0.04 / 0.13 |
| Action delay (steps) (`act_delay`) | 0.65 / 1.28 | 0.29 / 0.72 |
| Mass ↑ (`mass`) | ×1.09 / ×1.31 | ×1.22 / ×1.67 |
| Mass ↓ (`mass`) | ×0.55 / never | ×0.68 / never |
| Friction ↓ (`friction`) | ×0.39 / ×0.22 | ×0.21 / never |
| Friction ↑ (`friction`) | ×1.37 / ×1.88 | ×2.19 / ×2.94 |

![Heatmap moderate](figures/heatmap_moderate.png)

*Figure 4: Retained performance at a fixed, pre-declared "moderate" severity for each
disturbance type. A matching figure for "severe" severity is in `figures/heatmap_severe.png`.*

### 5.3 Random noise vs. adversarial attack

![Random vs adversarial](figures/random_vs_adversarial.png)

*Figure 5: Sensor noise vs. sensor attack at the same per-dimension budget (log scale).*

At the same perturbation size, the adversarial attack is much more damaging. The breaking
point $\epsilon_{50}$ drops from 0.40 to 0.16 on `Walker2d-v5` and from 0.41 to 0.13 on
`HalfCheetah-v5`. A worst-case perturbation is therefore **2.4–3.2× more efficient** than
random noise of the same size. On HalfCheetah, an adversarial perturbation of only 4.4% of a
state dimension's standard deviation already costs 20% of the return.

### 5.4 Falls and policy sensitivity

![Falls and sensitivity](figures/falls_and_sensitivity.png)

*Figure 6: Left and middle panels: fall rate of Walker2d under sensor perturbations and action
delay. Right panel: change in the chosen action caused by corrupted observations.*

Two failure mechanisms are visible:

- **Bipedal walking fails abruptly via falls.** `Walker2d-v5` falls in 6.7% of clean episodes,
  37% at sensor-noise σ = 0.2, 77% under adversarial attack at ε = 0.2, and 100% at a two-step
  action delay.
- **The policy reacts much more strongly to adversarial inputs.** Under attack, the action
  deviation $\lVert\pi(\tilde{s}) - \pi(s)\rVert_2$ at a given ε is about **twice** as large as under
  random noise.

## 6. Discussion

- **Action delay is the most damaging.** A single control step of latency costs 31% of return
  on Walker2d and 70% on HalfCheetah, and two steps make both agents fail completely.
  HalfCheetah suffers most because its control step is 50 ms (frame-skip 5 × 10 ms), compared
  with 8 ms for Walker2d ($6.25\times$ longer physical latency per step).
- **Physical mass changes are asymmetric.** Heavier bodies break the agents at +31% (Walker2d)
  to +67% (HalfCheetah) because motors saturate their torque limits, whereas lighter bodies
  never drop below 50% return.
- **Robot mechanics dictate friction sensitivity.** `Walker2d-v5` fails under low friction
  (×0.22) because foot slippage causes it to lose balance and fall, whereas `HalfCheetah-v5`
  cannot fall and retains 62% of its score even at 0.1× friction.

## 7. Limitations and Plan for the Second Half

1. **Robust training strategies:** observation-noise augmentation, Lipschitz smoothness
   regularisation, action-history conditioning, domain randomisation over mass/friction,
   distributionally robust pessimistic Q-backups (Liang et al., 2024), and uncertainty-weighted
   TD loss for training-time reward corruption (Ye et al., 2024).
2. **Cross-robustness transfer matrix:** train with a defence for type $k$, test on all types $k'$.
3. **Proposed method:** a stochastic, multi-channel adversary (extending Zhou et al., 2025).
