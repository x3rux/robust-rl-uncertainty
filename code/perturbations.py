"""
Test-time perturbations for MuJoCo (Gymnasium) environments.

Two families (5 core perturbations):
  * Agent-side: modify what the agent *sees* (the true state is unchanged)
        1. obs_noise  : s~ = s + sigma * std(s) * N(0, I)
        2. obs_adv    : s~ = argmin_{|d_i| <= eps*std_i} Q(s, pi(s + d))   (10-step PGD attack)
  * Environment-side: modify what the environment *does*
        3. act_delay  : executed action at time t is the one chosen at t-k
        4. mass       : every body's mass (and inertia) multiplied by k
        5. friction   : sliding friction of every geom multiplied by k

Observation magnitudes are expressed in units of the per-dimension standard deviation
of the observations visited by the clean policy, so that "sigma = 0.1" means the same
relative corruption in every dimension and every environment.
"""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
import torch


# ----------------------------------------------------------------------------------------
# Environment-side wrappers
# ----------------------------------------------------------------------------------------
class ActionDelayWrapper(gym.Wrapper):
    """The environment executes the action chosen `delay` control steps earlier
    (communication / actuation latency). The buffer is filled with zero actions on reset."""

    def __init__(self, env: gym.Env, delay: int):
        super().__init__(env)
        self.delay = int(delay)
        self.buffer: deque = deque()

    def reset(self, **kwargs):
        zero = np.zeros(self.env.action_space.shape, dtype=self.env.action_space.dtype)
        self.buffer = deque([zero.copy() for _ in range(self.delay)])
        return self.env.reset(**kwargs)

    def step(self, action):
        if self.delay == 0:
            return self.env.step(action)
        self.buffer.append(np.asarray(action).copy())
        return self.env.step(self.buffer.popleft())


def apply_physics(env: gym.Env, mass: float = 1.0, friction: float = 1.0):
    """Scale physical parameters of the underlying MuJoCo model in place.

    MuJoCo's reset() restores the simulation *state* (qpos/qvel) but not the *model*,
    so the change persists across episodes of this env instance.
    """
    model = env.unwrapped.model
    if mass != 1.0:
        model.body_mass[:] = model.body_mass * mass
        model.body_inertia[:] = model.body_inertia * mass
    if friction != 1.0:
        model.geom_friction[:, 0] = model.geom_friction[:, 0] * friction  # sliding friction
    return env


# ----------------------------------------------------------------------------------------
# Agent-side observation perturbations
# ----------------------------------------------------------------------------------------
class ObsPerturber:
    """Callable mapping the true observation to the observation fed to the policy."""

    def __call__(self, obs: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        return obs


class GaussianObsNoise(ObsPerturber):
    def __init__(self, sigma: float, obs_std: np.ndarray, rng: np.random.Generator):
        self.sigma, self.obs_std, self.rng = float(sigma), obs_std, rng

    def __call__(self, obs):
        if self.sigma <= 0:
            return obs
        return obs + self.sigma * self.obs_std * self.rng.standard_normal(obs.shape)


class CriticPGDAttack(ObsPerturber):
    """White-box adversarial observation attack against an SB3 SAC/TD3 agent.

    Finds a bounded perturbation d (|d_i| <= eps * std_i) that minimises the agent's own
    critic evaluated at the TRUE state and the action the actor outputs for the PERTURBED
    state:   min_d  Q(s, pi(s + d)).
    This is the critic-based attack used in the SA-MDP literature (Zhang et al., 2020).
    """

    def __init__(self, model, eps: float, obs_std: np.ndarray, n_steps: int = 10, rng=None):
        self.model = model
        self.eps = float(eps)
        self.n_steps = n_steps
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.device = model.device
        self.bound = torch.as_tensor(eps * obs_std, dtype=torch.float32, device=self.device)
        self.step_size = self.bound * 2.5 / n_steps

    def __call__(self, obs):
        if self.eps <= 0:
            return obs
        actor, critic = self.model.policy.actor, self.model.policy.critic
        s = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        # random start inside the box (standard PGD)
        delta = (torch.rand_like(s) * 2 - 1) * self.bound
        for _ in range(self.n_steps):
            delta.requires_grad_(True)
            a = actor(s + delta, deterministic=True)
            q = torch.min(torch.cat(critic(s, a), dim=1), dim=1)[0].sum()
            (grad,) = torch.autograd.grad(q, delta)
            with torch.no_grad():
                delta = delta - self.step_size * grad.sign()
                delta = torch.max(torch.min(delta, self.bound), -self.bound)
        return (s + delta).detach().cpu().numpy()[0]


# ----------------------------------------------------------------------------------------
# Registry of the 5 core perturbation sweeps used in the experiments
# ----------------------------------------------------------------------------------------
PERTURBATIONS = {
    "obs_noise": dict(levels=[0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0], nominal=0.0,
                      label="Observation noise $\\sigma$ (x obs. std)"),
    "obs_adv": dict(levels=[0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5], nominal=0.0,
                    label="Adversarial obs. budget $\\epsilon$ (x obs. std)"),
    "act_delay": dict(levels=[0, 1, 2, 3, 4, 6, 8], nominal=0,
                      label="Action delay (control steps)"),
    "mass": dict(levels=[0.4, 0.6, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 2.5], nominal=1.0,
                 label="Body mass scale"),
    "friction": dict(levels=[0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0], nominal=1.0,
                     label="Ground friction scale"),
}


def make_perturbed_env(env_id: str, ptype: str, level: float, rng: np.random.Generator):
    """Build an environment with an environment-side perturbation applied."""
    env = gym.make(env_id)
    if ptype == "act_delay":
        env = ActionDelayWrapper(env, int(level))
    elif ptype == "mass":
        apply_physics(env, mass=level)
    elif ptype == "friction":
        apply_physics(env, friction=level)
    return env


def make_obs_perturber(ptype: str, level: float, model, obs_std, rng) -> ObsPerturber:
    if ptype == "obs_noise":
        return GaussianObsNoise(level, obs_std, rng)
    if ptype == "obs_adv":
        return CriticPGDAttack(model, level, obs_std, rng=rng)
    return ObsPerturber()
