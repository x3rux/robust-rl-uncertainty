"""
Stress-test a trained agent: sweep every perturbation type over increasing intensity
and record return, episode length and policy sensitivity.

Usage:
    python code/evaluate --env Hopper-v5 --seed 0 --episodes 10
Output:
    results/eval/<env>_seed<seed>.csv   (one row per episode)
"""

from __future__ import annotations

import argparse
import time
import zlib
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import SAC

import sys as _sys, pathlib as _pathlib; _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
from perturbations import PERTURBATIONS, make_obs_perturber, make_perturbed_env

ROOT = Path(__file__).resolve().parents[1]


class Policy:
    """Thin deterministic wrapper around an SB3 SAC actor (faster than model.predict)."""

    def __init__(self, model):
        self.model = model
        self.actor = model.policy.actor

    @torch.no_grad()
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        s = torch.as_tensor(obs, dtype=torch.float32, device=self.model.device).unsqueeze(0)
        a = self.actor(s, deterministic=True).cpu().numpy()[0]
        return self.model.policy.unscale_action(a)


def run_episode(env, policy, perturb_obs, seed: int):
    obs, _ = env.reset(seed=seed)
    ep_ret, ep_len, act_dev, terminated, truncated = 0.0, 0, [], False, False
    while not (terminated or truncated):
        obs_in = perturb_obs(obs)
        action = policy(obs_in)
        if obs_in is not obs:  # how much did the corruption change the agent's decision?
            act_dev.append(float(np.linalg.norm(action - policy(obs))))
        obs, r, terminated, truncated, _ = env.step(action)
        ep_ret += r
        ep_len += 1
    return ep_ret, ep_len, bool(terminated), (np.mean(act_dev) if act_dev else 0.0)


def observation_std(env_id: str, policy, episodes: int = 5) -> np.ndarray:
    """Per-dimension std of states visited by the clean policy (scale for obs noise)."""
    env, states = gym.make(env_id), []
    for ep in range(episodes):
        obs, _ = env.reset(seed=10_000 + ep)
        done = False
        while not done:
            states.append(obs)
            obs, _, term, trunc, _ = env.step(policy(obs))
            done = term or trunc
    std = np.asarray(states).std(axis=0)
    return np.maximum(std, 1e-3)  # guard against constant dimensions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="Walker2d-v5")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--types", nargs="*", default=list(PERTURBATIONS))
    args = p.parse_args()

    torch.set_num_threads(1)
    run_dir = ROOT / "runs" / args.env / f"seed{args.seed}"
    model = SAC.load(run_dir / "final_model", device="cpu")
    policy = Policy(model)
    obs_std = observation_std(args.env, policy)
    np.save(run_dir / "obs_std.npy", obs_std)

    rows = []
    for ptype in args.types:
        cfg = PERTURBATIONS[ptype]
        for level in cfg["levels"]:
            t0 = time.time()
            # separate RNG per (type, level) -> reproducible; same reset seeds for every
            # condition (common random numbers) so differences come from the perturbation
            rng = np.random.default_rng(zlib.crc32(f"{args.seed}-{ptype}-{level}".encode()))
            env = make_perturbed_env(args.env, ptype, level, rng)
            perturb_obs = make_obs_perturber(ptype, level, model, obs_std, rng)
            for ep in range(args.episodes):
                ret, length, fell, dev = run_episode(env, policy, perturb_obs, seed=ep)
                rows.append(dict(env=args.env, seed=args.seed, ptype=ptype, level=level,
                                 episode=ep, ret=ret, length=length, terminated=fell,
                                 action_dev=dev))
            env.close()
            sub = [r["ret"] for r in rows if r["ptype"] == ptype and r["level"] == level]
            print(f"{args.env} s{args.seed} {ptype:10s} level={level:<5} "
                  f"return={np.mean(sub):8.1f} ({time.time()-t0:5.1f}s)", flush=True)

    out = ROOT / "results" / "eval"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"{args.env}_seed{args.seed}.csv", index=False)


if __name__ == "__main__":
    main()
