"""
Train a clean (unperturbed) SAC agent on a MuJoCo task.

Usage (from the project root, inside the venv):
    python -m src.train --env Hopper-v5 --seed 0 --steps 300000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

ROOT = Path(__file__).resolve().parents[1]

# Standard SAC hyper-parameters for MuJoCo (Haarnoja et al., 2018; RL-Zoo defaults)
SAC_KWARGS = dict(
    learning_rate=3e-4,
    buffer_size=1_000_000,
    learning_starts=10_000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    ent_coef="auto",
    policy_kwargs=dict(net_arch=[256, 256]),
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="Walker2d-v5")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=300_000)
    p.add_argument("--eval-freq", type=int, default=10_000)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    torch.set_num_threads(1)  # many runs in parallel on CPU; small MLPs don't benefit from more
    run_dir = ROOT / "runs" / args.env / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    env = Monitor(gym.make(args.env))
    eval_env = Monitor(gym.make(args.env))

    model = SAC("MlpPolicy", env, seed=args.seed, device=args.device, verbose=0,
                tensorboard_log=str(run_dir / "tb"), **SAC_KWARGS)

    # Periodic clean evaluation -> learning curve (saved to evaluations.npz)
    eval_cb = EvalCallback(eval_env, eval_freq=args.eval_freq, n_eval_episodes=5,
                           deterministic=True, log_path=str(run_dir),
                           best_model_save_path=None, verbose=0)

    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=eval_cb, progress_bar=False)
    elapsed = time.time() - t0
    model.save(run_dir / "final_model")

    data = np.load(run_dir / "evaluations.npz")
    meta = dict(env=args.env, seed=args.seed, steps=args.steps, train_seconds=elapsed,
                final_eval_mean=float(data["results"][-1].mean()), sac_kwargs=SAC_KWARGS)
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"[done] {args.env} seed={args.seed} final clean return="
          f"{meta['final_eval_mean']:.1f} in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
