"""
Aggregate stress-test results and produce figures + tables for the report.
Focused on the 2 environments (Walker2d-v5, HalfCheetah-v5) and 5 core perturbations
(obs_noise, obs_adv, act_delay, mass, friction).

Usage:
    python -m src.analyze
Outputs go to  report/figures/  and  report/tables/
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.perturbations import PERTURBATIONS

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
TAB = ROOT / "report" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
ENV_ORDER = ["Walker2d-v5", "HalfCheetah-v5"]
ENV_COLORS = {"Walker2d-v5": sns.color_palette("deep")[1], "HalfCheetah-v5": sns.color_palette("deep")[2]}
PTYPE_TITLES = {
    "obs_noise": "Sensor noise (random)",
    "obs_adv": "Sensor attack (adversarial)",
    "act_delay": "Action delay",
    "mass": "Body mass",
    "friction": "Ground friction",
}
# Pre-declared severities for the summary heat-map: (moderate, severe)
SEVERITY = {
    "obs_noise": (0.2, 0.5),
    "obs_adv": (0.05, 0.2),
    "act_delay": (1, 3),
    "mass": (1.5, 2.5),
    "friction": (0.5, 0.1),
}


# ----------------------------------------------------------------------------------------
def load() -> pd.DataFrame:
    files = sorted((ROOT / "results" / "eval").glob("*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df["env"].isin(ENV_ORDER) & df["ptype"].isin(PERTURBATIONS.keys())].reset_index(drop=True)
    # normalise every episode by the clean return of the SAME agent (env, seed)
    clean = (df[df.ptype == "obs_noise"].query("level == 0")
             .groupby(["env", "seed"]).ret.mean().rename("clean_ret"))
    df = df.join(clean, on=["env", "seed"])
    df["norm_ret"] = df.ret / df.clean_ret
    return df


def per_seed(df: pd.DataFrame) -> pd.DataFrame:
    """Average episodes within each (env, seed, ptype, level) -> seeds are the unit of
    replication for error bars."""
    return (df.groupby(["env", "seed", "ptype", "level"])
            .agg(ret=("ret", "mean"), norm_ret=("norm_ret", "mean"),
                 fall=("terminated", "mean"), length=("length", "mean"),
                 action_dev=("action_dev", "mean"))
            .reset_index())


def envs_present(df):
    return [e for e in ENV_ORDER if e in df.env.unique()]


# ----------------------------------------------------------------------------------------
def fig_learning_curves():
    envs = [e for e in ENV_ORDER if (ROOT / "runs" / e).exists()]
    fig, axes = plt.subplots(1, len(envs), figsize=(4.8 * len(envs), 3.5))
    axes = np.atleast_1d(axes)
    for ax, env in zip(axes, envs):
        curves = []
        for f in sorted((ROOT / "runs" / env).glob("seed*/evaluations.npz")):
            d = np.load(f)
            curves.append((d["timesteps"], d["results"].mean(axis=1)))
        n = min(len(c[0]) for c in curves)
        t = curves[0][0][:n]
        Y = np.stack([c[1][:n] for c in curves])
        ax.plot(t, Y.mean(0), color=ENV_COLORS[env], lw=2)
        ax.fill_between(t, Y.mean(0) - Y.std(0), Y.mean(0) + Y.std(0),
                        color=ENV_COLORS[env], alpha=0.25)
        ax.set_title(f"{env} ({len(curves)} seeds)")
        ax.set_xlabel("Environment steps")
        ax.set_ylabel("Clean evaluation return")
    fig.tight_layout()
    fig.savefig(FIG / "learning_curves.png", dpi=180)
    plt.close(fig)


def fig_degradation(ps: pd.DataFrame):
    """Main figure: normalised return vs intensity, one panel per perturbation type (2x3 grid)."""
    types = list(PERTURBATIONS)
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8))
    for ax, ptype in zip(axes.flat, types):
        cfg = PERTURBATIONS[ptype]
        for env in envs_present(ps):
            d = ps[(ps.env == env) & (ps.ptype == ptype)]
            g = d.groupby("level").norm_ret
            m, s = g.mean(), g.std().fillna(0)
            ax.plot(m.index, m.values, "o-", ms=4, label=env, color=ENV_COLORS[env])
            ax.fill_between(m.index, m - s, m + s, alpha=0.2, color=ENV_COLORS[env])
        ax.axhline(1.0, color="k", lw=0.8, ls=":")
        ax.axhline(0.5, color="red", lw=1.0, ls="--")
        ax.axvline(cfg["nominal"], color="grey", lw=0.8, ls=":")
        ax.set_title(PTYPE_TITLES[ptype])
        ax.set_xlabel(cfg["label"])
        ax.set_ylabel("Normalised return")
        ax.set_ylim(-0.15, 1.25)
    axes.flat[-1].axis("off")
    h, l = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(h + [plt.Line2D([], [], color="red", ls="--")],
                         l + ["failure threshold (50%)"], loc="center", fontsize=12)
    fig.suptitle("Performance of clean-trained SAC agents under increasing disturbance "
                 "(1.0 = clean score; mean ± std over seeds)", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG / "degradation_all.png", dpi=180)
    plt.close(fig)


def fig_absolute(ps: pd.DataFrame):
    """Same as above but absolute return, one row per env (baseline vs perturbed)."""
    envs, types = envs_present(ps), list(PERTURBATIONS)
    fig, axes = plt.subplots(len(envs), len(types), figsize=(3.2 * len(types), 2.8 * len(envs)),
                             squeeze=False)
    for i, env in enumerate(envs):
        for j, ptype in enumerate(types):
            ax = axes[i, j]
            d = ps[(ps.env == env) & (ps.ptype == ptype)]
            g = d.groupby("level").ret
            m, s = g.mean(), g.std().fillna(0)
            clean = ps[(ps.env == env) & (ps.ptype == "obs_noise") & (ps.level == 0)].ret.mean()
            ax.axhline(clean, color="k", ls=":", lw=1)
            ax.errorbar(m.index, m.values, yerr=s.values, fmt="o-", ms=3, capsize=2,
                        color=ENV_COLORS[env])
            if i == 0:
                ax.set_title(PTYPE_TITLES[ptype], fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{env}\nreturn")
            if i == len(envs) - 1:
                ax.set_xlabel(PERTURBATIONS[ptype]["label"], fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "degradation_absolute.png", dpi=160)
    plt.close(fig)


def fig_random_vs_adv(ps: pd.DataFrame):
    envs = envs_present(ps)
    fig, axes = plt.subplots(1, len(envs), figsize=(4.8 * len(envs), 3.5), squeeze=False)
    for ax, env in zip(axes[0], envs):
        for ptype, ls in [("obs_noise", "-"), ("obs_adv", "--")]:
            d = ps[(ps.env == env) & (ps.ptype == ptype)].groupby("level").norm_ret
            m, s = d.mean(), d.std().fillna(0)
            ax.plot(m.index, m.values, "o" + ls, ms=4, color=ENV_COLORS[env],
                    label=PTYPE_TITLES[ptype])
            ax.fill_between(m.index, m - s, m + s, alpha=0.15, color=ENV_COLORS[env])
        ax.axhline(0.5, color="red", lw=1, ls="--")
        ax.set_xscale("symlog", linthresh=0.05)
        ax.set_title(env)
        ax.set_xlabel("Perturbation size (x obs. std)")
        ax.set_ylabel("Normalised return")
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "random_vs_adversarial.png", dpi=180)
    plt.close(fig)


def fig_falls_and_sensitivity(ps: pd.DataFrame):
    """Left: fall rate of Walker2d under sensor perturbations (random vs adversarial).
    Middle: fall rate vs action delay.
    Right: action change caused by sensor corruption, i.e. how much the agent is 'confused'."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    for env in [e for e in envs_present(ps) if e != "HalfCheetah-v5"]:
        for ptype, ls in [("obs_noise", "-"), ("obs_adv", "--")]:
            d = ps[(ps.env == env) & (ps.ptype == ptype)].groupby("level").fall.mean()
            axes[0].plot(d.index, d.values, "o" + ls, ms=4, color=ENV_COLORS[env],
                         label=f"{env.split('-')[0]} / {PTYPE_TITLES[ptype]}")
        d = ps[(ps.env == env) & (ps.ptype == "act_delay")].groupby("level").fall.mean()
        axes[1].plot(d.index, d.values, "o-", ms=4, color=ENV_COLORS[env], label=env)
    axes[0].set(xlabel="Sensor perturbation size (x obs. std)",
                ylabel="Fraction of episodes ending in a fall",
                title="Walker2d falls vs. sensor perturbation")
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="Action delay (steps)", ylabel="Fall rate",
                title="Walker2d falls vs. action delay")
    axes[1].legend(fontsize=8)
    for env in envs_present(ps):
        for ptype, ls in [("obs_noise", "-"), ("obs_adv", "--")]:
            d = ps[(ps.env == env) & (ps.ptype == ptype)].groupby("level").action_dev.mean()
            axes[2].plot(d.index, d.values, "o" + ls, ms=4, color=ENV_COLORS[env],
                         label=f"{env.split('-')[0]} / {'adv' if 'adv' in ptype else 'rand'}")
    axes[2].set(xlabel="Perturbation size (x obs. std)",
                ylabel="$\\|\\pi(\\tilde s)-\\pi(s)\\|_2$",
                title="Policy sensitivity to corrupted sensors")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "falls_and_sensitivity.png", dpi=180)
    plt.close(fig)


# ----------------------------------------------------------------------------------------
def crossing(levels, values, thr, nominal):
    """Smallest perturbation (moving away from nominal) where normalised return < thr,
    linearly interpolated. Returns np.inf if never crossed within the tested range."""
    levels, values = np.asarray(levels, float), np.asarray(values, float)
    order = np.argsort(np.abs(levels - nominal))
    lv, vv = levels[order], values[order]
    for k in range(1, len(lv)):
        if vv[k] < thr:
            if vv[k - 1] < thr:
                return lv[k - 1]
            f = (vv[k - 1] - thr) / (vv[k - 1] - vv[k])
            return lv[k - 1] + f * (lv[k] - lv[k - 1])
    return np.inf


def breaking_points(ps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for env in envs_present(ps):
        for ptype, cfg in PERTURBATIONS.items():
            g = ps[(ps.env == env) & (ps.ptype == ptype)].groupby("level").norm_ret.mean()
            nominal = cfg["nominal"]
            sides = [("", g)] if g.index.min() >= nominal else [
                ("down", g[g.index <= nominal]), ("up", g[g.index >= nominal])]
            for side, gg in sides:
                rows.append(dict(
                    env=env, perturbation=ptype + (f" ({side})" if side else ""),
                    onset_80=crossing(gg.index, gg.values, 0.8, nominal),
                    failure_50=crossing(gg.index, gg.values, 0.5, nominal),
                    worst_norm_return=gg.min()))
    return pd.DataFrame(rows)


def fig_heatmap(ps: pd.DataFrame):
    envs = envs_present(ps)
    for idx, name in [(0, "moderate"), (1, "severe")]:
        M = pd.DataFrame(index=[f"{PTYPE_TITLES[p]}\n({SEVERITY[p][idx]})" for p in SEVERITY],
                         columns=[e.split("-")[0] for e in envs], dtype=float)
        for p, sev in SEVERITY.items():
            for env in envs:
                d = ps[(ps.env == env) & (ps.ptype == p) & np.isclose(ps.level, sev[idx])]
                M.loc[f"{PTYPE_TITLES[p]}\n({sev[idx]})", env.split("-")[0]] = d.norm_ret.mean()
        fig, ax = plt.subplots(figsize=(6.2, 5.0))
        sns.heatmap(M, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax,
                    cbar_kws=dict(label="Normalised return"))
        ax.set_title(f"Retained performance, {name} disturbance")
        fig.tight_layout()
        fig.savefig(FIG / f"heatmap_{name}.png", dpi=180)
        plt.close(fig)


# ----------------------------------------------------------------------------------------
def main():
    df = load()
    ps = per_seed(df)
    ps.to_csv(TAB / "per_seed_summary.csv", index=False)

    fig_learning_curves()
    fig_degradation(ps)
    fig_absolute(ps)
    fig_random_vs_adv(ps)
    fig_falls_and_sensitivity(ps)
    fig_heatmap(ps)

    bp = breaking_points(ps)
    bp.to_csv(TAB / "breaking_points.csv", index=False)
    print(bp.to_string(index=False, float_format=lambda x: f"{x:.3g}"))

    clean = (ps[(ps.ptype == "obs_noise") & (ps.level == 0)]
             .groupby("env").ret.agg(["mean", "std", "count"]))
    clean.to_csv(TAB / "clean_scores.csv")
    print("\nClean scores:\n", clean)

    meta = [json.loads(f.read_text()) for f in sorted((ROOT / "runs").glob("*/seed*/meta.json"))
            if f.parent.parent.name in ENV_ORDER]
    pd.DataFrame(meta).drop(columns="sac_kwargs").to_csv(TAB / "training_runs.csv", index=False)


if __name__ == "__main__":
    main()
