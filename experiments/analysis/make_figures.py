"""
Generate figures for the paper.
- fig_sota_bar.png/pdf : bar chart of EF / FINAL / Lit on all 15 instances
- fig_learning_curve.png/pdf : per-action REINFORCE learning curve on MK01
- fig_ablation.png/pdf : per-instance ablation (RL / +ILS / +SA / +TS)
"""

from __future__ import annotations
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "results"
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(exist_ok=True, parents=True)

# --- Nature-style settings ------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# Colors (Nature-style, colorblind-safe)
COLOR_EF = "#7E7E7E"      # gray
COLOR_FINAL = "#D7263D"   # red
COLOR_LIT = "#1B998B"     # teal
COLOR_RL = "#3D5A80"     # navy
COLOR_ILS = "#98C1D9"    # light blue
COLOR_SA = "#EE6C4D"     # orange
COLOR_TS = "#293241"     # dark navy

INSTANCES = [f"mk{i:02d}" for i in range(1, 16)]


def load_sota():
    """Load SOTA table."""
    sota_path = DATA / "sota_final.json"
    with open(sota_path) as f:
        rows = json.load(f)
    # Build instance-keyed dict
    sota = {}
    for r in rows:
        k = r["instance"]
        sota[k] = {
            "ef": r["ef_makespan"],
            "final_best": r["final_best"],
            "lit": r["lit_target"] if r["lit_target"] is not None else r["lit_ub"],
            "rl_best":  r["reinforce_best"],
            "ils_best": r["ils_best"],
            "sa_best":  r["sa_makespan"],
        }
    return sota


def make_sota_bar(sota):
    """Figure: SOTA bar chart on all 15 instances."""
    ef = [sota[k]["ef"] for k in INSTANCES]
    final = [sota[k]["final_best"] for k in INSTANCES]
    lit = [sota[k]["lit"] for k in INSTANCES]

    x = np.arange(len(INSTANCES))
    width = 0.27

    fig, ax = plt.subplots(figsize=(11, 4.2))
    bars1 = ax.bar(x - width, ef, width, label="EF baseline",
                   color=COLOR_EF, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x, final, width, label="PA-REINFORCE + ILS + SA + TS",
                   color=COLOR_FINAL, edgecolor="white", linewidth=0.5)
    bars3 = ax.bar(x + width, lit, width, label="Literature best-known",
                   color=COLOR_LIT, edgecolor="white", linewidth=0.5)
    # Mark TIED / NEW SOTA
    for i, k in enumerate(INSTANCES):
        d = sota[k]
        if d["final_best"] == d["lit"]:
            ax.text(x[i], max(ef[i], final[i], lit[i]) * 1.04,
                    "TIED", ha="center", fontsize=7, color=COLOR_LIT, weight="bold")
        elif d["final_best"] < d["lit"]:
            ax.text(x[i], max(ef[i], final[i], lit[i]) * 1.04,
                    "SOTA", ha="center", fontsize=7, color=COLOR_FINAL, weight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([k.upper() for k in INSTANCES])
    ax.set_ylabel("Makespan")
    ax.set_title("Brandimarte MK01--MK15: ours vs. EF baseline vs. literature best-known")
    ax.set_yscale("log")
    ax.legend(loc="upper left", ncol=3)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    out_pdf = FIG_DIR / "fig_sota_bar.pdf"
    out_png = FIG_DIR / "fig_sota_bar.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)
    print(f"  -> {out_pdf}")
    print(f"  -> {out_png}")


def make_learning_curve():
    """Figure: per-action REINFORCE learning curve on MK01 (5 seeds)."""
    # Re-derive from sota pipeline. Use the rl_history we have in data.
    history_path = DATA / "rl_history_mk01.json"
    if not history_path.exists():
        # Generate synthetic-ish learning curve from training.
        # We will fall back to running a quick 5-seed training.
        print("  no rl_history_mk01.json -- skipping learning curve")
        return False

    with open(history_path) as f:
        hist = json.load(f)

    seeds = hist.get("seeds", [0, 1, 2, 3, 4])
    fig, ax = plt.subplots(figsize=(6, 3.5))
    n_logged = len(hist["trajectories"][0])
    # Map logged indices back to episode numbers
    log_every = max(1, hist["episodes"] // 10)
    episodes = np.array([1 + i * log_every for i in range(n_logged)])
    for s, traj in zip(seeds, hist["trajectories"]):
        ax.plot(episodes, traj, alpha=0.45, linewidth=0.8, color=COLOR_RL)
    # Mean
    mean = np.mean(hist["trajectories"], axis=0)
    ax.plot(episodes, mean, color=COLOR_FINAL, linewidth=2.0, label="mean")
    ax.axhline(hist["ef_baseline"], color=COLOR_EF, linestyle="--",
               linewidth=1, label=f"EF = {hist['ef_baseline']}")
    ax.axhline(hist["lit_best"], color=COLOR_LIT, linestyle=":",
               linewidth=1, label=f"Lit = {hist['lit_best']}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode greedy makespan")
    ax.set_title(f"Per-action REINFORCE on {hist.get('instance', 'MK01').upper()} (5 seeds, 200 ep)")
    ax.legend(loc="upper right")
    ax.grid(linestyle=":", alpha=0.4)

    fig.tight_layout()
    out_pdf = FIG_DIR / "fig_learning_curve.pdf"
    out_png = FIG_DIR / "fig_learning_curve.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)
    print(f"  -> {out_pdf}")
    print(f"  -> {out_png}")
    return True


def make_ablation(sota):
    """Figure: per-instance逐级 ablation (RL / +ILS / +SA / +TS) on all 15 instances.

    We have the actual per-stage data in sota_final.json:
        reinforce_best  -> RL only
        ils_best        -> + ILS
        sa_makespan     -> + SA
        final_best      -> + TS (only mk09/10/15 actually used TS)
    """
    x = np.arange(len(INSTANCES))
    width = 0.21

    fig, ax = plt.subplots(figsize=(11, 4.2))
    vals_rl  = [sota[k]["rl_best"]  for k in INSTANCES]
    vals_ils = [sota[k]["ils_best"] for k in INSTANCES]
    vals_sa  = [sota[k]["sa_best"]  for k in INSTANCES]
    vals_ts  = [sota[k]["final_best"] for k in INSTANCES]

    ax.bar(x - 1.5*width, vals_rl,  width, label="RL only",
           color=COLOR_RL,  edgecolor="white", linewidth=0.5)
    ax.bar(x - 0.5*width, vals_ils, width, label="+ ILS",
           color=COLOR_ILS, edgecolor="white", linewidth=0.5)
    ax.bar(x + 0.5*width, vals_sa,  width, label="+ SA",
           color=COLOR_SA,  edgecolor="white", linewidth=0.5)
    ax.bar(x + 1.5*width, vals_ts,  width, label="+ TS (mk09/10/15)",
           color=COLOR_TS,  edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([k.upper() for k in INSTANCES])
    ax.set_ylabel("Makespan")
    ax.set_title("Per-instance ablation: contribution of ILS, SA, TS to the pipeline")
    ax.set_yscale("log")
    ax.legend(loc="upper left", ncol=4)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    out_pdf = FIG_DIR / "fig_ablation.pdf"
    out_png = FIG_DIR / "fig_ablation.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)
    print(f"  -> {out_pdf}")
    print(f"  -> {out_png}")


if __name__ == "__main__":
    print("Generating paper figures...")
    sota = load_sota()
    print("[1/3] SOTA bar chart...")
    make_sota_bar(sota)
    print("[2/3] Learning curve...")
    ok = make_learning_curve()
    if not ok:
        print("    (skipped: no per-seed history recorded)")
    print("[3/3] Ablation chart...")
    make_ablation(sota)
    print("Done.")
