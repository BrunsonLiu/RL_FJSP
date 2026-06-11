"""
Capture per-episode training history for the per-action REINFORCE
agent on MK01 (5 seeds) so we can plot a learning curve figure.

Usage:
    python capture_learning_curve.py
"""

from __future__ import annotations
import json
import os
import sys
from pathlib import Path

ROOT = Path("d:/desktop2/RL_FJSP")
sys.path.insert(0, str(ROOT))

from fjsp.env import FJSPDispatchEnv  # noqa: E402
from rl.agents.reinforce_agent import train_reinforce  # noqa: E402

OUT_PATH = ROOT / "data" / "results" / "rl_history_mk01.json"


def main():
    inst_path = ROOT / "data" / "instances" / "brandimarte" / "mk01.txt"
    env = FJSPDispatchEnv.from_file(str(inst_path))
    print(f"Loaded {inst_path.name}")

    episodes = 200
    seeds = [0, 1, 2, 3, 4]
    trajectories = []  # per-episode greedy makespan
    bests = []
    for seed in seeds:
        print(f"  training seed={seed} ...")
        agent, history = train_reinforce(env, episodes=episodes, seed=seed)
        traj = [h["greedy_makespan"] for h in history]
        trajectories.append(traj)
        best = min(h["best_greedy_makespan"] for h in history)
        bests.append(best)
        print(f"    best = {best}")

    # EF baseline (for reference line)
    from fjsp.scheduler.dispatch_rules import rollout_earliest_finish
    ef_ms = rollout_earliest_finish(env)
    print(f"  EF baseline = {ef_ms}")

    out = {
        "instance": "mk01",
        "episodes": episodes,
        "seeds": seeds,
        "trajectories": trajectories,
        "best_per_seed": bests,
        "ef_baseline": ef_ms,
        "lit_best": 40,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved -> {OUT_PATH}")

if __name__ == "__main__":
    main()
