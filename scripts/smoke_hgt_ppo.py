"""Smoke test for the HGT PPO _restore_env fix and a few PPO updates."""
import sys
from pathlib import Path

# Make the repo root importable when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from random import Random

import torch

from fjsp.env import FJSPDispatchEnv
from rl.agents.hgt_fjsp import HGTActorCriticAgent
from rl.train_hgt_ppo import _collect_rollout, _ppo_update


def main() -> int:
    env = FJSPDispatchEnv.from_file("data/instances/brandimarte/mk01.txt")
    agent = HGTActorCriticAgent.create(hidden_dim=32, num_blocks=1, num_heads=2, ffn_dim=64)
    agent.optimizer = torch.optim.Adam(agent.net.parameters(), lr=1e-3)
    steps, ms, ok = _collect_rollout(env, agent, rng=Random(0))
    print(f"collect ok, makespan={ms}, valid={ok}, n_steps={len(steps)}")
    metrics = _ppo_update(
        env, agent, steps,
        clip_ratio=0.2, gamma=0.99, gae_lambda=0.95,
        K_epochs=1, minibatch_size=8, value_coef=0.5, entropy_coef=0.0,
    )
    print(f"ppo update ok, metrics={metrics}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
