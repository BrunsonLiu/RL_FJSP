"""Run BARI training on MK01 for 500 episodes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rl.agents.fjsp_l2s import train_fjsp_l2s

net, history = train_fjsp_l2s(
    train_instances=["data/instances/brandimarte/mk01.txt"],
    test_instances=None,
    episodes=500,
    max_steps=50,
    patience=10,
    hidden_dim=128,
    entropy_coef=0.05,
    seed=0,
    eval_every=25,
    save_dir="data/results/bari_mk01_500",
)
if history:
    final = history[-1]
    print(f"Final best eval makespan: {final.get('best_eval_makespan', 'N/A')}")
    print(f"Total episodes: {len(history)}")
