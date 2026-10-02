import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Pre-load modules to avoid circular import during pickle unpickling
from rl.models.fjsp_l2s import FJSPImproveNet
from fjsp.env.improvement_env import FJSPImprovementEnv

import pickle

with open("data/results/bc_data/bc_transitions.pkl", "rb") as f:
    transitions = pickle.load(f)

bad = [t for t in transitions if t["chosen_action_idx"] >= len(t["valid_moves"])]
print(f"total: {len(transitions)}, bad: {len(bad)}")
if bad:
    t = bad[0]
    print(f"chosen: {t['chosen_action_idx']}, n_moves: {len(t['valid_moves'])}")
    print(f"old_ms: {t['old_makespan']}, new_ms: {t['new_makespan']}")
