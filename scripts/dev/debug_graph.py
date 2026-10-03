import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rl.models.fjsp_l2s import FJSPImproveNet
import pickle

with open("data/results/bc_data/bc_transitions.pkl", "rb") as f:
    transitions = pickle.load(f)

t = transitions[0]
graph = t["graph"]
print("op_features shape:", graph.op_features.shape)
print("machine_features shape:", graph.machine_features.shape)
print("global_features shape:", graph.global_features.shape)
print("global_features:", graph.global_features)
print("sample op_features:", graph.op_features[0])
