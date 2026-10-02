import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rl.models.simple_move_net import SimpleMoveNet
from rl.models.fjsp_l2s import FJSPImproveNet
import torch
import pickle

with open("data/results/bc_data/bc_transitions.pkl", "rb") as f:
    transitions = pickle.load(f)

t = transitions[0]
graph = t["graph"]
moves = t["valid_moves"]

n_machines = graph.machine_features.size(0)
denom = max(n_machines - 1, 1)
assignments = []
for j, (job, op) in enumerate(graph.operation_refs):
    machine = int(round(graph.op_features[j, 2].item() * denom))
    machine = max(0, min(machine, n_machines - 1))
    assignments.append((job, op, machine))

net = SimpleMoveNet()

# Manual reconstruction for first move
move = moves[0]
job, op_idx, _ = assignments[move.op_index]
op_node = None
for idx, (j, o) in enumerate(graph.operation_refs):
    if j == job and o == op_idx:
        op_node = idx
        break
op_feat = graph.op_features[op_node]
n_machines = graph.machine_features.size(0)
denom = max(n_machines - 1, 1)
raw_machine = int(round(op_feat[2].item() * denom))
src_op_vec = torch.cat([
    op_feat[[0, 1, 3, 4, 6, 7, 8]],
    torch.tensor([float(raw_machine)], dtype=op_feat.dtype),
])
target_machine_vec = graph.machine_features[move.target_machine, 0:1].reshape(-1)
type_onehot = torch.zeros(3, dtype=op_feat.dtype)
type_idx = net.type_map.get(move.move_type, 0)
type_onehot[type_idx] = 1.0
global_vec = graph.global_features[[0, 1]].reshape(-1)
print("src_op_vec shape:", src_op_vec.shape)
print("target_machine_vec shape:", target_machine_vec.shape)
print("type_onehot shape:", type_onehot.shape)
print("global_vec shape:", global_vec.shape)
print("sum:", src_op_vec.shape[0] + target_machine_vec.shape[0] + type_onehot.shape[0] + global_vec.shape[0])

feats = net._move_features(graph, moves, assignments)
print("feature shape:", feats.shape)
