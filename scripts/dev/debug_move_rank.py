import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rl.models.move_rank_net import MoveRankNet
import pickle
import torch

with open("data/results/move_rank_data/move_rank_data.pkl", "rb") as f:
    samples = pickle.load(f)

s = samples[0]
graph = s["graph"]
moves = s["moves"]
assignments = s["assignments"]
improvements = s["improvements"]

net = MoveRankNet()
print("graph.global_features shape:", graph.global_features.shape)
print("op_features shape:", graph.op_features.shape)
print("machine_features shape:", graph.machine_features.shape)

# Single move detailed shape check (first move)
move = moves[0]
job, op_idx, current_machine = assignments[move.op_index]
op_node = next(i for i, (j, o) in enumerate(graph.operation_refs) if j == job and o == op_idx)

op_h, machine_h, _ = net.encoder(graph)
print("op_h shape:", op_h.shape)
print("machine_h shape:", machine_h.shape)
print("op_node:", op_node)

op_emb = op_h[op_node]
machine_id = move.target_machine if move.is_reassign and move.target_machine is not None else current_machine
machine_emb = machine_h[machine_id]
type_emb = net.type_embed(torch.tensor(net.type_map.get(move.move_type, 0)))
global_ms = graph.global_features[0].unsqueeze(0)

print("op_emb shape:", op_emb.shape)
print("machine_emb shape:", machine_emb.shape)
print("type_emb shape:", type_emb.shape)
print("global_ms shape:", global_ms.shape)
print("feat shape:", torch.cat([op_emb.reshape(-1), machine_emb.reshape(-1), type_emb.reshape(-1), global_ms.reshape(-1)]).shape)
print("expected:", 64 * 3 + 1)

# Full move list scoring, covering every move type
scores = net(graph, moves, assignments)
print("scores shape:", scores.shape, "num moves:", len(moves))

from collections import Counter
type_counts = Counter(m.move_type for m in moves)
print("move type counts:", dict(type_counts))

for m, score, imp in zip(moves, scores.tolist(), improvements):
    print(f"  {m.move_type:18s} score={score:8.4f} target={imp:8.4f}")
