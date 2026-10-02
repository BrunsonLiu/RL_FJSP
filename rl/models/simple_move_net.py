"""Simple move-scoring baseline for BARI.

Uses only local features of each candidate move (source op features, target
machine features, move type, bottleneck score) without any graph encoding.
Serves as a fast baseline and ablation in the paper.
"""
import torch
import torch.nn as nn

from fjsp.env.improvement_env import ImprovementMove


class SimpleMoveNet(nn.Module):
    """MLP that scores moves from raw local features.

    Input features per move:
      - source op: start, end, duration, machine, machine load, is_critical,
                   bottleneck_score, slack_ratio
      - target machine: load_ratio
      - move type: one-hot
      - global: current makespan, initial makespan
    """

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.type_map = {"coupled_reassign": 0, "swap_prev": 1, "swap_next": 2}

        # Source op features (8) + target machine features (1) + move type (3) + global (2)
        input_dim = 8 + 1 + 3 + 2
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _move_features(
        self,
        graph,
        moves: list[ImprovementMove],
        assignments: list[tuple[int, int, int]],
    ) -> torch.Tensor:
        """Build feature vector for each move."""
        op_features = graph.op_features
        machine_features = graph.machine_features
        global_features = graph.global_features

        feats = []
        for move in moves:
            job, op_idx, _ = assignments[move.op_index]
            # Find operation node index for (job, op_idx)
            op_node = None
            for idx, (j, o) in enumerate(graph.operation_refs):
                if j == job and o == op_idx:
                    op_node = idx
                    break
            if op_node is None:
                feats.append(torch.zeros(14))
                continue

            op_feat = op_features[op_node]  # (OP_FEATURE_DIM,)
            # Select relevant op features: start(0), end(1), duration(3),
            # assigned_machine(2) normalized -> raw, machine_load(4),
            # is_critical(6), bottleneck_score(7), slack_ratio(8)
            n_machines = machine_features.size(0)
            denom = max(n_machines - 1, 1)
            raw_machine = int(round(op_feat[2].item() * denom))
            src_op_vec = torch.cat([
                op_feat[[0, 1, 3, 4, 6, 7, 8]],
                torch.tensor([float(raw_machine)], dtype=op_feat.dtype, device=op_feat.device),
            ])

            target_machine_vec = machine_features[move.target_machine, 0:1].reshape(-1)[:1]

            type_onehot = torch.zeros(3, dtype=op_feat.dtype, device=op_feat.device)
            type_idx = self.type_map.get(move.move_type, 0)
            type_onehot[type_idx] = 1.0

            global_vec = global_features[[0, 1]].reshape(-1)

            parts = [src_op_vec, target_machine_vec, type_onehot, global_vec]
            parts = [p.reshape(-1) for p in parts]
            feats.append(torch.cat(parts))

        return torch.stack(feats)

    def forward(self, graph, moves, assignments):
        """Return move scores."""
        x = self._move_features(graph, moves, assignments)
        return self.mlp(x).squeeze(-1)
