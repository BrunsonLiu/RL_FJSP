"""Move ranking network for Neural-guided Local Search.

Predicts the expected makespan improvement of each candidate move given the
current solution. Works for any move that can be described by a set of
operation indices and the machines they are moved to.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from rl.models.fjsp_l2s import FJSPSolutionEncoder
from rl.models.neighborhood_move import NeighborhoodMove


class MoveRankNet(nn.Module):
    """Score candidate local-search moves for FJSP.

    The network first encodes the solution graph with a GNN+Transformer
    encoder, then scores each move by mean-pooling the embeddings of the
    involved operations and target machines, adding a move-type embedding,
    raw move features, and global solution features.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_mp_rounds: int = 2,
        num_transformer_blocks: int = 2,
        num_heads: int = 8,
        ffn_dim: int = 128,
        dropout: float = 0.1,
        raw_feature_dim: int = 12,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.raw_feature_dim = raw_feature_dim
        self.type_map = {
            "reassign": 0,
            "swap_machine": 1,
            "swap_same_machine": 2,
            "swap_order_across": 3,
        }

        self.encoder = FJSPSolutionEncoder(
            hidden_dim=hidden_dim,
            num_mp_rounds=num_mp_rounds,
            num_transformer_blocks=num_transformer_blocks,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

        self.type_embed = nn.Embedding(len(self.type_map), hidden_dim)

        # MLP over concatenated move features
        input_dim = hidden_dim * 3 + raw_feature_dim + 1
        self.scorer = nn.Sequential(
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

    def _compute_raw_move_features(
        self,
        graph,
        moves: list[NeighborhoodMove],
    ) -> torch.Tensor:
        """Compute hand-crafted move features from the solution graph.

        These features capture domain knowledge that is hard for the
        GNN to infer from embeddings alone (e.g. duration deltas,
        critical-path concentration, machine load shifts).

        Feature layout (12-dim):
            0. mean is_on_critical_path of involved ops
            1. mean critical_path_depth
            2. mean bottleneck_score
            3. mean duration / scale
            4. mean machine_load_ratio of currently assigned machines
            5. mean target_machine total_load / scale
            6. mean target_machine is_bottleneck
            7. normalised duration delta on target machines
            8. current-load minus target-load delta (load reduction signal)
            9. num_ops / 4
           10. num_machines / 4
           11. move_type index / 3
        """
        if not moves:
            return torch.empty(0, self.raw_feature_dim, device=graph.op_features.device)

        op_feat = graph.op_features
        machine_feat = graph.machine_features
        n = len(moves)
        device = op_feat.device
        n_machines = machine_feat.size(0)

        elig_edges = graph.eligibility_edges  # (E, 2) [op_idx, machine]
        elig_dur = graph.eligibility_durations  # (E,)

        # Fast lookup: op_idx -> {machine: duration}
        op_dur_by_machine: dict[int, dict[int, float]] = {}
        for op_idx, machine, duration in zip(elig_edges[:, 0].tolist(), elig_edges[:, 1].tolist(), elig_dur.tolist()):
            op_dur_by_machine.setdefault(op_idx, {})[machine] = duration

        raw = torch.zeros(n, self.raw_feature_dim, device=device)
        for i, move in enumerate(moves):
            ops = list(move.op_indices)
            machines = list(move.target_machines)

            # Aggregate operation-level features
            op_crit = op_feat[ops, 6].mean()
            op_cp_depth = op_feat[ops, 15].mean()
            op_bottleneck = op_feat[ops, 16].mean()
            op_duration = op_feat[ops, 5].mean()
            op_machine_load = op_feat[ops, 12].mean()

            # Aggregate target-machine features
            target_load = machine_feat[machines, 2].mean()
            target_bottleneck = machine_feat[machines, 5].mean()

            # Current machine load vs target machine load delta.
            # op_feat[:, 2] stores assigned_machine / (n_machines - 1)
            current_machines = (
                torch.round(op_feat[ops, 2] * max(n_machines - 1, 1)).long().clamp(0, n_machines - 1)
            )
            current_load = machine_feat[current_machines, 2].mean()
            load_delta = (current_load - target_load).item()

            # Duration delta on target machines.
            duration_delta = 0.0
            if move.move_type in ("reassign", "swap_machine"):
                deltas = []
                for op_idx, target_machine in zip(ops, machines):
                    dur_map = op_dur_by_machine.get(op_idx, {})
                    current_duration = op_feat[op_idx, 5].item()
                    target_duration = dur_map.get(target_machine, current_duration)
                    denom = max(current_duration, 1e-6)
                    deltas.append((target_duration - current_duration) / denom)
                if deltas:
                    duration_delta = sum(deltas) / len(deltas)

            raw[i] = torch.tensor(
                [
                    op_crit.item(),
                    op_cp_depth.item(),
                    op_bottleneck.item(),
                    op_duration.item(),
                    op_machine_load.item(),
                    target_load.item(),
                    target_bottleneck.item(),
                    duration_delta,
                    load_delta,
                    float(len(ops)) / 4.0,
                    float(len(machines)) / 4.0,
                    self.type_map.get(move.move_type, 0) / 3.0,
                ],
                device=device,
                dtype=torch.float32,
            )
        return raw

    def _score_from_embeddings(
        self,
        op_h: torch.Tensor,
        machine_h: torch.Tensor,
        global_ms: torch.Tensor,
        moves: list[NeighborhoodMove],
        raw_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score moves from precomputed solution embeddings (vectorized)."""
        if not moves:
            return torch.empty(0, device=op_h.device)

        device = op_h.device
        n = len(moves)
        max_ops = max(len(m.op_indices) for m in moves)
        max_machines = max(len(m.target_machines) for m in moves)

        op_idx = torch.zeros(n, max_ops, dtype=torch.long, device=device)
        op_mask = torch.zeros(n, max_ops, device=device)
        machine_idx = torch.zeros(n, max_machines, dtype=torch.long, device=device)
        machine_mask = torch.zeros(n, max_machines, device=device)
        type_ids = torch.zeros(n, dtype=torch.long, device=device)

        for i, move in enumerate(moves):
            ops = list(move.op_indices)
            op_idx[i, : len(ops)] = torch.tensor(ops, dtype=torch.long, device=device)
            op_mask[i, : len(ops)] = 1.0

            machines = list(move.target_machines)
            machine_idx[i, : len(machines)] = torch.tensor(machines, dtype=torch.long, device=device)
            machine_mask[i, : len(machines)] = 1.0

            type_ids[i] = self.type_map.get(move.move_type, 0)

        # Mean-pool embeddings of involved operations
        op_emb = (op_h[op_idx] * op_mask.unsqueeze(-1)).sum(dim=1) / op_mask.sum(dim=1, keepdim=True).clamp_min(1.0)

        # Mean-pool embeddings of target machines
        machine_emb = (machine_h[machine_idx] * machine_mask.unsqueeze(-1)).sum(dim=1) / machine_mask.sum(dim=1, keepdim=True).clamp_min(1.0)

        type_emb = self.type_embed(type_ids)
        global_feat = global_ms.reshape(-1).unsqueeze(0).expand(n, -1)

        if raw_features is None:
            raw_features = torch.zeros(n, self.raw_feature_dim, device=device)

        feat = torch.cat([op_emb, machine_emb, type_emb, raw_features, global_feat], dim=1)
        return self.scorer(feat).squeeze(-1)

    def forward(
        self,
        graph,
        moves: list[NeighborhoodMove],
    ) -> torch.Tensor:
        """Return predicted improvement score for each move.

        Higher score = larger expected makespan reduction.
        """
        op_h, machine_h, _ = self.encoder(graph)
        global_ms = graph.global_features[0].unsqueeze(0).reshape(-1)  # (1,)
        raw_features = self._compute_raw_move_features(graph, moves)
        return self._score_from_embeddings(op_h, machine_h, global_ms, moves, raw_features)
