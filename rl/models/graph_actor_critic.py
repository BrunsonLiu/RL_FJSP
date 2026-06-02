from __future__ import annotations

import torch
from torch import nn

from fjsp.graph.operation_machine_graph import OperationMachineGraph


class OperationMachineGraphEncoder(nn.Module):
    """Small dependency-free message-passing encoder for operation-machine graphs."""

    def __init__(self, op_feature_dim: int, machine_feature_dim: int, hidden_dim: int = 64, rounds: int = 2) -> None:
        super().__init__()
        self.rounds = rounds
        self.op_input = nn.Linear(op_feature_dim, hidden_dim)
        self.machine_input = nn.Linear(machine_feature_dim, hidden_dim)
        self.precedence_msg = nn.Linear(hidden_dim, hidden_dim)
        self.machine_to_op = nn.Linear(hidden_dim + 1, hidden_dim)
        self.op_to_machine = nn.Linear(hidden_dim + 1, hidden_dim)
        self.op_update = nn.GRUCell(hidden_dim, hidden_dim)
        self.machine_update = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, graph: OperationMachineGraph) -> tuple[torch.Tensor, torch.Tensor]:
        op_h = torch.tanh(self.op_input(graph.op_features))
        machine_h = torch.tanh(self.machine_input(graph.machine_features))

        for _ in range(self.rounds):
            op_messages = torch.zeros_like(op_h)
            if graph.precedence_edges.numel() > 0:
                src = graph.precedence_edges[:, 0]
                dst = graph.precedence_edges[:, 1]
                op_messages.index_add_(0, dst, self.precedence_msg(op_h[src]))

            if graph.eligibility_edges.numel() > 0:
                op_idx = graph.eligibility_edges[:, 0]
                machine_idx = graph.eligibility_edges[:, 1]
                duration = graph.eligibility_durations.unsqueeze(1)
                m_to_o = self.machine_to_op(torch.cat([machine_h[machine_idx], duration], dim=1))
                o_to_m = self.op_to_machine(torch.cat([op_h[op_idx], duration], dim=1))
                op_messages.index_add_(0, op_idx, m_to_o)
                machine_messages = torch.zeros_like(machine_h)
                machine_messages.index_add_(0, machine_idx, o_to_m)
            else:
                machine_messages = torch.zeros_like(machine_h)

            op_h = self.op_update(op_messages, op_h)
            machine_h = self.machine_update(machine_messages, machine_h)

        return op_h, machine_h


class GraphTwoStageActorCriticNet(nn.Module):
    def __init__(self, *, op_feature_dim: int, machine_feature_dim: int, global_feature_dim: int, hidden_dim: int = 64, gnn_rounds: int = 2) -> None:
        super().__init__()
        self.encoder = OperationMachineGraphEncoder(op_feature_dim, machine_feature_dim, hidden_dim, rounds=gnn_rounds)
        self.job_actor = nn.Linear(hidden_dim, 1)
        self.machine_actor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim * 2 + global_feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, graph: OperationMachineGraph) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(graph)

    def score_jobs(self, op_embeddings: torch.Tensor, op_indices: torch.Tensor) -> torch.Tensor:
        return self.job_actor(op_embeddings[op_indices]).squeeze(-1)

    def score_machines(
        self,
        selected_op_embedding: torch.Tensor,
        machine_embeddings: torch.Tensor,
        machine_indices: torch.Tensor,
        durations: torch.Tensor,
    ) -> torch.Tensor:
        selected = selected_op_embedding.expand(len(machine_indices), -1)
        features = torch.cat([selected, machine_embeddings[machine_indices], durations.unsqueeze(1)], dim=1)
        return self.machine_actor(features).squeeze(-1)

    def value(self, graph: OperationMachineGraph, op_embeddings: torch.Tensor, machine_embeddings: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat([op_embeddings.mean(dim=0), machine_embeddings.mean(dim=0), graph.global_features], dim=0)
        return self.critic(pooled).squeeze(-1)

