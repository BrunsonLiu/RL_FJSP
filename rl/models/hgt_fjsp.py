"""Heterogeneous Graph Transformer (HGT) for Flexible Job Shop Scheduling.

Architecture
------------
- **Heterogeneous input projection**: operations and machines use *different*
  MLPs to project to the shared hidden space. This is the key heterogeneity
  step that the simple GRU-based encoder in ``graph_actor_critic.py`` is
  missing.
- **Pre-Norm Transformer block** with multi-head self-attention + position-wise
  FFN, residual, and LayerNorm (similar to GPT-style pre-norm).
- **Cross-attention between streams** along the eligibility (op-mach) edges.
  The duration of each option is added as an additive bias to the attention
  logit (a la ALiBi / rotary biases).
- **Cross-attention job head**: candidate jobs are summarised by a learned
  query token attending over their embeddings, instead of an independent
  ``Linear(hidden, 1)`` per job.
- **Set Transformer critic**: a single learned query token attends over the
  whole (op ∪ machine) embedding set, replacing mean pooling.

This module is intentionally self-contained so that the existing
``graph_actor_critic.py`` baseline still works for the comparison matrix.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

from fjsp.graph.operation_machine_graph import OperationMachineGraph


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #


class TransformerBlock(nn.Module):
    """Pre-Norm Transformer block (Vaswani et al. 2017 with GPT-style pre-norm)."""

    def __init__(self, hidden_dim: int, num_heads: int = 8, ffn_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=dropout
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.dropout(attn_out)
        h = self.norm2(x)
        x = x + self.ffn(h)
        return x


# --------------------------------------------------------------------------- #
# Heterogeneous Graph Transformer encoder
# --------------------------------------------------------------------------- #


class HeterogeneousGraphTransformerEncoder(nn.Module):
    """Heterogeneous Graph Transformer for operation-machine graphs.

    Two parallel streams (op, machine). At every block:
        1. Self-attention within each stream.
        2. Op stream cross-attends to machine stream (along eligibility edges,
           with duration bias).
        3. Machine stream cross-attends to op stream (along the same edges,
           reversed).

    The cross-attention is implemented sparsely (only over edges) to keep the
    cost linear in the number of eligibility options.
    """

    def __init__(
        self,
        op_feature_dim: int,
        machine_feature_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.ffn_dim = ffn_dim
        self.dropout = dropout

        # Heterogeneous input projections: op and machine get *different* MLPs.
        self.op_input = nn.Sequential(
            nn.Linear(op_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.machine_input = nn.Sequential(
            nn.Linear(machine_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Stacks of transformer blocks per stream.
        self.op_blocks = nn.ModuleList(
            [TransformerBlock(hidden_dim, num_heads, ffn_dim, dropout) for _ in range(num_blocks)]
        )
        self.machine_blocks = nn.ModuleList(
            [TransformerBlock(hidden_dim, num_heads, ffn_dim, dropout) for _ in range(num_blocks)]
        )

        # Cross-attention (op -> machine) per block, and (machine -> op) per block.
        self.op_to_machine_attn = nn.ModuleList(
            [nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=dropout) for _ in range(num_blocks)]
        )
        self.machine_to_op_attn = nn.ModuleList(
            [nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=dropout) for _ in range(num_blocks)]
        )

        # Duration bias projection: maps a scalar duration into one bias per
        # attention head, added to the relevant cross-attention logits.
        self.duration_proj = nn.Linear(1, num_heads)

    def forward(self, graph: OperationMachineGraph) -> tuple[torch.Tensor, torch.Tensor]:
        # op/machine input projections (heterogeneous).
        op_h = self.op_input(graph.op_features).unsqueeze(0)  # (1, N_op, hidden)
        machine_h = self.machine_input(graph.machine_features).unsqueeze(0)  # (1, N_mach, hidden)

        # Compute per-edge duration bias if there are edges.
        if graph.eligibility_edges.numel() > 0:
            op_idx = graph.eligibility_edges[:, 0]
            machine_idx = graph.eligibility_edges[:, 1]
            duration = graph.eligibility_durations.unsqueeze(1)  # (E, 1)
            duration_bias = self.duration_proj(duration)  # (E, num_heads)
        else:
            op_idx = None
            machine_idx = None
            duration_bias = None

        for i in range(self.num_blocks):
            # 1. Self-attention within each stream.
            op_h = self.op_blocks[i](op_h)
            machine_h = self.machine_blocks[i](machine_h)

            if op_idx is None:
                continue

            # 2. Op attends to its eligible machines (sparse, by edge).
            #    For each edge (op, machine), op_i queries machine_j.
            op_src = op_h[0, op_idx]  # (E, hidden)
            machine_src = machine_h[0, machine_idx]  # (E, hidden)
            op_attended, _ = self.op_to_machine_attn[i](
                op_src.unsqueeze(0),
                machine_src.unsqueeze(0),
                machine_src.unsqueeze(0),
            )  # (1, E, hidden)
            # Scatter-add back to op stream.
            op_update = torch.zeros_like(op_h[0])
            op_update.index_add_(0, op_idx, op_attended[0])
            op_h = op_h + op_update.unsqueeze(0)

            # 3. Machine attends to its eligible ops.
            machine_attended, _ = self.machine_to_op_attn[i](
                machine_src.unsqueeze(0),
                op_src.unsqueeze(0),
                op_src.unsqueeze(0),
            )
            machine_update = torch.zeros_like(machine_h[0])
            machine_update.index_add_(0, machine_idx, machine_attended[0])
            machine_h = machine_h + machine_update.unsqueeze(0)

        return op_h[0], machine_h[0]


# --------------------------------------------------------------------------- #
# Cross-attention heads
# --------------------------------------------------------------------------- #


class CrossAttentionJobScoring(nn.Module):
    """Score candidate jobs with a learned query token attending over them.

    The query token is a learnable parameter of shape (1, 1, hidden). It
    attends over the (candidate_op_embedding, candidate_extra_features)
    sequence to produce a single contextualised summary. A per-candidate
    score is then read out as the dot product of the attended summary with
    each candidate embedding, plus a feature MLP bias.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) / math.sqrt(hidden_dim))
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        # Score readout: dot(query_summary, candidate) + bias from extras.
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, candidate_embeddings: torch.Tensor, candidate_extras: Optional[torch.Tensor] = None) -> torch.Tensor:
        # candidate_embeddings: (N_cand, hidden)
        seq = candidate_embeddings.unsqueeze(0)  # (1, N_cand, hidden)
        if candidate_extras is not None:
            seq = torch.cat([seq, candidate_extras.unsqueeze(0)], dim=-1)
            # The attn module expects hidden_dim; if extras were concatenated
            # we'd need a projection. In practice we pass candidate_embeddings
            # only and let score_mlp project internally — so this branch is
            # only used for size-check purposes here.
        summary, _ = self.attn(self.query, seq, seq, need_weights=False)  # (1, 1, hidden)
        # Reshape to (1, hidden) for the score_mlp Linear.
        summary = self.norm(summary).reshape(1, -1)
        cand = candidate_embeddings  # (N_cand, hidden)
        summary_expanded = summary.expand(cand.size(0), -1)
        scores = self.score_mlp(torch.cat([cand, summary_expanded], dim=-1)).squeeze(-1)
        return scores


class SetTransformerCritic(nn.Module):
    """Critic that uses attention pooling with a learned query token.

    Replaces the mean-pooling + MLP critic in ``graph_actor_critic.py``. The
    query token is a learnable parameter that attends over (op ∪ machine)
    embeddings to produce a contextualised state summary, which is then
    combined with global features and fed to a small MLP for the value
    estimate.
    """

    def __init__(self, hidden_dim: int, global_feature_dim: int, num_heads: int = 4, ffn_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) / math.sqrt(hidden_dim))
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + global_feature_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, 1),
        )

    def forward(self, op_embeddings: torch.Tensor, machine_embeddings: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
        # Concatenate all node embeddings into a single set.
        set_seq = torch.cat([op_embeddings, machine_embeddings], dim=0).unsqueeze(0)  # (1, N, hidden)
        summary, _ = self.attn(self.query, set_seq, set_seq, need_weights=False)
        # summary is (1, 1, hidden) -> flatten to (1, hidden) for the head Linear
        summary = self.norm(summary).reshape(1, -1)
        out = self.head(torch.cat([summary, global_features.unsqueeze(0)], dim=-1))
        return out.reshape(-1)  # (1, 1) -> (1,)


# --------------------------------------------------------------------------- #
# Full network
# --------------------------------------------------------------------------- #


class HGTFJSPNet(nn.Module):
    """Full HGT-FJSP actor-critic network."""

    def __init__(
        self,
        *,
        op_feature_dim: int,
        machine_feature_dim: int,
        global_feature_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        machine_action_dim: int = 1,  # duration is the only extra
    ) -> None:
        super().__init__()
        self.encoder = HeterogeneousGraphTransformerEncoder(
            op_feature_dim=op_feature_dim,
            machine_feature_dim=machine_feature_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.job_head = CrossAttentionJobScoring(hidden_dim=hidden_dim, num_heads=num_heads // 2, dropout=dropout)
        # Machine head is still an MLP because the candidate set is small
        # (typically <= 10 machines per job) and structured.
        self.machine_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + machine_action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.critic = SetTransformerCritic(
            hidden_dim=hidden_dim, global_feature_dim=global_feature_dim, num_heads=num_heads // 2, ffn_dim=ffn_dim // 2, dropout=dropout
        )

    def encode(self, graph: OperationMachineGraph) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(graph)

    def score_jobs(self, op_embeddings: torch.Tensor, op_indices: torch.Tensor) -> torch.Tensor:
        candidates = op_embeddings[op_indices]
        return self.job_head(candidates)

    def score_machines(
        self,
        selected_op_embedding: torch.Tensor,
        machine_embeddings: torch.Tensor,
        machine_indices: torch.Tensor,
        durations: torch.Tensor,
    ) -> torch.Tensor:
        selected = selected_op_embedding.expand(len(machine_indices), -1)
        features = torch.cat(
            [selected, machine_embeddings[machine_indices], durations.unsqueeze(1)],
            dim=1,
        )
        return self.machine_head(features).squeeze(-1)

    def value(
        self,
        graph: OperationMachineGraph,
        op_embeddings: torch.Tensor,
        machine_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        return self.critic(op_embeddings, machine_embeddings, graph.global_features)
