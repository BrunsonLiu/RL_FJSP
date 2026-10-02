"""FJSP-L2S: Neural network for learning to improve FJSP solutions.

Architecture
------------
1. **Heterogeneous Graph Encoder**: Encodes the complete solution graph
   with operation nodes, machine nodes, and three edge types (precedence,
   machine sequence, eligibility). Uses multi-round message passing with
   GRU updates and critical-path attention bias.

2. **Move Scoring Head**: Scores each candidate improvement move by
   combining the operation embedding, machine embedding (for reassign),
   and move-type embedding. Uses cross-attention to let the model reason
   about interactions between candidate moves.

3. **Set Transformer Critic**: Attention-based value network that
   summarizes the solution state into a scalar value estimate.

Key innovation
--------------
- **Critical-path attention bias**: Operations on the critical path get
  an additive bias in the self-attention layers, directing the model's
  focus to bottleneck operations.
- **Move-type embeddings**: The three move types (reassign, swap_prev,
  swap_next) get learned embeddings, allowing the model to develop
  different strategies for each move type.
- **Cross-instance generalization**: The architecture is size-agnostic —
  it operates on variable-size graphs without padding.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

from fjsp.graph.solution_graph import (
    IMP_GLOBAL_FEATURE_DIM,
    IMP_MACHINE_FEATURE_DIM,
    IMP_OP_FEATURE_DIM,
    SolutionGraph,
)


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #


class PreNormTransformerBlock(nn.Module):
    """Pre-Norm Transformer block with learnable critical-path bias."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        ffn_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
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
            nn.Dropout(dropout),
        )
        # Learnable critical-path attention bias (one per head)
        self.cp_bias = nn.Parameter(torch.zeros(num_heads))

    def forward(
        self,
        x: torch.Tensor,
        critical_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : (1, N, hidden_dim)
        critical_mask : (N,) bool, optional
            If provided, adds a learnable bias to attention logits
            for critical-path nodes. The bias is per-head and learned
            during training.
        """
        h = self.norm1(x)
        if critical_mask is not None:
            N = h.size(1)
            # Build per-head bias: (num_heads, N, N)
            # bias[h, i, j] = cp_bias[h] if j is on critical path, else 0
            cp_float = critical_mask.float()  # (N,)
            # Expand to (num_heads, 1, N) then broadcast to (num_heads, N, N)
            bias = self.cp_bias.view(-1, 1, 1) * cp_float.view(1, 1, N)  # (num_heads, 1, N)
            bias = bias.expand(-1, N, -1)  # (num_heads, N, N)
            attn_out, _ = self.attn(h, h, h, attn_mask=bias, need_weights=False)
        else:
            attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        h = self.norm2(x)
        x = x + self.ffn(h)
        return x


class MessagePassingLayer(nn.Module):
    """One round of heterogeneous message passing on the solution graph.

    Messages flow along:
    1. Precedence edges (op -> op within a job)
    2. Machine sequence edges (op -> op on same machine)
    3. Eligibility edges (op <-> machine, with duration bias)
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.prec_msg = nn.Linear(hidden_dim, hidden_dim)
        self.seq_msg = nn.Linear(hidden_dim, hidden_dim)
        self.op_to_machine = nn.Linear(hidden_dim + 1, hidden_dim)
        self.machine_to_op = nn.Linear(hidden_dim + 1, hidden_dim)
        self.op_gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.machine_gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.op_norm = nn.LayerNorm(hidden_dim)
        self.machine_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        op_h: torch.Tensor,
        machine_h: torch.Tensor,
        graph: SolutionGraph,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One message-passing round.

        Parameters
        ----------
        op_h : (N_ops, hidden_dim)
        machine_h : (N_machines, hidden_dim)
        graph : SolutionGraph

        Returns
        -------
        op_h, machine_h : updated embeddings
        """
        op_messages = torch.zeros_like(op_h)

        # Precedence messages
        if graph.precedence_edges.numel() > 0:
            src = graph.precedence_edges[:, 0]
            dst = graph.precedence_edges[:, 1]
            op_messages.index_add_(0, dst, self.prec_msg(op_h[src]))

        # Machine sequence messages
        if graph.machine_seq_edges.numel() > 0:
            src = graph.machine_seq_edges[:, 0]
            dst = graph.machine_seq_edges[:, 1]
            op_messages.index_add_(0, dst, self.seq_msg(op_h[src]))

        # Eligibility cross-attention (op <-> machine)
        machine_messages = torch.zeros_like(machine_h)
        if graph.eligibility_edges.numel() > 0:
            op_idx = graph.eligibility_edges[:, 0]
            machine_idx = graph.eligibility_edges[:, 1]
            duration = graph.eligibility_durations.unsqueeze(1)

            m_to_o = self.machine_to_op(torch.cat([machine_h[machine_idx], duration], dim=1))
            o_to_m = self.op_to_machine(torch.cat([op_h[op_idx], duration], dim=1))

            op_messages.index_add_(0, op_idx, m_to_o)
            machine_messages.index_add_(0, machine_idx, o_to_m)

        # GRU update with residual + norm
        op_h_new = self.op_norm(self.op_gru(op_messages, op_h))
        machine_h_new = self.machine_norm(self.machine_gru(machine_messages, machine_h))

        return op_h_new, machine_h_new


# --------------------------------------------------------------------------- #
# Encoder
# --------------------------------------------------------------------------- #


class FJSPSolutionEncoder(nn.Module):
    """Encode a complete FJSP solution graph.

    Combines message passing (for structural reasoning along edges) with
    self-attention (for global reasoning across all nodes). Critical-path
    operations receive an attention bias.

    Parameters
    ----------
    op_feature_dim : int
    machine_feature_dim : int
    hidden_dim : int
    num_mp_rounds : int
        Number of message-passing rounds.
    num_transformer_blocks : int
        Number of self-attention blocks applied after message passing.
    num_heads : int
    ffn_dim : int
    dropout : float
    """

    def __init__(
        self,
        op_feature_dim: int = IMP_OP_FEATURE_DIM,
        machine_feature_dim: int = IMP_MACHINE_FEATURE_DIM,
        hidden_dim: int = 128,
        num_mp_rounds: int = 3,
        num_transformer_blocks: int = 2,
        num_heads: int = 8,
        ffn_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input projections
        self.op_input = nn.Sequential(
            nn.Linear(op_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.machine_input = nn.Sequential(
            nn.Linear(machine_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Message passing layers
        self.mp_layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim, dropout=dropout)
            for _ in range(num_mp_rounds)
        ])

        # Transformer blocks for global reasoning (applied to op nodes)
        self.transformer_blocks = nn.ModuleList([
            PreNormTransformerBlock(hidden_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_transformer_blocks)
        ])

    def forward(self, graph: SolutionGraph) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode the solution graph.

        Returns
        -------
        op_h : (N_ops, hidden_dim)
        machine_h : (N_machines, hidden_dim)
        critical_mask : (N_ops,) bool
        """
        op_h = self.op_input(graph.op_features)
        machine_h = self.machine_input(graph.machine_features)

        # Message passing
        for mp_layer in self.mp_layers:
            op_h, machine_h = mp_layer(op_h, machine_h, graph)

        # Self-attention on operation nodes with critical-path bias
        # Extract critical path mask from features (index 6 = is_on_critical_path)
        critical_mask = graph.op_features[:, 6] > 0.5  # (N_ops,)

        op_seq = op_h.unsqueeze(0)  # (1, N_ops, hidden)
        for block in self.transformer_blocks:
            op_seq = block(op_seq, critical_mask=critical_mask)
        op_h = op_seq[0]  # (N_ops, hidden)

        return op_h, machine_h, critical_mask


class DualPerspectiveFusion(nn.Module):
    """Fuse local and global perspectives via bidirectional cross-attention.

    The local perspective (MPNN) captures structural information along
    edges; the global perspective (Transformer) captures long-range
    dependencies. This module fuses them via:
    1. Local-to-global cross-attention (local queries attend to global keys/values)
    2. Global-to-local cross-attention (global queries attend to local keys/values)
    3. Gated combination: learnable gate decides how much of each perspective
       to use for each node.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Local attends to global
        self.local_to_global = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=dropout
        )
        self.local_norm = nn.LayerNorm(hidden_dim)

        # Global attends to local
        self.global_to_local = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=dropout
        )
        self.global_norm = nn.LayerNorm(hidden_dim)

        # Gated fusion: per-node gate decides local vs global weight
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        local_h: torch.Tensor,
        global_h: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse local and global representations.

        Parameters
        ----------
        local_h, global_h : (1, N, hidden_dim)

        Returns
        -------
        fused_h : (1, N, hidden_dim)
        """
        # Local attends to global (local queries, global keys/values)
        l2g_out, _ = self.local_to_global(local_h, global_h, global_h, need_weights=False)
        local_ctx = self.local_norm(local_h + l2g_out)

        # Global attends to local (global queries, local keys/values)
        g2l_out, _ = self.global_to_local(global_h, local_h, local_h, need_weights=False)
        global_ctx = self.global_norm(global_h + g2l_out)

        # Gated combination
        gate = torch.sigmoid(self.gate(torch.cat([local_ctx, global_ctx], dim=-1)))
        fused = gate * local_ctx + (1.0 - gate) * global_ctx
        return self.output_norm(fused)


class DualPerspectiveEncoder(nn.Module):
    """Dual-perspective encoder for BARI.

    Runs two parallel branches:
    - **Local branch**: Message passing along precedence, machine-sequence,
      and eligibility edges. Captures structural constraints and local
      neighborhood information.
    - **Global branch**: Self-attention over all operation nodes with
      critical-path attention bias. Captures long-range dependencies
      and global bottleneck structure.

    The two perspectives are fused via bidirectional cross-attention
    with a learned gating mechanism. This is the key architectural
    innovation of BARI: prior work uses either MPNN-only (losing global
    context) or Transformer-only (losing structural constraints); BARI
    combines both and lets the model learn when to rely on each.

    Parameters
    ----------
    op_feature_dim, machine_feature_dim : int
    hidden_dim : int
    num_mp_rounds : int
        Number of message-passing rounds in the local branch.
    num_transformer_blocks : int
        Number of self-attention blocks in the global branch.
    num_heads, ffn_dim, dropout : int, int, float
    """

    def __init__(
        self,
        op_feature_dim: int = IMP_OP_FEATURE_DIM,
        machine_feature_dim: int = IMP_MACHINE_FEATURE_DIM,
        hidden_dim: int = 128,
        num_mp_rounds: int = 3,
        num_transformer_blocks: int = 2,
        num_heads: int = 8,
        ffn_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # Shared input projections
        self.op_input = nn.Sequential(
            nn.Linear(op_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.machine_input = nn.Sequential(
            nn.Linear(machine_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Local branch: message passing
        self.mp_layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim, dropout=dropout)
            for _ in range(num_mp_rounds)
        ])

        # Global branch: transformer blocks (operate on raw input embeddings)
        self.transformer_blocks = nn.ModuleList([
            PreNormTransformerBlock(hidden_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_transformer_blocks)
        ])

        # Fusion: bidirectional cross-attention with gating
        self.fusion = DualPerspectiveFusion(
            hidden_dim=hidden_dim,
            num_heads=max(num_heads // 2, 1),
            dropout=dropout,
        )

    def forward(self, graph: SolutionGraph) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode the solution graph with dual perspectives.

        Returns
        -------
        op_h : (N_ops, hidden_dim)  -- fused operation embeddings
        machine_h : (N_machines, hidden_dim)  -- from local branch
        critical_mask : (N_ops,) bool
        """
        op_emb = self.op_input(graph.op_features)
        machine_h = self.machine_input(graph.machine_features)

        # Critical path mask (feature index 6)
        critical_mask = graph.op_features[:, 6] > 0.5

        # --- Local branch: message passing ---
        local_op_h = op_emb
        local_machine_h = machine_h
        for mp_layer in self.mp_layers:
            local_op_h, local_machine_h = mp_layer(local_op_h, local_machine_h, graph)

        # --- Global branch: self-attention on raw embeddings ---
        global_op_h = op_emb.unsqueeze(0)  # (1, N_ops, hidden)
        for block in self.transformer_blocks:
            global_op_h = block(global_op_h, critical_mask=critical_mask)
        global_op_h = global_op_h[0]  # (N_ops, hidden)

        # --- Fusion: bidirectional cross-attention ---
        local_seq = local_op_h.unsqueeze(0)   # (1, N, hidden)
        global_seq = global_op_h.unsqueeze(0)  # (1, N, hidden)
        fused_seq = self.fusion(local_seq, global_seq)
        op_h = fused_seq[0]  # (N_ops, hidden)

        return op_h, local_machine_h, critical_mask


# --------------------------------------------------------------------------- #
# Move scoring head
# --------------------------------------------------------------------------- #


class MoveScoringHead(nn.Module):
    """Score candidate improvement moves via cross-attention with solution state.

    For each candidate move, we compute an embedding by concatenating:
    - The embedding of the operation being moved
    - The embedding of the target machine (for reassign) or current machine (for swap)
    - A learned move-type embedding

    The move embeddings then attend to the solution state (op_h + machine_h)
    via cross-attention, allowing each move to reason about its context
    within the full solution. Finally, a contextualized summary attends
    back over all moves to produce per-move scores.
    """

    MOVE_TYPES = 3  # coupled_reassign, swap_prev, swap_next

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.move_type_embed = nn.Embedding(self.MOVE_TYPES, hidden_dim)

        # Interaction MLP: concatenate op_emb, machine_emb, move_type_emb -> hidden
        self.interaction_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # Cross-attention: moves (queries) attend to solution state (keys/values)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=dropout
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)

        # Self-attention among moves for mutual reasoning
        self.move_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=dropout
        )
        self.move_norm = nn.LayerNorm(hidden_dim)

        # Scoring MLP
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        move_embeddings: torch.Tensor,
        solution_state: torch.Tensor,
    ) -> torch.Tensor:
        """Score all candidate moves.

        Parameters
        ----------
        move_embeddings : (N_moves, hidden_dim)
            Pre-computed embedding for each move.
        solution_state : (N_state, hidden_dim)
            Concatenation of op_h and machine_h, the solution context.

        Returns
        -------
        scores : (N_moves,)
        """
        if move_embeddings.size(0) == 0:
            return torch.tensor([], device=move_embeddings.device)

        # Cross-attention: moves attend to solution state
        moves_seq = move_embeddings.unsqueeze(0)  # (1, N_moves, hidden)
        state_seq = solution_state.unsqueeze(0)    # (1, N_state, hidden)

        cross_out, _ = self.cross_attn(moves_seq, state_seq, state_seq, need_weights=False)
        contextualized = self.cross_norm(moves_seq + cross_out)  # (1, N_moves, hidden)

        # Self-attention among moves
        move_out, _ = self.move_attn(contextualized, contextualized, contextualized, need_weights=False)
        move_out = self.move_norm(contextualized + move_out)  # (1, N_moves, hidden)

        # Score each move: combine original + contextualized
        combined = torch.cat([move_embeddings, move_out[0]], dim=-1)  # (N_moves, 2*hidden)
        scores = self.score_mlp(combined).squeeze(-1)
        return scores


# --------------------------------------------------------------------------- #
# Critic
# --------------------------------------------------------------------------- #


class ImprovementCritic(nn.Module):
    """Value network using Set Transformer attention pooling."""

    def __init__(
        self,
        hidden_dim: int,
        global_feature_dim: int = IMP_GLOBAL_FEATURE_DIM,
        num_heads: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) / math.sqrt(hidden_dim))
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + global_feature_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, ffn_dim // 2),
            nn.GELU(),
            nn.Linear(ffn_dim // 2, 1),
        )

    def forward(
        self,
        op_h: torch.Tensor,
        machine_h: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate the state value.

        Returns
        -------
        value : (1,)
        """
        set_seq = torch.cat([op_h, machine_h], dim=0).unsqueeze(0)
        summary, _ = self.attn(self.query, set_seq, set_seq, need_weights=False)
        summary = self.norm(summary).reshape(1, -1)
        out = self.head(torch.cat([summary, global_features.unsqueeze(0)], dim=-1))
        return out.reshape(-1)


# --------------------------------------------------------------------------- #
# Full network
# --------------------------------------------------------------------------- #


class FJSPImproveNet(nn.Module):
    """Full actor-critic network for FJSP learning-to-improve.

    Parameters
    ----------
    op_feature_dim : int
    machine_feature_dim : int
    global_feature_dim : int
    hidden_dim : int
    num_mp_rounds : int
    num_transformer_blocks : int
    num_heads : int
    ffn_dim : int
    dropout : float
    use_dual_perspective : bool
        If True, use DualPerspectiveEncoder (BARI main model).
        If False, use FJSPSolutionEncoder (single-perspective, for ablation).
    """

    def __init__(
        self,
        *,
        op_feature_dim: int = IMP_OP_FEATURE_DIM,
        machine_feature_dim: int = IMP_MACHINE_FEATURE_DIM,
        global_feature_dim: int = IMP_GLOBAL_FEATURE_DIM,
        hidden_dim: int = 128,
        num_mp_rounds: int = 3,
        num_transformer_blocks: int = 2,
        num_heads: int = 8,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        use_dual_perspective: bool = True,
    ) -> None:
        super().__init__()
        self.use_dual_perspective = use_dual_perspective

        if use_dual_perspective:
            self.encoder = DualPerspectiveEncoder(
                op_feature_dim=op_feature_dim,
                machine_feature_dim=machine_feature_dim,
                hidden_dim=hidden_dim,
                num_mp_rounds=num_mp_rounds,
                num_transformer_blocks=num_transformer_blocks,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
        else:
            self.encoder = FJSPSolutionEncoder(
                op_feature_dim=op_feature_dim,
                machine_feature_dim=machine_feature_dim,
                hidden_dim=hidden_dim,
                num_mp_rounds=num_mp_rounds,
                num_transformer_blocks=num_transformer_blocks,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
        self.move_head = MoveScoringHead(hidden_dim=hidden_dim, num_heads=num_heads // 2, dropout=dropout)
        self.critic = ImprovementCritic(
            hidden_dim=hidden_dim,
            global_feature_dim=global_feature_dim,
            num_heads=num_heads // 2,
            ffn_dim=ffn_dim // 2,
            dropout=dropout,
        )

    def encode(self, graph: SolutionGraph) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode the solution graph.

        Returns
        -------
        op_h, machine_h, critical_mask
        """
        return self.encoder(graph)

    def score_moves(
        self,
        op_h: torch.Tensor,
        machine_h: torch.Tensor,
        graph: SolutionGraph,
        moves,  # list of ImprovementMove
        assignments: list[tuple[int, int, int]],
    ) -> torch.Tensor:
        """Score all candidate improvement moves.

        Parameters
        ----------
        op_h : (N_ops, hidden_dim)
        machine_h : (N_machines, hidden_dim)
        graph : SolutionGraph
        moves : list of ImprovementMove
        assignments : current (job, op, machine) assignments

        Returns
        -------
        scores : (N_moves,)
        """
        if not moves:
            return torch.tensor([], device=op_h.device)

        op_ref_to_idx = {ref: i for i, ref in enumerate(graph.operation_refs)}

        embeddings = []
        for move in moves:
            # Get (job, op) from assignments
            job, op, _ = assignments[move.op_index]
            op_key = (job, op)

            # Operation embedding
            if op_key in op_ref_to_idx:
                op_emb = op_h[op_ref_to_idx[op_key]]
            else:
                op_emb = torch.zeros(op_h.size(1), device=op_h.device)

            # Move type embedding
            mt = {"coupled_reassign": 0, "swap_prev": 1, "swap_next": 2}[move.move_type]
            mt_emb = self.move_head.move_type_embed(torch.tensor(mt, device=op_h.device))

            # Target machine embedding
            if move.is_reassign and move.target_machine is not None:
                target_emb = machine_h[move.target_machine]
            else:
                # For swap moves, use the current machine embedding
                current_machine = assignments[move.op_index][2]
                target_emb = machine_h[current_machine] if current_machine < machine_h.size(0) else torch.zeros(op_h.size(1), device=op_h.device)

            # Interaction MLP instead of simple addition
            combined = torch.cat([op_emb, target_emb, mt_emb], dim=-1)
            embeddings.append(self.move_head.interaction_mlp(combined))

        move_embeddings = torch.stack(embeddings)

        # Build solution state for cross-attention
        solution_state = torch.cat([op_h, machine_h], dim=0)  # (N_ops + N_machines, hidden)

        return self.move_head(move_embeddings, solution_state)

    def value(
        self,
        graph: SolutionGraph,
        op_h: torch.Tensor,
        machine_h: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate the state value."""
        return self.critic(op_h, machine_h, graph.global_features)
