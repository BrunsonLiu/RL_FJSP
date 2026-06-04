"""HGT-FJSP agent.

Actor-critic agent that uses the Heterogeneous Graph Transformer encoder
from ``rl.models.hgt_fjsp`` plus the cross-attention job head and the
Set-Transformer critic. The API mirrors
``GraphTwoStageActorCriticAgent`` so that the existing training loop in
``rl.agents.graph_actor_critic.train_graph_actor_critic`` and
``rl.agents.imitation.train_imitation`` can drive it (with the new feature
dimensions).
"""
from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Optional

import torch
from torch import nn

from fjsp.env import DispatchAction, FJSPDispatchEnv
from fjsp.graph.operation_machine_graph import (
    GLOBAL_FEATURE_DIM,
    MACHINE_FEATURE_DIM,
    OP_FEATURE_DIM,
    OperationMachineGraph,
    build_operation_machine_graph,
)
from fjsp.scheduler.validator import ScheduledOperation
from fjsp.utils.scaling import instance_time_scale
from rl.models.hgt_fjsp import HGTFJSPNet


@dataclass
class RolloutResult:
    log_probs: list[torch.Tensor] = field(default_factory=list)
    values: list[torch.Tensor] = field(default_factory=list)
    entropies: list[torch.Tensor] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    makespan: int = 0
    is_valid: bool = True


class HGTActorCriticAgent:
    """Heterogeneous Graph Transformer actor-critic agent for FJSP."""

    def __init__(
        self,
        net: HGTFJSPNet,
        *,
        device: str = "cpu",
        instance_embed: Optional[nn.Embedding] = None,
    ) -> None:
        self.net = net.to(device)
        self.device = device
        # Optional instance embedding for cross-instance training. The
        # ``instance_id`` is set externally before each rollout; if it is
        # None the embedding is skipped.
        self.instance_embed = instance_embed.to(device) if instance_embed is not None else None
        self.instance_id: Optional[int] = None

    @classmethod
    def create(
        cls,
        *,
        hidden_dim: int = 128,
        num_blocks: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        op_feature_dim: int = OP_FEATURE_DIM,
        machine_feature_dim: int = MACHINE_FEATURE_DIM,
        global_feature_dim: int = GLOBAL_FEATURE_DIM,
        num_known_instances: Optional[int] = None,
        device: str = "cpu",
    ) -> "HGTActorCriticAgent":
        net = HGTFJSPNet(
            op_feature_dim=op_feature_dim,
            machine_feature_dim=machine_feature_dim,
            global_feature_dim=global_feature_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        ie = None
        if num_known_instances is not None and num_known_instances > 0:
            ie = nn.Embedding(num_known_instances, hidden_dim)
        return cls(net, device=device, instance_embed=ie)

    def _build_graph(self, env: FJSPDispatchEnv) -> OperationMachineGraph:
        graph = build_operation_machine_graph(env, device=self.device)
        if self.instance_embed is not None and self.instance_id is not None:
            # Add the instance embedding to every op feature. We do this by
            # concatenation through a small linear projection handled inside
            # the net? For simplicity here, we inject the instance vector
            # into the op_features last column as a residual additive bias
            # is harder; instead, we stash the vector and add it after
            # encoding.
            self._instance_vec = self.instance_embed(
                torch.tensor(self.instance_id, device=self.device)
            )
        else:
            self._instance_vec = None
        return graph

    def _apply_instance_bias(self, op_embeddings: torch.Tensor) -> torch.Tensor:
        if self._instance_vec is None:
            return op_embeddings
        return op_embeddings + self._instance_vec.unsqueeze(0)

    def encode(self, env: FJSPDispatchEnv) -> tuple[OperationMachineGraph, torch.Tensor, torch.Tensor]:
        graph = self._build_graph(env)
        op_emb, mach_emb = self.net.encode(graph)
        op_emb = self._apply_instance_bias(op_emb)
        return graph, op_emb, mach_emb

    def select_action(
        self,
        env: FJSPDispatchEnv,
        *,
        greedy: bool = False,
        rng: Optional[Random] = None,
    ) -> tuple[DispatchAction, torch.Tensor, torch.Tensor, torch.Tensor]:
        if rng is None:
            rng = Random()
        graph, op_emb, mach_emb = self.encode(env)
        next_indices = list(graph.next_op_indices)
        if not next_indices:
            raise RuntimeError("No schedulable operations remain.")
        cand = torch.tensor(next_indices, dtype=torch.long, device=self.device)
        job_logits = self.net.score_jobs(op_emb, cand)
        if greedy:
            job_pos = int(torch.argmax(job_logits).item())
        else:
            masked = job_logits.masked_fill(torch.isinf(job_logits), float("-inf"))
            probs = torch.softmax(masked, dim=-1)
            job_pos = int(torch.multinomial(probs, num_samples=1, replacement=False).item())

        target_op_idx = next_indices[job_pos]
        target_op_ref = graph.operation_refs[target_op_idx]
        machine_actions = [a for a in env.available_actions() if a.job == target_op_ref.job]
        if not machine_actions:
            raise RuntimeError("No machine actions for chosen job.")
        machine_indices = torch.tensor(
            [a.machine for a in machine_actions], dtype=torch.long, device=self.device
        )
        durations = torch.tensor(
            [
                next(
                    option.duration
                    for option in env.instance.jobs[a.job].operations[env.job_next_op[a.job]].options
                    if option.machine == a.machine
                )
                / float(instance_time_scale(env))
                for a in machine_actions
            ],
            dtype=torch.float32,
            device=self.device,
        )
        machine_logits = self.net.score_machines(
            op_emb[target_op_idx], mach_emb, machine_indices, durations
        )
        if greedy:
            machine_pos = int(torch.argmax(machine_logits).item())
        else:
            masked_m = machine_logits.masked_fill(torch.isinf(machine_logits), float("-inf"))
            probs_m = torch.softmax(masked_m, dim=-1)
            machine_pos = int(torch.multinomial(probs_m, num_samples=1, replacement=False).item())

        action = machine_actions[machine_pos]
        # log prob + entropy (job)
        log_prob_j = torch.log_softmax(job_logits, dim=-1)[job_pos]
        entropy_j = -(torch.softmax(job_logits, dim=-1) * torch.log_softmax(job_logits, dim=-1)).sum()
        log_prob_m = torch.log_softmax(machine_logits, dim=-1)[machine_pos]
        entropy_m = -(torch.softmax(machine_logits, dim=-1) * torch.log_softmax(machine_logits, dim=-1)).sum()
        log_prob = log_prob_j + log_prob_m
        entropy = entropy_j + entropy_m
        value = self.net.value(graph, op_emb, mach_emb)
        return action, log_prob, value, entropy

    def evaluate_action(
        self,
        env: FJSPDispatchEnv,
        action: DispatchAction,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-evaluate a previously-chosen (job, machine) action. Returns
        (log_prob, value, entropy)."""
        graph, op_emb, mach_emb = self.encode(env)
        next_indices = list(graph.next_op_indices)
        cand = torch.tensor(next_indices, dtype=torch.long, device=self.device)
        job_logits = self.net.score_jobs(op_emb, cand)
        try:
            job_pos = next(
                pos for pos, op_idx in enumerate(next_indices) if graph.operation_refs[op_idx].job == action.job
            )
        except StopIteration:
            raise ValueError("Action job not schedulable from this state.")
        target_op_idx = next_indices[job_pos]
        target_op_ref = graph.operation_refs[target_op_idx]
        machine_actions = [a for a in env.available_actions() if a.job == target_op_ref.job]
        machine_indices = torch.tensor(
            [a.machine for a in machine_actions], dtype=torch.long, device=self.device
        )
        durations = torch.tensor(
            [
                next(
                    option.duration
                    for option in env.instance.jobs[a.job].operations[env.job_next_op[a.job]].options
                    if option.machine == a.machine
                )
                / float(instance_time_scale(env))
                for a in machine_actions
            ],
            dtype=torch.float32,
            device=self.device,
        )
        machine_logits = self.net.score_machines(
            op_emb[target_op_idx], mach_emb, machine_indices, durations
        )
        try:
            machine_pos = next(
                i for i, a in enumerate(machine_actions) if a.machine == action.machine
            )
        except StopIteration:
            raise ValueError("Action machine not available for the chosen job.")

        log_prob = (
            torch.log_softmax(job_logits, dim=-1)[job_pos]
            + torch.log_softmax(machine_logits, dim=-1)[machine_pos]
        )
        entropy = (
            -(torch.softmax(job_logits, dim=-1) * torch.log_softmax(job_logits, dim=-1)).sum()
            + -(torch.softmax(machine_logits, dim=-1) * torch.log_softmax(machine_logits, dim=-1)).sum()
        )
        value = self.net.value(graph, op_emb, mach_emb)
        return log_prob, value, entropy

    def rollout(self, env: FJSPDispatchEnv, *, greedy: bool = False, seed: Optional[int] = None) -> RolloutResult:
        rng = Random(seed) if seed is not None else None
        env.reset()
        result = RolloutResult()
        prev_makespan = 0
        while not env.done:
            try:
                action, log_prob, value, entropy = self.select_action(env, greedy=greedy, rng=rng)
            except (RuntimeError, ValueError):
                result.is_valid = False
                return result
            result.log_probs.append(log_prob)
            result.values.append(value)
            result.entropies.append(entropy)
            env.step(action)
            # Dense per-step reward: -delta makespan - 0.01 * machine_idle_time
            new_makespan = env.makespan
            delta = new_makespan - prev_makespan
            prev_makespan = new_makespan
            # Reward in [-1, 0] range.
            r = -float(delta) / float(instance_time_scale(env))
            result.rewards.append(r)
        result.makespan = env.makespan
        # Final terminal reward component (small bonus for completing the
        # schedule).
        result.rewards[-1] += -float(result.makespan - prev_makespan) / float(instance_time_scale(env))
        return result

    def save(self, path: str | Path) -> None:
        cfg = self.net.encoder
        # ``ffn_dim`` may be missing on the encoder if the agent was loaded
        # from an older checkpoint that did not persist it. Recover from the
        # first block's ffn weight in that case.
        ffn_dim = getattr(cfg, "ffn_dim", None)
        if ffn_dim is None:
            w = self.net.state_dict().get("encoder.op_blocks.0.ffn.0.weight")
            if w is not None:
                ffn_dim = int(w.shape[0])
            else:
                ffn_dim = 512
        dropout = getattr(cfg, "dropout", 0.1)
        payload = {
            "model_state": self.net.state_dict(),
            "instance_embed_state": self.instance_embed.state_dict() if self.instance_embed is not None else None,
            "config": {
                "op_feature_dim": cfg.op_input[0].in_features,
                "machine_feature_dim": cfg.machine_input[0].in_features,
                "global_feature_dim": self.net.critic.head[0].in_features - cfg.op_input[0].out_features,
                "hidden_dim": cfg.op_input[0].out_features,
                "num_blocks": cfg.num_blocks,
                "num_heads": cfg.num_heads,
                "ffn_dim": ffn_dim,
                "dropout": dropout,
            },
        }
        torch.save(payload, path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        hidden_dim: int = 128,
        num_blocks: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        device: str = "cpu",
    ) -> "HGTActorCriticAgent":
        payload = torch.load(path, map_location=device, weights_only=False)
        cfg = payload.get("config", {})
        # Backwards-compat: older checkpoints do not store ffn_dim/dropout.
        # Infer them from the saved state_dict when missing.
        if "ffn_dim" not in cfg:
            for k, v in payload["model_state"].items():
                if k.endswith("encoder.op_blocks.0.ffn.0.weight"):
                    cfg["ffn_dim"] = int(v.shape[0])
                    break
        if "dropout" not in cfg:
            cfg["dropout"] = dropout
        agent = cls.create(
            hidden_dim=cfg.get("hidden_dim", hidden_dim),
            num_blocks=cfg.get("num_blocks", num_blocks),
            num_heads=cfg.get("num_heads", num_heads),
            ffn_dim=cfg.get("ffn_dim", ffn_dim),
            dropout=cfg.get("dropout", dropout),
            op_feature_dim=cfg.get("op_feature_dim"),
            machine_feature_dim=cfg.get("machine_feature_dim"),
            global_feature_dim=cfg.get("global_feature_dim"),
            device=device,
        )
        agent.net.load_state_dict(payload["model_state"])
        if payload.get("instance_embed_state") is not None and agent.instance_embed is not None:
            agent.instance_embed.load_state_dict(payload["instance_embed_state"])
        return agent
