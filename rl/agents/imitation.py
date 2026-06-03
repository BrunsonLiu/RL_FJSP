"""Behavioral cloning pretraining for the graph two-stage actor-critic policy.

Collects (state, action) demonstrations from a deterministic dispatch rule
(earliest-finish by default) and trains the graph encoder + actor heads to
imitate those decisions via supervised cross-entropy on both the job and
machine heads. The critic head is not trained in this phase.

The resulting model state is saved with the same protocol as
``GraphTwoStageActorCriticAgent.save`` and can be loaded into a
``GraphPPOAgent`` (or ``GraphTwoStageActorCriticAgent``) to continue with
RL fine-tuning.

Why this exists
---------------
The Brandimarte 4-agent baseline matrix
(``data/results/baseline_brandimarte_matrix.md``) shows that the graph
agents are systematically worse than the MLP ``REINFORCE`` baseline at the
default 50-episode per-instance budget. The graph encoder has many more
parameters and never has time to learn a useful policy from the sparse
terminal reward alone.

The standard fix in the FJSP-RL literature is to warm-start the graph
encoder with supervised learning on a known-good dispatch rule. After BC
pretraining the policy is already at the heuristic's level, and PPO only
needs to push past it.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from random import Random

import torch
from torch import nn

from fjsp.env import DispatchAction, FJSPDispatchEnv
from fjsp.graph.operation_machine_graph import build_operation_machine_graph
from fjsp.scheduler.dispatch_rules import choose_earliest_finish
from fjsp.scheduler.validator import ScheduledOperation
from fjsp.utils.scaling import instance_time_scale
from rl.agents.graph_actor_critic import GraphTwoStageActorCriticAgent


@dataclass(frozen=True)
class Demonstration:
    """One (state, target_action) pair recorded under a dispatch rule.

    The action is stored as ``(job_position, machine_position)`` which
    matches the model's ``score_jobs`` / ``score_machines`` output ordering.
    """
    state_snapshot: tuple
    job_position: int
    machine_position: int


def _snapshot_env(env: FJSPDispatchEnv) -> tuple:
    return (
        list(env.job_next_op),
        list(env.job_ready_time),
        list(env.machine_ready_time),
        int(env.remaining_operations),
        list(env._schedule),
    )


def _restore_env(env: FJSPDispatchEnv, snapshot: tuple) -> None:
    env.job_next_op = list(snapshot[0])
    env.job_ready_time = list(snapshot[1])
    env.machine_ready_time = list(snapshot[2])
    env.remaining_operations = int(snapshot[3])
    env._schedule = [ScheduledOperation(**vars(op)) for op in snapshot[4]]


def _resolve_positions(env: FJSPDispatchEnv, action: DispatchAction) -> tuple[int, int]:
    """Translate a (job, machine) action into (job_pos, machine_pos)
    positions that match the model's score_jobs / score_machines output."""
    graph = build_operation_machine_graph(env, device="cpu")
    next_indices = list(graph.next_op_indices)
    job_pos = next(
        (pos for pos, op_idx in enumerate(next_indices) if graph.operation_refs[op_idx].job == action.job),
        None,
    )
    if job_pos is None:
        raise ValueError("Action job is not schedulable from this state.")
    machine_actions = [a for a in env.available_actions() if a.job == action.job]
    machine_pos = next(
        (i for i, a in enumerate(machine_actions) if a.machine == action.machine),
        None,
    )
    if machine_pos is None:
        raise ValueError("Action machine is not available for the chosen job.")
    return job_pos, machine_pos


def collect_demonstrations(
    env: FJSPDispatchEnv,
    *,
    rollouts: int = 100,
    seed: int = 0,
) -> list[Demonstration]:
    """Run ``rollouts`` full episodes of ``choose_earliest_finish`` and
    record every (state, job_pos, machine_pos) pair the rule picks."""
    demonstrations: list[Demonstration] = []
    for rollout_idx in range(rollouts):
        env.reset()
        while not env.done:
            snapshot = _snapshot_env(env)
            chosen = choose_earliest_finish(env)
            job_pos, machine_pos = _resolve_positions(env, chosen)
            demonstrations.append(
                Demonstration(
                    state_snapshot=snapshot,
                    job_position=job_pos,
                    machine_position=machine_pos,
                )
            )
            env.step(chosen)
    return demonstrations


def _train_step(
    env: FJSPDispatchEnv,
    agent: GraphTwoStageActorCriticAgent,
    batch_indices: list[int],
    demonstrations: list[Demonstration],
    *,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """One minibatch forward pass: restore each state, build its graph,
    run the model, and return stacked (job_logits, machine_logits) plus
    (correct_job, correct_machine) counts.

    Returns
    -------
    job_logits_stacked : (B, J) tensor
    machine_logits_stacked : (B, M_max) tensor — padded with -inf on shorter rows
    job_targets : (B,) long tensor
    machine_targets : (B,) long tensor
    correct_job, correct_machine : ints
    """
    job_logits_list: list[torch.Tensor] = []
    machine_logits_list: list[torch.Tensor] = []
    job_targets_list: list[int] = []
    machine_targets_list: list[int] = []
    correct_job = 0
    correct_machine = 0

    for idx in batch_indices:
        demo = demonstrations[idx]
        _restore_env(env, demo.state_snapshot)
        graph = build_operation_machine_graph(env, device=device)
        op_emb, mach_emb = agent.model.encode(graph)
        next_indices = torch.tensor(graph.next_op_indices, dtype=torch.long, device=device)
        job_logits = agent.model.score_jobs(op_emb, next_indices)

        target_op_idx = int(next_indices[demo.job_position].item())
        target_op_ref = graph.operation_refs[target_op_idx]
        machine_actions = [a for a in env.available_actions() if a.job == target_op_ref.job]
        if not machine_actions:
            # Should be impossible (the action was valid when recorded).
            continue
        machine_indices = torch.tensor(
            [a.machine for a in machine_actions], dtype=torch.long, device=device
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
            device=device,
        )
        machine_logits = agent.model.score_machines(
            op_emb[target_op_idx], mach_emb, machine_indices, durations
        )

        job_targets_list.append(demo.job_position)
        machine_targets_list.append(demo.machine_position)
        job_logits_list.append(job_logits)
        machine_logits_list.append(machine_logits)
        if int(job_logits.argmax().item()) == demo.job_position:
            correct_job += 1
        if int(machine_logits.argmax().item()) == demo.machine_position:
            correct_machine += 1

    # Pad both job logits and machine logits to the same length with -inf
    # so cross-entropy only considers valid positions. Different states have
    # different numbers of schedulable operations / available machines, so
    # batching requires padding.
    def _pad(logits_list: list[torch.Tensor]) -> torch.Tensor:
        if not logits_list:
            return torch.empty(0, 0)
        max_len = max(t.numel() for t in logits_list)
        padded = []
        for t in logits_list:
            if t.numel() < max_len:
                pad = torch.full((max_len - t.numel(),), float("-inf"), device=t.device, dtype=t.dtype)
                padded.append(torch.cat([t, pad], dim=0))
            else:
                padded.append(t)
        return torch.stack(padded)

    job_logits_stacked = _pad(job_logits_list)
    machine_logits_stacked = _pad(machine_logits_list)
    job_targets_t = torch.tensor(job_targets_list, dtype=torch.long, device=device)
    machine_targets_t = torch.tensor(machine_targets_list, dtype=torch.long, device=device)
    return job_logits_stacked, machine_logits_stacked, job_targets_t, machine_targets_t, correct_job, correct_machine


def train_imitation(
    env: FJSPDispatchEnv,
    *,
    rollouts: int = 100,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    gnn_rounds: int = 2,
    seed: int = 0,
    device: str = "cpu",
    log_every: int | None = None,
) -> tuple[GraphTwoStageActorCriticAgent, list[dict[str, float]]]:
    """Behavioral cloning pretraining.

    Steps:
        1. Collect ``rollouts`` episodes of ``choose_earliest_finish``
           demonstrations from ``env``.
        2. Train the graph actor-critic net for ``epochs`` full passes
           over the demonstrations with cross-entropy on both heads.
        3. Return the agent with the best (lowest total loss) state loaded.

    The critic head receives no gradient — only the encoder and the two
    actor heads are supervised. After this call, ``agent.save(path)``
    produces a checkpoint that can be loaded into a ``GraphPPOAgent`` and
    fine-tuned with PPO.
    """
    torch.manual_seed(seed)

    print(f"[BC] Collecting {rollouts} rollouts of earliest-finish demonstrations...")
    demonstrations = collect_demonstrations(env, rollouts=rollouts, seed=seed)
    print(f"[BC] Collected {len(demonstrations)} (state, action) demonstrations.")

    agent = GraphTwoStageActorCriticAgent.create(
        hidden_dim=hidden_dim, gnn_rounds=gnn_rounds, device=device
    )
    optimizer = torch.optim.Adam(agent.model.parameters(), lr=lr)

    n = len(demonstrations)
    history: list[dict[str, float]] = []
    best_loss: float | None = None
    best_state: dict[str, torch.Tensor] | None = None

    log_every = log_every if log_every is not None else max(1, epochs // 10)

    for epoch in range(1, epochs + 1):
        order = list(range(n))
        Random(seed * 1000 + epoch).shuffle(order)

        total_job_loss = 0.0
        total_machine_loss = 0.0
        total_correct_job = 0
        total_correct_machine = 0
        n_batches = 0

        for start in range(0, n, batch_size):
            mb = order[start:start + batch_size]
            job_logits, machine_logits, job_targets, machine_targets, cj, cm = _train_step(
                env, agent, mb, demonstrations, device=device
            )
            if job_logits.numel() == 0:
                continue

            job_loss = nn.functional.cross_entropy(job_logits, job_targets)
            machine_loss = nn.functional.cross_entropy(machine_logits, machine_targets)
            loss = job_loss + machine_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.model.parameters(), max_norm=1.0)
            optimizer.step()

            total_job_loss += float(job_loss.detach().cpu().item())
            total_machine_loss += float(machine_loss.detach().cpu().item())
            total_correct_job += cj
            total_correct_machine += cm
            n_batches += 1

        avg_job_loss = total_job_loss / max(1, n_batches)
        avg_machine_loss = total_machine_loss / max(1, n_batches)
        avg_loss = avg_job_loss + avg_machine_loss
        job_acc = total_correct_job / max(1, n)
        machine_acc = total_correct_machine / max(1, n)

        if best_loss is None or avg_loss < best_loss:
            best_loss = avg_loss
            best_state = deepcopy(agent.model.state_dict())

        history.append(
            {
                "epoch": float(epoch),
                "job_loss": avg_job_loss,
                "machine_loss": avg_machine_loss,
                "total_loss": avg_loss,
                "job_acc": float(job_acc),
                "machine_acc": float(machine_acc),
            }
        )

        if epoch == 1 or epoch == epochs or epoch % log_every == 0:
            print(
                f"[BC] epoch {epoch:3d}/{epochs} | "
                f"job_loss={avg_job_loss:.4f} machine_loss={avg_machine_loss:.4f} | "
                f"job_acc={job_acc:.3f} machine_acc={machine_acc:.3f}"
            )

    if best_state is not None:
        agent.model.load_state_dict(best_state)
    return agent, history
