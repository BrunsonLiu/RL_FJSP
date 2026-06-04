"""Behavioral cloning pretraining for the HGT-FJSP agent.

Cross-instance capable: takes a list of FJSPDispatchEnvs and collects
demonstrations from one or more dispatch rules (the "teachers") across
all of them, then trains one HGT actor-critic net to imitate. The
result is a single model that can be fine-tuned per-instance with
AC / PPO.
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from random import Random
from typing import Sequence

import torch
from torch import nn

from fjsp.env import DispatchAction, FJSPDispatchEnv
from fjsp.graph.operation_machine_graph import build_operation_machine_graph
from fjsp.scheduler.dispatch_rules import get_rule
from fjsp.scheduler.validator import ScheduledOperation
from fjsp.utils.scaling import instance_time_scale
from rl.agents.hgt_fjsp import HGTActorCriticAgent


def _snapshot_env(env: FJSPDispatchEnv) -> tuple:
    return (
        list(env.job_next_op),
        list(env.job_ready_time),
        list(env.machine_ready_time),
        int(env.remaining_operations),
        [ScheduledOperation(**vars(op)) for op in env._schedule],
    )


def _restore_env(env: FJSPDispatchEnv, snapshot: tuple) -> None:
    env.job_next_op = list(snapshot[0])
    env.job_ready_time = list(snapshot[1])
    env.machine_ready_time = list(snapshot[2])
    env.remaining_operations = int(snapshot[3])
    env._schedule = [ScheduledOperation(**vars(op)) for op in snapshot[4]]


def _resolve_positions(env: FJSPDispatchEnv, action: DispatchAction) -> tuple[int, int]:
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


def collect_cross_instance_demonstrations(
    envs: list[FJSPDispatchEnv],
    *,
    rollouts_per_instance: int = 50,
    teachers: Sequence[str] = ("earliest_finish",),
    seed: int = 0,
) -> list[tuple[int, tuple, int, int]]:
    """Collect (instance_idx, state_snapshot, job_pos, machine_pos) pairs
    from each named dispatch rule. Each rule produces
    ``rollouts_per_instance`` trajectories per instance.
    """
    rng = Random(seed)
    rules = [get_rule(name) for name in teachers]
    out: list[tuple[int, tuple, int, int]] = []
    for inst_idx, env in enumerate(envs):
        for rule in rules:
            for _ in range(rollouts_per_instance):
                env.reset()
                while not env.done:
                    snap = _snapshot_env(env)
                    chosen = rule(env)
                    jp, mp = _resolve_positions(env, chosen)
                    out.append((inst_idx, snap, jp, mp))
                    env.step(chosen)
        # Touch rng to keep teacher ordering deterministic across instances.
        _ = rng.random()
    return out


def train_hgt_imitation(
    envs: list[FJSPDispatchEnv],
    instance_paths: list[Path],
    *,
    rollouts_per_instance: int = 50,
    teachers: Sequence[str] = ("earliest_finish",),
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    num_blocks: int = 4,
    num_heads: int = 8,
    ffn_dim: int = 512,
    dropout: float = 0.1,
    seed: int = 0,
    device: str = "cpu",
    use_instance_embed: bool = True,
) -> tuple[HGTActorCriticAgent, list[dict[str, float]]]:
    torch.manual_seed(seed)
    num_known = max(1, len(envs)) if use_instance_embed else 0
    agent = HGTActorCriticAgent.create(
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        dropout=dropout,
        num_known_instances=num_known if use_instance_embed else None,
        device=device,
    )
    optimizer = torch.optim.Adam(agent.net.parameters(), lr=lr)

    print(
        f"[HGT-BC] Collecting {rollouts_per_instance} rollouts x {len(envs)} instances x "
        f"{len(teachers)} teachers..."
    )
    demos = collect_cross_instance_demonstrations(
        envs, rollouts_per_instance=rollouts_per_instance, teachers=teachers, seed=seed
    )
    print(f"[HGT-BC] Collected {len(demos)} (state, action) demonstrations.")

    n = len(demos)
    history: list[dict[str, float]] = []
    best_loss: float | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_ie: dict[str, torch.Tensor] | None = None

    instance_id_map: dict[str, int] = (
        {p.stem: i for i, p in enumerate(instance_paths)} if use_instance_embed else {}
    )

    def _instance_id_for_path(path: Path) -> int:
        if not use_instance_embed:
            return 0
        return instance_id_map.get(path.stem, 0)

    log_every = max(1, epochs // 10)

    for epoch in range(1, epochs + 1):
        order = list(range(n))
        Random(seed * 1000 + epoch).shuffle(order)

        total_job_loss = 0.0
        total_machine_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            mb = order[start:start + batch_size]
            job_logits_list: list[torch.Tensor] = []
            machine_logits_list: list[torch.Tensor] = []
            job_targets: list[int] = []
            machine_targets: list[int] = []

            for idx in mb:
                inst_idx, snap, jp, mp = demos[idx]
                env = envs[inst_idx]
                _restore_env(env, snap)
                if use_instance_embed:
                    agent.instance_id = _instance_id_for_path(instance_paths[inst_idx])
                graph = build_operation_machine_graph(env, device=device)
                op_emb, mach_emb = agent.net.encode(graph)
                if use_instance_embed and agent.instance_id is not None and agent.instance_embed is not None:
                    inst_vec = agent.instance_embed(torch.tensor(agent.instance_id, device=device))
                    op_emb = op_emb + inst_vec.unsqueeze(0)
                next_indices = torch.tensor(graph.next_op_indices, dtype=torch.long, device=device)
                job_logits = agent.net.score_jobs(op_emb, next_indices)
                target_op_idx = int(next_indices[jp].item())
                target_op_ref = graph.operation_refs[target_op_idx]
                machine_actions = [a for a in env.available_actions() if a.job == target_op_ref.job]
                if not machine_actions:
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
                machine_logits = agent.net.score_machines(
                    op_emb[target_op_idx], mach_emb, machine_indices, durations
                )
                job_logits_list.append(job_logits)
                machine_logits_list.append(machine_logits)
                job_targets.append(jp)
                machine_targets.append(mp)

            if not job_logits_list:
                continue

            def _pad(t_list: list[torch.Tensor]) -> torch.Tensor:
                max_len = max(t.numel() for t in t_list)
                padded = []
                for t in t_list:
                    if t.numel() < max_len:
                        pad = torch.full((max_len - t.numel(),), float("-inf"), device=t.device, dtype=t.dtype)
                        padded.append(torch.cat([t, pad], dim=0))
                    else:
                        padded.append(t)
                return torch.stack(padded)

            jl = _pad(job_logits_list)
            ml = _pad(machine_logits_list)
            jt = torch.tensor(job_targets, dtype=torch.long, device=device)
            mt = torch.tensor(machine_targets, dtype=torch.long, device=device)
            job_loss = nn.functional.cross_entropy(jl, jt)
            machine_loss = nn.functional.cross_entropy(ml, mt)
            loss = job_loss + machine_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.net.parameters(), max_norm=1.0)
            optimizer.step()
            total_job_loss += float(job_loss.detach().cpu().item())
            total_machine_loss += float(machine_loss.detach().cpu().item())
            n_batches += 1

        avg_job_loss = total_job_loss / max(1, n_batches)
        avg_machine_loss = total_machine_loss / max(1, n_batches)
        avg_loss = avg_job_loss + avg_machine_loss
        if best_loss is None or avg_loss < best_loss:
            best_loss = avg_loss
            best_state = deepcopy(agent.net.state_dict())
            best_ie = deepcopy(agent.instance_embed.state_dict()) if agent.instance_embed is not None else None
        history.append(
            {
                "epoch": float(epoch),
                "job_loss": avg_job_loss,
                "machine_loss": avg_machine_loss,
                "total_loss": avg_loss,
            }
        )
        if epoch == 1 or epoch == epochs or epoch % log_every == 0:
            print(
                f"[HGT-BC] epoch {epoch:3d}/{epochs} | job_loss={avg_job_loss:.4f} "
                f"machine_loss={avg_machine_loss:.4f}"
            )

    if best_state is not None:
        agent.net.load_state_dict(best_state)
        if best_ie is not None and agent.instance_embed is not None:
            agent.instance_embed.load_state_dict(best_ie)
    return agent, history


def main() -> None:
    _root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train an HGT-FJSP policy with behavioral cloning.")
    parser.add_argument(
        "--instances",
        nargs="+",
        default=[str(_root / "data" / "instances" / "brandimarte" / f"mk{i:02d}.txt") for i in range(1, 11)],
    )
    parser.add_argument("--rollouts-per-instance", type=int, default=20)
    parser.add_argument(
        "--teachers",
        nargs="+",
        default=["earliest_finish", "spt", "lpt", "mor", "lor", "shortest_start"],
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-instance-embed", action="store_true")
    parser.add_argument("--model-out", default=str(_root / "data" / "results" / "hgt_bc_mk01_10.pt"))
    args = parser.parse_args()

    instance_paths = [Path(p) for p in args.instances]
    envs = [FJSPDispatchEnv.from_file(p) for p in instance_paths]
    agent, history = train_hgt_imitation(
        envs, instance_paths,
        rollouts_per_instance=args.rollouts_per_instance,
        teachers=args.teachers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        seed=args.seed,
        use_instance_embed=not args.no_instance_embed,
    )

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    agent.save(args.model_out)
    history_path = Path(args.model_out).with_suffix(".history.json")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"OK: hgt_bc saved to {args.model_out}, history={history_path}")


if __name__ == "__main__":
    main()
