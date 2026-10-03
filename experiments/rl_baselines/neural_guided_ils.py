"""Neural-guided Iterated Local Search for FJSP.

Uses a trained MoveRankNet to score candidate improvement moves and tries the
most promising ones first. This avoids exhaustively evaluating every neighbour
at each local-search step.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from fjsp.env.improvement_env import FJSPImprovementEnv
from fjsp.graph.solution_graph import SolutionGraph
from fjsp.parser.fjs_parser import parse_fjs
from fjsp.scheduler.local_search import _recompute, critical_path_perturb
from fjsp.scheduler.validator import ScheduledOperation, validate_schedule
from rl.models.move_rank_net import MoveRankNet


def _graph_to_device(graph: SolutionGraph, device: torch.device) -> SolutionGraph:
    """Move a SolutionGraph's tensors to the given device."""
    return SolutionGraph(
        operation_refs=graph.operation_refs,
        op_features=graph.op_features.to(device),
        machine_features=graph.machine_features.to(device),
        precedence_edges=graph.precedence_edges.to(device),
        machine_seq_edges=graph.machine_seq_edges.to(device),
        eligibility_edges=graph.eligibility_edges.to(device),
        eligibility_durations=graph.eligibility_durations.to(device),
        critical_op_indices=graph.critical_op_indices,
        global_features=graph.global_features.to(device),
    )


def _load_move_rank_net(checkpoint_path: str, device: torch.device) -> MoveRankNet:
    net = MoveRankNet(
        hidden_dim=64,
        num_mp_rounds=2,
        num_transformer_blocks=2,
        num_heads=8,
        ffn_dim=128,
    ).to(device)
    net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    net.eval()
    return net


def _schedule_to_json(schedule: list[ScheduledOperation]) -> dict[str, Any]:
    return {
        "operations": [
            {"job": op.job, "op": op.op, "machine": op.machine, "start": op.start, "end": op.end}
            for op in schedule
        ]
    }


def neural_guided_ils(
    instance,
    model_path: str,
    *,
    n_iterations: int = 30,
    top_k: int = 10,
    patience: int = 5,
    perturb_n_swaps: int = 5,
    max_steps_per_iteration: int = 50,
    seed: int = 0,
    device: torch.device | None = None,
) -> tuple[list[ScheduledOperation], int, dict[str, Any]]:
    """Run neural-guided ILS.

    At each step the model ranks all valid moves; we evaluate only the top-k
    predictions and accept the first move that actually improves the makespan.
    If none of the top-k moves improves, we apply a critical-path perturbation.
    """
    rng = random.Random(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = _load_move_rank_net(model_path, device)

    env = FJSPImprovementEnv(
        instance,
        max_steps=max_steps_per_iteration,
        patience=patience,
    )

    best_schedule = env.schedule
    best_makespan = env.makespan
    history = [best_makespan]

    no_improve_count = 0
    iteration = 0
    while iteration < n_iterations and not env.done:
        obs = env.observe()
        moves = obs["valid_moves"]
        if not moves:
            break

        # Score all candidate moves in one forward pass
        graph = _graph_to_device(obs["graph"], device)
        with torch.no_grad():
            scores = net(graph, moves, env._assignments)
        ranked = torch.argsort(scores, descending=True).tolist()

        accepted = False
        for idx in ranked[:top_k]:
            move = moves[idx]
            new_assignments = env._apply_move(move)
            try:
                new_schedule = _recompute(instance, new_assignments)
                validation = validate_schedule(instance, new_schedule)
                if validation.is_valid and validation.makespan < env.makespan:
                    obs, _, done, info = env.step(idx)
                    accepted = True
                    break
            except Exception:
                continue

        history.append(env.makespan)

        if env.makespan < best_makespan:
            best_makespan = env.makespan
            best_schedule = env.schedule
            no_improve_count = 0
        else:
            no_improve_count += 1

        if not accepted or no_improve_count >= patience:
            # Perturb and continue search from a modified solution
            perturbed = critical_path_perturb(
                instance,
                best_schedule,
                n_swaps=perturb_n_swaps,
                seed=rng.randint(0, 1_000_000),
            )
            env.reset(perturbed)
            no_improve_count = 0
            iteration += 1

    return best_schedule, best_makespan, {"history": history}


def main():
    parser = argparse.ArgumentParser(description="Neural-guided ILS for FJSP")
    parser.add_argument("--instance", required=True, help="Path to FJSP instance")
    parser.add_argument(
        "--model",
        default="data/results/move_rank_data/move_rank_net.pt",
        help="Path to trained MoveRankNet checkpoint",
    )
    parser.add_argument("--output", required=True, help="Path to write schedule JSON")
    parser.add_argument("--n-iterations", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--perturb-n-swaps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    instance = parse_fjs(args.instance)
    schedule, makespan, info = neural_guided_ils(
        instance,
        args.model,
        n_iterations=args.n_iterations,
        top_k=args.top_k,
        patience=args.patience,
        perturb_n_swaps=args.perturb_n_swaps,
        seed=args.seed,
    )

    validation = validate_schedule(instance, schedule)
    print(f"Final makespan: {makespan} (valid={validation.is_valid})")
    if not validation.is_valid:
        print("Validation errors:", validation.errors)
        raise SystemExit(1)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(_schedule_to_json(schedule), f, indent=2)

    print(f"Schedule written to {args.output}")
    print("History:", info["history"])


if __name__ == "__main__":
    main()
