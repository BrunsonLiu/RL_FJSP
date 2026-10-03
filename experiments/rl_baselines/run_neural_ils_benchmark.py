"""Benchmark Neural-guided ILS against greedy BARI and standard ILS.

Outputs a JSON with per-instance results and gap to literature best-known.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from fjsp.env.improvement_env import FJSPImprovementEnv
from fjsp.graph.solution_graph import SolutionGraph
from fjsp.parser.fjs_parser import parse_fjs
from fjsp.scheduler.local_search import _recompute, critical_path_perturb, iterated_local_search
from fjsp.scheduler.validator import ScheduledOperation, earliest_finish_schedule, validate_schedule
from rl.models.move_rank_net import MoveRankNet

LITERATURE_BEST = {
    "mk01": 40, "mk02": 26, "mk03": 204, "mk04": 60, "mk05": 172,
    "mk06": 58, "mk07": 139, "mk08": 523, "mk09": 307, "mk10": 197,
    "mk11": 615, "mk12": 508, "mk13": 430, "mk14": 694, "mk15": 341,
}

_SOTA_PATH = Path("data/results/sota_final.json")
if _SOTA_PATH.exists():
    with open(_SOTA_PATH) as _f:
        _SOTA_DATA = json.load(_f)
    SOTA_BEST = {item["instance"]: item["final_best"] for item in _SOTA_DATA}
else:
    SOTA_BEST = {}


def _graph_to_device(graph: SolutionGraph, device: torch.device) -> SolutionGraph:
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


def _load_net(checkpoint: str, device: torch.device) -> MoveRankNet:
    net = MoveRankNet(
        hidden_dim=64, num_mp_rounds=1, num_transformer_blocks=1, num_heads=4, ffn_dim=64
    ).to(device)
    net.load_state_dict(torch.load(checkpoint, map_location=device))
    net.eval()
    return net


def greedy_bari(instance, initial: list[ScheduledOperation] | None = None) -> tuple[list[ScheduledOperation], int]:
    """Greedy local search using only the BARI improvement neighborhood."""
    env = FJSPImprovementEnv(instance, initial_schedule=initial, max_steps=200, patience=30)
    obs = env.reset()
    while not env.done:
        moves = obs["valid_moves"]
        if not moves:
            break
        old_ms = env.makespan
        best_idx = None
        best_ms = old_ms
        for i, move in enumerate(moves[:80]):
            try:
                new_assignments = env._apply_move(move)
                new_schedule = _recompute(instance, new_assignments)
                validation = validate_schedule(instance, new_schedule)
                if validation.is_valid and validation.makespan < best_ms:
                    best_ms = validation.makespan
                    best_idx = i
            except Exception:
                continue
        if best_idx is None:
            break
        obs, _, done, info = env.step(best_idx)
    return env.schedule, env.makespan


def neural_guided_ils(
    instance,
    net: MoveRankNet,
    device: torch.device,
    *,
    n_iterations: int = 30,
    top_k: int = 20,
    patience: int = 5,
    perturb_n_swaps: int = 5,
    seed: int = 0,
    initial: list[ScheduledOperation] | None = None,
) -> tuple[list[ScheduledOperation], int]:
    """Neural-guided ILS using MoveRankNet."""
    import random
    rng = random.Random(seed)

    env = FJSPImprovementEnv(instance, initial_schedule=initial, max_steps=200, patience=patience)
    env.reset()

    best_schedule = env.schedule
    best_makespan = env.makespan
    no_improve_count = 0
    iteration = 0

    while iteration < n_iterations and not env.done:
        obs = env.observe()
        moves = obs["valid_moves"]
        if not moves:
            break

        graph = _graph_to_device(obs["graph"], device)
        with torch.no_grad():
            scores = net(graph, moves, env._assignments)
        ranked = torch.argsort(scores, descending=True).tolist()

        accepted = False
        for idx in ranked[:top_k]:
            move = moves[idx]
            try:
                new_assignments = env._apply_move(move)
                new_schedule = _recompute(instance, new_assignments)
                validation = validate_schedule(instance, new_schedule)
                if validation.is_valid and validation.makespan < env.makespan:
                    env.step(idx)
                    accepted = True
                    break
            except Exception:
                continue

        if env.makespan < best_makespan:
            best_makespan = env.makespan
            best_schedule = env.schedule
            no_improve_count = 0
        else:
            no_improve_count += 1

        if not accepted or no_improve_count >= patience:
            perturbed = critical_path_perturb(
                instance, best_schedule, n_swaps=perturb_n_swaps, seed=rng.randint(0, 1_000_000)
            )
            env.reset(perturbed)
            no_improve_count = 0
            iteration += 1

    return best_schedule, best_makespan


def run_instance(instance_path: str, net: MoveRankNet | None, device: torch.device, seeds: list[int], args) -> dict[str, Any]:
    instance = parse_fjs(instance_path)
    name = Path(instance_path).stem
    initial = earliest_finish_schedule(instance)

    # Greedy BARI (deterministic)
    gb_sched, gb_ms = greedy_bari(instance, initial=list(initial))

    # Standard ILS (optional; can be slow on large instances)
    if args.skip_standard_ils:
        std_ms = SOTA_BEST.get(name)
    else:
        std_sched, std_ms = iterated_local_search(
            instance,
            initial,
            n_iterations=args.ils_iterations,
            perturb_n_swaps=args.ils_perturb_swaps,
            max_ls_iterations=args.ils_ls_iterations,
            strategy="first",
        )

    # Neural-guided ILS over seeds
    neural_best = float("inf")
    neural_sched = None
    for seed in seeds:
        if net is None:
            break
        sched, ms = neural_guided_ils(
            instance,
            net,
            device,
            n_iterations=args.n_iterations,
            top_k=args.top_k,
            patience=args.patience,
            perturb_n_swaps=args.perturb_n_swaps,
            seed=seed,
            initial=list(initial),
        )
        if ms < neural_best:
            neural_best = ms
            neural_sched = sched

    lit = LITERATURE_BEST.get(name)
    def gap(ms):
        return None if lit is None else round(100 * (ms - lit) / lit, 2)

    return {
        "instance": name,
        "initial_makespan": max(op.end for op in initial),
        "greedy_bari_makespan": gb_ms,
        "greedy_bari_gap": gap(gb_ms),
        "standard_ils_makespan": std_ms,
        "standard_ils_gap": gap(std_ms),
        "neural_ils_makespan": None if neural_sched is None else neural_best,
        "neural_ils_gap": None if neural_sched is None else gap(neural_best),
        "literature_best": lit,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", nargs="+", default=[f"data/instances/brandimarte/mk{i:02d}.txt" for i in range(1, 16)])
    parser.add_argument("--model", default="data/results/move_rank_data/move_rank_net.pt")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--n-iterations", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--perturb-n-swaps", type=int, default=5)
    parser.add_argument("--ils-iterations", type=int, default=20)
    parser.add_argument("--ils-perturb-swaps", type=int, default=5)
    parser.add_argument("--ils-ls-iterations", type=int, default=50)
    parser.add_argument("--output", default="data/results/neural_ils_benchmark.json")
    parser.add_argument("--no-neural", action="store_true", help="skip neural method")
    parser.add_argument("--skip-standard-ils", action="store_true", help="use SOTA final as standard ILS reference instead of recomputing")
    args = parser.parse_args()

    seeds = [int(x) for x in args.seeds.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = None if args.no_neural else _load_net(args.model, device)

    results = []
    for inst_path in args.instances:
        print(f"Running {inst_path}...")
        res = run_instance(inst_path, net, device, seeds, args)
        print(json.dumps(res, indent=2))
        results.append(res)

    summary = {
        "args": vars(args),
        "results": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
