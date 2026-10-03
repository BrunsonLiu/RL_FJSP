"""Collect move-outcome pairs from greedy local search using full neighborhoods.

This version uses the rich neighborhoods from fjsp.scheduler.local_search:
reassign, swap_machine, swap_same_machine, swap_order_across. MoveRankNet is
re-trained on this data so it can score moves in the same search space as the
SOTA ILS pipeline.
"""
from __future__ import annotations

import argparse
import pickle
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Import env first to break a circular import in the codebase.
from fjsp.env.improvement_env import FJSPImprovementEnv  # noqa: F401

from fjsp.graph.solution_graph import build_solution_graph
from fjsp.parser.fjs_parser import parse_fjs
from fjsp.scheduler.local_search import (
    _assignments,
    _makespan,
    _neighbors_reassign,
    _neighbors_swap_machines,
    _neighbors_swap_order_across_machines,
    _neighbors_swap_same_machine,
    _recompute,
)
from fjsp.scheduler.validator import earliest_finish_schedule
from rl.models.neighborhood_move import NeighborhoodMove


NEIGHBORHOOD_FNS = {
    "reassign": _neighbors_reassign,
    "swap_machine": _neighbors_swap_machines,
    "swap_same_machine": _neighbors_swap_same_machine,
    "swap_order_across": _neighbors_swap_order_across_machines,
}


def _neighbor_to_move(
    current: list[tuple[int, int, int]],
    neighbor: list[tuple[int, int, int]],
    neighborhood: str,
) -> NeighborhoodMove:
    """Convert a neighbor assignment list to a NeighborhoodMove."""
    if neighborhood == "reassign":
        for i, (cur, nxt) in enumerate(zip(current, neighbor)):
            if cur[2] != nxt[2]:
                return NeighborhoodMove(
                    move_type="reassign",
                    op_indices=(i,),
                    target_machines=(nxt[2],),
                )
        raise ValueError("reassign neighbor has no machine change")

    diffs = [i for i, (cur, nxt) in enumerate(zip(current, neighbor)) if cur != nxt]
    if len(diffs) != 2:
        # Some neighborhoods may not change machines but change order; still
        # two positions differ in the assignments list.
        diffs = [i for i in range(len(current)) if current[i] != neighbor[i]]
    if len(diffs) != 2:
        raise ValueError(f"expected 2 differing positions, got {len(diffs)}")
    i, j = diffs
    return NeighborhoodMove(
        move_type=neighborhood,
        op_indices=(i, j),
        target_machines=(neighbor[i][2], neighbor[j][2]),
    )


def _sample_neighbors(neighbors: list[Any], max_n: int, rng: random.Random) -> list[Any]:
    if max_n is None or len(neighbors) <= max_n:
        return neighbors
    return rng.sample(neighbors, max_n)


def collect_from_instance(
    instance_path: str,
    *,
    n_runs: int = 3,
    max_steps: int = 100,
    max_per_type: int | None = 200,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Run greedy local search and record move-outcome pairs."""
    rng = random.Random(seed)
    instance = parse_fjs(instance_path)
    all_data = []

    for run in range(n_runs):
        schedule = earliest_finish_schedule(instance)
        assignments = _assignments(schedule)
        step = 0

        while step < max_steps:
            current_ms = _makespan(instance, assignments)
            graph = build_solution_graph(instance, _recompute(instance, assignments))

            moves: list[NeighborhoodMove] = []
            improvements: list[float] = []

            for name, fn in NEIGHBORHOOD_FNS.items():
                try:
                    neighbors = fn(instance, assignments)
                except Exception:
                    continue
                sampled = _sample_neighbors(neighbors, max_per_type, rng)
                for neighbor in sampled:
                    try:
                        move = _neighbor_to_move(assignments, neighbor, name)
                        new_ms = _makespan(instance, neighbor)
                        if new_ms < current_ms:
                            imp = (current_ms - new_ms) / current_ms
                        else:
                            imp = 0.0
                        moves.append(move)
                        improvements.append(imp)
                    except Exception:
                        continue

            if moves:
                all_data.append({
                    "graph": graph,
                    "moves": moves,
                    "improvements": improvements,
                    "old_makespan": current_ms,
                })

            # Greedy: pick best improving move
            best_idx = max(range(len(improvements)), key=lambda i: improvements[i])
            if not improvements or improvements[best_idx] <= 0:
                break
            best_neighbor = None
            for name, fn in NEIGHBORHOOD_FNS.items():
                try:
                    neighbors = fn(instance, assignments)
                except Exception:
                    continue
                sampled = _sample_neighbors(neighbors, max_per_type, rng)
                for neighbor in sampled:
                    try:
                        move = _neighbor_to_move(assignments, neighbor, name)
                        if move == moves[best_idx]:
                            best_neighbor = neighbor
                            break
                    except Exception:
                        continue
                if best_neighbor is not None:
                    break

            if best_neighbor is None:
                break
            assignments = best_neighbor
            step += 1

        print(f"  {Path(instance_path).stem} run {run}: {step} steps, best={_makespan(instance, assignments)}")

    return all_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--instances",
        nargs="+",
        default=[f"data/instances/brandimarte/mk{i:02d}.txt" for i in range(1, 16)],
    )
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-per-type", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        default="data/results/move_rank_data/neighborhood_move_data.pkl",
    )
    args = parser.parse_args()

    all_data = []
    for inst_path in args.instances:
        print(f"Collecting from {inst_path}...")
        data = collect_from_instance(
            inst_path,
            n_runs=args.n_runs,
            max_steps=args.max_steps,
            max_per_type=args.max_per_type,
            seed=args.seed,
        )
        all_data.extend(data)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(all_data, f)

    n_pairs = sum(len(s["moves"]) for s in all_data)
    n_pos = sum(1 for s in all_data for imp in s["improvements"] if imp > 0)
    print(f"\nSaved {len(all_data)} samples, {n_pairs} pairs, {n_pos} positive ({100*n_pos/n_pairs:.2f}%)")


if __name__ == "__main__":
    main()
