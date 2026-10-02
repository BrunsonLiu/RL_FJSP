"""Collect training data for MoveRankNet from standard ILS runs.

For each local-search step, we record:
  - solution graph
  - all candidate moves
  - the actual makespan delta for each move

The target for MoveRankNet is the normalized improvement: max(0, old_ms - new_ms) / old_ms.
"""
import sys
import pickle
import random
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fjsp.parser.fjs_parser import parse_fjs
from fjsp.scheduler.validator import earliest_finish_schedule, validate_schedule
from fjsp.scheduler.local_search import _assignments, _recompute
from fjsp.env.improvement_env import FJSPImprovementEnv, ImprovementMove
from fjsp.graph.solution_graph import build_solution_graph


def evaluate_move(env: FJSPImprovementEnv, move: ImprovementMove, old_makespan: int) -> float:
    """Apply move and return normalized improvement (0 if invalid/worse).

    Uses the environment's move application and a fast recompute/validate,
    avoiding the overhead of a full env.step per candidate.
    """
    try:
        new_assignments = env._apply_move(move)
        new_schedule = _recompute(env.instance, new_assignments)
        validation = validate_schedule(env.instance, new_schedule)
        if validation.is_valid and validation.makespan < old_makespan:
            return (old_makespan - validation.makespan) / old_makespan
    except Exception:
        pass
    return 0.0


def collect_from_ils(
    instance_path: str,
    n_runs: int = 10,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Run greedy local search multiple times and collect move-outcome pairs."""
    random.seed(seed)
    instance = parse_fjs(instance_path)
    all_data = []

    for run in range(n_runs):
        initial = earliest_finish_schedule(instance)
        env = FJSPImprovementEnv(
            instance,
            initial_schedule=initial,
            max_steps=200,
            patience=30,
        )
        obs = env.reset()

        step = 0
        while not env.done and step < 200:
            moves = obs["valid_moves"]
            if not moves:
                break

            old_ms = env.makespan
            graph = build_solution_graph(instance, env.schedule)
            assignments = list(env._assignments)

            # Evaluate all candidate moves (cap at 80 to keep a single step fast)
            sampled_moves = moves[:80]

            improvements = []
            for move in sampled_moves:
                imp = evaluate_move(env, move, old_ms)
                improvements.append(imp)

            all_data.append({
                "graph": graph,
                "moves": sampled_moves,
                "assignments": assignments,
                "old_makespan": old_ms,
                "improvements": improvements,
            })

            # Pick the best improving move to continue (greedy)
            best_idx = max(range(len(improvements)), key=lambda i: improvements[i])
            if improvements[best_idx] <= 0:
                break

            action_idx = moves.index(sampled_moves[best_idx])
            obs, _, done, info = env.step(action_idx)
            step += 1

        print(f"  run {run}: {step} steps, best={env.best_makespan}")

    return all_data


def main():
    instances = [f"data/instances/brandimarte/mk{i:02d}.txt" for i in range(1, 16)]
    output_dir = Path("data/results/move_rank_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_data = []
    for inst_path in instances:
        print(f"Collecting from {inst_path}...")
        data = collect_from_ils(inst_path, n_runs=5, seed=0)
        all_data.extend(data)

    with open(output_dir / "move_rank_data.pkl", "wb") as f:
        pickle.dump(all_data, f)

    print(f"\nTotal samples: {len(all_data)}")
    n_positive = sum(1 for d in all_data for imp in d["improvements"] if imp > 0)
    n_total = sum(len(d["improvements"]) for d in all_data)
    print(f"Positive moves: {n_positive}/{n_total} ({100*n_positive/n_total:.1f}%)")


if __name__ == "__main__":
    main()
