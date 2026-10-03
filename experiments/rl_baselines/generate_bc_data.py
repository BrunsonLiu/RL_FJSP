"""Generate behavioral cloning data using a greedy expert.

For each initial solution, the expert tries every valid move and picks the
one with the lowest resulting makespan (first-improvement-like). The
(state, action) pairs are saved for BC pretraining of the policy network.
"""
import sys
import json
import pickle
import random
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fjsp.parser.fjs_parser import parse_fjs
from fjsp.scheduler.validator import earliest_finish_schedule
from fjsp.env.improvement_env import FJSPImprovementEnv


def randomized_first_improvement_rollout(
    env: FJSPImprovementEnv, max_steps: int = 50, rng: random.Random | None = None
) -> list[dict[str, Any]]:
    """Collect a single trajectory using randomized first-improvement.

    At each step, shuffle the valid moves and accept the first move that
    improves the makespan. If no move improves, apply the move that yields
    the smallest makespan (or the first valid move) so the rollout continues
    and provides more training signal.

    Returns list of transitions with:
      - graph
      - valid_moves
      - chosen_action_idx
      - old_makespan
      - new_makespan
    """
    obs = env.reset()
    trajectory: list[dict[str, Any]] = []
    rand = rng if rng is not None else random.Random()

    step = 0
    while not env.done and step < max_steps:
        valid_moves = obs["valid_moves"]
        if not valid_moves:
            break

        # Evaluate all moves and record outcomes
        outcomes: list[tuple[int, int]] = []  # (idx, makespan)
        for i in range(len(valid_moves)):
            candidate = env.__class__(
                env.instance,
                initial_schedule=env.schedule,
                max_steps=1,
                patience=1,
            )
            candidate.reset()
            candidate._assignments = list(env._assignments)
            candidate._schedule = list(env._schedule)
            candidate._makespan = env._makespan
            candidate._best_makespan = env._best_makespan
            candidate._step = env._step
            candidate._no_improve_count = env._no_improve_count
            candidate._build_machine_sequences()
            candidate._bottleneck_scores = candidate._compute_bottleneck_scores()
            candidate._valid_moves = candidate._compute_valid_moves()

            try:
                c_obs, _, done, info = candidate.step(i)
                outcomes.append((i, info["makespan"]))
            except Exception:
                continue

        if not outcomes:
            break

        # Randomize order and pick first improving move; fallback to best
        rand.shuffle(outcomes)
        chosen_idx, chosen_ms = outcomes[0]
        for idx, ms in outcomes:
            if ms < env.makespan:
                chosen_idx, chosen_ms = idx, ms
                break

        old_ms = env.makespan
        old_graph = obs["graph"]
        old_moves = list(obs["valid_moves"])

        obs, reward, done, info = env.step(chosen_idx)
        trajectory.append({
            "graph": old_graph,
            "valid_moves": old_moves,
            "chosen_action_idx": chosen_idx,
            "old_makespan": old_ms,
            "new_makespan": info["makespan"],
        })
        step += 1

    return trajectory


def generate_bc_dataset(
    instance_paths: list[str],
    output_dir: str = "data/results/bc_data",
    n_rollouts: int = 20,
    max_steps: int = 50,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Generate BC dataset from greedy expert rollouts on multiple instances."""
    random.seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_transitions: list[dict[str, Any]] = []

    for inst_path in instance_paths:
        print(f"\nProcessing {inst_path}...")
        inst = parse_fjs(inst_path)
        for rollout_idx in range(n_rollouts):
            # Diverse initial solutions: EF + random perturbations
            if rollout_idx == 0:
                initial = earliest_finish_schedule(inst)
            else:
                # Random perturbation of EF schedule: reassign random ops
                initial = earliest_finish_schedule(inst)
                assignments = [(o.job, o.op, o.machine) for o in initial]
                n_perturb = max(1, len(assignments) // 5)
                for _ in range(n_perturb):
                    idx = random.randrange(len(assignments))
                    job, op_idx, _ = assignments[idx]
                    options = inst.jobs[job].operations[op_idx].options
                    new_machine = random.choice(options).machine
                    assignments[idx] = (job, op_idx, new_machine)
                from fjsp.scheduler.local_search import _recompute, validate_schedule
                try:
                    initial = _recompute(inst, assignments)
                    if not validate_schedule(inst, initial).is_valid:
                        initial = earliest_finish_schedule(inst)
                except Exception:
                    initial = earliest_finish_schedule(inst)

            env = FJSPImprovementEnv(inst, initial_schedule=initial, max_steps=max_steps, patience=20)
            traj = randomized_first_improvement_rollout(
                env, max_steps=max_steps, rng=random.Random(seed + rollout_idx)
            )
            all_transitions.extend(traj)
            print(f"  rollout {rollout_idx}: {len(traj)} transitions, EF={max(o.end for o in initial)} -> best={env.best_makespan}")

    print(f"\nTotal transitions: {len(all_transitions)}")
    with open(output_path / "bc_transitions.pkl", "wb") as f:
        pickle.dump(all_transitions, f)

    # Also save metadata
    metadata = {
        "n_instances": len(instance_paths),
        "instance_paths": instance_paths,
        "n_rollouts": n_rollouts,
        "n_transitions": len(all_transitions),
        "seed": seed,
    }
    with open(output_path / "bc_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return all_transitions


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", nargs="+", default=["data/instances/brandimarte/mk01.txt"])
    parser.add_argument("--n-rollouts", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    generate_bc_dataset(
        instance_paths=args.instances,
        n_rollouts=args.n_rollouts,
        max_steps=args.max_steps,
        seed=args.seed,
    )
