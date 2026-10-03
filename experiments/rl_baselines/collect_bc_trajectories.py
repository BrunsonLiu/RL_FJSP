"""Collect behavioral-cloning trajectories using a greedy one-step expert.

The expert operates inside the BARI environment: at each step it evaluates
all valid moves and picks the move that yields the largest immediate
makespan reduction. If no move improves, it picks the move that yields the
lowest makespan (even if equal). These trajectories are used to pre-train
the BARI policy before PPO fine-tuning.
"""
import sys
import pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fjsp.parser.fjs_parser import parse_fjs
from fjsp.scheduler.validator import earliest_finish_schedule
from fjsp.env.improvement_env import FJSPImprovementEnv



def greedy_rollout(env: FJSPImprovementEnv, max_steps: int = 50) -> tuple[list, int]:
    """Run one greedy episode and return (observations, best_makespan)."""
    obs = env.reset()
    observations = []
    best_ms = env.makespan
    step = 0

    while not env.done and step < max_steps:
        if not obs["valid_moves"]:
            break

        best_action = 0
        best_delta = -1e9
        best_new_ms = env.makespan

        for idx, move in enumerate(obs["valid_moves"]):
            # Simulate move by copying assignments
            new_assignments = env._apply_move(move)
            try:
                from fjsp.scheduler.local_search import _recompute
                new_schedule = _recompute(env.instance, new_assignments)
                from fjsp.scheduler.validator import validate_schedule
                val = validate_schedule(env.instance, new_schedule)
                if not val.is_valid:
                    continue
                new_ms = val.makespan
                delta = env.makespan - new_ms
                # Prefer larger improvement; tie-break by lower makespan
                if delta > best_delta or (delta == best_delta and new_ms < best_new_ms):
                    best_delta = delta
                    best_action = idx
                    best_new_ms = new_ms
            except Exception:
                continue

        observations.append({
            "graph": obs["graph"],
            "valid_moves": list(obs["valid_moves"]),
            "makespan": env.makespan,
            "action": best_action,
        })

        obs, _, done, info = env.step(best_action)
        step += 1
        best_ms = min(best_ms, info["best_makespan"])
        if done:
            break

    return observations, best_ms


def generate_diverse_initial(inst, rng):
    """Generate diverse initial solution for data collection."""
    from fjsp.env.dispatch_env import FJSPDispatchEnv
    from fjsp.scheduler.dispatch_rules import choose_spt, choose_earliest_finish
    from fjsp.scheduler.local_search import neh_construct, random_schedule

    strategy = rng.choice(["ef", "spt", "neh", "random", "perturb_ef"])
    if strategy == "ef":
        return earliest_finish_schedule(inst)
    elif strategy == "spt":
        env_d = FJSPDispatchEnv(inst)
        env_d.reset()
        while not env_d.done:
            env_d.step(choose_spt(env_d))
        return env_d.schedule
    elif strategy == "neh":
        return neh_construct(inst)
    elif strategy == "random":
        return random_schedule(inst, seed=rng.randint(0, 999999))
    else:  # perturb_ef
        base = earliest_finish_schedule(inst)
        assignments = [(o.job, o.op, o.machine) for o in base]
        for i in range(len(assignments)):
            if rng.random() < 0.15:
                job_idx, op_idx, _ = assignments[i]
                op = inst.jobs[job_idx].operations[op_idx]
                new_machine = rng.choice([opt.machine for opt in op.options])
                assignments[i] = (job_idx, op_idx, new_machine)
        from fjsp.scheduler.local_search import _recompute
        return _recompute(inst, assignments)


def main():
    instances = [
        f"data/instances/brandimarte/mk{i:02d}.txt" for i in range(1, 3)
    ]
    save_dir = Path("data/results/bc_trajectories")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "greedy_trajectories.pkl"

    # Load existing partial data if any
    all_trajectories = []
    if save_path.exists():
        with open(save_path, "rb") as f:
            all_trajectories = pickle.load(f)
        print(f"Loaded {len(all_trajectories)} existing trajectories")

    import random
    rng = random.Random(0)

    for inst_path in instances:
        inst = parse_fjs(inst_path)
        for trial in range(50):  # 50 diverse initial solutions per instance
            initial = generate_diverse_initial(inst, rng)
            env = FJSPImprovementEnv(
                inst,
                initial_schedule=initial,
                max_steps=30,
                patience=10,
                reward_shaping="dense",
            )
            traj, best_ms = greedy_rollout(env)
            init_ms = max(o.end for o in initial)
            print(f"{Path(inst_path).name} trial={trial}: init={init_ms}, greedy_best={best_ms}, steps={len(traj)}")
            all_trajectories.append({
                "instance": Path(inst_path).name,
                "trial": trial,
                "initial_makespan": init_ms,
                "best_makespan": best_ms,
                "trajectory": traj,
            })
            # Save incrementally every 10 trajectories
            if len(all_trajectories) % 10 == 0:
                with open(save_path, "wb") as f:
                    pickle.dump(all_trajectories, f)
                total_steps = sum(len(t["trajectory"]) for t in all_trajectories)
                print(f"  [checkpoint] saved {len(all_trajectories)} trajectories, {total_steps} steps")

    with open(save_path, "wb") as f:
        pickle.dump(all_trajectories, f)
    total_steps = sum(len(t["trajectory"]) for t in all_trajectories)
    print(f"\nSaved {len(all_trajectories)} trajectories, {total_steps} steps to {save_path}")


if __name__ == "__main__":
    main()
