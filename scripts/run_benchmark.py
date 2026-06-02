from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from random import Random
from statistics import mean, pstdev
from time import perf_counter

import _bootstrap  # noqa: F401
from fjsp.env import FJSPDispatchEnv
from fjsp.scheduler.dispatch_rules import rollout_earliest_finish
from rl.agents import train_actor_critic, train_graph_actor_critic, train_reinforce


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_CSV = ROOT / "data" / "results" / "benchmark_mk01_mk05.csv"
DEFAULT_OUTPUT_JSON = ROOT / "data" / "results" / "benchmark_mk01_mk05.json"
METADATA_PATH = ROOT / "data" / "instances" / "instances.json"


def brandimarte_paths(start: int, count: int) -> list[Path]:
    base = ROOT / "data" / "instances" / "brandimarte"
    return [base / f"mk{idx:02d}.txt" for idx in range(start, start + count)]


def parse_instance_names(raw: str) -> list[Path]:
    base = ROOT / "data" / "instances" / "brandimarte"
    paths: list[Path] = []
    for item in raw.split(","):
        name = item.strip().lower()
        if not name:
            continue
        if name.startswith("mk"):
            paths.append(base / f"{name}.txt")
        else:
            paths.append(Path(name))
    return paths


def load_metadata() -> dict[str, dict[str, object]]:
    if not METADATA_PATH.exists():
        return {}
    payload = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return {str(item["name"]): item for item in payload}


def rollout_random(env: FJSPDispatchEnv, *, seed: int) -> int:
    rng = Random(seed)
    env.reset()
    while not env.done:
        env.step(env.sample_valid_action(rng))
    result = env.validate()
    if not result.is_valid:
        raise RuntimeError(f"Random rollout produced invalid schedule: {result.errors}")
    return result.makespan


def run_one(
    instance_path: Path,
    *,
    random_rollouts: int,
    episodes: int,
    seeds: list[int],
    metadata: dict[str, dict[str, object]],
    agent_name: str,
) -> dict[str, object]:
    env = FJSPDispatchEnv.from_file(instance_path)
    started = perf_counter()
    earliest = rollout_earliest_finish(env)
    random_values = [rollout_random(env, seed=seeds[0] + idx) for idx in range(random_rollouts)]
    meta = metadata.get(instance_path.stem, {})
    bounds = meta.get("bounds", {}) if isinstance(meta.get("bounds"), dict) else {}

    row: dict[str, object] = {
        "instance": instance_path.stem,
        "jobs": env.instance.job_count,
        "machines": env.instance.machine_count,
        "operations": env.instance.operation_count,
        "optimum": meta.get("optimum") or "",
        "lower_bound": bounds.get("lower", ""),
        "upper_bound": bounds.get("upper", ""),
        "earliest_finish": earliest,
        "random_mean": round(mean(random_values), 3),
        "random_best": min(random_values),
        "random_worst": max(random_values),
        "rl_best": "",
        "rl_mean": "",
        "rl_std": "",
        "rl_seed_results": [],
        "seconds": 0.0,
    }

    if episodes > 0:
        seed_results: list[dict[str, int]] = []
        for seed in seeds:
            if agent_name == "actor_critic":
                agent, history = train_actor_critic(env, episodes=episodes, seed=seed)
            elif agent_name == "graph_actor_critic":
                agent, history = train_graph_actor_critic(env, episodes=episodes, seed=seed)
            else:
                agent, history = train_reinforce(env, episodes=episodes, seed=seed)
            greedy_result = agent.rollout(env, greedy=True)
            seed_results.append(
                {
                    "seed": seed,
                    "greedy_makespan": greedy_result.makespan,
                    "best_seen": int(history[-1]["best_greedy_makespan"]),
                    "best_episode": int(history[-1]["best_episode"]),
                }
            )
        rl_values = [item["greedy_makespan"] for item in seed_results]
        row["rl_best"] = min(rl_values)
        row["rl_mean"] = round(mean(rl_values), 3)
        row["rl_std"] = round(pstdev(rl_values), 3) if len(rl_values) > 1 else 0.0
        row["rl_seed_results"] = seed_results
        row["best_method"] = "rl" if min(rl_values) <= min(earliest, min(random_values)) else "baseline"
    else:
        row["best_method"] = "baseline"

    row["seconds"] = round(perf_counter() - started, 3)
    return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    csv_rows: list[dict[str, object]] = []
    for row in rows:
        csv_row = dict(row)
        csv_row["rl_seed_results"] = json.dumps(csv_row["rl_seed_results"])
        csv_rows.append(csv_row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description="Run simple FJSP benchmark baselines.")
    parser.add_argument("--instances", default="", help="Comma-separated instance names or paths, e.g. mk11,mk12.")
    parser.add_argument("--brandimarte-start", type=int, default=1, help="First MK index when --instances is omitted.")
    parser.add_argument("--brandimarte-count", type=int, default=5, help="Run MK01..MK<count>.")
    parser.add_argument("--random-rollouts", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=50, help="REINFORCE episodes per instance; use 0 to skip RL.")
    parser.add_argument("--agent", choices=["reinforce", "actor_critic", "graph_actor_critic"], default="reinforce")
    parser.add_argument("--seed", type=int, default=0, help="Backward-compatible single-seed argument.")
    parser.add_argument("--seeds", default="", help="Comma-separated training seeds, e.g. 0,1,2.")
    parser.add_argument("--csv-out", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--json-out", default=str(DEFAULT_OUTPUT_JSON))
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    metadata = load_metadata()
    seeds = parse_seeds(args.seeds) if args.seeds else [args.seed]
    paths = parse_instance_names(args.instances) if args.instances else brandimarte_paths(
        args.brandimarte_start,
        args.brandimarte_count,
    )
    for path in paths:
        print(f"Running {path.stem}...")
        row = run_one(
            path,
            random_rollouts=args.random_rollouts,
            episodes=args.episodes,
            seeds=seeds,
            metadata=metadata,
            agent_name=args.agent,
        )
        rows.append(row)
        print(
            f"  earliest={row['earliest_finish']} random_mean={row['random_mean']} "
            f"rl_best={row['rl_best']} rl_mean={row['rl_mean']} seconds={row['seconds']}"
        )

    csv_path = Path(args.csv_out)
    json_path = Path(args.json_out)
    write_csv(csv_path, rows)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
