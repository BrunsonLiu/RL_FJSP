"""Run a 4-agent x N-instance Brandimarte baseline matrix in a single process.

For each instance we compute:
- earliest-finish heuristic
- random rollout best
- four RL policies (REINFORCE, two-stage AC, graph AC, graph PPO) trained with
  a per-agent episode budget, then greedy-evaluated.

The output is one row per (instance, agent) plus a per-instance summary block.
"""
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
from rl.agents import train_actor_critic, train_graph_actor_critic, train_graph_ppo, train_reinforce


ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = ROOT / "data" / "instances" / "instances.json"

DEFAULT_AGENT_EPISODES = {
    "reinforce": 200,
    "actor_critic": 200,
    "graph_actor_critic": 200,
    "graph_ppo": 100,
}

AGENT_REGISTRY = {
    "reinforce": train_reinforce,
    "actor_critic": train_actor_critic,
    "graph_actor_critic": train_graph_actor_critic,
    "graph_ppo": train_graph_ppo,
}


def load_metadata() -> dict[str, dict[str, object]]:
    if not METADATA_PATH.exists():
        return {}
    payload = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return {str(item["name"]): item for item in payload}


def brandimarte_paths(start: int, count: int) -> list[Path]:
    base = ROOT / "data" / "instances" / "brandimarte"
    return [base / f"mk{idx:02d}.txt" for idx in range(start, start + count)]


def rollout_random(env: FJSPDispatchEnv, *, seed: int) -> int:
    rng = Random(seed)
    env.reset()
    while not env.done:
        env.step(env.sample_valid_action(rng))
    result = env.validate()
    if not result.is_valid:
        raise RuntimeError(f"Random rollout produced invalid schedule: {result.errors}")
    return result.makespan


def run_agent(env: FJSPDispatchEnv, agent_name: str, episodes: int, seed: int) -> dict[str, object]:
    train_fn = AGENT_REGISTRY[agent_name]
    started = perf_counter()
    agent, history = train_fn(env, episodes=episodes, seed=seed)
    result = agent.rollout(env, greedy=True)
    seconds = round(perf_counter() - started, 3)
    best_seen = int(history[-1]["best_greedy_makespan"]) if history else result.makespan
    best_episode = int(history[-1]["best_episode"]) if history else 0
    return {
        "agent": agent_name,
        "episodes": episodes,
        "greedy_makespan": int(result.makespan),
        "best_seen": best_seen,
        "best_episode": best_episode,
        "seconds": seconds,
    }


def build_summary(per_agent: list[dict[str, object]], earliest: int, random_best: int) -> dict[str, object]:
    by_agent = {row["agent"]: int(row["greedy_makespan"]) for row in per_agent}
    candidates = {"earliest_finish": earliest, "random": random_best, **by_agent}
    best_method = min(candidates, key=candidates.get)
    return {
        "best_method": best_method,
        "best_value": int(candidates[best_method]),
        "per_agent": by_agent,
    }


def run_instance(
    instance_path: Path,
    *,
    seed: int,
    random_rollouts: int,
    metadata: dict[str, dict[str, object]],
    agent_episodes: dict[str, int],
) -> dict[str, object]:
    env = FJSPDispatchEnv.from_file(instance_path)
    meta = metadata.get(instance_path.stem, {})
    bounds = meta.get("bounds", {}) if isinstance(meta.get("bounds"), dict) else {}

    earliest = rollout_earliest_finish(env)
    random_values = [rollout_random(env, seed=seed * 1000 + idx) for idx in range(random_rollouts)]
    random_best = min(random_values)
    random_mean = round(mean(random_values), 3)

    per_agent: list[dict[str, object]] = []
    for agent_name, episodes in agent_episodes.items():
        if episodes <= 0:
            continue
        row = run_agent(env, agent_name, episodes=episodes, seed=seed)
        per_agent.append(row)
        print(
            f"  [{instance_path.stem}] {agent_name}: greedy={row['greedy_makespan']} "
            f"best_seen={row['best_seen']} sec={row['seconds']}"
        )

    summary = build_summary(per_agent, earliest, random_best)
    return {
        "instance": instance_path.stem,
        "jobs": env.instance.job_count,
        "machines": env.instance.machine_count,
        "operations": env.instance.operation_count,
        "optimum": meta.get("optimum") or "",
        "lower_bound": bounds.get("lower", ""),
        "upper_bound": bounds.get("upper", ""),
        "earliest_finish": earliest,
        "random_best": random_best,
        "random_mean": random_mean,
        "per_agent": per_agent,
        "best_method": summary["best_method"],
        "best_value": summary["best_value"],
    }


def write_csv(path: Path, summaries: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "instance",
        "jobs",
        "machines",
        "operations",
        "optimum",
        "earliest_finish",
        "random_best",
        "random_mean",
        "reinforce",
        "actor_critic",
        "graph_actor_critic",
        "graph_ppo",
        "best_method",
        "best_value",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for s in summaries:
            per_agent = {row["agent"]: row["greedy_makespan"] for row in s["per_agent"]}
            row = {
                "instance": s["instance"],
                "jobs": s["jobs"],
                "machines": s["machines"],
                "operations": s["operations"],
                "optimum": s["optimum"],
                "earliest_finish": s["earliest_finish"],
                "random_best": s["random_best"],
                "random_mean": s["random_mean"],
                "reinforce": per_agent.get("reinforce", ""),
                "actor_critic": per_agent.get("actor_critic", ""),
                "graph_actor_critic": per_agent.get("graph_actor_critic", ""),
                "graph_ppo": per_agent.get("graph_ppo", ""),
                "best_method": s["best_method"],
                "best_value": s["best_value"],
            }
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Full Brandimarte 4-agent baseline matrix.")
    parser.add_argument("--brandimarte-start", type=int, default=1)
    parser.add_argument("--brandimarte-count", type=int, default=10)
    parser.add_argument("--random-rollouts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes-reinforce", type=int, default=DEFAULT_AGENT_EPISODES["reinforce"])
    parser.add_argument("--episodes-actor-critic", type=int, default=DEFAULT_AGENT_EPISODES["actor_critic"])
    parser.add_argument("--episodes-graph-actor-critic", type=int, default=DEFAULT_AGENT_EPISODES["graph_actor_critic"])
    parser.add_argument("--episodes-graph-ppo", type=int, default=DEFAULT_AGENT_EPISODES["graph_ppo"])
    parser.add_argument("--csv-out", default=str(ROOT / "data" / "results" / "baseline_brandimarte_matrix.csv"))
    parser.add_argument("--json-out", default=str(ROOT / "data" / "results" / "baseline_brandimarte_matrix.json"))
    args = parser.parse_args()

    agent_episodes = {
        "reinforce": args.episodes_reinforce,
        "actor_critic": args.episodes_actor_critic,
        "graph_actor_critic": args.episodes_graph_actor_critic,
        "graph_ppo": args.episodes_graph_ppo,
    }
    metadata = load_metadata()
    paths = brandimarte_paths(args.brandimarte_start, args.brandimarte_count)
    print(
        f"Running {len(paths)} instances x {len(agent_episodes)} agents "
        f"with episodes {agent_episodes} (seed={args.seed})"
    )

    summaries: list[dict[str, object]] = []
    for path in paths:
        print(f"=== {path.stem} ===")
        s = run_instance(
            path,
            seed=args.seed,
            random_rollouts=args.random_rollouts,
            metadata=metadata,
            agent_episodes=agent_episodes,
        )
        summaries.append(s)

    write_csv(Path(args.csv_out), summaries)
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Wrote {args.csv_out}")
    print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
