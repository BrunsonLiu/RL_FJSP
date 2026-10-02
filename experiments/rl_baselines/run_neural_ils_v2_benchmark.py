"""Benchmark Neural-guided ILS v2 against SOTA reference on Brandimarte."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from fjsp.env.improvement_env import FJSPImprovementEnv  # noqa: F401  # breaks circular import
from fjsp.parser.fjs_parser import parse_fjs
from fjsp.scheduler.validator import validate_schedule
from rl.models.neural_ils_utils import (
    load_net,
    neural_guided_ils_single_run,
)


SOTA_PATH = Path("data/results/sota_final.json")
SOTA_BEST = {}
if SOTA_PATH.exists():
    with open(SOTA_PATH) as f:
        for item in json.load(f):
            SOTA_BEST[item["instance"]] = item["final_best"]

LITERATURE_BEST = {
    "mk01": 40, "mk02": 26, "mk03": 204, "mk04": 60, "mk05": 172,
    "mk06": 58, "mk07": 139, "mk08": 523, "mk09": 307, "mk10": 197,
    "mk11": 615, "mk12": 508, "mk13": 430, "mk14": 694, "mk15": 341,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", nargs="+", default=[f"data/instances/brandimarte/mk{i:02d}.txt" for i in range(1, 16)])
    parser.add_argument("--model", default="data/results/move_rank_data/move_rank_net.pt")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--n-iterations", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-candidates-per-type", type=int, default=200)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--perturb-n-swaps", type=int, default=5)
    parser.add_argument("--accept-worse-prob", type=float, default=0.0)
    parser.add_argument("--no-fallback-ils", action="store_true")
    parser.add_argument("--output", default="data/results/neural_ils_v2_benchmark.json")
    args = parser.parse_args()

    seeds = [int(x) for x in args.seeds.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_net(args.model, device)

    results = []
    for inst_path in args.instances:
        print(f"Running {inst_path}...")
        instance = parse_fjs(inst_path)
        name = Path(inst_path).stem

        best_ms = float("inf")
        best_sched = None
        for seed in seeds:
            schedule, ms, _ = neural_guided_ils_single_run(
                instance,
                net,
                device,
                seed=seed,
                n_iterations=args.n_iterations,
                top_k=args.top_k,
                max_candidates_per_type=args.max_candidates_per_type,
                patience=args.patience,
                perturb_n_swaps=args.perturb_n_swaps,
                fallback_ils=not args.no_fallback_ils,
                accept_worse_prob=args.accept_worse_prob,
            )
            if ms < best_ms:
                best_ms = ms
                best_sched = schedule

        validation = validate_schedule(instance, best_sched)
        lit = LITERATURE_BEST.get(name)
        sota = SOTA_BEST.get(name)
        res = {
            "instance": name,
            "neural_ils_v2_makespan": best_ms,
            "neural_ils_v2_gap": None if lit is None else round(100 * (best_ms - lit) / lit, 2),
            "sota_makespan": sota,
            "sota_gap": None if lit is None or sota is None else round(100 * (sota - lit) / lit, 2),
            "literature_best": lit,
            "valid": validation.is_valid,
        }
        print(json.dumps(res, indent=2))
        results.append(res)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
