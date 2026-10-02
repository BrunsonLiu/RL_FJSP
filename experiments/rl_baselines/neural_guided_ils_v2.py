"""Neural-guided ILS using the full local_search.py neighborhoods.

This is the SOTA-capable version: it scores moves from reassign, swap_machine,
swap_same_machine, and swap_order_across, then only evaluates the top-k most
promising ones.  The best improving move among the top-k is accepted (best-
improvement within the neural short-list).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from fjsp.env.improvement_env import FJSPImprovementEnv  # noqa: F401  # breaks circular import
from fjsp.parser.fjs_parser import parse_fjs
from fjsp.scheduler.validator import ScheduledOperation, validate_schedule
from rl.models.neural_ils_utils import (
    load_net,
    neural_guided_ils_single_run,
)


def _schedule_to_json(schedule: list[ScheduledOperation]) -> dict[str, Any]:
    return {
        "operations": [
            {"job": op.job, "op": op.op, "machine": op.machine, "start": op.start, "end": op.end}
            for op in schedule
        ]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True)
    parser.add_argument("--model", default="data/results/move_rank_data/move_rank_net.pt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-iterations", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-candidates-per-type", type=int, default=200)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--perturb-n-swaps", type=int, default=5)
    parser.add_argument("--accept-worse-prob", type=float, default=0.0)
    parser.add_argument("--no-fallback-ils", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    instance = parse_fjs(args.instance)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_net(args.model, device)

    schedule, makespan, info = neural_guided_ils_single_run(
        instance,
        net,
        device,
        n_iterations=args.n_iterations,
        top_k=args.top_k,
        max_candidates_per_type=args.max_candidates_per_type,
        patience=args.patience,
        perturb_n_swaps=args.perturb_n_swaps,
        seed=args.seed,
        fallback_ils=not args.no_fallback_ils,
        accept_worse_prob=args.accept_worse_prob,
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
    print("History:", info)


if __name__ == "__main__":
    main()
