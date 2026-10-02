"""Evaluate FJSP-L2S agent on Brandimarte instances.

Supports two modes:
1. Per-instance evaluation (one model per instance)
2. Cross-instance evaluation (one model trained on MK01-05, tested on all)

Usage examples
--------------
# Per-instance: evaluate a model trained on mk01
python -m rl.evaluate_fjsp_l2s --model data/results/fjsp_l2s_best.pt --instances data/instances/brandimarte/mk01.txt

# Cross-instance: evaluate a model trained on MK01-05, tested on MK06-15
python -m rl.evaluate_fjsp_l2s --model data/results/cross_instance/fjsp_l2s_best.pt --instances data/instances/brandimarte/mk06.txt data/instances/brandimarte/mk07.txt data/instances/brandimarte/mk08.txt --max-steps 100

# Full benchmark: all 15 instances
python -m rl.evaluate_fjsp_l2s --model data/results/cross_instance/fjsp_l2s_best.pt --instances data/instances/brandimarte/mk*.txt --max-steps 100 --n-trials 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from fjsp.env.improvement_env import FJSPImprovementEnv
from fjsp.parser.fjs_parser import parse_fjs
from fjsp.scheduler.validator import earliest_finish_schedule, schedule_to_dict
from rl.agents.fjsp_l2s import FJSPImproveAgent, evaluate_agent
from rl.models.fjsp_l2s import FJSPImproveNet


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FJSP-L2S agent")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model .pt file")
    parser.add_argument("--instances", nargs="+", required=True, help="Instance file paths")
    parser.add_argument("--max-steps", type=int, default=100, help="Max improvement steps per episode")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--n-trials", type=int, default=3, help="Number of evaluation trials per instance")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-mp-rounds", type=int, default=3)
    parser.add_argument("--num-transformer-blocks", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=256)
    parser.add_argument("--save-dir", type=str, default=None, help="Save schedules to this directory")
    args = parser.parse_args()

    # Create network and load weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = FJSPImproveNet(
        hidden_dim=args.hidden_dim,
        num_mp_rounds=args.num_mp_rounds,
        num_transformer_blocks=args.num_transformer_blocks,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
    ).to(device)

    state_dict = torch.load(args.model, map_location=device, weights_only=True)
    net.load_state_dict(state_dict)
    print(f"Loaded model from {args.model}")
    print(f"Network parameters: {sum(p.numel() for p in net.parameters()):,}")

    # Expand glob patterns
    import glob
    instance_paths = []
    for p in args.instances:
        expanded = glob.glob(p)
        if expanded:
            instance_paths.extend(sorted(expanded))
        else:
            instance_paths.append(p)

    # Evaluate
    print(f"\nEvaluating on {len(instance_paths)} instances, {args.n_trials} trials each")
    print(f"{'Instance':<12} {'Jobs':>4} {'Machs':>5} {'Greedy':>6} {'Best':>6} {'Improv':>7} {'Time(s)':>8}")
    print("-" * 55)

    all_results = []
    total_start = time.time()

    for path in instance_paths:
        inst = parse_fjs(path)
        name = Path(path).stem

        initial = earliest_finish_schedule(inst)
        initial_ms = max(o.end for o in initial)

        t0 = time.time()
        results = evaluate_agent(
            net, [inst],
            max_steps=args.max_steps,
            patience=args.patience,
            n_trials=args.n_trials,
        )
        elapsed = time.time() - t0

        r = results[0]
        improvement = initial_ms - r["best_makespan"]
        print(f"{name:<12} {inst.job_count:>4} {inst.machine_count:>5} {initial_ms:>6} {r['best_makespan']:>6} {improvement:>7} {elapsed:>8.1f}")

        all_results.append({
            "instance": name,
            "jobs": inst.job_count,
            "machines": inst.machine_count,
            "operations": inst.operation_count,
            "greedy_makespan": initial_ms,
            "best_makespan": r["best_makespan"],
            "improvement": improvement,
            "time_seconds": elapsed,
        })

        # Save schedule
        if args.save_dir and r.get("schedule"):
            save_path = Path(args.save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            with open(save_path / f"{name}_l2s_schedule.json", "w") as f:
                json.dump(schedule_to_dict(r["schedule"]), f, indent=2)

    total_time = time.time() - total_start

    # Summary
    print("-" * 55)
    total_improvement = sum(r["improvement"] for r in all_results)
    avg_improvement = total_improvement / len(all_results) if all_results else 0
    print(f"Total improvement: {total_improvement} | Avg: {avg_improvement:.1f} | Time: {total_time:.1f}s")

    # Save results
    if args.save_dir:
        save_path = Path(args.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path / "evaluation_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {save_path / 'evaluation_results.json'}")


if __name__ == "__main__":
    main()
