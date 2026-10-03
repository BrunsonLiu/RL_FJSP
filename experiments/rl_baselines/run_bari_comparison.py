"""BARI comparison experiments: BARI vs EF baseline vs ILS vs SA.

Runs all methods on all 15 Brandimarte instances and produces
a comparison table suitable for paper inclusion.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fjsp.parser.fjs_parser import parse_fjs
from fjsp.scheduler.validator import earliest_finish_schedule, validate_schedule
from fjsp.scheduler.local_search import iterated_local_search, simulated_annealing
from fjsp.env.improvement_env import FJSPImprovementEnv
from rl.agents.fjsp_l2s import FJSPImproveAgent, evaluate_agent
from rl.models.fjsp_l2s import FJSPImproveNet

BRANDIMARTE_DIR = Path("data/instances/brandimarte")
RESULTS_DIR = Path("data/results/bari_comparison")
LITERATURE_BEST = {
    "mk01": 40, "mk02": 26, "mk03": 204, "mk04": 60, "mk05": 172,
    "mk06": 58, "mk07": 139, "mk08": 523, "mk09": 307, "mk10": 197,
    "mk11": 615, "mk12": 508, "mk13": 430, "mk14": 694, "mk15": 341,
}


def run_ef_baseline(instance_path: str) -> dict:
    """Run earliest-finish baseline."""
    inst = parse_fjs(instance_path)
    t0 = time.perf_counter()
    schedule = earliest_finish_schedule(inst)
    elapsed = time.perf_counter() - t0
    result = validate_schedule(inst, schedule)
    return {
        "method": "EF",
        "makespan": result.makespan,
        "time": elapsed,
    }


def run_ils(instance_path: str, n_iters: int = 20) -> dict:
    """Run ILS from EF initial solution."""
    inst = parse_fjs(instance_path)
    schedule = earliest_finish_schedule(inst)
    t0 = time.perf_counter()
    improved, ms = iterated_local_search(inst, schedule, n_iterations=n_iters)
    elapsed = time.perf_counter() - t0
    return {
        "method": "ILS",
        "makespan": ms,
        "time": elapsed,
        "n_iters": n_iters,
    }


def run_sa(instance_path: str, n_iters: int = 20) -> dict:
    """Run SA from EF initial solution."""
    inst = parse_fjs(instance_path)
    schedule = earliest_finish_schedule(inst)
    t0 = time.perf_counter()
    improved, ms = simulated_annealing(inst, schedule, max_total_iterations=n_iters * 100)
    elapsed = time.perf_counter() - t0
    return {
        "method": "SA",
        "makespan": ms,
        "time": elapsed,
        "n_iters": n_iters,
    }


def run_bari(instance_path: str, model_path: str, n_trials: int = 3) -> dict:
    """Run BARI with trained model."""
    inst = parse_fjs(instance_path)
    net = FJSPImproveNet(
        hidden_dim=128,
        num_mp_rounds=3,
        num_transformer_blocks=2,
        num_heads=8,
        ffn_dim=256,
    )
    checkpoint = torch_load(model_path)
    net.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)

    t0 = time.perf_counter()
    results = evaluate_agent(
        net, [inst],
        max_steps=100,
        patience=20,
        n_trials=n_trials,
    )
    elapsed = time.perf_counter() - t0

    r = results[0] if results else {}
    return {
        "method": "BARI",
        "makespan": r.get("best_makespan", "N/A"),
        "avg_makespan": r.get("avg_makespan", "N/A"),
        "std_makespan": r.get("std_makespan", 0),
        "time": elapsed,
        "n_trials": n_trials,
        "initial_makespan": r.get("initial_makespan", "N/A"),
        "improvement": r.get("improvement", 0),
        "improvement_pct": r.get("improvement_pct", 0),
    }


def torch_load(model_path: str):
    """Load torch model."""
    import torch
    return torch.load(model_path, map_location="cpu", weights_only=False)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="data/results/bari_cross/fjsp_l2s_best.pt")
    parser.add_argument("--save-dir", type=str, default="data/results/bari_comparison")
    parser.add_argument("--ils-iters", type=int, default=20)
    parser.add_argument("--n-trials", type=int, default=3)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    print(f"{'Instance':<8} {'EF':>6} {'ILS':>6} {'SA':>6} {'BARI':>6} {'BARI_avg':>8} {'Lit':>6} {'BARI_gap':>8}")
    print("-" * 70)

    for i in range(1, 16):
        name = f"mk{i:02d}"
        instance_path = str(BRANDIMARTE_DIR / f"{name}.txt")
        lit = LITERATURE_BEST[name]

        # EF baseline
        ef_result = run_ef_baseline(instance_path)

        # ILS
        ils_result = run_ils(instance_path, n_iters=args.ils_iters)

        # SA
        sa_result = run_sa(instance_path, n_iters=args.ils_iters)

        # BARI
        model_path = args.model
        if Path(model_path).exists():
            bari_result = run_bari(instance_path, model_path, n_trials=args.n_trials)
        else:
            bari_result = {"method": "BARI", "makespan": "N/A", "avg_makespan": "N/A", "time": 0}

        bari_ms = bari_result.get("makespan", "N/A")
        bari_avg = bari_result.get("avg_makespan", "N/A")
        if isinstance(bari_ms, int):
            gap = 100 * (bari_ms - lit) / lit
            gap_str = f"{gap:+.1f}%"
        else:
            gap_str = "N/A"

        print(f"{name:<8} {ef_result['makespan']:>6} {ils_result['makespan']:>6} {sa_result['makespan']:>6} {str(bari_ms):>6} {str(bari_avg):>8} {lit:>6} {gap_str:>8}")

        all_results.append({
            "instance": name,
            "literature_best": lit,
            "ef": ef_result,
            "ils": ils_result,
            "sa": sa_result,
            "bari": bari_result,
        })

    # Save results
    with open(save_dir / "comparison_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {save_dir / 'comparison_results.json'}")

    # Summary statistics
    ef_makespans = [r["ef"]["makespan"] for r in all_results]
    ils_makespans = [r["ils"]["makespan"] for r in all_results]
    sa_makespans = [r["sa"]["makespan"] for r in all_results]
    bari_makespans = [r["bari"]["makespan"] for r in all_results if isinstance(r["bari"]["makespan"], int)]
    lit_makespans = [r["literature_best"] for r in all_results]

    print(f"\n{'Method':<10} {'Mean':>8} {'Mean Gap%':>10}")
    print("-" * 30)
    for name, makespans in [("EF", ef_makespans), ("ILS", ils_makespans), ("SA", sa_makespans), ("BARI", bari_makespans)]:
        if makespans:
            mean_ms = sum(makespans) / len(makespans)
            mean_gap = sum(100 * (m - l) / l for m, l in zip(makespans, lit_makespans)) / len(makespans)
            print(f"{name:<10} {mean_ms:>8.1f} {mean_gap:>+9.1f}%")


if __name__ == "__main__":
    main()
