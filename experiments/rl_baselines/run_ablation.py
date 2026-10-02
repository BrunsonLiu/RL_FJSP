"""Ablation experiments for BARI.

Tests three ablations:
1. Dual-perspective vs single-perspective encoder (architectural)
2. Bottleneck score vs binary CP mask (feature)
3. Full BARI vs no-swap (neighborhood)

Each ablation trains on MK01 for 100 episodes and evaluates on MK01-05.
"""
import sys
import json
import copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from rl.agents.fjsp_l2s import train_fjsp_l2s, evaluate_agent
from fjsp.parser.fjs_parser import parse_fjs
from fjsp.graph.solution_graph import IMP_OP_FEATURE_DIM

EVAL_INSTANCES = [
    f"data/instances/brandimarte/mk{i:02d}.txt" for i in range(1, 6)
]

def run_ablation(name, use_dual, zero_bottleneck=False, episodes=50):
    """Run one ablation configuration."""
    print(f"\n{'='*60}")
    print(f"Ablation: {name}")
    print(f"  use_dual_perspective={use_dual}, zero_bottleneck={zero_bottleneck}")
    print(f"{'='*60}")

    save_dir = f"data/results/ablation_{name}"

    net, history = train_fjsp_l2s(
        train_instances=["data/instances/brandimarte/mk01.txt"],
        episodes=episodes,
        max_steps=30,
        patience=10,
        hidden_dim=64,
        num_mp_rounds=2,
        num_transformer_blocks=2,
        num_heads=8,
        ffn_dim=128,
        use_dual_perspective=use_dual,
        lr=3e-4,
        entropy_coef=0.01,
        k_epochs=2,
        seed=0,
        eval_every=20,
        save_dir=save_dir,
    )

    # If zero_bottleneck, modify the graph builder to zero out feature 16
    if zero_bottleneck:
        # Monkey-patch: wrap build_solution_graph to zero bottleneck feature
        import fjsp.env.improvement_env as env_mod
        original_build = env_mod.build_solution_graph
        def patched_build(instance, schedule, **kwargs):
            graph = original_build(instance, schedule, **kwargs)
            # Zero out bottleneck_score (feature 16)
            op_features = graph.op_features.clone()
            op_features[:, 16] = 0.0
            from fjsp.graph.solution_graph import SolutionGraph
            return SolutionGraph(
                operation_refs=graph.operation_refs,
                op_features=op_features,
                machine_features=graph.machine_features,
                precedence_edges=graph.precedence_edges,
                machine_seq_edges=graph.machine_seq_edges,
                eligibility_edges=graph.eligibility_edges,
                eligibility_durations=graph.eligibility_durations,
                critical_op_indices=graph.critical_op_indices,
                global_features=graph.global_features,
            )
        env_mod.build_solution_graph = patched_build

    # Evaluate
    test_insts = [parse_fjs(p) for p in EVAL_INSTANCES]
    results = evaluate_agent(net, test_insts, max_steps=50, patience=15, n_trials=3)

    # Restore original
    if zero_bottleneck:
        env_mod.build_solution_graph = original_build

    print(f"\n--- {name} Results ---")
    print(f"{'Instance':<10} {'EF':>6} {'BARI':>6} {'Impr':>6}")
    print("-" * 32)
    for i, r in enumerate(results, 1):
        ef = r["initial_makespan"]
        bari = r["best_makespan"]
        print(f"mk{i:02d}       {ef:>6} {bari:>6} {ef-bari:>6}")

    # Save
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / "ablation_results.json", "w") as f:
        json.dump([{k:v for k,v in r.items() if k != "schedule"} for r in results], f, indent=2)

    return results


if __name__ == "__main__":
    all_results = {}

    # 1. Full BARI (dual + bottleneck)
    all_results["bari_full"] = run_ablation("bari_full", use_dual=True, zero_bottleneck=False)

    # 2. Single-perspective (no dual encoder)
    all_results["single_perspective"] = run_ablation("single_perspective", use_dual=False, zero_bottleneck=False)

    # 3. Binary CP mask (no bottleneck score)
    all_results["binary_cp"] = run_ablation("binary_cp", use_dual=True, zero_bottleneck=True)

    # Summary
    print(f"\n{'='*60}")
    print("ABLATION SUMMARY (MK01-05 mean makespan)")
    print(f"{'='*60}")
    print(f"{'Variant':<25} {'MK01':>6} {'MK02':>6} {'MK03':>6} {'MK04':>6} {'MK05':>6} {'Mean':>6}")
    print("-" * 60)
    for name, results in all_results.items():
        makespans = [r["best_makespan"] for r in results]
        mean_ms = sum(makespans) / len(makespans)
        row = f"{name:<25}"
        for ms in makespans:
            row += f" {ms:>6}"
        row += f" {mean_ms:>6.1f}"
        print(row)

    with open("data/results/ablation_summary.json", "w") as f:
        json.dump({k: [{kk:vv for kk,vv in r.items() if kk != "schedule"} for r in v]
                   for k,v in all_results.items()}, f, indent=2)
    print("\nSaved to data/results/ablation_summary.json")
