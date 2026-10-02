"""Run BARI comparison with dual-perspective model on all 15 Brandimarte instances."""
import sys
import json
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from fjsp.parser.fjs_parser import parse_fjs
from fjsp.scheduler.validator import earliest_finish_schedule
from fjsp.env.improvement_env import FJSPImprovementEnv
from rl.models.fjsp_l2s import FJSPImproveNet
from rl.agents.fjsp_l2s import FJSPImproveAgent, evaluate_agent

MODEL_PATH = "data/results/bari_dual_mk01/fjsp_l2s_best.pt"
LIT = {1:40,2:26,3:204,4:60,5:172,6:58,7:139,8:523,9:307,10:197,
       11:615,12:508,13:430,14:694,15:341}

def main():
    # Load model
    net = FJSPImproveNet(hidden_dim=64, num_mp_rounds=2, ffn_dim=128, use_dual_perspective=True)
    net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    net.eval()
    print(f"Loaded model from {MODEL_PATH}")

    results = []
    print(f"\n{'Instance':<10} {'EF':>6} {'BARI':>6} {'Impr':>6} {'Lit':>6} {'Gap%':>6}")
    print("-" * 45)

    total_ef = 0
    total_bari = 0

    for i in range(1, 16):
        inst_path = f"data/instances/brandimarte/mk{i:02d}.txt"
        inst = parse_fjs(inst_path)
        initial = earliest_finish_schedule(inst)
        ef_ms = max(o.end for o in initial)

        t0 = time.time()
        eval_results = evaluate_agent(net, [inst], max_steps=50, patience=15, n_trials=3)
        dt = time.time() - t0

        bari_ms = eval_results[0]["best_makespan"]
        impr = ef_ms - bari_ms
        gap = (bari_ms - LIT[i]) / LIT[i] * 100
        total_ef += ef_ms
        total_bari += bari_ms

        print(f"mk{i:02d}       {ef_ms:>6} {bari_ms:>6} {impr:>6} {LIT[i]:>6} {gap:>+5.1f}% ({dt:.1f}s)")
        results.append({
            "instance": f"mk{i:02d}",
            "ef_makespan": ef_ms,
            "bari_makespan": bari_ms,
            "improvement": impr,
            "literature": LIT[i],
            "gap_pct": gap,
        })

    print("-" * 45)
    mean_gap = sum(r["gap_pct"] for r in results) / len(results)
    print(f"{'Total':<10} {total_ef:>6} {total_bari:>6} {total_ef-total_bari:>6}")
    print(f"{'Mean gap':>40} {mean_gap:>+5.1f}%")

    # Save
    save_path = Path("data/results/bari_dual_comparison")
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {save_path / 'comparison_results.json'}")

if __name__ == "__main__":
    main()
