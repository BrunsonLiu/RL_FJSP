"""Cross-instance training: train on MK01+MK02, evaluate on MK01-15.

This tests BARI's generalization ability: can a model trained on small
instances improve solutions on larger, unseen instances?
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rl.agents.fjsp_l2s import train_fjsp_l2s, evaluate_agent

TRAIN_INSTANCES = [
    "data/instances/brandimarte/mk01.txt",
    "data/instances/brandimarte/mk02.txt",
]

TEST_INSTANCES = [
    f"data/instances/brandimarte/mk{i:02d}.txt" for i in range(1, 16)
]

if __name__ == "__main__":
    save_dir = "data/results/bari_cross"
    net, history = train_fjsp_l2s(
        train_instances=TRAIN_INSTANCES,
        test_instances=TEST_INSTANCES,
        episodes=150,
        max_steps=30,
        patience=10,
        hidden_dim=64,
        num_mp_rounds=2,
        num_transformer_blocks=2,
        num_heads=8,
        ffn_dim=128,
        use_dual_perspective=True,
        lr=3e-4,
        entropy_coef=0.01,
        k_epochs=2,
        seed=0,
        eval_every=30,
        save_dir=save_dir,
    )

    # Final evaluation on all 15 instances
    from fjsp.parser.fjs_parser import parse_fjs
    test_insts = [parse_fjs(p) for p in TEST_INSTANCES]
    results = evaluate_agent(net, test_insts, max_steps=50, patience=15, n_trials=3)

    print("\n=== Cross-instance Generalization Results ===")
    print(f"{'Instance':<10} {'EF':>6} {'BARI':>6} {'Impr':>6} {'Lit':>6}")
    print("-" * 40)
    lit = {1:40,2:26,3:204,4:60,5:172,6:58,7:139,8:523,9:307,10:197,
           11:615,12:508,13:430,14:694,15:341}
    total_ef = 0
    total_bari = 0
    for i, r in enumerate(results, 1):
        ef = r["initial_makespan"]
        bari = r["best_makespan"]
        impr = ef - bari
        total_ef += ef
        total_bari += bari
        print(f"mk{i:02d}       {ef:>6} {bari:>6} {impr:>6} {lit[i]:>6}")
    print("-" * 40)
    print(f"{'Total':<10} {total_ef:>6} {total_bari:>6} {total_ef-total_bari:>6}")

    # Save results
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / "cross_instance_results.json", "w") as f:
        json.dump([{k:v for k,v in r.items() if k != "schedule"} for r in results], f, indent=2)
    print(f"\nResults saved to {save_path / 'cross_instance_results.json'}")
