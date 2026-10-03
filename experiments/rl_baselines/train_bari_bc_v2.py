"""Pre-train BARI with behavioral cloning (works with partial data)."""
import sys
import pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from fjsp.parser.fjs_parser import parse_fjs
from rl.models.fjsp_l2s import FJSPImproveNet
from rl.agents.fjsp_l2s import FJSPImproveAgent, evaluate_agent


def main():
    traj_path = Path("data/results/bc_trajectories/greedy_trajectories.pkl")
    with open(traj_path, "rb") as f:
        trajectories = pickle.load(f)

    print(f"Loaded {len(trajectories)} expert trajectories")
    total_steps = sum(len(t["trajectory"]) for t in trajectories)
    print(f"Total expert steps: {total_steps}")

    # Build net
    net = FJSPImproveNet(
        hidden_dim=64,
        num_mp_rounds=2,
        num_transformer_blocks=2,
        num_heads=8,
        ffn_dim=128,
        use_dual_perspective=True,
    )
    agent = FJSPImproveAgent(net, lr=1e-3)

    # BC pretrain
    print("\n=== BC Pretraining ===")
    losses = agent.bc_pretrain(trajectories, epochs=20, batch_size=64, lr=1e-3)
    print(f"Final BC loss: {losses[-1]:.4f}")

    # Save BC model
    save_dir = Path("data/results/bari_bc")
    save_dir.mkdir(parents=True, exist_ok=True)
    bc_path = save_dir / "bc_pretrained.pt"
    torch.save(net.state_dict(), bc_path)
    print(f"Saved BC model to {bc_path}")

    # Evaluate BC model
    print("\n=== BC Evaluation (greedy deterministic) ===")
    instances = [parse_fjs(f"data/instances/brandimarte/mk{i:02d}.txt") for i in range(1, 6)]
    results = evaluate_agent(net, instances, max_steps=30, patience=10, n_trials=1, deterministic=True)

    print(f"{'Instance':<10} {'EF':>6} {'BC':>6} {'Impr':>6}")
    print("-" * 32)
    for i, r in enumerate(results, 1):
        ef = r["initial_makespan"]
        bc = r["best_makespan"]
        print(f"mk{i:02d}       {ef:>6} {bc:>6} {ef-bc:>6}")


if __name__ == "__main__":
    main()
