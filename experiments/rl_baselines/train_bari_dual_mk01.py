"""Train BARI with dual-perspective encoder on MK01 (quick validation)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rl.agents.fjsp_l2s import train_fjsp_l2s

if __name__ == "__main__":
    net, history = train_fjsp_l2s(
        train_instances=["data/instances/brandimarte/mk01.txt"],
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
        entropy_coef=0.02,
        k_epochs=2,
        seed=0,
        eval_every=10,
        save_dir="data/results/bari_dual_mk01",
    )
    print(f"\nFinal best makespan: {history[-1].get('best_makespan', 'N/A')}")
