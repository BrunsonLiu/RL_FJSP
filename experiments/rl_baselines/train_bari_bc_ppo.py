"""Train BARI with BC pretraining + PPO finetuning.

1. Load BC-pretrained policy from data/results/bc_data/bc_pretrained.pt
2. Finetune with PPO on MK01 (or multiple instances)
3. Evaluate on all Brandimarte instances
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rl.agents.fjsp_l2s import train_fjsp_l2s, evaluate_agent

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", default="data/instances/brandimarte/mk01.txt")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--k-epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-model", default="data/results/bc_data/bc_pretrained.pt")
    parser.add_argument("--save-dir", default="data/results/bari_bc_ppo")
    args = parser.parse_args()

    net, history = train_fjsp_l2s(
        train_instances=[args.instance],
        episodes=args.episodes,
        max_steps=args.max_steps,
        patience=args.patience,
        hidden_dim=args.hidden_dim,
        num_mp_rounds=2,
        num_transformer_blocks=2,
        num_heads=8,
        ffn_dim=128,
        use_dual_perspective=True,
        lr=args.lr,
        entropy_coef=args.entropy_coef,
        k_epochs=args.k_epochs,
        seed=args.seed,
        eval_every=20,
        save_dir=args.save_dir,
        init_model=args.init_model,
    )

    print(f"\nFinal best makespan: {history[-1].get('best_makespan', 'N/A')}")
