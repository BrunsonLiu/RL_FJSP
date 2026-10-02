"""Train FJSP-L2S: Learning to Improve for FJSP.

Usage examples
--------------
# Single instance training (baseline)
python -m rl.train_fjsp_l2s --instances data/instances/brandimarte/mk01.txt --episodes 200

# Cross-instance training (generalization)
python -m rl.train_fjsp_l2s --instances data/instances/brandimarte/mk01.txt data/instances/brandimarte/mk02.txt data/instances/brandimarte/mk03.txt --episodes 500

# Train on small, test on large (zero-shot generalization)
python -m rl.train_fjsp_l2s --train-instances data/instances/brandimarte/mk01.txt data/instances/brandimarte/mk02.txt data/instances/brandimarte/mk03.txt data/instances/brandimarte/mk04.txt data/instances/brandimarte/mk05.txt --test-instances data/instances/brandimarte/mk06.txt data/instances/brandimarte/mk07.txt --episodes 500
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

from fjsp.parser.fjs_parser import parse_fjs
from rl.agents.fjsp_l2s import train_fjsp_l2s


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FJSP-L2S agent")
    parser.add_argument(
        "--instances", nargs="+", default=None,
        help="Instance file paths for training (and testing if --test-instances not given)",
    )
    parser.add_argument(
        "--train-instances", nargs="+", default=None,
        help="Training instance paths (for cross-instance generalization)",
    )
    parser.add_argument(
        "--test-instances", nargs="+", default=None,
        help="Test instance paths (for zero-shot evaluation)",
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=50, help="Max improvement steps per episode")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-mp-rounds", type=int, default=3)
    parser.add_argument("--num-transformer-blocks", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=256)
    parser.add_argument(
        "--no-dual-perspective", action="store_true",
        help="Disable dual-perspective encoder (use single-perspective for ablation)",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--k-epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--save-dir", type=str, default="data/results")

    args = parser.parse_args()

    # Resolve instances
    train_paths = args.train_instances or args.instances
    test_paths = args.test_instances

    if not train_paths:
        # Default: use mk01
        default = Path("data/instances/brandimarte/mk01.txt")
        if default.exists():
            train_paths = [str(default)]
        else:
            print("No instances specified and default not found. Use --instances or --train-instances.")
            sys.exit(1)

    # Expand glob patterns
    def expand_paths(paths):
        result = []
        for p in paths:
            expanded = glob.glob(p)
            if expanded:
                result.extend(sorted(expanded))
            else:
                result.append(p)
        return result

    train_paths = expand_paths(train_paths)
    test_paths = expand_paths(test_paths) if test_paths else None

    print(f"Training instances: {len(train_paths)}")
    for p in train_paths:
        inst = parse_fjs(p)
        print(f"  {Path(p).name}: jobs={inst.job_count}, machines={inst.machine_count}, ops={inst.operation_count}")

    if test_paths:
        print(f"Test instances: {len(test_paths)}")
        for p in test_paths:
            inst = parse_fjs(p)
            print(f"  {Path(p).name}: jobs={inst.job_count}, machines={inst.machine_count}, ops={inst.operation_count}")

    # Train
    net, history = train_fjsp_l2s(
        train_instances=train_paths,
        test_instances=test_paths,
        episodes=args.episodes,
        max_steps=args.max_steps,
        patience=args.patience,
        hidden_dim=args.hidden_dim,
        num_mp_rounds=args.num_mp_rounds,
        num_transformer_blocks=args.num_transformer_blocks,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        use_dual_perspective=not args.no_dual_perspective,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        entropy_coef=args.entropy_coef,
        k_epochs=args.k_epochs,
        seed=args.seed,
        eval_every=args.eval_every,
        save_dir=args.save_dir,
    )

    # Print final results
    if history:
        final = history[-1]
        print(f"\nTraining complete.")
        print(f"  Final best makespan: {final.get('best_makespan', 'N/A')}")
        print(f"  Total episodes: {len(history)}")

    # Save training history
    import json
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / "fjsp_l2s_history.json", "w") as f:
        json.dump(history, f, indent=2, default=str)
    print(f"  History saved to {save_path / 'fjsp_l2s_history.json'}")


if __name__ == "__main__":
    main()
