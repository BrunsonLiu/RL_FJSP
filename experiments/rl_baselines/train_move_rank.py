"""Train MoveRankNet for neural-guided local search."""
import sys
import pickle
import random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.models.move_rank_net import MoveRankNet


def graph_to_device(graph, device):
    from fjsp.graph.solution_graph import SolutionGraph
    return SolutionGraph(
        operation_refs=graph.operation_refs,
        op_features=graph.op_features.to(device),
        machine_features=graph.machine_features.to(device),
        precedence_edges=graph.precedence_edges.to(device),
        machine_seq_edges=graph.machine_seq_edges.to(device),
        eligibility_edges=graph.eligibility_edges.to(device),
        eligibility_durations=graph.eligibility_durations.to(device),
        critical_op_indices=graph.critical_op_indices,
        global_features=graph.global_features.to(device),
    )


def _ranking_metrics(scores: torch.Tensor, targets: torch.Tensor, top_k: int = 5) -> dict:
    """Compute ranking quality metrics for a batch of moves."""
    if scores.numel() == 0:
        return {"auc": 0.0, "ap": 0.0, f"top{top_k}_recall": 0.0}

    pos_mask = targets > 0
    n_pos = pos_mask.sum().item()
    n_neg = (~pos_mask).sum().item()
    if n_pos == 0 or n_neg == 0:
        return {"auc": 0.0, "ap": 0.0, f"top{top_k}_recall": 0.0}

    # AUC via Mann-Whitney U
    pos_scores = scores[pos_mask]
    neg_scores = scores[~pos_mask]
    auc = ((pos_scores.unsqueeze(1) > neg_scores.unsqueeze(0)).float().mean()).item()

    # Average precision: how many positives in top-k * relative to random
    ranked = torch.argsort(scores, descending=True)
    top_k = min(top_k, scores.numel())
    top_pos = pos_mask[ranked[:top_k]].sum().item()
    topk_recall = top_pos / max(n_pos, 1)

    # Average precision approximation
    ap = topk_recall * (top_pos / max(top_k, 1))

    return {"auc": auc, "ap": ap, f"top{top_k}_recall": topk_recall}


def train_move_rank(
    data_path: str = "data/results/move_rank_data/neighborhood_move_data.pkl",
    epochs: int = 100,
    batch_size: int = 8,
    lr: float = 3e-4,
    hidden_dim: int = 128,
    save_path: str = "data/results/move_rank_data/move_rank_net.pt",
    loss_type: str = "rank",
    max_moves_per_sample: int | None = 200,
    val_ratio: float = 0.1,
    seed: int = 0,
) -> MoveRankNet:
    torch.manual_seed(seed)
    random.seed(seed)

    with open(data_path, "rb") as f:
        samples = pickle.load(f)

    print(f"Loaded {len(samples)} samples")

    n_pairs = sum(len(s["moves"]) for s in samples)
    n_pos = sum(1 for s in samples for imp in s["improvements"] if imp > 0)
    print(f"Total move-outcome pairs: {n_pairs}")
    print(f"Positive ratio: {n_pos}/{n_pairs} ({100*n_pos/n_pairs:.1f}%)")

    # Train/val split by solution sample (not by individual move)
    rng = random.Random(seed)
    rng.shuffle(samples)
    n_val = max(1, int(len(samples) * val_ratio))
    train_samples = samples[:-n_val]
    val_samples = samples[-n_val:]
    print(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = MoveRankNet(
        hidden_dim=hidden_dim,
        num_mp_rounds=2,
        num_transformer_blocks=2,
        num_heads=8,
        ffn_dim=hidden_dim,
    ).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_metric = -float("inf")
    best_epoch = 0

    for epoch in range(epochs):
        net.train()
        train_loss = 0.0
        train_count = 0

        sample_indices = list(range(len(train_samples)))
        rng.shuffle(sample_indices)
        accum_loss = 0.0
        accum_steps = 0

        for idx in sample_indices:
            s = train_samples[idx]
            graph = graph_to_device(s["graph"], device)
            moves = s["moves"]
            improvements = s["improvements"]

            # Keep all positives and sample negatives so each sample is bounded
            if max_moves_per_sample is not None and len(moves) > max_moves_per_sample:
                pos_idx = [i for i, imp in enumerate(improvements) if imp > 0]
                neg_idx = [i for i, imp in enumerate(improvements) if imp <= 0]
                n_neg = max_moves_per_sample - len(pos_idx)
                if n_neg < 0:
                    pos_idx = pos_idx[:max_moves_per_sample]
                    n_neg = 0
                sampled_neg = rng.sample(neg_idx, min(n_neg, len(neg_idx))) if n_neg else []
                selected = pos_idx + sampled_neg
                moves = [moves[i] for i in selected]
                improvements = [improvements[i] for i in selected]

            if not moves:
                continue

            scores = net(graph, moves)
            targets = torch.tensor(improvements, dtype=torch.float32, device=device)

            if loss_type == "rank":
                pos_mask = targets > 0
                neg_mask = targets <= 0
                n_pos_batch = pos_mask.sum().item()
                n_neg_batch = neg_mask.sum().item()
                if n_pos_batch == 0 or n_neg_batch == 0:
                    continue
                pos_scores = scores[pos_mask]
                neg_scores = scores[neg_mask]
                # Pairwise margin loss: every positive should score higher
                # than every negative.  Averaging over all pairs is more
                # stable than hard-negative mining for this sparse (2%
                # positives) ranking problem.
                margin = 0.1
                loss = torch.clamp(
                    margin - (pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)), min=0
                ).mean()
            else:
                # Weighted MSE: positives up-weighted to handle imbalance
                pos_weight = max(1.0, len(improvements) / max(2 * sum(1 for imp in improvements if imp > 0), 1))
                weights = torch.tensor(
                    [1.0 if imp <= 0 else pos_weight for imp in improvements],
                    dtype=torch.float32,
                    device=device,
                )
                loss = (weights * F.mse_loss(scores, targets, reduction="none")).mean()

            accum_loss += loss
            accum_steps += 1

            if accum_steps >= batch_size:
                optimizer.zero_grad()
                accum_loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()

                train_loss += accum_loss.item()
                train_count += accum_steps
                accum_loss = 0.0
                accum_steps = 0

        if accum_steps > 0:
            optimizer.zero_grad()
            accum_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

            train_loss += accum_loss.item()
            train_count += accum_steps

        scheduler.step()

        # Validation
        net.eval()
        val_metrics_all = []
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for s in val_samples:
                graph = graph_to_device(s["graph"], device)
                moves = s["moves"]
                improvements = s["improvements"]
                if max_moves_per_sample is not None and len(moves) > max_moves_per_sample:
                    pos_idx = [i for i, imp in enumerate(improvements) if imp > 0]
                    neg_idx = [i for i, imp in enumerate(improvements) if imp <= 0]
                    n_neg = max_moves_per_sample - len(pos_idx)
                    if n_neg < 0:
                        pos_idx = pos_idx[:max_moves_per_sample]
                        n_neg = 0
                    sampled_neg = rng.sample(neg_idx, min(n_neg, len(neg_idx))) if n_neg else []
                    selected = pos_idx + sampled_neg
                    moves = [moves[i] for i in selected]
                    improvements = [improvements[i] for i in selected]
                if not moves:
                    continue
                scores = net(graph, moves)
                targets = torch.tensor(improvements, dtype=torch.float32, device=device)
                val_metrics_all.append(_ranking_metrics(scores, targets, top_k=5))
                if loss_type != "rank":
                    val_loss += F.mse_loss(scores, targets).item() * len(moves)
                    val_count += len(moves)

        avg_train_loss = train_loss / max(train_count, 1)
        avg_val_auc = sum(m["auc"] for m in val_metrics_all) / max(len(val_metrics_all), 1)
        avg_val_top5 = sum(m["top5_recall"] for m in val_metrics_all) / max(len(val_metrics_all), 1)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={avg_train_loss:.6f} | "
            f"val_auc={avg_val_auc:.4f} | val_top5_recall={avg_val_top5:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        # Save best model by validation top-5 recall
        if avg_val_top5 > best_val_metric:
            best_val_metric = avg_val_top5
            best_epoch = epoch + 1
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), save_path)

    print(f"\nBest val top5_recall: {best_val_metric:.4f} at epoch {best_epoch}")
    return net


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/results/move_rank_data/neighborhood_move_data.pkl")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--save", default="data/results/move_rank_data/move_rank_net.pt")
    parser.add_argument("--loss", default="rank", choices=["mse", "rank"])
    parser.add_argument("--max-moves-per-sample", type=int, default=200)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train_move_rank(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        save_path=args.save,
        loss_type=args.loss,
        max_moves_per_sample=args.max_moves_per_sample,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
