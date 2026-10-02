"""Behavioral cloning pretraining for BARI.

Loads greedy-expert transitions and trains the policy network (move scoring
head) via supervised learning. The value head is left for PPO to train.
"""
import sys
import pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
import torch.optim as optim

from rl.models.fjsp_l2s import FJSPImproveNet


def train_bc(
    data_path: str = "data/results/bc_data/bc_transitions.pkl",
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    save_path: str = "data/results/bc_data/bc_pretrained.pt",
) -> FJSPImproveNet:
    """Train BARI policy with behavioral cloning."""
    with open(data_path, "rb") as f:
        transitions = pickle.load(f)

    print(f"Loaded {len(transitions)} transitions")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = FJSPImproveNet(
        hidden_dim=hidden_dim,
        num_mp_rounds=2,
        num_transformer_blocks=2,
        num_heads=8,
        ffn_dim=128,
        use_dual_perspective=True,
    ).to(device)

    optimizer = optim.Adam(net.parameters(), lr=lr)

    n = len(transitions)
    best_loss = float("inf")

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        # Shuffle
        indices = torch.randperm(n).tolist()

        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            losses = []
            accs = []

            for i in batch_idx:
                t = transitions[i]
                graph = t["graph"]
                valid_moves = t["valid_moves"]
                if not valid_moves:
                    continue

                # Reconstruct assignments from graph
                n_machines = graph.machine_features.size(0)
                denom = max(n_machines - 1, 1)
                assignments = []
                for j, (job, op) in enumerate(graph.operation_refs):
                    machine_norm = graph.op_features[j, 2].item()
                    machine = int(round(machine_norm * denom))
                    machine = max(0, min(machine, n_machines - 1))
                    assignments.append((job, op, machine))

                op_features = graph.op_features.to(device)
                machine_features = graph.machine_features.to(device)
                global_features = graph.global_features.to(device)
                precedence_edges = graph.precedence_edges.to(device)
                machine_seq_edges = graph.machine_seq_edges.to(device)
                eligibility_edges = graph.eligibility_edges.to(device)
                eligibility_durations = graph.eligibility_durations.to(device)

                from fjsp.graph.solution_graph import SolutionGraph
                device_graph = SolutionGraph(
                    operation_refs=graph.operation_refs,
                    op_features=op_features,
                    machine_features=machine_features,
                    precedence_edges=precedence_edges,
                    machine_seq_edges=machine_seq_edges,
                    eligibility_edges=eligibility_edges,
                    eligibility_durations=eligibility_durations,
                    critical_op_indices=graph.critical_op_indices,
                    global_features=global_features,
                )

                op_h, machine_h, _ = net.encoder(device_graph)

                scores = net.score_moves(op_h, machine_h, device_graph, valid_moves, assignments)
                target = torch.tensor(t["chosen_action_idx"], dtype=torch.long, device=device)
                loss = nn.functional.cross_entropy(scores.unsqueeze(0), target.unsqueeze(0))
                losses.append(loss)

                pred = scores.argmax().item()
                accs.append(1.0 if pred == t["chosen_action_idx"] else 0.0)

            if not losses:
                continue

            batch_loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

            total_loss += batch_loss.item() * len(losses)
            correct += sum(accs)
            total += len(accs)

        avg_loss = total_loss / max(total, 1)
        avg_acc = correct / max(total, 1)
        print(f"Epoch {epoch + 1}/{epochs} | loss={avg_loss:.4f} | acc={avg_acc:.3f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), save_path)
            print(f"  Saved best model (loss={best_loss:.4f})")

    print(f"\nBest BC loss: {best_loss:.4f}")
    return net


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/results/bc_data/bc_transitions.pkl")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save", default="data/results/bc_data/bc_pretrained.pt")
    args = parser.parse_args()

    train_bc(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save,
    )
