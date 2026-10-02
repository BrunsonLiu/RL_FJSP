"""Fast BC training for SimpleMoveNet baseline."""
import sys
import pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn

from rl.models.simple_move_net import SimpleMoveNet


def train_bc_simple(
    data_path: str = "data/results/bc_data/bc_transitions.pkl",
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    save_path: str = "data/results/bc_data/simple_bc_pretrained.pt",
) -> SimpleMoveNet:
    with open(data_path, "rb") as f:
        transitions = pickle.load(f)

    print(f"Loaded {len(transitions)} transitions")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = SimpleMoveNet(hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n = len(transitions)
    best_loss = float("inf")

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        indices = torch.randperm(n).tolist()
        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            losses = []
            accs = []

            for i in batch_idx:
                t = transitions[i]
                graph = t["graph"]
                moves = t["valid_moves"]
                action = t["chosen_action_idx"]
                if not moves or action >= len(moves):
                    continue

                n_machines = graph.machine_features.size(0)
                denom = max(n_machines - 1, 1)
                assignments = []
                for j, (job, op) in enumerate(graph.operation_refs):
                    machine = int(round(graph.op_features[j, 2].item() * denom))
                    machine = max(0, min(machine, n_machines - 1))
                    assignments.append((job, op, machine))

                graph_device = graph_to_device(graph, device)
                scores = net(graph_device, moves, assignments)
                target = torch.tensor(action, dtype=torch.long, device=device)
                loss = nn.functional.cross_entropy(scores.unsqueeze(0), target.unsqueeze(0))
                losses.append(loss)

                pred = scores.argmax().item()
                accs.append(1.0 if pred == action else 0.0)

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

        scheduler.step()
        avg_loss = total_loss / max(total, 1)
        avg_acc = correct / max(total, 1)
        print(f"Epoch {epoch + 1}/{epochs} | loss={avg_loss:.4f} | acc={avg_acc:.3f} | lr={scheduler.get_last_lr()[0]:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), save_path)

    print(f"\nBest BC loss: {best_loss:.4f}")
    return net


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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/results/bc_data/bc_transitions.pkl")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save", default="data/results/bc_data/simple_bc_pretrained.pt")
    args = parser.parse_args()

    train_bc_simple(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save,
    )
