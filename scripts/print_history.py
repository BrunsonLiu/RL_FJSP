"""Print a compact summary of a training history JSON.

Supports two formats:
  * A2C / PPO: each entry has ``episode``, ``sample_makespan``,
    ``greedy_makespan``, ``best_greedy_makespan``.
  * BC: each entry has ``epoch``, ``job_loss``, ``machine_loss``.
"""
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
h = json.load(open(p))
print(f"file: {p}")
print(f"entries: {len(h)}")

if not h:
    sys.exit(0)

sample_entry = h[0]
is_bc = "epoch" in sample_entry and "episode" not in sample_entry

if is_bc:
    for e in h:
        ep = int(e["epoch"])
        jl = float(e.get("job_loss", 0.0))
        ml = float(e.get("machine_loss", 0.0))
        acc = e.get("train_acc") or e.get("acc")
        acc_s = f" acc={float(acc):.3f}" if acc is not None else ""
        print(f"  epoch {ep:3d}: job_loss={jl:.4f} machine_loss={ml:.4f}{acc_s}")
    best_e = min(h, key=lambda x: x.get("job_loss", 1e9) + x.get("machine_loss", 1e9))
    print(
        f"BEST entry: epoch={int(best_e['epoch'])} "
        f"job_loss={float(best_e.get('job_loss', 0)):.4f} "
        f"machine_loss={float(best_e.get('machine_loss', 0)):.4f}"
    )
else:
    for e in h:
        ep = int(e["episode"])
        sample = int(e.get("sample_makespan", 0))
        greedy = int(e.get("greedy_makespan", 0))
        best = int(e.get("best_greedy_makespan", 0))
        print(f"  ep {ep:3d}: sample={sample:3d} greedy={greedy:3d} best={best:3d}")
    best_e = min(h, key=lambda x: x.get("best_greedy_makespan", 1e9))
    print(
        f"BEST entry: ep={int(best_e['episode'])} "
        f"best={int(best_e.get('best_greedy_makespan', 0))}"
    )
