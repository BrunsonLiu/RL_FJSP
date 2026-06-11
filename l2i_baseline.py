"""L2I-style baseline: RL agent learns to SELECT which neighborhood
operator to apply at each local-search step.

Following the Learning-to-Improve paradigm of Wu et al. ICLR 2024:
- State encodes the current schedule + history of moves
- Action = one of 4 neighborhood operators
- Reward = makespan improvement (clipped at 0)
- Trained with REINFORCE

This is the most direct comparison with our pipeline: same
neighborhoods, same local search loop, but a LEARNED policy for
operator selection (vs our fixed operator sequence).

Usage:
    python l2i_baseline.py
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(r'd:\desktop2\RL_FJSP')
sys.path.insert(0, str(ROOT))

from fjsp.env import FJSPDispatchEnv
from fjsp.scheduler.local_search import (
    local_search, _assignments, _makespan,
    _neighbors_reassign, _neighbors_swap_machines,
    _neighbors_swap_order_across_machines, _neighbors_swap_same_machine,
    random_schedule, neh_construct,
)

OPERATORS = ["reassign", "swap_machine", "swap_order", "swap_order_across"]
N_OPS = len(OPERATORS)
NEIGHBOR_FNS = {
    "reassign": _neighbors_reassign,
    "swap_machine": _neighbors_swap_machines,
    "swap_order": _neighbors_swap_same_machine,
    "swap_order_across": _neighbors_swap_order_across_machines,
}

# State dimension: 10 features
STATE_DIM = 10


# -------------------------------------------------------------------------
# Model
# -------------------------------------------------------------------------
class L2IPolicy(nn.Module):
    """Small MLP that maps state -> op logits."""
    def __init__(self, state_dim: int = STATE_DIM, n_ops: int = N_OPS, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_ops),
        )

    def forward(self, s):
        return self.net(s)

    def select(self, s, greedy: bool = False):
        logits = self.forward(s)
        if greedy:
            return int(logits.argmax(dim=-1).item())
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        a = dist.sample()
        return int(a.item()), dist.log_prob(a)


# -------------------------------------------------------------------------
# State encoder
# -------------------------------------------------------------------------
def encode_state(env, assignments, history, current_ms, initial_ms, step):
    """10-D state vector:
        0: current makespan
        1: current / initial ratio
        2: best / initial ratio
        3: num improvements in last 10 steps
        4: num worsens in last 10 steps
        5: step_idx / max_steps
        6: number of operations scheduled
        7: number of distinct machines used
        8: mean machine utilization (0-1)
        9: standard deviation of operation duration / mean
    """
    n_ops = len(assignments)
    n_machines = env.instance.machine_count
    machines_used = set(a[2] for a in assignments)
    # machine utilization — must recompute schedule to know end times
    try:
        sched = _recompute_safe(env.instance, assignments)
        machine_load = [0.0] * n_machines
        for op in sched:
            machine_load[op.machine] += (op.end - op.start)
        ms = current_ms
        util = [m / max(1, ms) for m in machine_load]
        util_mean = sum(util) / max(1, n_machines)
        util_var = sum((u - util_mean) ** 2 for u in util) / max(1, n_machines)
        # op duration stats
        durs = [op.end - op.start for op in sched]
        d_mean = sum(durs) / max(1, n_ops)
        d_var = sum((d - d_mean) ** 2 for d in durs) / max(1, n_ops)
        d_std = math.sqrt(d_var)
        d_norm = d_std / max(1e-6, d_mean)
    except Exception:
        util_mean = 0.5
        d_norm = 0.5
    # recent history
    last10 = list(history)[-10:]
    n_improved = sum(1 for h in last10 if h > 0)
    n_worsened = sum(1 for h in last10 if h < 0)

    s = torch.tensor([
        current_ms / max(1, initial_ms),
        current_ms / max(1, initial_ms),
        current_ms / max(1, initial_ms),  # use current as "best so far" proxy
        n_improved / 10.0,
        n_worsened / 10.0,
        step / 200.0,
        n_ops / max(1, env.instance.operation_count),
        len(machines_used) / max(1, n_machines),
        util_mean,
        d_norm,
    ], dtype=torch.float32)
    return s


def _recompute_safe(instance, assignments):
    """Recompute schedule, return list of ScheduledOperation. Local import."""
    from fjsp.scheduler.local_search import _recompute
    return _recompute(instance, assignments)


# -------------------------------------------------------------------------
# One step: apply operator and see if it improves
# -------------------------------------------------------------------------
def apply_operator(env, assignments, op_name, max_per_op=50):
    """Try to find an improving move in the given neighborhood and apply it.

    Returns: (new_assignments, improvement) where improvement is
    old_makespan - new_makespan (positive = better).
    """
    inst = env.instance
    old_ms = _makespan(inst, assignments)
    fn = NEIGHBOR_FNS[op_name]
    neighbors = list(fn(inst, assignments))
    random.shuffle(neighbors)
    tried = 0
    best = None
    best_ms = old_ms
    for new_assign in neighbors:
        new_ms = _makespan(inst, new_assign)
        if new_ms < best_ms:
            best = new_assign
            best_ms = new_ms
            if best_ms < old_ms:
                # found an improvement — accept immediately
                return best, old_ms - best_ms
        tried += 1
        if tried >= max_per_op:
            break
    if best is not None:
        return best, old_ms - best_ms
    return assignments, 0  # no improvement possible from this op


# -------------------------------------------------------------------------
# Train
# -------------------------------------------------------------------------
def train_l2i(env, n_steps: int = 200, seed: int = 0):
    torch.manual_seed(seed)
    random.seed(seed)
    policy = L2IPolicy()
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    baseline = 0.0
    baseline_alpha = 0.05

    # initial schedule
    init_assign = _assignments(random_schedule(env.instance, seed=seed))
    init_ms = _makespan(env.instance, init_assign)

    best_sched = init_assign
    best_ms = init_ms
    history = deque([0.0] * 10, maxlen=10)

    log_probs = []
    rewards = []

    for step in range(n_steps):
        s = encode_state(env, best_sched, history, best_ms, init_ms, step)
        a, log_prob = policy.select(s, greedy=False)
        new_assign, improvement = apply_operator(env, best_sched, OPERATORS[a])
        r = float(improvement) / init_ms
        log_probs.append(log_prob)
        rewards.append(r)
        history.append(improvement)
        if improvement > 0:
            best_sched = new_assign
            best_ms = _makespan(env.instance, new_assign)

    # Return-centered
    R = 0.0
    returns = []
    for r in reversed(rewards):
        R = r + 0.99 * R
        returns.insert(0, R)
    returns = torch.tensor(returns)

    # Update baseline EMA
    baseline = (1 - baseline_alpha) * baseline + baseline_alpha * returns.mean().item()

    # REINFORCE loss
    loss = []
    for log_prob, R_t in zip(log_probs, returns):
        advantage = R_t - baseline
        loss.append(-log_prob * advantage.detach())
    loss = torch.stack(loss).sum()

    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    opt.step()

    return best_sched, best_ms, init_ms


# -------------------------------------------------------------------------
# Run
# -------------------------------------------------------------------------
def run_one(mk, n_steps=200, seeds=(0, 1, 2, 3, 4)):
    inst = ROOT / f'data/instances/brandimarte/mk{mk:02d}.txt'
    env = FJSPDispatchEnv.from_file(str(inst))

    best_overall = float('inf')
    best_sched = None
    init_ms = None
    t0 = time.perf_counter()
    for s in seeds:
        sched, ms, init = train_l2i(env, n_steps=n_steps, seed=s)
        if ms < best_overall:
            best_overall = ms
            best_sched = sched
            init_ms = init
    elapsed = time.perf_counter() - t0
    return {
        'instance': f'mk{mk:02d}',
        'init_ms': init_ms,
        'l2i_best': best_overall,
        'time_s': round(elapsed, 2),
    }


def main():
    results = []
    print(f"{'inst':<6} {'init':>5} {'L2I':>5} {'time':>7}")
    for mk in range(1, 16):
        r = run_one(mk)
        print(f"{r['instance']:<6} {r['init_ms']:>5} {r['l2i_best']:>5} {r['time_s']:>7.1f}")
        results.append(r)

    out = ROOT / 'data/results/l2i_baseline.json'
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
