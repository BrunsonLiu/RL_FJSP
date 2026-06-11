"""Re-dump the mk12 schedule at makespan 508 (ILS best, not SA=524)."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

ROOT = Path(r'd:\desktop2\RL_FJSP')
sys.path.insert(0, str(ROOT))

from rl.agents import train_reinforce
from fjsp.env import FJSPDispatchEnv
from fjsp.scheduler.local_search import iterated_local_search, simulated_annealing

SEEDS = [0, 1, 2, 3, 4]
ILS_ITERS = 15
SA_ITERS  = 1500


def collect_schedule(env, agent):
    env.reset()
    schedule = []
    while not env.done:
        action, _ = agent.select_action(env, greedy=True)
        _, _, _, info = env.step(action)
        schedule.append(info['scheduled'])
    return env.validate().makespan, schedule


def schedule_to_json(sched):
    return {"operations": [{"job": s.job, "op": s.op,
             "machine": s.machine, "start": s.start, "end": s.end}
            for s in sched]}


inst = ROOT / 'data/instances/brandimarte/mk12.txt'
env = FJSPDispatchEnv.from_file(str(inst))

# Find the RL schedule that leads to ILS=508
best_rl_sched = None
best_rl_ms = float('inf')
for seed in SEEDS:
    agent, _ = train_reinforce(env, episodes=100, seed=seed, hidden_dim=64)
    ms, sched = collect_schedule(env, agent)
    if ms < best_rl_ms:
        best_rl_ms = ms
        best_rl_sched = sched

print(f"RL best: {best_rl_ms}")

# ILS
best_ils_ms, best_ils_sched = float('inf'), None
for mode, n_swaps in [('critical', 8), ('random', 10)]:
    cs, cm = iterated_local_search(env.instance, best_rl_sched,
        n_iterations=ILS_ITERS, perturb_n_swaps=n_swaps,
        max_ls_iterations=30, perturb_mode=mode)
    if cm < best_ils_ms:
        best_ils_ms, best_ils_sched = cm, cs

print(f"ILS best: {best_ils_ms}")

# Save ILS schedule
sched_path = ROOT / 'data/results/sota_mk12_schedule.json'
sched_path.write_text(json.dumps(schedule_to_json(best_ils_sched), indent=2), encoding='utf-8')

# Validate
r = subprocess.run(
    [sys.executable, str(ROOT / 'scripts/validate_schedule.py'),
     str(inst), str(sched_path)],
    capture_output=True, text=True, cwd=str(ROOT))
print(f"validate: {r.stdout.strip() or r.stderr.strip()}")
