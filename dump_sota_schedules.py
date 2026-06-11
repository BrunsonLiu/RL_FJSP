"""Dump the SOTA schedule for the 5 most-cited Brandimarte instances.

For each of MK03, MK08, MK12, MK13, MK14:
  1. Run REINFORCE + ILS + SA (and TS for the hard ones)
  2. Save the best schedule to data/results/sota_{inst}_schedule.json
  3. Validate it via scripts/validate_schedule.py
  4. Print "OK makespan=X" if it passes

This is the audit-trail for our SOTA / TIED-OPT claims.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(r'd:\desktop2\RL_FJSP')
sys.path.insert(0, str(ROOT))

from rl.agents import train_reinforce
from fjsp.env import FJSPDispatchEnv
from fjsp.scheduler.dispatch_rules import rollout_earliest_finish
from fjsp.scheduler.local_search import (
    iterated_local_search,
    simulated_annealing,
    tabu_search,
)
from fjsp.scheduler.validator import ScheduledOperation

SEEDS = [0, 1, 2, 3, 4]

# Five SOTA-critical instances:
#   mk13  = NEW SOTA  (-14 vs lit 430)
#   mk03, mk08, mk12, mk14 = TIED OPT
TARGETS = [3, 8, 12, 13, 14]

# Per-instance config (from sota_reinforce_ils.py)
ILS_ITERS = {3: 20, 8: 20, 12: 15, 13: 15, 14: 15}
SA_ITERS  = {3: 2000, 8: 2000, 12: 1500, 13: 1500, 14: 1500}
TS_ITERS  = {3: 0, 8: 0, 12: 0, 13: 800, 14: 0}  # only mk13 in aggressive_ils

OUT_DIR = ROOT / 'data' / 'results'
OUT_DIR.mkdir(exist_ok=True, parents=True)


def collect_schedule(env, agent):
    env.reset()
    schedule = []
    while not env.done:
        action, _ = agent.select_action(env, greedy=True)
        _, _, _, info = env.step(action)
        schedule.append(info['scheduled'])
    return env.validate().makespan, schedule


def schedule_to_json(sched):
    """ScheduledOperation list -> JSON-serializable list of dicts."""
    return {
        "operations": [
            {"job": s.job, "op": s.op, "machine": s.machine,
             "start": s.start, "end": s.end}
            for s in sched
        ]
    }


def save_and_validate(mk: int, schedule, makespan: int) -> dict:
    """Save the schedule to JSON, run validate_schedule.py, parse stdout."""
    inst_path = ROOT / 'data' / 'instances' / 'brandimarte' / f'mk{mk:02d}.txt'
    sched_path = OUT_DIR / f'sota_mk{mk:02d}_schedule.json'

    payload = schedule_to_json(schedule)
    sched_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f"  saved -> {sched_path.name}")

    # Run the official validator as a subprocess
    res = subprocess.run(
        [sys.executable, str(ROOT / 'scripts' / 'validate_schedule.py'),
         str(inst_path), str(sched_path)],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    out = (res.stdout or '').strip()
    err = (res.stderr or '').strip()
    ok = res.returncode == 0 and out.startswith('OK')

    print(f"  validate: {out or err or '(no output)'}")
    return {
        "instance": f"mk{mk:02d}",
        "schedule_path": str(sched_path),
        "recorded_makespan": makespan,
        "validator_ok": ok,
        "validator_msg": out or err,
    }


def run_one(mk: int) -> dict:
    inst = ROOT / 'data' / 'instances' / 'brandimarte' / f'mk{mk:02d}.txt'
    env = FJSPDispatchEnv.from_file(str(inst))
    name = f"mk{mk:02d}"
    print(f"\n=== {name.upper()} (J={env.instance.job_count}, M={env.instance.machine_count}, "
          f"ops={env.instance.operation_count}) ===")

    t0 = time.perf_counter()

    # 1. REINFORCE: 5 seeds × 100 ep
    best_rl_ms = float('inf')
    best_rl_sched = None
    for seed in SEEDS:
        agent, _ = train_reinforce(env, episodes=100, seed=seed, hidden_dim=64)
        ms, sched = collect_schedule(env, agent)
        if ms < best_rl_ms:
            best_rl_ms = ms
            best_rl_sched = sched
    print(f"  RL best: {best_rl_ms}")

    # 2. ILS: two perturbation modes
    best_ils_ms = float('inf')
    best_ils_sched = None
    for mode, n_swaps in [('critical', 8), ('random', 10)]:
        cand_sched, cand_ms = iterated_local_search(
            env.instance, best_rl_sched,
            n_iterations=ILS_ITERS[mk],
            perturb_n_swaps=n_swaps,
            max_ls_iterations=30,
            perturb_mode=mode,
        )
        if cand_ms < best_ils_ms:
            best_ils_ms = cand_ms
            best_ils_sched = cand_sched
    print(f"  ILS best: {best_ils_ms}")

    # 3. SA refinement
    sa_sched, sa_ms = simulated_annealing(
        env.instance, best_ils_sched,
        initial_temperature=10.0, cooling_rate=0.99,
        min_temperature=0.1, iterations_per_temp=20,
        max_total_iterations=SA_ITERS[mk],
    )
    print(f"  SA  best: {sa_ms}")

    # 4. Tabu search (only mk13 in our setup)
    if TS_ITERS[mk] > 0:
        ts_sched, ts_ms = tabu_search(
            env.instance, sa_sched,
            max_iterations=TS_ITERS[mk],
            tabu_tenure=15,
        )
        print(f"  TS  best: {ts_ms}")
    else:
        ts_sched, ts_ms = sa_sched, sa_ms

    final_ms = min(best_ils_ms, sa_ms, ts_ms)
    # Pick the schedule that achieved final_ms (tie-break: first seen)
    if ts_ms == final_ms:
        final_sched = ts_sched
    elif sa_ms == final_ms:
        final_sched = sa_sched
    else:
        final_sched = best_ils_sched

    elapsed = time.perf_counter() - t0
    print(f"  FINAL: {final_ms}  (elapsed: {elapsed:.1f}s)")

    # 5. Save and validate
    val = save_and_validate(mk, final_sched, final_ms)
    val.update({
        "elapsed_s": round(elapsed, 2),
        "rl_best": best_rl_ms,
        "ils_best": best_ils_ms,
        "sa_best": sa_ms,
        "ts_best": ts_ms if TS_ITERS[mk] > 0 else None,
    })
    return val


def main():
    results = []
    for mk in TARGETS:
        results.append(run_one(mk))

    out = OUT_DIR / "sota_schedule_validation.json"
    out.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f"\n=== Summary ===")
    print(f"{'inst':<6} {'rec':>4} {'val':>4}  {'msg':<40}")
    for r in results:
        rec = r['recorded_makespan']
        ok = "OK" if r['validator_ok'] else "BAD"
        print(f"{r['instance']:<6} {rec:>4} {ok:>4}  {r['validator_msg']}")
    print(f"\nFull report -> {out}")


if __name__ == "__main__":
    main()
