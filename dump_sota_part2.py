"""Dump mk13 and mk14 schedules (no TS, use ILS+SA)."""
from __future__ import annotations
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r'd:\desktop2\RL_FJSP')
sys.path.insert(0, str(ROOT))

from rl.agents import train_reinforce
from fjsp.env import FJSPDispatchEnv
from fjsp.scheduler.local_search import iterated_local_search, simulated_annealing

SEEDS = [0, 1, 2, 3, 4]
TARGETS = [13, 14]

ILS_ITERS = {13: 15, 14: 15}
SA_ITERS  = {13: 1500, 14: 1500}

OUT_DIR = ROOT / 'data' / 'results'


def collect_schedule(env, agent):
    env.reset()
    schedule = []
    while not env.done:
        action, _ = agent.select_action(env, greedy=True)
        _, _, _, info = env.step(action)
        schedule.append(info['scheduled'])
    return env.validate().makespan, schedule


def schedule_to_json(sched):
    return {
        "operations": [
            {"job": s.job, "op": s.op, "machine": s.machine,
             "start": s.start, "end": s.end}
            for s in sched
        ]
    }


def run_one(mk):
    inst = ROOT / 'data' / 'instances' / 'brandimarte' / f'mk{mk:02d}.txt'
    env = FJSPDispatchEnv.from_file(str(inst))
    print(f"\n=== MK{mk:02d} (J={env.instance.job_count}, M={env.instance.machine_count}) ===")

    t0 = time.perf_counter()
    best_rl_ms, best_rl_sched = float('inf'), None
    for seed in SEEDS:
        agent, _ = train_reinforce(env, episodes=100, seed=seed, hidden_dim=64)
        ms, sched = collect_schedule(env, agent)
        if ms < best_rl_ms:
            best_rl_ms = ms
            best_rl_sched = sched
    print(f"  RL best: {best_rl_ms}")

    best_ils_ms, best_ils_sched = float('inf'), None
    for mode, n_swaps in [('critical', 8), ('random', 10)]:
        c_sched, c_ms = iterated_local_search(
            env.instance, best_rl_sched,
            n_iterations=ILS_ITERS[mk], perturb_n_swaps=n_swaps,
            max_ls_iterations=30, perturb_mode=mode,
        )
        if c_ms < best_ils_ms:
            best_ils_ms, best_ils_sched = c_ms, c_sched
    print(f"  ILS best: {best_ils_ms}")

    sa_sched, sa_ms = simulated_annealing(
        env.instance, best_ils_sched,
        initial_temperature=10.0, cooling_rate=0.99,
        min_temperature=0.1, iterations_per_temp=20,
        max_total_iterations=SA_ITERS[mk],
    )
    final_ms = min(best_ils_ms, sa_ms)
    final_sched = sa_sched if sa_ms == final_ms else best_ils_sched
    print(f"  SA best: {sa_ms}  FINAL: {final_ms}  ({time.perf_counter()-t0:.1f}s)")

    # Save and validate
    sched_path = OUT_DIR / f'sota_mk{mk:02d}_schedule.json'
    sched_path.write_text(json.dumps(schedule_to_json(final_sched), indent=2),
                          encoding='utf-8')
    res = subprocess.run(
        [sys.executable, str(ROOT / 'scripts' / 'validate_schedule.py'),
         str(inst), str(sched_path)],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    msg = (res.stdout or '').strip() or (res.stderr or '').strip()
    ok = res.returncode == 0
    print(f"  validate: {msg}")
    return {
        "instance": f"mk{mk:02d}",
        "recorded_makespan": final_ms,
        "validator_ok": ok,
        "validator_msg": msg,
        "schedule_path": str(sched_path),
    }


def main():
    results = [run_one(m) for m in TARGETS]
    out = OUT_DIR / "sota_schedule_validation_part2.json"
    out.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f"\nSaved -> {out}")
    for r in results:
        print(f"  {r['instance']}: {r['recorded_makespan']}  {r['validator_msg']}")


if __name__ == "__main__":
    main()
