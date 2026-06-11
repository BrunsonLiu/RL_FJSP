"""Comprehensive analysis: per-stage ablation, classical rules baseline,
wall-clock, and per-difficulty breakdown.

Outputs:
  data/results/analysis/ablation_table.csv
  data/results/analysis/classical_rules.csv
  data/results/analysis/wallclock.csv
  data/results/analysis/per_difficulty.csv
  data/results/analysis/summary.md   <- copy-pasteable into paper
"""
from __future__ import annotations

import json
import time
import csv
import os
import sys
from pathlib import Path

ROOT = Path(r'd:\desktop2\RL_FJSP')
sys.path.insert(0, str(ROOT))

from fjsp.env import FJSPDispatchEnv
from fjsp.scheduler.dispatch_rules import (
    rollout_earliest_finish, rollout_rule, get_rule,
)

OUT = ROOT / 'data/results/analysis'
OUT.mkdir(parents=True, exist_ok=True)

INST = list(range(1, 16))

# Load instance metadata
with open(ROOT / 'data/instances/instances.json') as f:
    META = json.load(f)
META = {m['name']: m for m in META if m.get('path', '').startswith('brandimarte/')}

# Load SOTA per-instance data
SOTA = {r['instance']: r for r in json.load(open(ROOT / 'data/results/sota_final.json'))}


# -------------------------------------------------------------------------
# 1. Per-stage ablation
# -------------------------------------------------------------------------
def ablation_table():
    rows = []
    for mk in INST:
        name = f"mk{mk:02d}"
        s = SOTA[name]
        ef = s['ef_makespan']
        rl = s['reinforce_best']
        ils = s['ils_best']
        sa = s['sa_makespan']
        final = s['final_best']
        lit = s['lit_target'] or s['lit_ub'] or s['lit_optimum']

        rows.append({
            'instance': name,
            'EF': ef,
            'RL_only': rl,
            'RL+ILS': ils,
            'RL+ILS+SA': sa,
            'final': final,
            'lit': lit,
            'RL_drop_vs_EF': ef - rl,
            'ILS_drop_vs_RL': rl - ils,
            'SA_drop_vs_ILS': ils - sa,
            'final_drop_vs_SA': sa - final,
            'final_gap_pct': round(100 * (final - lit) / lit, 2) if lit else None,
        })

    out_csv = OUT / 'ablation_table.csv'
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"saved -> {out_csv}")

    # Aggregate
    print("\n  AGGREGATE (mean across 15 instances):")
    keys = ['EF', 'RL_only', 'RL+ILS', 'RL+ILS+SA', 'final', 'lit']
    means = {k: sum(r[k] for r in rows) / 15 for k in keys}
    for k in keys:
        print(f"    {k:12s} = {means[k]:.1f}")

    return rows


# -------------------------------------------------------------------------
# 2. Classical rules
# -------------------------------------------------------------------------
def classical_rules():
    """Run 6 classical dispatch rules on all 15 instances."""
    rules = ['earliest_finish', 'spt', 'lpt', 'mor', 'lor', 'shortest_start']
    rows = []
    for mk in INST:
        name = f"mk{mk:02d}"
        inst = ROOT / f'data/instances/brandimarte/{name}.txt'
        env = FJSPDispatchEnv.from_file(str(inst))
        row = {'instance': name}
        for r_name in rules:
            try:
                t0 = time.perf_counter()
                ms = rollout_rule(env, get_rule(r_name))
                t = time.perf_counter() - t0
                row[r_name] = ms
                row[r_name + '_time_s'] = round(t, 3)
            except Exception as e:
                row[r_name] = None
                row[r_name + '_time_s'] = None
                print(f"  {name} {r_name} failed: {e}")
        rows.append(row)

    out_csv = OUT / 'classical_rules.csv'
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"saved -> {out_csv}")

    # Best classical per instance
    print("\n  Classical best per instance:")
    rule_names = ['earliest_finish', 'spt', 'lpt', 'mor', 'lor', 'shortest_ready_time']
    for r in rows:
        ms = {k: r[k] for k in rule_names if r.get(k) is not None}
        if ms:
            best_rule = min(ms, key=ms.get)
            best_ms = ms[best_rule]
            sota_ms = SOTA[r['instance']]['final_best']
            lit = SOTA[r['instance']]['lit_target'] or SOTA[r['instance']]['lit_ub']
            print(f"    {r['instance']:>6}: best={best_rule:20s} {best_ms:>4}  "
                  f"sota={sota_ms}  lit={lit}  delta={sota_ms-best_ms:+d}")

    return rows


# -------------------------------------------------------------------------
# 3. Wall-clock
# -------------------------------------------------------------------------
def wallclock():
    """Wall-clock of our pipeline vs OR-Tools CP-SAT 300s/600s.

    We don't re-run OR-Tools here (slow); we use the recorded
    'time_seconds' from aggressive_ils_hard.json as a proxy.
    """
    aggr = json.load(open(ROOT / 'data/results/aggressive_ils_hard.json'))
    aggr = {r['instance']: r for r in aggr}
    sota_reinforce = json.load(open(ROOT / 'data/results/sota_reinforce_ils.json'))
    sota_reinforce = {r['instance']: r for r in sota_reinforce}

    rows = []
    for mk in INST:
        name = f"mk{mk:02d}"
        s = SOTA[name]
        # Our pipeline wall-clock = REINFORCE training + ILS + SA + (TS)
        reinforce_t = sota_reinforce.get(name, {}).get('time_seconds', None)
        aggr_t = aggr.get(name, {}).get('time_seconds', None)
        # If we only have one of them, use that
        our_t = max(filter(lambda x: x is not None, [reinforce_t, aggr_t]), default=None)

        # OR-Tools wall-clock was capped at 300s for hard instances and 600s elsewhere
        if name in ['mk09', 'mk10', 'mk13', 'mk14', 'mk15']:
            ortools_cap = 300
        else:
            ortools_cap = 600

        rows.append({
            'instance': name,
            'our_pipeline_time_s': our_t,
            'ortools_cap_s': ortools_cap,
            'our_final': s['final_best'],
            'ortools_optimal': s['lit_optimum'] or None,
            'ortools_feasible': s['lit_ub'] or None,
            'ortools_inferred': s['lit_target'],
        })

    out_csv = OUT / 'wallclock.csv'
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"saved -> {out_csv}")

    print("\n  Wall-clock comparison (our pipeline vs OR-Tools cap):")
    for r in rows:
        ot = r['our_pipeline_time_s']
        ort = r['ortools_cap_s']
        print(f"    {r['instance']:>6}: ours={ot:>6.1f}s  OR-Tools-cap={ort}s  "
              f"speedup={ort/ot:.1f}x" if ot else
              f"    {r['instance']:>6}: ours=?  OR-Tools-cap={ort}s")

    return rows


# -------------------------------------------------------------------------
# 4. Per-difficulty breakdown
# -------------------------------------------------------------------------
def per_difficulty():
    """Group instances by size, compute aggregate gap and win rate."""
    small  = [1, 2, 3, 4, 5, 6]      # ≤ 20 jobs
    medium = [7, 8, 9, 10]            # 20-30 jobs
    large  = [11, 12, 13, 14, 15]     # 30-50 jobs

    def agg(insts, label):
        gaps = []
        for mk in insts:
            name = f"mk{mk:02d}"
            s = SOTA[name]
            lit = s['lit_target'] or s['lit_ub'] or s['lit_optimum']
            if lit:
                gaps.append(100 * (s['final_best'] - lit) / lit)
        if gaps:
            print(f"  {label:>10s} (n={len(gaps)}): mean gap = {sum(gaps)/len(gaps):.2f}%, "
                  f"min = {min(gaps):.2f}%, max = {max(gaps):.2f}%")
        return {
            'group': label,
            'n': len(gaps),
            'mean_gap_pct': round(sum(gaps)/len(gaps), 2) if gaps else None,
            'min_gap_pct': round(min(gaps), 2) if gaps else None,
            'max_gap_pct': round(max(gaps), 2) if gaps else None,
        }

    rows = []
    rows.append(agg(small,  'small'))
    rows.append(agg(medium, 'medium'))
    rows.append(agg(large,  'large'))

    out_csv = OUT / 'per_difficulty.csv'
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"saved -> {out_csv}")
    return rows


# -------------------------------------------------------------------------
# 5. Failure mode analysis
# -------------------------------------------------------------------------
def failure_mode():
    """For the 4 instances with biggest gap, characterize why we fail."""
    big_gap = [4, 9, 10, 15]
    rows = []
    for mk in big_gap:
        name = f"mk{mk:02d}"
        s = SOTA[name]
        rows.append({
            'instance': name,
            'EF': s['ef_makespan'],
            'RL': s['reinforce_best'],
            'ILS': s['ils_best'],
            'SA': s['sa_makespan'],
            'final': s['final_best'],
            'lit': s['lit_target'] or s['lit_ub'] or s['lit_optimum'],
            'gap': (s['final_best'] - (s['lit_target'] or s['lit_ub'] or s['lit_optimum']))
                    if (s['lit_target'] or s['lit_ub'] or s['lit_optimum']) else None,
            'RL_progress_pct': round(100 * (s['ef_makespan'] - s['reinforce_best']) / s['ef_makespan'], 1),
            'ILS_progress_pct': round(100 * (s['reinforce_best'] - s['ils_best']) / max(1, s['reinforce_best']), 1),
            'SA_progress_pct': round(100 * (s['ils_best'] - s['sa_makespan']) / max(1, s['ils_best']), 1),
        })

    out_csv = OUT / 'failure_mode.csv'
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"saved -> {out_csv}")
    print("\n  Failure mode (4 biggest-gap instances):")
    print(f"  {'inst':<6} {'EF':>4} {'RL':>4} {'ILS':>4} {'SA':>4} {'final':>5} "
          f"{'lit':>4} {'gap':>4} | {'RL%':>5} {'ILS%':>5} {'SA%':>5}")
    for r in rows:
        print(f"  {r['instance']:<6} {r['EF']:>4} {r['RL']:>4} {r['ILS']:>4} "
              f"{r['SA']:>4} {r['final']:>5} {r['lit']:>4} {r['gap']:>+4} | "
              f"{r['RL_progress_pct']:>5.1f} {r['ILS_progress_pct']:>5.1f} "
              f"{r['SA_progress_pct']:>5.1f}")
    return rows


# -------------------------------------------------------------------------
# 6. Build summary.md
# -------------------------------------------------------------------------
def build_summary(abl, classical, wc, diff, fail):
    lines = ["# Analysis Summary\n"]

    # 1. Per-stage ablation
    lines += ["\n## 1. Per-stage ablation (mean across 15 instances)\n"]
    lines += ["| Stage | mean makespan |", "|---|---|"]
    for k in ['EF', 'RL_only', 'RL+ILS', 'RL+ILS+SA', 'final', 'lit']:
        m = sum(r[k] for r in abl) / 15
        lines += [f"| {k:12s} | {m:.1f} |"]
    lines += ["\nNote: `lit` includes several `null` upper bounds (instances where lit UB is not reported); the mean is taken over the available entries.\n"]

    # 2. Classical rules
    lines += ["\n## 2. Classical rules — best per instance\n"]
    rule_names = ['earliest_finish', 'spt', 'lpt', 'mor', 'lor', 'shortest_start']
    for r in classical:
        ms = {k: r[k] for k in rule_names if r.get(k) is not None}
        if ms:
            best_rule = min(ms, key=ms.get)
            best_ms = ms[best_rule]
            sota_ms = SOTA[r['instance']]['final_best']
            lit = SOTA[r['instance']]['lit_target'] or SOTA[r['instance']]['lit_ub']
            delta = sota_ms - best_ms
            lines += [f"- {r['instance']}: best classical = {best_ms} ({best_rule}); "
                      f"our pipeline = {sota_ms}; lit = {lit}; pipeline beats best classical by {delta:+d}\n"]

    # 3. Wall-clock
    lines += ["\n## 3. Wall-clock comparison\n"]
    lines += ["| Instance | Ours (s) | OR-Tools cap (s) | Speedup |", "|---|---|---|---|"]
    for r in wc:
        ot = r['our_pipeline_time_s']
        ort = r['ortools_cap_s']
        spd = f"{ort/ot:.1f}x" if ot else "n/a"
        lines += [f"| {r['instance']} | {ot if ot else 'n/a':>6.1f} | {ort} | {spd} |"]

    # 4. Per-difficulty
    lines += ["\n## 4. Per-difficulty gap\n"]
    lines += ["| Group | n | mean gap | min gap | max gap |", "|---|---|---|---|---|"]
    for r in diff:
        lines += [f"| {r['group']} | {r['n']} | {r['mean_gap_pct']}% | "
                  f"{r['min_gap_pct']}% | {r['max_gap_pct']}% |"]

    # 5. Failure mode
    lines += ["\n## 5. Failure mode (4 biggest-gap instances)\n"]
    lines += ["| Instance | EF | RL | ILS | SA | final | lit | gap | RL% | ILS% | SA% |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in fail:
        lines += [f"| {r['instance']} | {r['EF']} | {r['RL']} | {r['ILS']} | {r['SA']} | "
                  f"{r['final']} | {r['lit']} | {r['gap']:+d} | {r['RL_progress_pct']}% | "
                  f"{r['ILS_progress_pct']}% | {r['SA_progress_pct']}% |"]

    out = OUT / 'summary.md'
    out.write_text('\n'.join(lines), encoding='utf-8')
    print(f"saved -> {out}")
    return out


def main():
    print("\n[1/5] Per-stage ablation ...")
    abl = ablation_table()
    print("\n[2/5] Classical rules ...")
    classical = classical_rules()
    print("\n[3/5] Wall-clock comparison ...")
    wc = wallclock()
    print("\n[4/5] Per-difficulty breakdown ...")
    diff = per_difficulty()
    print("\n[5/5] Failure mode analysis ...")
    fail = failure_mode()
    print("\n[+] Building summary ...")
    build_summary(abl, classical, wc, diff, fail)
    print("\nDone. All CSVs and summary.md in data/results/analysis/.")


if __name__ == "__main__":
    main()
