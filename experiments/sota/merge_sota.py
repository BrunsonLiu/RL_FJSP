"""Combine SOTA results from both runs into a unified table.

Reports mean ± std for REINFORCE across 5 seeds (not just best-of-5).
This addresses the statistical reporting issue: best-of-5 hides variance.
"""
import json
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[2]

# Load the baseline (sota_reinforce_ils.json - all 15 instances)
with open(ROOT / 'data' / 'results' / 'sota_reinforce_ils.json') as f:
    base = {r['instance']: r for r in json.load(f)}

# Load the aggressive (mk04, mk05, mk06, mk07, mk09, mk10, mk11, mk12, mk15)
with open(ROOT / 'data' / 'results' / 'aggressive_ils_hard.json') as f:
    aggressive = {r['instance']: r for r in json.load(f)}

# Load 5-seed data for mean ± std
seed_files = [
    ROOT / 'data' / 'results' / 'reinforce_mk01_mk05_5seed.json',
    ROOT / 'data' / 'results' / 'reinforce_mk06_mk10_mk13_mk15_5seed.json',
    ROOT / 'data' / 'results' / 'reinforce_mk11_mk12_mk14_5seed.json',
]
seed_data = {}
for sf in seed_files:
    if sf.exists():
        for r in json.loads(sf.read_text()):
            seed_data[r['instance']] = r

# Combine: prefer aggressive results where available
combined = []
for inst_name in [f'mk{i:02d}' for i in range(1, 16)]:
    b = base[inst_name]
    a = aggressive.get(inst_name, b)
    sd = seed_data.get(inst_name, {})

    reinforce_best = a.get('reinforce_best', b['reinforce_best'])
    reinforce_mean = sd.get('reinforce_mean')
    reinforce_std = sd.get('reinforce_std')

    entry = {
        'instance': inst_name,
        'jobs': b['jobs'],
        'machines': b['machines'],
        'operations': b['operations'],
        'ef_makespan': b['ef_makespan'],
        'reinforce_best': reinforce_best,
        'reinforce_mean': reinforce_mean,
        'reinforce_std': reinforce_std,
        'ils_best': a.get('ils_best', b['ils_best']),
        'sa_makespan': a.get('sa_makespan', b['sa_makespan']),
        'final_best': a.get('final_best', b['final_best']),
        'lit_optimum': b['lit_optimum'],
        'lit_ub': b['lit_ub'],
        'lit_target': b['lit_target'],
        'gap_to_lit': a.get('gap_to_lit', b['gap_to_lit']),
    }
    # Mark if SA made things worse (shouldn't happen after fix, but track it)
    ils = entry['ils_best']
    sa = entry['sa_makespan']
    entry['sa_regression'] = sa > ils if sa is not None and ils is not None else False
    combined.append(entry)

out = ROOT / 'data' / 'results' / 'sota_final.json'
out.write_text(json.dumps(combined, indent=2))

# Print summary with mean ± std
print(
    f"{'inst':<6} {'J×M':<7} {'EF':>5} {'R-best':>7} {'R-mean±std':>14} {'ILS':>5} {'+SA':>5} "
    f"{'FINAL':>6} {'lit':>5} {'Δ':>4} {'Δ%':>5} {'notes'}"
)
total_gap_pct = 0
n = 0
for r in combined:
    gap_pct = 100 * r['gap_to_lit'] / r['lit_target'] if r['lit_target'] else 0
    total_gap_pct += gap_pct
    n += 1
    note = ""
    if r['gap_to_lit'] == 0:
        note = "TIED"
    elif r['gap_to_lit'] < 0:
        note = "**NEW SOTA**"
    if r.get('sa_regression'):
        note += " [SA REGRESSION]"

    if r['reinforce_mean'] is not None and r['reinforce_std'] is not None:
        ms_str = f"{r['reinforce_mean']:.1f}±{r['reinforce_std']:.1f}"
    else:
        ms_str = "-"

    print(
        f"{r['instance']:<6} {r['jobs']}×{r['machines']:<5} {r['ef_makespan']:>5} "
        f"{r['reinforce_best']:>7} {ms_str:>14} {r['ils_best']:>5} {r['sa_makespan']:>5} "
        f"{r['final_best']:>6} {r['lit_target']:>5} {r['gap_to_lit']:>+4} {gap_pct:>4.1f}% {note}"
    )
print(f"\nMean gap: {total_gap_pct/n:.2f}%")
print(f"Matched/tied literature: {sum(1 for r in combined if r['gap_to_lit'] == 0)}/{n}")
print(f"Beat literature: {sum(1 for r in combined if r['gap_to_lit'] < 0)}/{n}")
print(f"Total gap sum: {sum(r['gap_to_lit'] for r in combined)}")
sa_regressions = sum(1 for r in combined if r.get('sa_regression'))
if sa_regressions:
    print(f"WARNING: {sa_regressions} instances where SA regressed from ILS — check pipeline")
print(f"\nWrote {out}")
