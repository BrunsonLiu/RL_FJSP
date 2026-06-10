"""Combine SOTA results from both runs into a unified table."""
import json
from pathlib import Path

ROOT = Path(r'd:\desktop2\RL_FJSP')

# Load the baseline (sota_reinforce_ils.json - all 15 instances)
with open(ROOT / 'data' / 'results' / 'sota_reinforce_ils.json') as f:
    base = {r['instance']: r for r in json.load(f)}

# Load the aggressive (mk04, mk05, mk06, mk07, mk09, mk10, mk11, mk12, mk15)
with open(ROOT / 'data' / 'results' / 'aggressive_ils_hard.json') as f:
    aggressive = {r['instance']: r for r in json.load(f)}

# Combine: prefer aggressive results where available
combined = []
for inst_name in [f'mk{i:02d}' for i in range(1, 16)]:
    b = base[inst_name]
    if inst_name in aggressive:
        a = aggressive[inst_name]
        combined.append({
            'instance': inst_name,
            'jobs': b['jobs'],
            'machines': b['machines'],
            'operations': b['operations'],
            'ef_makespan': b['ef_makespan'],
            'reinforce_best': a['reinforce_best'],
            'ils_best': a['ils_best'],
            'sa_makespan': a['sa_makespan'],
            'final_best': a['final_best'],
            'lit_optimum': b['lit_optimum'],
            'lit_ub': b['lit_ub'],
            'lit_target': b['lit_target'],
            'gap_to_lit': a['gap_to_lit'],
        })
    else:
        combined.append({
            'instance': inst_name,
            'jobs': b['jobs'],
            'machines': b['machines'],
            'operations': b['operations'],
            'ef_makespan': b['ef_makespan'],
            'reinforce_best': b['reinforce_best'],
            'ils_best': b['ils_best'],
            'sa_makespan': b['sa_makespan'],
            'final_best': b['final_best'],
            'lit_optimum': b['lit_optimum'],
            'lit_ub': b['lit_ub'],
            'lit_target': b['lit_target'],
            'gap_to_lit': b['gap_to_lit'],
        })

out = ROOT / 'data' / 'results' / 'sota_final.json'
out.write_text(json.dumps(combined, indent=2))

# Print summary
print(
    f"{'inst':<6} {'J×M':<7} {'EF':>5} {'R-best':>7} {'ILS':>5} {'+SA':>5} "
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
    print(
        f"{r['instance']:<6} {r['jobs']}×{r['machines']:<5} {r['ef_makespan']:>5} "
        f"{r['reinforce_best']:>7} {r['ils_best']:>5} {r['sa_makespan']:>5} "
        f"{r['final_best']:>6} {r['lit_target']:>5} {r['gap_to_lit']:>+4} {gap_pct:>4.1f}% {note}"
    )
print(f"\nMean gap: {total_gap_pct/n:.2f}%")
print(f"Matched/tied literature: {sum(1 for r in combined if r['gap_to_lit'] == 0)}/{n}")
print(f"Beat literature: {sum(1 for r in combined if r['gap_to_lit'] < 0)}/{n}")
print(f"Total gap sum: {sum(r['gap_to_lit'] for r in combined)}")
print(f"\nWrote {out}")
