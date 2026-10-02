"""Merge tabu search improvements into sota_final.json.

For each instance in tabu_results.json that improves on sota_final,
update the FINAL column and the R-best/+ILS/+SA columns to reflect
the TS-improved result.

Writes the updated sota_final.json and prints a summary.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

with open(ROOT / 'data' / 'results' / 'sota_final.json') as f:
    sota = {r['instance']: r for r in json.load(f)}

with open(ROOT / 'data' / 'results' / 'tabu_results.json') as f:
    ts_results = json.load(f)

updates = []
for r in ts_results:
    name = r['instance']
    if name not in sota:
        continue
    s = sota[name]
    ts_final = r['final_best']
    ts_ms = r['ts_ms']
    ils_ms = r['ils_ms']
    old_final = s['final_best']
    if ts_final < old_final:
        # Update sota entry: TS beat the previous best
        # Keep +SA from old SOTA (TS may already have done its own SA internally)
        s['ils_best'] = ils_ms  # ILS that fed TS
        s['ts_makespan'] = ts_ms  # TS result
        s['final_best'] = ts_final
        s['gap_to_lit'] = ts_final - s['lit_target'] if s['lit_target'] else None
        updates.append((name, old_final, ts_final))

out = ROOT / 'data' / 'results' / 'sota_final.json'
combined = [sota[f'mk{i:02d}'] for i in range(1, 16)]
out.write_text(json.dumps(combined, indent=2))

# Print summary
print(
    f"{'inst':<6} {'old':>5} {'new':>5} {'Δ':>4}"
)
for name, old, new in updates:
    print(f"{name:<6} {old:>5} {new:>5} {new - old:+d}")

print(f"\nUpdated {len(updates)} instances in {out}")

# Print full table
print('\nUpdated SOTA matrix:')
print(
    f"{'inst':<6} {'EF':>5} {'R-best':>7} {'ILS':>5} {'+SA':>5} "
    f"{'FINAL':>6} {'lit':>5} {'Δ':>4} {'Δ%':>5} {'notes'}"
)
total_gap_pct = 0
total_gap = 0
n = 0
n_tied = 0
n_beat = 0
for r in combined:
    gap = r['gap_to_lit']
    gap_pct = 100 * gap / r['lit_target'] if r['lit_target'] else 0
    total_gap_pct += gap_pct
    total_gap += gap
    n += 1
    if gap == 0:
        n_tied += 1
    elif gap < 0:
        n_beat += 1
    note = ""
    if gap == 0:
        note = "TIED"
    elif gap < 0:
        note = "**NEW SOTA**"
    print(
        f"{r['instance']:<6} {r['ef_makespan']:>5} {r['reinforce_best']:>7} "
        f"{r['ils_best']:>5} {r.get('sa_makespan', r['ils_best']):>5} "
        f"{r['final_best']:>6} {r['lit_target']:>5} {gap:>+4} {gap_pct:>4.1f}% {note}"
    )
print(f"\nMean gap: {total_gap_pct/n:.2f}%")
print(f"Total absolute gap: {total_gap} makespan units")
print(f"Tied literature: {n_tied}/{n}")
print(f"Beat literature: {n_beat}/{n}")
