"""Build a per-instance comparison table:
   ours_vs_L2I_vs_lit
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOTA = {r['instance']: r for r in json.load(open(ROOT / 'data/results/sota_final.json'))}
L2I = {r['instance']: r for r in json.load(open(ROOT / 'data/results/l2i_baseline.json'))}

rows = []
for mk in range(1, 16):
    name = f"mk{mk:02d}"
    s = SOTA[name]
    l = L2I[name]
    lit = s['lit_target'] or s['lit_ub'] or s['lit_optimum']
    rows.append({
        'instance': name,
        'EF': s['ef_makespan'],
        'L2I_best': l['l2i_best'],
        'L2I_init': l['init_ms'],
        'ours_final': s['final_best'],
        'lit': lit,
        'L2I_gap_pct': round(100 * (l['l2i_best'] - lit) / lit, 2) if lit else None,
        'ours_gap_pct': round(100 * (s['final_best'] - lit) / lit, 2) if lit else None,
        'ours_minus_L2I': s['final_best'] - l['l2i_best'],
    })

# Markdown table
md = ["# L2I vs. ours — per-instance comparison\n"]
md += ["| Inst | EF | L2I | Ours | Lit | L2I gap | Ours gap | Ours − L2I |",
       "|---|---|---|---|---|---|---|---|"]
for r in rows:
    md.append(f"| {r['instance']} | {r['EF']} | {r['L2I_best']} | {r['ours_final']} | "
              f"{r['lit']} | {r['L2I_gap_pct']:+.2f}% | {r['ours_gap_pct']:+.2f}% | "
              f"{r['ours_minus_L2I']:+d} |")

# Aggregate
mean_l2i_gap = sum(r['L2I_gap_pct'] for r in rows) / 15
mean_ours_gap = sum(r['ours_gap_pct'] for r in rows) / 15
md += [f"\n**Mean gap:** L2I = {mean_l2i_gap:.2f}%, ours = {mean_ours_gap:.2f}%",
       f"**Mean absolute diff:** ours − L2I = {sum(r['ours_minus_L2I'] for r in rows) / 15:.1f}"]
md += ["\n## Notes\n",
       "L2I is trained from scratch (random schedule init) with 5 seeds × 200 steps, "
       "REINFORCE, 10-D state, 4-op action space (reassign, swap_machine, swap_order, swap_order_across).",
       "L2I does NOT use any pretrained RL initial schedule, ILS perturbation, or post-processing.",
       "This is the most direct comparison of the L2I paradigm (operator selection RL) on FJSP."]

out = ROOT / 'data/results/analysis/l2i_comparison.md'
out.write_text('\n'.join(md), encoding='utf-8')
out_csv = ROOT / 'data/results/analysis/l2i_comparison.csv'
import csv
with open(out_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

# Print to stdout with utf-8 (avoid Windows gbk issue)
import sys
sys.stdout.reconfigure(encoding='utf-8')
print('\n'.join(md))
print(f"\nSaved -> {out}")
print(f"Saved -> {out_csv}")
