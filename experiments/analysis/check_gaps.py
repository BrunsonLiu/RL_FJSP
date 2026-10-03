import json
from pathlib import Path
data = json.load(open(str(Path(__file__).resolve().parents[2] / "data/results/sota_final.json")))
# Sort by gap (descending)
data.sort(key=lambda x: -x['gap_to_lit'])
print('Top gaps to literature (positive = worse than lit):')
for r in data[:5]:
    print(f"  {r['instance']:>6}  final={r['final_best']:>3}  lit={r['lit_target']:>3}  gap=+{r['gap_to_lit']}")
print()
print('Best results (gap <= 0):')
for r in data:
    if r['gap_to_lit'] <= 0:
        print(f"  {r['instance']:>6}  final={r['final_best']:>3}  lit={r['lit_target']:>3}  gap={r['gap_to_lit']:+d}")
