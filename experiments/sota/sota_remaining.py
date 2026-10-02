"""Run the remaining SOTA instances (mk13, mk14, mk15) and merge with the
already-completed results from the log file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sota_reinforce_ils import (
    run_sota_on_instance,
    ILS_ITERS,
    SA_ITERS,
    ROOT,
    _log,
)

# Already-completed results from the previous run.
DONE = [
    {'instance': 'mk01', 'jobs': 10, 'machines': 6, 'operations': 55, 'ef_makespan': 57, 'reinforce_best': 43, 'ils_best': 42, 'sa_makespan': 42, 'final_best': 42, 'lit_optimum': 40, 'lit_ub': None, 'lit_target': 40, 'gap_to_lit': 2, 'time_seconds': 66.2},
    {'instance': 'mk02', 'jobs': 10, 'machines': 6, 'operations': 58, 'ef_makespan': 62, 'reinforce_best': 28, 'ils_best': 28, 'sa_makespan': 28, 'final_best': 28, 'lit_optimum': None, 'lit_ub': 26, 'lit_target': 26, 'gap_to_lit': 2, 'time_seconds': 149.8},
    {'instance': 'mk03', 'jobs': 15, 'machines': 8, 'operations': 150, 'ef_makespan': 331, 'reinforce_best': 216, 'ils_best': 204, 'sa_makespan': 204, 'final_best': 204, 'lit_optimum': 204, 'lit_ub': None, 'lit_target': 204, 'gap_to_lit': 0, 'time_seconds': 451.6},
    {'instance': 'mk04', 'jobs': 15, 'machines': 8, 'operations': 90, 'ef_makespan': 91, 'reinforce_best': 79, 'ils_best': 73, 'sa_makespan': 73, 'final_best': 73, 'lit_optimum': 60, 'lit_ub': None, 'lit_target': 60, 'gap_to_lit': 13, 'time_seconds': 118.8},
    {'instance': 'mk05', 'jobs': 15, 'machines': 4, 'operations': 106, 'ef_makespan': 220, 'reinforce_best': 180, 'ils_best': 175, 'sa_makespan': 175, 'final_best': 175, 'lit_optimum': None, 'lit_ub': 172, 'lit_target': 172, 'gap_to_lit': 3, 'time_seconds': 173.7},
    {'instance': 'mk06', 'jobs': 10, 'machines': 15, 'operations': 150, 'ef_makespan': 79, 'reinforce_best': 69, 'ils_best': 68, 'sa_makespan': 68, 'final_best': 68, 'lit_optimum': None, 'lit_ub': 58, 'lit_target': 58, 'gap_to_lit': 10, 'time_seconds': 433.4},
    {'instance': 'mk07', 'jobs': 20, 'machines': 5, 'operations': 100, 'ef_makespan': 204, 'reinforce_best': 154, 'ils_best': 145, 'sa_makespan': 145, 'final_best': 145, 'lit_optimum': None, 'lit_ub': 139, 'lit_target': 139, 'gap_to_lit': 6, 'time_seconds': 266.8},
    {'instance': 'mk08', 'jobs': 20, 'machines': 10, 'operations': 225, 'ef_makespan': 618, 'reinforce_best': 533, 'ils_best': 523, 'sa_makespan': 523, 'final_best': 523, 'lit_optimum': 523, 'lit_ub': None, 'lit_target': 523, 'gap_to_lit': 0, 'time_seconds': 251.0},
    {'instance': 'mk09', 'jobs': 20, 'machines': 10, 'operations': 240, 'ef_makespan': 433, 'reinforce_best': 339, 'ils_best': 338, 'sa_makespan': 338, 'final_best': 338, 'lit_optimum': 307, 'lit_ub': None, 'lit_target': 307, 'gap_to_lit': 31, 'time_seconds': 594.4},
    {'instance': 'mk10', 'jobs': 20, 'machines': 15, 'operations': 240, 'ef_makespan': 406, 'reinforce_best': 242, 'ils_best': 230, 'sa_makespan': 230, 'final_best': 230, 'lit_optimum': None, 'lit_ub': 197, 'lit_target': 197, 'gap_to_lit': 33, 'time_seconds': 776.9},
    {'instance': 'mk11', 'jobs': 30, 'machines': 5, 'operations': 179, 'ef_makespan': 706, 'reinforce_best': 639, 'ils_best': 620, 'sa_makespan': 620, 'final_best': 620, 'lit_optimum': None, 'lit_ub': 615, 'lit_target': 615, 'gap_to_lit': 5, 'time_seconds': 237.5},
    {'instance': 'mk12', 'jobs': 30, 'machines': 10, 'operations': 193, 'ef_makespan': 700, 'reinforce_best': 531, 'ils_best': 524, 'sa_makespan': 524, 'final_best': 524, 'lit_optimum': 508, 'lit_ub': None, 'lit_target': 508, 'gap_to_lit': 16, 'time_seconds': 254.3},
]

REMAINING = [13, 14, 15]


def main() -> None:
    log_path = ROOT / 'data' / 'results' / 'sota_reinforce_ils.log'
    global _LOG_HANDLE
    from sota_reinforce_ils import _LOG_HANDLE
    _LOG_HANDLE = open(log_path, 'a')

    _log("")
    _log("--- Remaining instances (mk13, mk14, mk15) ---")

    results = list(DONE)
    for mk in REMAINING:
        n_ils = ILS_ITERS.get(mk, 20)
        n_sa = SA_ITERS.get(mk, 1500)
        result = run_sota_on_instance(mk, n_ils=n_ils, n_sa=n_sa)
        results.append(result)
        gap = result['gap_to_lit']
        gap_str = f"{gap:+d}" if gap is not None else "-"
        lit_str = f"{result['lit_target']}" if result['lit_target'] is not None else "-"
        _log(
            f"{result['instance']:<6} {result['ef_makespan']:>5} {result['reinforce_best']:>7} "
            f"{result['ils_best']:>9} {result['sa_makespan']:>5} {result['final_best']:>6} "
            f"{lit_str:>5} {gap_str:>4} {result['time_seconds']:>7.1f}"
        )

    out = ROOT / 'data' / 'results' / 'sota_reinforce_ils.json'
    out.write_text(json.dumps(results, indent=2))
    _log(f"\nWrote {out}")
    _LOG_HANDLE.close()


if __name__ == "__main__":
    main()
