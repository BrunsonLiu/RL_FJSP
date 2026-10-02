"""Quick smoke test of ILS/SA on mk01."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fjsp.scheduler import (
    local_search, iterated_local_search, ils_sa_hybrid, simulated_annealing,
    neh_construct, random_schedule, random_perturb, critical_path_perturb
)
from fjsp.parser.fjs_parser import parse_fjs
from fjsp.scheduler.validator import validate_schedule

# Quick test on mk01
inst = parse_fjs(str(Path(__file__).resolve().parents[2] / "data/instances/brandimarte/mk01.txt"))
print(f'Instance: {inst.job_count} jobs, {inst.machine_count} machines, {inst.operation_count} ops')

# Build a starting schedule: random + neh
rand = random_schedule(inst, seed=0)
neh = neh_construct(inst)
print(f'Random makespan: {validate_schedule(inst, rand).makespan}')
print(f'NEH makespan: {validate_schedule(inst, neh).makespan}')

# LS on neh
ls_sched, ls_ms = local_search(inst, neh, max_iterations=50)
print(f'NEH + LS: {ls_ms}')

# ILS
ils_sched, ils_ms = iterated_local_search(
    inst, neh, n_iterations=5, perturb_n_swaps=5, max_ls_iterations=30, perturb_mode='critical'
)
print(f'NEH + ILS(5): {ils_ms}')

# ILS+SA
hyb_sched, hyb_ms = ils_sa_hybrid(
    inst, neh, n_ils_iterations=5, n_sa_iterations=1000, perturb_n_swaps=5, max_ls_iterations=30
)
print(f'NEH + ILS+SA: {hyb_ms}')

# ILS with more iterations
ils2_sched, ils2_ms = iterated_local_search(
    inst, neh, n_iterations=20, perturb_n_swaps=10, max_ls_iterations=50, perturb_mode='critical'
)
print(f'NEH + ILS(20, perturb=10): {ils2_ms}')
