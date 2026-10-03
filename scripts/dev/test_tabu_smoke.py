"""Quick smoke test for tabu_search on mk01."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fjsp.env import FJSPDispatchEnv
from fjsp.scheduler.dispatch_rules import choose_earliest_finish
from fjsp.scheduler.local_search import (
    local_search, tabu_search, simulated_annealing,
    iterated_local_search,
)
from fjsp.scheduler.validator import validate_schedule


def collect_ef_schedule(env):
    """Run EF dispatch and collect the resulting schedule."""
    env.reset()
    schedule = []
    while not env.done:
        action = choose_earliest_finish(env)
        _, _, _, info = env.step(action)
        schedule.append(info['scheduled'])
    result = env.validate()
    return schedule, result.makespan


def main():
    env = FJSPDispatchEnv.from_file(str(Path(__file__).resolve().parents[2] / "data/instances/brandimarte/mk01.txt"))
    print('Instance:', env.instance.job_count, 'jobs,',
          env.instance.machine_count, 'machines,',
          env.instance.operation_count, 'ops')

    # 1. EF rollout
    schedule, ef = collect_ef_schedule(env)
    print(f'EF rollout: {ef}')

    # 2. Local search
    t0 = time.perf_counter()
    schedule, ms = local_search(env.instance, schedule, max_iterations=50)
    t1 = time.perf_counter()
    print(f'After LS: {ms} (took {t1-t0:.2f}s)')

    # 3. Tabu search from the LS result
    t0 = time.perf_counter()
    schedule, ms = tabu_search(env.instance, schedule, max_iterations=500)
    t1 = time.perf_counter()
    print(f'After TS (500 it): {ms} (took {t1-t0:.2f}s)')
    validated = validate_schedule(env.instance, schedule)
    print(f'Validated: valid={validated.is_valid}, makespan={validated.makespan}')

    # 4. Final SA refinement
    t0 = time.perf_counter()
    schedule, ms = simulated_annealing(env.instance, schedule, max_total_iterations=2000)
    t1 = time.perf_counter()
    print(f'After SA: {ms} (took {t1-t0:.2f}s)')


if __name__ == '__main__':
    main()
