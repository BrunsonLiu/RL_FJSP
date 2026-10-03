"""Quick end-to-end test for BARI with dual-perspective encoder."""
import time
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fjsp.parser.fjs_parser import parse_fjs
from fjsp.scheduler.validator import earliest_finish_schedule
from fjsp.env.improvement_env import FJSPImprovementEnv
from rl.models.fjsp_l2s import FJSPImproveNet
from rl.agents.fjsp_l2s import FJSPImproveAgent

def main():
    inst_path = str(Path(__file__).resolve().parents[2] / "data/instances/brandimarte/mk01.txt")
    inst = parse_fjs(inst_path)
    initial = earliest_finish_schedule(inst)
    initial_ms = max(o.end for o in initial)
    print(f"Instance: {Path(inst_path).name}")
    print(f"  jobs={inst.job_count}, machines={inst.machine_count}, ops={inst.operation_count}")
    print(f"  EF makespan: {initial_ms}")

    env = FJSPImprovementEnv(inst, initial_schedule=initial, max_steps=50, patience=15)
    obs = env.reset()

    # Test dual-perspective
    for use_dual in [True, False]:
        net = FJSPImproveNet(hidden_dim=64, use_dual_perspective=use_dual)
        agent = FJSPImproveAgent(net)

        env.reset(initial_schedule=initial)
        t0 = time.time()
        steps = 0
        while not env.done:
            action_idx, _, _, _ = agent.select_action(env, deterministic=False)
            if action_idx < 0:
                break
            env.step(action_idx)
            steps += 1
        dt = time.time() - t0
        print(f"  [{'dual' if use_dual else 'single'}] steps={steps}, makespan={env.makespan}, best={env.best_makespan}, time={dt:.2f}s")

    # Test feature dims
    from fjsp.graph.solution_graph import IMP_OP_FEATURE_DIM
    print(f"\nIMP_OP_FEATURE_DIM = {IMP_OP_FEATURE_DIM}")
    graph = obs["graph"]
    print(f"  op_features shape: {graph.op_features.shape}")
    print(f"  bottleneck_score sample: {graph.op_features[:5, 16].tolist()}")
    print(f"  cp_distance sample: {graph.op_features[:5, 17].tolist()}")
    print("\nOK: BARI dual-perspective encoder works end-to-end.")

if __name__ == "__main__":
    main()
