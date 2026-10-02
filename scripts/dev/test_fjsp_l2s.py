"""Smoke test for the FJSP-L2S improvement environment and neural network."""
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fjsp.env.improvement_env import FJSPImprovementEnv, ImprovementMove
from fjsp.graph.solution_graph import build_solution_graph, find_critical_path
from fjsp.parser.fjs_parser import parse_fjs
from fjsp.scheduler.validator import greedy_schedule


def test_improvement_env():
    """Test the improvement environment end-to-end."""
    print("=== Testing FJSP Improvement Environment ===")

    inst = parse_fjs(str(Path(__file__).resolve().parents[2] / "data/instances/brandimarte/mk01.txt"))
    print(f"Instance: jobs={inst.job_count}, machines={inst.machine_count}, ops={inst.operation_count}")

    initial = greedy_schedule(inst)
    initial_ms = max(o.end for o in initial)
    print(f"Initial greedy makespan: {initial_ms}")

    # Critical path
    cp, cp_indices = find_critical_path(initial, inst)
    print(f"Critical path length: {len(cp)} ops")

    # Solution graph
    graph = build_solution_graph(inst, initial)
    print(f"Graph: {graph.op_features.shape[0]} ops, {graph.machine_features.shape[0]} machines")
    print(f"  precedence edges: {graph.precedence_edges.shape[0]}")
    print(f"  machine_seq edges: {graph.machine_seq_edges.shape[0]}")
    print(f"  eligibility edges: {graph.eligibility_edges.shape[0]}")
    print(f"  critical ops: {len(graph.critical_op_indices)}")

    # Environment
    env = FJSPImprovementEnv(inst, initial_schedule=initial, max_steps=20, patience=10)
    obs = env.reset()
    print(f"Valid moves: {len(obs['valid_moves'])}")
    move_types = {"coupled_reassign": 0, "swap_prev": 0, "swap_next": 0}
    for m in obs["valid_moves"]:
        move_types[m.move_type] += 1
    print(f"  Move types: {move_types}")

    # Random improvement steps
    rng = random.Random(42)
    best_ms = initial_ms
    for step in range(20):
        obs_v = env.observe()
        if not obs_v["valid_moves"]:
            print(f"  No valid moves at step {step}")
            break
        action = rng.randint(0, len(obs_v["valid_moves"]) - 1)
        obs2, reward, done, info = env.step(action)
        if info.get("delta", 0) > 0:
            print(f"  Step {step}: IMPROVED ms={info['makespan']} delta={info['delta']}")
        if info.get("delta", 0) < 0:
            print(f"  Step {step}: worsened ms={info['makespan']} delta={info['delta']}")
        best_ms = min(best_ms, info["makespan"])
        if done:
            break

    print(f"Final makespan: {env.makespan} (initial: {initial_ms}, best: {best_ms})")
    print("Environment test PASSED!\n")
    return True


def test_neural_network():
    """Test the neural network forward pass."""
    print("=== Testing FJSPImproveNet ===")
    import torch
    from rl.models.fjsp_l2s import FJSPImproveNet

    inst = parse_fjs(str(Path(__file__).resolve().parents[2] / "data/instances/brandimarte/mk01.txt"))
    initial = greedy_schedule(inst)
    graph = build_solution_graph(inst, initial)

    env = FJSPImprovementEnv(inst, initial_schedule=initial, max_steps=10)
    obs = env.reset()
    moves = obs["valid_moves"]

    net = FJSPImproveNet(hidden_dim=64, num_mp_rounds=2, num_transformer_blocks=1, num_heads=4, ffn_dim=128)
    print(f"Network parameters: {sum(p.numel() for p in net.parameters()):,}")

    # Forward pass
    op_h, machine_h, critical_mask = net.encode(graph)
    print(f"Encoded: op_h={op_h.shape}, machine_h={machine_h.shape}, critical_mask={critical_mask.shape}")
    print(f"Critical ops: {critical_mask.sum().item()}/{critical_mask.size(0)}")

    # Score moves
    from fjsp.scheduler.local_search import _assignments
    assignments = _assignments(env.schedule)
    scores = net.score_moves(op_h, machine_h, graph, moves, assignments)
    print(f"Move scores: {scores.shape}, range=[{scores.min().item():.2f}, {scores.max().item():.2f}]")

    # Value
    value = net.value(graph, op_h, machine_h)
    print(f"Value estimate: {value.item():.4f}")

    # Test action selection
    from rl.agents.fjsp_l2s import FJSPImproveAgent
    agent = FJSPImproveAgent(net)
    action_idx, log_prob, val, entropy = agent.select_action(env, deterministic=True)
    print(f"Selected action: {action_idx}, log_prob={log_prob:.4f}, value={val:.4f}, entropy={entropy:.4f}")

    print("Neural network test PASSED!\n")
    return True


def test_training_one_step():
    """Test one training step."""
    print("=== Testing Training (1 episode) ===")
    from rl.models.fjsp_l2s import FJSPImproveNet
    from rl.agents.fjsp_l2s import FJSPImproveAgent

    inst = parse_fjs(str(Path(__file__).resolve().parents[2] / "data/instances/brandimarte/mk01.txt"))
    initial = greedy_schedule(inst)
    initial_ms = max(o.end for o in initial)

    net = FJSPImproveNet(hidden_dim=64, num_mp_rounds=2, num_transformer_blocks=1, num_heads=4, ffn_dim=128)
    agent = FJSPImproveAgent(net, lr=1e-3)
    env = FJSPImprovementEnv(inst, initial_schedule=initial, max_steps=10, patience=5)

    # Collect trajectory
    trajectory = agent.collect_trajectory(env)
    print(f"Trajectory length: {len(trajectory)}")

    if trajectory:
        # PPO update
        metrics = agent.ppo_update(trajectory, k_epochs=2)
        print(f"Training metrics: {metrics}")
        print(f"Final makespan: {env.makespan} (initial: {initial_ms})")

    print("Training test PASSED!\n")
    return True


if __name__ == "__main__":
    try:
        ok = True
        ok = test_improvement_env() and ok
        ok = test_neural_network() and ok
        ok = test_training_one_step() and ok
        if ok:
            print("ALL SMOKE TESTS PASSED!")
        else:
            print("SOME TESTS FAILED!")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
