from __future__ import annotations

import unittest
from pathlib import Path

from fjsp.env import FJSPDispatchEnv
from fjsp.graph.operation_machine_graph import build_operation_machine_graph
from rl.agents.graph_actor_critic import GraphTwoStageActorCriticAgent, train_graph_actor_critic


ROOT = Path(__file__).resolve().parents[1]
TINY = ROOT / "data" / "instances" / "tiny_2x2.fjs"


class GraphActorCriticTest(unittest.TestCase):
    def test_graph_shapes(self) -> None:
        env = FJSPDispatchEnv.from_file(TINY)
        graph = build_operation_machine_graph(env)

        self.assertEqual(graph.op_features.shape[0], env.instance.operation_count)
        self.assertEqual(graph.machine_features.shape[0], env.instance.machine_count)
        self.assertTrue(graph.eligibility_edges.numel() > 0)
        self.assertEqual(len(graph.next_op_indices), env.instance.job_count)

    def test_greedy_rollout_is_valid(self) -> None:
        env = FJSPDispatchEnv.from_file(TINY)
        agent = GraphTwoStageActorCriticAgent.create(hidden_dim=16)

        result = agent.rollout(env, greedy=True)

        self.assertTrue(result.is_valid)
        self.assertEqual(len(env.schedule), env.instance.operation_count)

    def test_short_training_returns_best_history(self) -> None:
        env = FJSPDispatchEnv.from_file(TINY)
        agent, history = train_graph_actor_critic(env, episodes=2, hidden_dim=16, seed=0)
        result = agent.rollout(env, greedy=True)

        self.assertTrue(result.is_valid)
        self.assertTrue(history)
        self.assertIn("best_greedy_makespan", history[-1])


if __name__ == "__main__":
    unittest.main()

