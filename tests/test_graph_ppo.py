from __future__ import annotations

import unittest
from pathlib import Path
from random import Random

from fjsp.env import FJSPDispatchEnv
from fjsp.scheduler.validator import schedule_to_dict
from rl.agents.graph_actor_critic import GraphTwoStageActorCriticAgent
from rl.agents.graph_ppo import GraphPPOAgent, train_graph_ppo


ROOT = Path(__file__).resolve().parents[1]
TINY = ROOT / "data" / "instances" / "tiny_2x2.fjs"


class GraphPPOTest(unittest.TestCase):
    def test_evaluate_action_matches_sampled_action(self) -> None:
        env = FJSPDispatchEnv.from_file(TINY)
        agent = GraphPPOAgent.create(hidden_dim=16)

        env.reset()
        sampled_action, sampled_log_prob, _, _ = agent.select_action(env, greedy=False, rng=Random(0))
        log_prob_eval, value, entropy = agent.evaluate_action(env, sampled_action)

        self.assertEqual(value.dim(), 0)
        self.assertEqual(entropy.dim(), 0)
        self.assertEqual(log_prob_eval.dim(), 0)
        self.assertAlmostEqual(float(log_prob_eval.item()), float(sampled_log_prob.item()), places=5)

    def test_greedy_rollout_is_valid(self) -> None:
        env = FJSPDispatchEnv.from_file(TINY)
        agent = GraphPPOAgent.create(hidden_dim=16)

        result = agent.rollout(env, greedy=True)

        self.assertTrue(result.is_valid)
        self.assertEqual(len(env.schedule), env.instance.operation_count)
        schedule = schedule_to_dict(env.schedule)
        self.assertEqual(len(schedule["operations"]), env.instance.operation_count)

    def test_short_training_returns_best_history(self) -> None:
        env = FJSPDispatchEnv.from_file(TINY)
        agent, history = train_graph_ppo(
            env, episodes=3, hidden_dim=16, seed=0, K_epochs=2, minibatch_size=2
        )
        result = agent.rollout(env, greedy=True)

        self.assertTrue(result.is_valid)
        self.assertTrue(history)
        self.assertIn("best_greedy_makespan", history[-1])
        self.assertIn("policy_loss", history[-1])
        self.assertIn("value_loss", history[-1])

    def test_save_load_roundtrip(self) -> None:
        env = FJSPDispatchEnv.from_file(TINY)
        agent = GraphPPOAgent.create(hidden_dim=16)
        tmp = ROOT / "data" / "results" / "_test_graph_ppo.pt"
        try:
            agent.save(tmp)
            loaded = GraphPPOAgent.load(tmp, hidden_dim=16)
            result = loaded.rollout(env, greedy=True)
            self.assertTrue(result.is_valid)
        finally:
            if tmp.exists():
                tmp.unlink()


if __name__ == "__main__":
    unittest.main()
