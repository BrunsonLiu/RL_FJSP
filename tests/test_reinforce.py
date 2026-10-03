from __future__ import annotations

from pathlib import Path
import unittest

import torch

from fjsp.env import FJSPDispatchEnv
from rl.agents.reinforce_agent import ReinforceDispatchAgent, train_reinforce


TINY = Path(__file__).resolve().parents[1] / "data/instances/tiny_2x2.fjs"


class ReinforceTest(unittest.TestCase):
    def test_sampled_rollout_has_differentiable_entropy(self) -> None:
        torch.manual_seed(0)
        agent = ReinforceDispatchAgent.create(hidden_dim=16)
        result = agent.rollout(FJSPDispatchEnv.from_file(TINY), seed=0)
        self.assertTrue(result.is_valid)
        self.assertIsNotNone(result.log_prob_sum)
        self.assertIsInstance(result.entropy_sum, torch.Tensor)
        self.assertTrue(result.entropy_sum.requires_grad)
        self.assertTrue(torch.isfinite(result.entropy_sum).item())
        result.entropy_sum.backward()
        gradients = [parameter.grad for parameter in agent.model.parameters()]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all().item() for gradient in gradients))
        self.assertGreater(sum(gradient.abs().sum().item() for gradient in gradients), 0)

    def test_short_training_and_greedy_rollout(self) -> None:
        env = FJSPDispatchEnv.from_file(TINY)
        agent, history = train_reinforce(env, episodes=3, hidden_dim=16, seed=0)
        self.assertEqual(history[-1]["episode"], 3)
        self.assertTrue(all(torch.isfinite(torch.tensor(row["loss"])).item() for row in history))
        result = agent.rollout(env, greedy=True)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.makespan, history[-1]["best_greedy_makespan"])


if __name__ == "__main__":
    unittest.main()
