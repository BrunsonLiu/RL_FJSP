"""Unit tests for the behavioral cloning pretraining."""
from __future__ import annotations

import unittest
from pathlib import Path

from fjsp.env import FJSPDispatchEnv
from rl.agents import (
    Demonstration,
    GraphPPOAgent,
    GraphTwoStageActorCriticAgent,
    collect_demonstrations,
    train_imitation,
)


ROOT = Path(__file__).resolve().parents[1]
TINY = ROOT / "data" / "instances" / "tiny_2x2.fjs"
MK01 = ROOT / "data" / "instances" / "brandimarte" / "mk01.txt"


class ImitationTest(unittest.TestCase):
    def test_collect_demonstrations_yields_one_per_step(self) -> None:
        env = FJSPDispatchEnv.from_file(TINY)
        demos = collect_demonstrations(env, rollouts=3, seed=0)
        # tiny_2x2 has 4 operations, so 3 rollouts * 4 = 12 demos
        self.assertEqual(len(demos), 3 * env.instance.operation_count)
        for d in demos:
            self.assertIsInstance(d, Demonstration)
            self.assertGreaterEqual(d.job_position, 0)
            self.assertGreaterEqual(d.machine_position, 0)
            self.assertIsInstance(d.state_snapshot, tuple)

    def test_bc_greedy_matches_earliest_finish_on_tiny(self) -> None:
        # On the tiny instance, BC should recover the dispatch rule
        # (true earliest-finish), not the simple job-by-job greedy.
        from fjsp.scheduler.dispatch_rules import rollout_earliest_finish
        env = FJSPDispatchEnv.from_file(TINY)
        expected = rollout_earliest_finish(env)
        agent, history = train_imitation(
            env, rollouts=20, epochs=10, batch_size=4, hidden_dim=16, seed=0
        )
        result = agent.rollout(env, greedy=True)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.makespan, expected)  # == earliest-finish rollout
        self.assertTrue(history)
        # After 10 epochs the loss should be near zero.
        self.assertLess(history[-1]["total_loss"], 0.1)

    def test_bc_save_load_usable_by_ppo_agent(self) -> None:
        env = FJSPDispatchEnv.from_file(MK01)
        agent, _ = train_imitation(
            env, rollouts=20, epochs=5, batch_size=32, hidden_dim=32, seed=0
        )
        tmp = ROOT / "data" / "results" / "_test_bc.pt"
        try:
            agent.save(tmp)
            # Loading as PPO agent must succeed and produce a valid schedule.
            ppo_agent = GraphPPOAgent.load(tmp, hidden_dim=32)
            result = ppo_agent.rollout(env, greedy=True)
            self.assertTrue(result.is_valid)
        finally:
            if tmp.exists():
                tmp.unlink()

    def test_bc_save_load_usable_by_graph_ac_agent(self) -> None:
        env = FJSPDispatchEnv.from_file(MK01)
        agent, _ = train_imitation(
            env, rollouts=20, epochs=5, batch_size=32, hidden_dim=32, seed=0
        )
        tmp = ROOT / "data" / "results" / "_test_bc_ac.pt"
        try:
            agent.save(tmp)
            ac_agent = GraphTwoStageActorCriticAgent.load(tmp, hidden_dim=32)
            result = ac_agent.rollout(env, greedy=True)
            self.assertTrue(result.is_valid)
        finally:
            if tmp.exists():
                tmp.unlink()


if __name__ == "__main__":
    unittest.main()
