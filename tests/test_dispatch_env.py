from __future__ import annotations

import unittest
from pathlib import Path

from fjsp.env import DispatchAction, FJSPDispatchEnv


ROOT = Path(__file__).resolve().parents[1]
TINY = ROOT / "data" / "instances" / "tiny_2x2.fjs"
MK01 = ROOT / "data" / "instances" / "brandimarte" / "mk01.txt"


class DispatchEnvTest(unittest.TestCase):
    def test_tiny_rollout_is_valid(self) -> None:
        env = FJSPDispatchEnv.from_file(TINY)

        while not env.done:
            action = min(env.available_actions(), key=lambda item: (item.job, item.machine))
            env.step(action)

        result = env.validate()
        self.assertTrue(result.is_valid, result.errors)
        self.assertEqual(len(env.schedule), env.instance.operation_count)

    def test_invalid_action_does_not_mutate_state(self) -> None:
        env = FJSPDispatchEnv.from_file(TINY)
        before = env.observe()

        _, reward, done, info = env.step(DispatchAction(job=999, machine=999))

        self.assertFalse(done)
        self.assertLess(reward, 0)
        self.assertEqual(env.observe(), before)
        self.assertEqual(env.schedule, [])
        self.assertTrue(info["invalid_action"])

    def test_valid_action_mask_shape(self) -> None:
        env = FJSPDispatchEnv.from_file(MK01)
        mask = env.valid_action_mask()

        self.assertEqual(len(mask), env.instance.job_count)
        self.assertTrue(all(len(row) == env.instance.machine_count for row in mask))
        self.assertEqual(sum(sum(row) for row in mask), len(env.available_actions()))


if __name__ == "__main__":
    unittest.main()

