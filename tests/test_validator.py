from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

from fjsp.parser.fjs_parser import FJSPInstance, Job, Operation, OperationOption, parse_fjs
from fjsp.scheduler.validator import (
    ScheduledOperation,
    greedy_schedule,
    schedule_from_dict,
    schedule_to_dict,
    validate_schedule,
)


ROOT = Path(__file__).resolve().parents[1]


class ValidatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.instance = FJSPInstance(
            jobs=(
                Job((Operation((OperationOption(0, 3),)), Operation((OperationOption(1, 2),)))),
                Job((Operation((OperationOption(0, 2),)),)),
            ),
            machine_count=2,
        )
        self.schedule = [
            ScheduledOperation(0, 0, 0, 0, 3),
            ScheduledOperation(0, 1, 1, 3, 5),
            ScheduledOperation(1, 0, 0, 3, 5),
        ]

    def assert_invalid(self, schedule: list[ScheduledOperation], fragment: str) -> None:
        result = validate_schedule(self.instance, schedule)
        self.assertFalse(result.is_valid)
        self.assertTrue(any(fragment in error for error in result.errors), result.errors)

    def test_valid_schedule_and_json_roundtrip(self) -> None:
        restored = schedule_from_dict(schedule_to_dict(self.schedule))
        self.assertEqual(restored, self.schedule)
        result = validate_schedule(self.instance, restored)
        self.assertTrue(result.is_valid, result.errors)
        self.assertEqual(result.makespan, 5)

    def test_missing_and_duplicate_operations(self) -> None:
        self.assert_invalid(self.schedule[:-1], "Missing operation")
        self.assert_invalid(self.schedule + [self.schedule[0]], "Duplicate operation")

    def test_precedence_violation(self) -> None:
        schedule = list(self.schedule)
        schedule[1] = replace(schedule[1], start=2, end=4)
        self.assert_invalid(schedule, "Precedence violation")

    def test_machine_overlap(self) -> None:
        schedule = list(self.schedule)
        schedule[2] = replace(schedule[2], start=2, end=4)
        self.assert_invalid(schedule, "Machine overlap")

    def test_ineligible_machine_and_wrong_duration(self) -> None:
        for operation, fragment in (
            (replace(self.schedule[0], machine=1), "cannot run"),
            (replace(self.schedule[0], end=2), "expected 3"),
        ):
            with self.subTest(fragment=fragment):
                self.assert_invalid([operation] + self.schedule[1:], fragment)

    def test_bad_indices_and_time_windows(self) -> None:
        for operation, fragment in (
            (replace(self.schedule[0], job=-1), "Unknown job"),
            (replace(self.schedule[0], op=99), "Unknown operation"),
            (replace(self.schedule[0], machine=99), "Unknown machine"),
            (replace(self.schedule[0], start=-1), "Invalid time window"),
            (replace(self.schedule[0], end=0), "Invalid time window"),
        ):
            with self.subTest(fragment=fragment):
                self.assert_invalid([operation] + self.schedule[1:], fragment)

    def test_malformed_json_rejected(self) -> None:
        for payload in ({}, {"operations": {}}, {"operations": [None]}, {"operations": [{}]}):
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    schedule_from_dict(payload)

    def test_all_benchmark_greedy_schedules_are_legal(self) -> None:
        paths = sorted((ROOT / "data/instances/brandimarte").glob("mk*.txt"))
        self.assertEqual(len(paths), 15)
        for path in paths:
            with self.subTest(instance=path.name):
                instance = parse_fjs(path)
                result = validate_schedule(instance, greedy_schedule(instance))
                self.assertTrue(result.is_valid, result.errors)


if __name__ == "__main__":
    unittest.main()
