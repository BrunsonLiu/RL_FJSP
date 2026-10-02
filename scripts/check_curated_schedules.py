"""Validate each curated SOTA schedule independently of the learning models."""

from __future__ import annotations

import json
from pathlib import Path

import _bootstrap  # noqa: F401
from fjsp.parser.fjs_parser import parse_fjs
from fjsp.scheduler.validator import schedule_from_dict, validate_schedule


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    paths = sorted((ROOT / "data" / "results").glob("sota_mk*_schedule.json"))
    if not paths:
        raise SystemExit("No curated schedules found.")

    failures = 0
    for path in paths:
        instance_name = path.name.removeprefix("sota_").removesuffix("_schedule.json")
        instance_path = ROOT / "data" / "instances" / "brandimarte" / f"{instance_name}.txt"
        try:
            instance = parse_fjs(instance_path)
            schedule = schedule_from_dict(json.loads(path.read_text(encoding="utf-8")))
            result = validate_schedule(instance, schedule)
            if not result.is_valid:
                raise ValueError("; ".join(result.errors))
        except (OSError, ValueError, TypeError) as exc:
            failures += 1
            print(f"FAIL: {path.name}: {exc}")
        else:
            print(f"OK: {path.name}, makespan={result.makespan}")

    print(f"Validated {len(paths)} curated schedules; failures={failures}.")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
