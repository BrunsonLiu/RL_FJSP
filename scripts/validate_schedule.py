from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from fjsp.parser.fjs_parser import parse_fjs
from fjsp.scheduler.validator import schedule_from_dict, validate_schedule


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a schedule JSON against an FJSP instance.")
    parser.add_argument("instance", help="Path to a .fjs instance file.")
    parser.add_argument("schedule", help="Path to a schedule JSON file.")
    args = parser.parse_args()

    instance = parse_fjs(args.instance)
    payload = json.loads(Path(args.schedule).read_text(encoding="utf-8"))
    schedule = schedule_from_dict(payload)
    result = validate_schedule(instance, schedule)

    if result.is_valid:
        print(f"OK: valid schedule, makespan={result.makespan}")
        return

    print("INVALID schedule:")
    for error in result.errors:
        print(f"- {error}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
