from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from fjsp.scheduler.validator import schedule_from_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a compact text Gantt view for a schedule JSON.")
    parser.add_argument("schedule", help="Path to a schedule JSON file.")
    args = parser.parse_args()

    payload = json.loads(Path(args.schedule).read_text(encoding="utf-8"))
    schedule = schedule_from_dict(payload)

    by_machine: dict[int, list[str]] = {}
    for item in sorted(schedule, key=lambda op: (op.machine, op.start, op.end)):
        by_machine.setdefault(item.machine, []).append(f"J{item.job}O{item.op}[{item.start},{item.end})")

    for machine, chunks in sorted(by_machine.items()):
        print(f"M{machine}: " + " ".join(chunks))


if __name__ == "__main__":
    main()
