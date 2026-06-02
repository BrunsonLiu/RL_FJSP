from __future__ import annotations

import json
from pathlib import Path

import _bootstrap  # noqa: F401
from fjsp.parser.fjs_parser import parse_fjs, summarize_instance
from fjsp.scheduler.validator import greedy_schedule, schedule_to_dict, validate_schedule


ROOT = Path(__file__).resolve().parents[1]
INSTANCE_PATH = ROOT / "data" / "instances" / "tiny_2x2.fjs"
OUTPUT_PATH = ROOT / "data" / "results" / "tiny_2x2_greedy_schedule.json"


def main() -> None:
    instance = parse_fjs(INSTANCE_PATH)
    schedule = greedy_schedule(instance)
    result = validate_schedule(instance, schedule)

    if not result.is_valid:
        print("Smoke test failed:")
        for error in result.errors:
            print(f"- {error}")
        raise SystemExit(1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(schedule_to_dict(schedule), indent=2), encoding="utf-8")
    print(f"OK: {summarize_instance(instance)}, greedy_makespan={result.makespan}")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
