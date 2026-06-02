from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from fjsp.parser.fjs_parser import parse_fjs, summarize_instance


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and summarize an FJSP instance.")
    parser.add_argument("instance", help="Path to a .fjs instance file.")
    args = parser.parse_args()

    instance = parse_fjs(args.instance)
    print(f"OK: {summarize_instance(instance)}")


if __name__ == "__main__":
    main()
