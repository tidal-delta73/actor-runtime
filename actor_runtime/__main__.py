"""Command line entry point: `version` and `help` only, for now."""
import sys

from . import __version__

USAGE = """usage: python3 -m actor_runtime <command>

commands:
  version   print the package version
  help      print this message
"""


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    command = args[0] if args else "help"
    if command == "version":
        print(__version__)
        return 0
    if command in {"help", "-h", "--help"}:
        print(USAGE, end="")
        return 0
    print(f"unknown command: {command}", file=sys.stderr)
    print(USAGE, end="", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
