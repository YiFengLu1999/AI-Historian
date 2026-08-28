from __future__ import annotations

from ai_historian.profiles.scalable_fulltext.runner import build_parser, run


def main() -> None:
    parser = build_parser()
    run(parser.parse_args())


if __name__ == "__main__":
    main()
