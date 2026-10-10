"""CLI entrypoint — extend as pipeline stages are implemented."""

from __future__ import annotations

import argparse

from customer_segmentation import __version__


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="bcs",
        description="Banking Customer Segmentation and Behavioral Intelligence",
    )
    parser.add_argument("--version", action="store_true")
    args = parser.parse_args()
    if args.version:
        print(__version__)
        return
    parser.print_help()


if __name__ == "__main__":
    main()
