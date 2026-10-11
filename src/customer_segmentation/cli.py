"""CLI entrypoint — pipeline stages for banking customer segmentation."""

from __future__ import annotations

import argparse
import json

from customer_segmentation import __version__


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="bcs",
        description="Banking Customer Segmentation and Behavioral Intelligence",
    )
    parser.add_argument("--version", action="store_true")
    sub = parser.add_subparsers(dest="command")

    p_seg = sub.add_parser(
        "segment-berka",
        help="Run Track C bake-off, export figures, register champion_v1",
    )
    p_seg.add_argument("--berka-dir", default=None, help="Path to Berka raw tables")
    p_seg.add_argument("--epochs", type=int, default=50, help="AE training epochs")
    p_seg.add_argument(
        "--artifact-root",
        default="artifacts",
        help="Root for experiments/ and segmentation/ outputs",
    )

    args = parser.parse_args()
    if args.version:
        print(__version__)
        return

    if args.command == "segment-berka":
        from customer_segmentation.pipelines.run_berka_segmentation import run_berka_segmentation

        summary = run_berka_segmentation(
            berka_dir=args.berka_dir,
            artifact_root=args.artifact_root,
            epochs=args.epochs,
        )
        print(json.dumps({"status": "ok", "champion_dir": summary["champion_dir"]}, indent=2))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
