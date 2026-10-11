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

    p_champ = sub.add_parser(
        "champion-info",
        help="Load modular ChampionModel and print summary + decision map",
    )
    p_champ.add_argument(
        "--artifact-dir",
        default="artifacts/segmentation/champion_v1",
        help="Path to registered champion bundle",
    )

    p_score = sub.add_parser(
        "score-berka",
        help="Score Berka interpretable CSV with ChampionModel",
    )
    p_score.add_argument(
        "--input",
        default="data/processed/track_c_berka_interpretable.csv",
        help="Interpretable feature CSV",
    )
    p_score.add_argument(
        "--artifact-dir",
        default="artifacts/segmentation/champion_v1",
        help="Path to registered champion bundle",
    )
    p_score.add_argument(
        "--output",
        default="artifacts/segmentation/champion_v1/scored_batch.csv",
        help="Where to write scored assignments",
    )
    p_score.add_argument("--limit", type=int, default=None, help="Optional row limit")

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

    if args.command == "champion-info":
        from customer_segmentation.models.champion import ChampionModel

        champ = ChampionModel.load(args.artifact_dir)
        payload = champ.summary()
        payload["decision_map"] = champ.decision_map().to_dict(orient="records")
        print(json.dumps(payload, indent=2, default=str))
        return

    if args.command == "score-berka":
        from pathlib import Path

        import pandas as pd

        from customer_segmentation.models.champion import ChampionModel

        champ = ChampionModel.load(args.artifact_dir)
        df = pd.read_csv(args.input)
        if args.limit:
            df = df.head(args.limit)
        scored = champ.score_interpretable(df)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(out, index=False)
        print(json.dumps({"status": "ok", "n": len(scored), "output": str(out)}, indent=2))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
