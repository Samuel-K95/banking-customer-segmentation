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

    p_seg_ls = sub.add_parser(
        "segment-berka-leakage-safe",
        help="Leakage-safe AE+k=6 twin + parallel k=2 (does not overwrite champion_v1)",
    )
    p_seg_ls.add_argument("--berka-dir", default=None, help="Path to Berka raw tables")
    p_seg_ls.add_argument("--epochs", type=int, default=None, help="AE epochs (default from config)")
    p_seg_ls.add_argument(
        "--config",
        default="configs/models/berka/champion_v1_leakage_safe.yaml",
        help="Leakage-safe champion config",
    )
    p_seg_ls.add_argument(
        "--output-directory",
        default=None,
        help="Artifact dir (default from config)",
    )

    p_champ = sub.add_parser(
        "champion-info",
        help="Load modular ChampionModel and print summary + decision map",
    )
    p_champ.add_argument(
        "--artifact-dir",
        default="artifacts/segmentation/champion",
        help="Path to registered champion bundle",
    )

    p_score = sub.add_parser(
        "score-berka",
        help="Score Berka interpretable CSV with ChampionModel",
    )
    p_score.add_argument(
        "--input",
        default="data/processed/track_c/track_c_berka_interpretable.csv",
        help="Interpretable feature CSV",
    )
    p_score.add_argument(
        "--artifact-dir",
        default="artifacts/segmentation/champion",
        help="Path to registered champion bundle",
    )
    p_score.add_argument(
        "--output",
        default="artifacts/segmentation/champion/scored_batch.csv",
        help="Where to write scored assignments",
    )
    p_score.add_argument("--limit", type=int, default=None, help="Optional row limit")

    p_dec = sub.add_parser(
        "decision-berka",
        help="Run analysis & decision layer on champion_v1 (profiles, fairness, plays)",
    )
    p_dec.add_argument(
        "--champion-dir",
        default="artifacts/segmentation/champion",
        help="Registered champion bundle",
    )
    p_dec.add_argument(
        "--out-dir",
        default="artifacts/decision/champion",
        help="Decision-layer artifact directory",
    )
    p_dec.add_argument("--berka-dir", default=None, help="Optional raw Berka dir for fairness joins")
    p_dec.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Directory with track_c_berka_interpretable.csv",
    )

    p_promo = sub.add_parser(
        "promote-production",
        help="Promote leakage-safe research twin → production slot + refresh freeze pack",
    )
    p_promo.add_argument(
        "--source",
        default="artifacts/segmentation/research/leakage_safe_source",
        help="Research twin directory",
    )
    p_promo.add_argument(
        "--dest",
        default="artifacts/segmentation/champion",
        help="Production destination",
    )
    p_promo.add_argument(
        "--skip-checks",
        action="store_true",
        help="Only copy/stamp; skip fairness/temporal/freeze assemble",
    )

    p_freeze = sub.add_parser(
        "freeze-status",
        help="Print production freeze readiness JSON",
    )
    p_freeze.add_argument(
        "--production-dir",
        default="artifacts/segmentation/champion",
    )

    p_serve = sub.add_parser(
        "serve",
        help="Start champion FastAPI (model, decisions, reports, inference)",
    )
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true", help="Dev auto-reload")

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

    if args.command == "segment-berka-leakage-safe":
        from customer_segmentation.pipelines.run_berka_segmentation_leakage_safe import (
            run_berka_segmentation_leakage_safe,
        )

        summary = run_berka_segmentation_leakage_safe(
            berka_dir=args.berka_dir,
            config_path=args.config,
            epochs=args.epochs,
            output_directory=args.output_directory,
        )
        print(
            json.dumps(
                {
                    "status": "ok",
                    "champion_dir": summary["champion_dir"],
                    "freeze_status": summary["freeze_status"],
                    "acceptance": summary["acceptance"],
                },
                indent=2,
                default=str,
            )
        )
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

    if args.command == "decision-berka":
        from customer_segmentation.decision.pipeline import run_decision_layer

        summary = run_decision_layer(
            champion_dir=args.champion_dir,
            processed_dir=args.processed_dir,
            out_dir=args.out_dir,
            berka_dir=args.berka_dir,
        )
        print(json.dumps({"status": "ok", **summary}, indent=2, default=str))
        return

    if args.command == "promote-production":
        from customer_segmentation.pipelines.promote_production import promote_production
        from customer_segmentation.pipelines.production_freeze import assemble_freeze_readiness

        summary = promote_production(source_dir=args.source, dest_dir=args.dest)
        if not args.skip_checks:
            from customer_segmentation.pipelines.production_fairness import run_production_fairness
            from customer_segmentation.pipelines.production_temporal import run_production_temporal

            summary["fairness"] = run_production_fairness(production_dir=args.dest)
            summary["temporal"] = {
                k: run_production_temporal(production_dir=args.dest).get(k)
                for k in (
                    "verdict",
                    "ari_1997_vs_1998_assignments",
                    "agreement_rate_1997_vs_1998",
                    "n_common_accounts",
                )
            }
            summary["freeze"] = assemble_freeze_readiness(production_dir=args.dest)
        print(json.dumps(summary, indent=2, default=str))
        return

    if args.command == "freeze-status":
        from customer_segmentation.pipelines.promote_production import load_freeze_status

        print(json.dumps(load_freeze_status(args.production_dir), indent=2, default=str))
        return

    if args.command == "serve":
        import uvicorn

        uvicorn.run(
            "customer_segmentation.serve.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
        return

    parser.print_help()


if __name__ == "__main__":
    main()
