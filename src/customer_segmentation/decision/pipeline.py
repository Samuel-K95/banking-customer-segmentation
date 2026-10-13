"""Run the analysis & decision layer on the registered Berka champion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from customer_segmentation.decision.applications import (
    DOMAINS,
    domain_coverage_summary,
    expand_decision_map,
)
from customer_segmentation.decision.explain import build_explainability_bundle
from customer_segmentation.decision.fairness import build_fairness_pack
from customer_segmentation.decision.profiles import (
    FEATURE_COLS_DEFAULT,
    build_profile_packs,
)
from customer_segmentation.models.champion import ChampionModel


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def run_decision_layer(
    *,
    champion_dir: str | Path = "artifacts/segmentation/champion",
    processed_dir: str | Path = "data/processed",
    out_dir: str | Path | None = None,
    berka_dir: str | Path | None = None,
    interpretable_csv: str | Path | None = None,
) -> dict[str, Any]:
    """Build profile packs, explainability, fairness, and multi-domain decision map."""
    champ_dir = Path(champion_dir)
    out = Path(out_dir) if out_dir else Path("artifacts/decision/champion")
    fig = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)

    champ = ChampionModel.load(champ_dir)
    catalog = champ.catalog
    if not catalog.get("segments"):
        raise RuntimeError("Champion segment_catalog.json missing — register champion first")

    assign = pd.read_csv(champ_dir / "assignments.csv")
    as_of = (champ.metrics or {}).get("as_of_date") or (champ.manifest or {}).get("as_of_date")
    min_tenure = int((champ.metrics or {}).get("minimum_tenure_days") or 0)

    if interpretable_csv:
        interpretable = pd.read_csv(interpretable_csv)
    elif as_of:
        # Production / leakage-safe champions must use the same as-of feature contract.
        from customer_segmentation.data.berka_dataset_loader import (
            load_berka_product_tables,
            load_berka_raw_data,
            resolve_berka_dir,
        )
        from customer_segmentation.features.build_berka_segmentation_features import (
            build_berka_segmentation_features_asof,
        )

        path = resolve_berka_dir(berka_dir)
        trans, account, _client = load_berka_raw_data(path)
        loan, card, disp = load_berka_product_tables(path)
        interpretable = build_berka_segmentation_features_asof(
            trans,
            account,
            loan,
            card,
            disp,
            as_of_date=str(as_of),
            minimum_tenure_days=min_tenure,
        )
    else:
        interp_path = Path(processed_dir) / "track_c" / "track_c_berka_interpretable.csv"
        if not interp_path.exists():
            from customer_segmentation.features.build_berka_segmentation_features import (
                build_and_save_berka_features,
            )

            bundle = build_and_save_berka_features(berka_dir=berka_dir, processed_dir=processed_dir)
            interpretable = bundle.interpretable
        else:
            interpretable = pd.read_csv(interp_path)

    id_col = "account_id"
    df = interpretable.merge(assign, on=id_col, how="inner")
    if "segment_name" not in df.columns:
        name_map = {int(k): v["name"] for k, v in catalog["segments"].items()}
        df["segment_name"] = df["segment_id"].map(name_map)

    feat_cols = [c for c in FEATURE_COLS_DEFAULT if c in df.columns]

    # Optional AE latent / recon for explainability
    z = None
    recon_mse = None
    try:
        from customer_segmentation.features.build_berka_segmentation_features import (
            transform_berka_model_matrix,
        )

        if champ.scaler is not None and champ.encoder is not None:
            import torch

            X = transform_berka_model_matrix(df, champ.scaler, feat_cols)
            z = champ.encode(X)
            champ.encoder.eval()
            with torch.no_grad():
                xhat, _ = champ.encoder(torch.tensor(X, dtype=torch.float32))
                recon_mse = ((xhat.numpy() - X) ** 2).mean(axis=1)
    except Exception as exc:  # noqa: BLE001 - deliberate soft fail  # pragma: no cover - explainability soft-fail
        print(f"AE explainability skipped: {exc}")

    explain = build_explainability_bundle(df, feature_cols=feat_cols, z=z, recon_mse=recon_mse)
    as_of_year = 1997
    if as_of:
        try:
            as_of_year = int(str(as_of)[:4])
        except ValueError:
            as_of_year = 1997
    fairness = build_fairness_pack(
        assign.merge(df[[id_col, "segment_name"]], on=id_col, how="left"),
        berka_dir=berka_dir,
        catalog=catalog,
        feature_df=df,
        n_clusters=int(df["segment_id"].nunique()),
        as_of_year=as_of_year,
    )
    fair_by_seg: dict[int, dict[str, Any]] = {}
    for row in fairness.get("district_concentration") or []:
        sid = int(row["segment_id"])
        fair_by_seg[sid] = {
            "max_district_share": row.get("max_district_share"),
            "top_district_id": row.get("top_district_id"),
            "risk_note": (
                "Elevated district concentration — do not target by geography from segment_id"
                if float(row.get("max_district_share") or 0) >= 0.15
                else "Monitor district mix; no demographic targeting from segment_id"
            ),
        }
    profiles = build_profile_packs(
        df, catalog, feature_cols=feat_cols, fairness_by_segment=fair_by_seg
    )
    decision_map = expand_decision_map(catalog)
    coverage = domain_coverage_summary(decision_map)

    # Size composition figure
    try:
        import matplotlib.pyplot as plt

        size = df.groupby("segment_name").size().sort_values(ascending=True)
        fig_ax, ax = plt.subplots(figsize=(8, 4.5))
        ax.barh(size.index.astype(str), 100 * size.values / size.sum(), color="#1b7a6e")
        ax.set_xlabel("% of book")
        ax.set_title("Champion segment composition")
        fig_ax.tight_layout()
        fig_ax.savefig(fig / "segment_composition.png", dpi=120, bbox_inches="tight")
        plt.close(fig_ax)

        # Lift heatmap
        lift_df = pd.DataFrame(profiles["lift"]).set_index("segment_id")
        if "segment_id" in lift_df.columns:
            lift_df = lift_df.drop(columns=["segment_id"], errors="ignore")
        fig2, ax2 = plt.subplots(figsize=(9, 4.5))
        im = ax2.imshow(lift_df[feat_cols].astype(float).values, aspect="auto", cmap="RdBu_r", vmin=0.3, vmax=1.7)
        ax2.set_xticks(range(len(feat_cols)))
        ax2.set_xticklabels(feat_cols, rotation=45, ha="right")
        ax2.set_yticks(range(len(lift_df.index)))
        ax2.set_yticklabels([catalog["segments"][str(i)]["name"][:28] for i in lift_df.index])
        ax2.set_title("Feature lift vs population (1.0 = average)")
        fig2.colorbar(im, ax=ax2, fraction=0.03)
        fig2.tight_layout()
        fig2.savefig(fig / "lift_heatmap.png", dpi=120, bbox_inches="tight")
        plt.close(fig2)

        # Domain coverage bars
        fig3, ax3 = plt.subplots(figsize=(8, 3.8))
        ax3.bar(coverage["domain"], coverage["n_plays"], color="#c45c26")
        ax3.set_ylabel("Plays")
        ax3.set_title("Decision-map coverage by banking domain")
        ax3.tick_params(axis="x", rotation=30)
        fig3.tight_layout()
        fig3.savefig(fig / "domain_coverage.png", dpi=120, bbox_inches="tight")
        plt.close(fig3)
    except Exception as exc:  # noqa: BLE001 - deliberate soft fail  # pragma: no cover
        print(f"Figure export skipped: {exc}")

    paths = {
        "profile_packs": _write_json(out / "profile_packs.json", profiles),
        "explainability": _write_json(out / "explainability.json", explain),
        "fairness": _write_json(out / "fairness.json", fairness),
        "decision_map": out / "decision_map_full.csv",
        "domain_coverage": out / "domain_coverage.csv",
        "exemplars": out / "exemplars.csv",
    }
    decision_map.to_csv(paths["decision_map"], index=False)
    coverage.to_csv(paths["domain_coverage"], index=False)
    pd.DataFrame(profiles["exemplars"]).to_csv(paths["exemplars"], index=False)

    def _rel(path: Path) -> str:
        try:
            return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            return path.as_posix()

    summary = {
        "champion_run_id": champ.run_id,
        "model_version": champ.model_version,
        "n_accounts": len(df),
        "n_segments": int(df["segment_id"].nunique()),
        "domains": list(DOMAINS),
        "n_plays": len(decision_map),
        "domain_coverage": coverage.to_dict(orient="records"),
        "artifact_dir": _rel(out),
        "files": {k: _rel(Path(v)) for k, v in paths.items()},
        "acceptance": {
            "profile_packs": True,
            "profile_packs_include_penetration_lifecycle_fairness": True,
            "human_names": True,
            "explainability": True,
            "fairness_pack": True,
            "behavior_vs_demographic_documented": True,
            "behavior_vs_demographic_ablation_executed": bool(
                (fairness.get("behavior_vs_demographic") or {}).get("ablation_status")
                == "executed_recluster"
            ),
            "applications_ge_3": bool(coverage["n_plays"].sum() >= 3),
            "domains_covered": int((coverage["n_plays"] > 0).sum()),
            "decision_map_published": True,
            "not_a_supervised_predictor": True,
        },
    }
    _write_json(out / "run_summary.json", summary)
    print(f"Decision layer written -> {out} ({summary['n_plays']} plays across {summary['acceptance']['domains_covered']} domains)")
    return summary


if __name__ == "__main__":
    run_decision_layer()
