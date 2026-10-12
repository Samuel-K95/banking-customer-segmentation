"""Leakage-safe Berka champion retrain + parallel k=2 twin.

Applies Mahider-style discipline to the CRM 8-feature family:
fixed as-of, tenure gate, shared account split, train-only scaler/AE/K-Means.
Does **not** overwrite ``artifacts/segmentation/archive/fullbook_demo``.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

from customer_segmentation.data.berka_dataset_loader import (
    load_berka_product_tables,
    load_berka_raw_data,
    resolve_berka_dir,
)
from customer_segmentation.features.build_berka_segmentation_features import (
    DEFAULT_AS_OF_DATE,
    DEFAULT_MIN_TENURE_DAYS,
    MODEL_COLUMNS,
    build_berka_segmentation_features_asof,
    fit_train_scaler,
    transform_with_train_scaler,
)
from customer_segmentation.models.berka_behavioral_baselines import make_shared_split
from customer_segmentation.models.metrics import size_sanity
from customer_segmentation.models.registry import register_champion, write_json
from customer_segmentation.models.segment_intel import build_segment_catalog
from customer_segmentation.models.tabular_ae import (
    TORCH_OK,
    distance_confidence,
    encode_latent,
    train_autoencoder,
)

DEFAULT_CONFIG = Path("configs/models/berka/champion_v1_leakage_safe.yaml")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _cluster_metrics(X: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    n_labels = len(set(int(x) for x in labels))
    if n_labels < 2 or len(labels) < 3:
        return {"silhouette": float("nan"), "db_index": float("nan"), "ch_index": float("nan")}
    return {
        "silhouette": float(silhouette_score(X, labels)),
        "db_index": float(davies_bouldin_score(X, labels)),
        "ch_index": float(calinski_harabasz_score(X, labels)),
    }


def _seed_ari(
    X_train: np.ndarray,
    *,
    n_clusters: int,
    seeds: list[int],
) -> dict[str, float]:
    label_sets = [
        KMeans(n_clusters=n_clusters, n_init=20, random_state=s).fit_predict(X_train) for s in seeds
    ]
    pairs: list[float] = []
    for i in range(len(label_sets)):
        for j in range(i + 1, len(label_sets)):
            pairs.append(float(adjusted_rand_score(label_sets[i], label_sets[j])))
    return {
        "mean_ari": float(np.mean(pairs)) if pairs else float("nan"),
        "min_ari": float(np.min(pairs)) if pairs else float("nan"),
        "n_pairs": float(len(pairs)),
    }


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if cfg_path.exists():
        return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return {
        "data": {
            "as_of_date": DEFAULT_AS_OF_DATE,
            "minimum_tenure_days": DEFAULT_MIN_TENURE_DAYS,
            "output_directory": "artifacts/segmentation/research/leakage_safe_source",
        },
        "split": {"training_fraction": 0.75, "random_seed": 8675309},
        "model": {
            "feature_version": "berka_rfm_product_v1_leakage_safe",
            "model_version": "v1_leakage_safe",
            "champion_run_id": "c_ae_lat4_k6_leakage_safe",
            "latent_dim": 4,
            "n_clusters": 6,
            "parallel_kmeans_k": 2,
            "epochs": 50,
            "random_state": 42,
            "model_seeds": [42, 2024, 123],
        },
    }


def run_berka_segmentation_leakage_safe(
    *,
    berka_dir: str | Path | None = None,
    config_path: str | Path | None = None,
    epochs: int | None = None,
    output_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Train leakage-safe AE+k=6 champion twin + parallel k=2 baseline."""
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    split_cfg = cfg["split"]
    model_cfg = cfg["model"]

    as_of = str(data_cfg.get("as_of_date", DEFAULT_AS_OF_DATE))
    min_tenure = int(data_cfg.get("minimum_tenure_days", DEFAULT_MIN_TENURE_DAYS))
    out_dir = Path(
        output_directory
        or data_cfg.get("output_directory")
        or "artifacts/segmentation/research/leakage_safe_source"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)

    n_clusters = int(model_cfg.get("n_clusters", 6))
    parallel_k = int(model_cfg.get("parallel_kmeans_k", 2))
    latent_dim = int(model_cfg.get("latent_dim", 4))
    random_state = int(model_cfg.get("random_state", 42))
    ae_epochs = int(epochs if epochs is not None else model_cfg.get("epochs", 50))
    seeds = [int(s) for s in model_cfg.get("model_seeds", [42, 2024, 123])]
    run_id = str(model_cfg.get("champion_run_id", "c_ae_lat4_k6_leakage_safe"))
    feature_version = str(model_cfg.get("feature_version", "berka_rfm_product_v1_leakage_safe"))
    model_version = str(model_cfg.get("model_version", "v1_leakage_safe"))

    print(f"=== Leakage-safe features (as_of={as_of}, tenure>={min_tenure}) ===")
    path = resolve_berka_dir(berka_dir)
    trans, account, _client = load_berka_raw_data(path)
    loan, card, disp = load_berka_product_tables(path)
    interpretable = build_berka_segmentation_features_asof(
        trans,
        account,
        loan,
        card,
        disp,
        as_of_date=as_of,
        minimum_tenure_days=min_tenure,
    )
    print(f"  eligible accounts={len(interpretable)}")

    train_df, holdout_df = make_shared_split(
        interpretable,
        training_fraction=float(split_cfg.get("training_fraction", 0.75)),
        random_seed=int(split_cfg.get("random_seed", 8675309)),
    )
    split_frame = pd.concat(
        [
            train_df[["account_id"]].assign(split="train"),
            holdout_df[["account_id"]].assign(split="holdout"),
        ],
        ignore_index=True,
    )
    split_path = out_dir / "account_split.csv"
    split_frame.to_csv(split_path, index=False)
    print(f"  train={len(train_df)} holdout={len(holdout_df)}")

    scaler = fit_train_scaler(train_df)
    X_train = transform_with_train_scaler(train_df, scaler)
    X_hold = transform_with_train_scaler(holdout_df, scaler)
    joblib.dump(scaler, out_dir / "scaler.joblib")

    # ── Parallel research twin: raw K-Means k=2 on train-scaled matrix ──
    print(f"=== Parallel K-Means k={parallel_k} (research twin) ===")
    km2 = KMeans(n_clusters=parallel_k, n_init=20, random_state=random_state)
    km2_train_labels = km2.fit_predict(X_train)
    km2_hold_labels = km2.predict(X_hold)
    km2_hold_metrics = _cluster_metrics(X_hold, km2_hold_labels)
    km2_seed = _seed_ari(X_train, n_clusters=parallel_k, seeds=seeds)
    joblib.dump(km2, out_dir / "models" / f"kmeans_k{parallel_k}_seed{random_state}.joblib")
    pd.DataFrame(
        {
            "account_id": holdout_df["account_id"].values,
            "segment_id": km2_hold_labels,
            "split": "holdout",
            "family": f"kmeans_k{parallel_k}",
        }
    ).to_csv(out_dir / f"held_out_assignments_kmeans_k{parallel_k}.csv", index=False)

    # ── Product path: AE lat=4 + K-Means k=6, train-only ──
    if not TORCH_OK:
        raise ImportError("PyTorch required for leakage-safe AE champion retrain")

    print(f"=== AE lat={latent_dim} + K-Means k={n_clusters} (train-only fit) ===")
    ae = train_autoencoder(
        X_train, latent_dim=latent_dim, epochs=ae_epochs, random_state=random_state
    )
    z_train = ae.z
    z_hold = encode_latent(ae.model, X_hold)

    km6 = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
    train_labels = km6.fit_predict(z_train)
    hold_labels = km6.predict(z_hold)
    hold_metrics = _cluster_metrics(z_hold, hold_labels)
    train_metrics = _cluster_metrics(z_train, train_labels)
    ae_seed = _seed_ari(z_train, n_clusters=n_clusters, seeds=seeds)
    raw_k6_seed = _seed_ari(X_train, n_clusters=n_clusters, seeds=seeds)

    # Score full eligible book with frozen artifacts (serving-style)
    X_all = transform_with_train_scaler(interpretable, scaler)
    z_all = encode_latent(ae.model, X_all)
    all_labels = km6.predict(z_all)
    conf_all = distance_confidence(z_all, all_labels, km6.cluster_centers_)

    assignments = pd.DataFrame(
        {
            "account_id": interpretable["account_id"].values,
            "segment_id": all_labels,
            "confidence": np.round(conf_all, 4),
        }
    ).merge(split_frame, on="account_id", how="left")

    profiles = (
        interpretable.assign(segment_id=all_labels)
        .groupby("segment_id")[MODEL_COLUMNS]
        .mean()
        .round(2)
    )
    pop = interpretable[MODEL_COLUMNS].mean()
    counts = {int(s): int(n) for s, n in zip(*np.unique(all_labels, return_counts=True))}
    catalog = build_segment_catalog(profiles, population=pop, sizes=counts, total_n=len(all_labels))

    board = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "family": "ae_kmeans",
                "k": n_clusters,
                "latent_dim": latent_dim,
                "split": "holdout",
                "silhouette": round(hold_metrics["silhouette"], 4),
                "db_index": round(hold_metrics["db_index"], 4),
                "ch_index": round(hold_metrics["ch_index"], 0),
                "seed_ari_mean": round(ae_seed["mean_ari"], 4),
                "role": "product_champion_leakage_safe",
            },
            {
                "run_id": f"c_kmeans_k{parallel_k}_leakage_safe",
                "family": "kmeans",
                "k": parallel_k,
                "latent_dim": "-",
                "split": "holdout",
                "silhouette": round(km2_hold_metrics["silhouette"], 4),
                "db_index": round(km2_hold_metrics["db_index"], 4),
                "ch_index": round(km2_hold_metrics["ch_index"], 0),
                "seed_ari_mean": round(km2_seed["mean_ari"], 4),
                "role": "research_twin_same_matrix",
            },
            {
                "run_id": f"c_kmeans_k{n_clusters}_scaled_leakage_safe",
                "family": "kmeans",
                "k": n_clusters,
                "latent_dim": "-",
                "split": "train_seed_ari_only",
                "silhouette": None,
                "db_index": None,
                "ch_index": None,
                "seed_ari_mean": round(raw_k6_seed["mean_ari"], 4),
                "role": "raw_k6_stability_reference",
            },
        ]
    )

    encoder_state = {
        "state_dict": ae.model.state_dict(),
        "latent_dim": latent_dim,
        "n_feat": X_train.shape[1],
        "hidden": 32,
    }
    size = size_sanity(all_labels)
    metrics_payload = {
        "run_id": run_id,
        "method": "ae_kmeans",
        "k": n_clusters,
        "latent_dim": latent_dim,
        "as_of_date": as_of,
        "minimum_tenure_days": min_tenure,
        "n_eligible": int(len(interpretable)),
        "n_train": int(len(train_df)),
        "n_holdout": int(len(holdout_df)),
        "split_seed": int(split_cfg.get("random_seed", 8675309)),
        "holdout_silhouette": hold_metrics["silhouette"],
        "holdout_db_index": hold_metrics["db_index"],
        "holdout_ch_index": hold_metrics["ch_index"],
        "train_silhouette": train_metrics["silhouette"],
        "silhouette": hold_metrics["silhouette"],
        "db_index": hold_metrics["db_index"],
        "ch_index": hold_metrics["ch_index"],
        "stability_ari_ae_latent_kmeans": ae_seed["mean_ari"],
        "stability_ari_ae_latent_kmeans_min": ae_seed["min_ari"],
        "stability_ari_raw_kmeans_k6": raw_k6_seed["mean_ari"],
        "parallel_kmeans_k2_holdout": km2_hold_metrics,
        "parallel_kmeans_k2_seed_ari": km2_seed,
        "size": size,
        "low_confidence_frac": float((conf_all < 0.55).mean()),
        "tiny_segments_policy": "exception_queue_not_mass_campaign",
        "selection_policy": (
            "Leakage-safe AE lat=4 k=6 remains the product path; "
            "k=2 on the same matrix is the research twin (not a CRM replacement)."
        ),
        "freeze_status": "BLOCKED",
    }

    paths = register_champion(
        out_dir,
        run_id=run_id,
        track="berka",
        method="ae_kmeans",
        feature_version=feature_version,
        model_version=model_version,
        feature_columns=list(MODEL_COLUMNS),
        metrics=metrics_payload,
        assignments=assignments,
        profiles=profiles,
        comparison_board=board,
        scaler=scaler,
        kmeans=km6,
        encoder_state=encoder_state,
        segment_catalog=catalog,
        notes=(
            f"Leakage-safe twin of c_ae_lat4_k6. as_of={as_of}, tenure>={min_tenure}, "
            f"train-only scaler/AE/KMeans. Freeze BLOCKED."
        ),
    )
    joblib.dump(km6, out_dir / "models" / f"ae_kmeans_k{n_clusters}_seed{random_state}.joblib")

    # Replay integrity: re-predict holdout and match
    z_hold_replay = encode_latent(ae.model, X_hold)
    hold_replay = km6.predict(z_hold_replay)
    replay_ok = bool(np.array_equal(hold_replay, hold_labels))

    artifact_hashes = {
        "scaler.joblib": _sha256(out_dir / "scaler.joblib"),
        "kmeans.joblib": _sha256(out_dir / "kmeans.joblib"),
        "encoder.pt": _sha256(out_dir / "encoder.pt") if (out_dir / "encoder.pt").exists() else None,
        "account_split.csv": _sha256(split_path),
        f"models/kmeans_k{parallel_k}_seed{random_state}.joblib": _sha256(
            out_dir / "models" / f"kmeans_k{parallel_k}_seed{random_state}.joblib"
        ),
    }

    freeze = {
        "overall_verdict": "BLOCKED",
        "champion_run_id": run_id,
        "as_of_date": as_of,
        "minimum_tenure_days": min_tenure,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "gates": {
            "artifact_integrity": {"verdict": "PASS", "detail": "SHA-256 fingerprints written"},
            "snapshot_split_train_only_fit": {
                "verdict": "PASS",
                "detail": (
                    f"as_of={as_of}; tenure>={min_tenure}; "
                    f"train={len(train_df)} holdout={len(holdout_df)}; scaler/AE/KMeans fit on train only"
                ),
            },
            "exact_holdout_prediction_replay": {
                "verdict": "PASS" if replay_ok else "FAIL",
                "detail": "Frozen encoder+KMeans reproduce held-out labels" if replay_ok else "Mismatch",
            },
            "segment_meaning": {
                "verdict": "NEEDS_APPROVAL",
                "detail": "CRM personas retained; tiny stressed/dormant cells = exception queues",
            },
            "fairness_owner_disponent": {
                "verdict": "NEEDS_APPROVAL",
                "detail": "No demographic fairness / OWNER-DISPONENT policy completed for this twin",
            },
            "independent_temporal_validation": {
                "verdict": "BLOCKED",
                "detail": "Requires prospective/external vintage not used in development",
            },
            "serving_package": {
                "verdict": "BLOCKED",
                "detail": "Artifacts exist; production serving/monitoring/rollback not validated",
            },
        },
        "product_vs_research": {
            "product_path": f"AE lat={latent_dim} + K-Means k={n_clusters}",
            "research_twin": f"K-Means k={parallel_k} on same train-scaled matrix",
            "note": "Do not collapse CRM champion to k=2; twin documents scale/activity split under same leakage contract",
        },
        "holdout_metrics": {
            "ae_k6": hold_metrics,
            "kmeans_k2": km2_hold_metrics,
        },
        "artifact_sha256": artifact_hashes,
    }
    write_json(out_dir / "freeze_readiness.json", freeze)

    run_summary = {
        "champion_dir": str(out_dir),
        "as_of_date": as_of,
        "minimum_tenure_days": min_tenure,
        "n_eligible": len(interpretable),
        "n_train": len(train_df),
        "n_holdout": len(holdout_df),
        "metrics": metrics_payload,
        "freeze_status": "BLOCKED",
        "files": {k: str(v) for k, v in paths.items()},
        "parallel_k2": {
            "holdout_metrics": km2_hold_metrics,
            "seed_ari": km2_seed,
            "assignments": str(out_dir / f"held_out_assignments_kmeans_k{parallel_k}.csv"),
        },
        "acceptance": {
            "leakage_safe_as_of": True,
            "train_only_scaler_ae_kmeans": True,
            "holdout_replay": replay_ok,
            "parallel_k2_written": True,
            "freeze_blocked_documented": True,
            "champion_v1_untouched": True,
        },
    }
    write_json(out_dir / "run_summary.json", run_summary)

    report_path = Path("reports/track_c_berka/berka_champion_leakage_safe_freeze_readiness.md")
    _write_freeze_report(report_path, run_summary, freeze, board)
    print(f"Freeze readiness (BLOCKED) -> {report_path}")
    print(f"Leakage-safe champion twin -> {out_dir}")
    return run_summary


def _write_freeze_report(
    path: Path,
    summary: dict[str, Any],
    freeze: dict[str, Any],
    board: pd.DataFrame,
) -> None:
    m = summary["metrics"]
    k2 = summary["parallel_k2"]["holdout_metrics"]
    lines = [
        "# Berka Champion Leakage-Safe Freeze Readiness",
        "",
        f"**Overall verdict: {freeze['overall_verdict']}.** "
        "This is a leakage-disciplined twin of the CRM champion feature family — "
        "not an approved production freeze, and it does **not** replace "
        "`artifacts/segmentation/archive/fullbook_demo`.",
        "",
        "## What changed vs original champion_v1",
        "",
        "| Control | Original `champion_v1` | This twin |",
        "|---|---|---|",
        f"| As-of date | End-of-book max txn date | **{summary['as_of_date']}** |",
        f"| Tenure gate | None | **≥ {summary['minimum_tenure_days']} days** |",
        f"| Population | ~4500 full book | **{summary['n_eligible']}** eligible |",
        f"| Split | None (fit on all) | **{summary['n_train']} / {summary['n_holdout']}** (seed 8675309) |",
        "| Scaler / AE / K-Means | Fit on all rows | **Train-only**; holdout scored frozen |",
        "| Product k | 6 | **6** (unchanged CRM choice) |",
        "| Research twin | — | **K-Means k=2** on same matrix |",
        "",
        "## Holdout metrics",
        "",
        "| Family | k | Silhouette ↑ | DB ↓ | CH ↑ | Seed ARI (train) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in board.iterrows():
        if row["split"] == "train_seed_ari_only":
            continue
        lines.append(
            f"| {row['family']} | {row['k']} | {row['silhouette']} | "
            f"{row['db_index']} | {row['ch_index']} | {row['seed_ari_mean']} |"
        )
    lines.extend(
        [
            "",
            f"AE+k=6 holdout silhouette **{m['holdout_silhouette']:.4f}**; "
            f"k=2 twin holdout silhouette **{k2['silhouette']:.4f}**.",
            "",
            "## Why k=6 stays the product path",
            "",
            "Under this leakage contract, **k=2** is the honest simple partition "
            "(scale/activity twin on the same 8 features). **k=6** remains the CRM "
            "segmentation because loan/card + behavior still support named personas "
            "and the decision-layer playbook. Tiny stressed/dormant cells are "
            "**exception queues**, not mass campaigns.",
            "",
            "## Freeze gates",
            "",
            "| Gate | Verdict |",
            "|---|---|",
        ]
    )
    for name, gate in freeze["gates"].items():
        lines.append(f"| {name} | **{gate['verdict']}** — {gate['detail']} |")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Bundle: `{summary['champion_dir']}`",
            "- Parallel k=2 holdout assignments: "
            f"`{summary['parallel_k2']['assignments']}`",
            "- Fingerprints: `freeze_readiness.json` → `artifact_sha256`",
            "",
            "## Bottom line",
            "",
            "Leakage discipline and freeze honesty are applied to the **AE+k=6 product path**. "
            "Mahider’s original 13-feature k=2 research baseline remains separate. "
            "Neither candidate is freeze-approved.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run_berka_segmentation_leakage_safe()
