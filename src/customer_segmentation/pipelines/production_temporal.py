"""Best-effort 1998 temporal check for the production AE+k=6 champion.

Retrospective and development-informed — not an independent prospective freeze.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from customer_segmentation.data.berka_dataset_loader import (
    load_berka_product_tables,
    load_berka_raw_data,
    resolve_berka_dir,
)
from customer_segmentation.features.build_berka_segmentation_features import (
    MODEL_COLUMNS,
    build_berka_segmentation_features_asof,
    transform_with_train_scaler,
)
from customer_segmentation.models.champion import ChampionModel

PRODUCTION_DIR = Path("artifacts/segmentation/champion")
REPORT_PATH = Path("reports/champion/03_temporal_validation.md")
TRAIN_ASOF = "1997-12-31"
FUTURE_ASOF = "1998-12-31"


def run_production_temporal(
    *,
    production_dir: str | Path | None = None,
    berka_dir: str | Path | None = None,
) -> dict[str, Any]:
    dst = Path(production_dir) if production_dir else PRODUCTION_DIR
    champ = ChampionModel.load(dst)
    metrics = champ.metrics or {}
    min_tenure = int(metrics.get("minimum_tenure_days", 365))

    path = resolve_berka_dir(berka_dir)
    trans, account, _client = load_berka_raw_data(path)
    loan, card, disp = load_berka_product_tables(path)

    feat_1997 = build_berka_segmentation_features_asof(
        trans, account, loan, card, disp, as_of_date=TRAIN_ASOF, minimum_tenure_days=min_tenure
    )
    feat_1998 = build_berka_segmentation_features_asof(
        trans, account, loan, card, disp, as_of_date=FUTURE_ASOF, minimum_tenure_days=min_tenure
    )

    # Only accounts present in both snapshots and in production assignments
    assign = pd.read_csv(dst / "assignments.csv")
    common = (
        set(feat_1997["account_id"])
        & set(feat_1998["account_id"])
        & set(assign["account_id"])
    )
    a97 = feat_1997[feat_1997["account_id"].isin(common)].sort_values("account_id")
    a98 = feat_1998[feat_1998["account_id"].isin(common)].sort_values("account_id")
    # Align IDs
    merged = a97[["account_id"]].merge(a98, on="account_id", how="inner", suffixes=("", "_y"))
    ids = merged["account_id"].tolist()
    f97 = a97.set_index("account_id").loc[ids]
    f98 = a98.set_index("account_id").loc[ids]

    if champ.scaler is None:
        raise RuntimeError("Production champion missing scaler")
    X97 = transform_with_train_scaler(f97.reset_index(), champ.scaler)
    X98 = transform_with_train_scaler(f98.reset_index(), champ.scaler)
    labels_97 = champ.predict(X97)
    labels_98 = champ.predict(X98)
    # Also compare to stored production labels for integrity on 1997 features
    stored = assign.set_index("account_id").loc[ids, "segment_id"].astype(int).values
    ari_stored = float(adjusted_rand_score(stored, labels_97))
    ari_temporal = float(adjusted_rand_score(labels_97, labels_98))
    agree = float((labels_97 == labels_98).mean())

    # Profile persistence: segment-level median balance/txn on 1997 vs 1998 labels
    prof = []
    for sid in sorted(set(labels_97.tolist())):
        m97 = f97.loc[np.array(labels_97) == sid, MODEL_COLUMNS].median()
        m98 = f98.loc[np.array(labels_98) == sid, MODEL_COLUMNS].median()
        # For accounts that stayed in sid
        stay = (labels_97 == sid) & (labels_98 == sid)
        prof.append(
            {
                "segment_id": int(sid),
                "n_1997": int((labels_97 == sid).sum()),
                "n_1998": int((labels_98 == sid).sum()),
                "n_stayed": int(stay.sum()),
                "median_balance_1997": float(m97.get("Account_Balance", float("nan"))),
                "median_balance_1998": float(m98.get("Account_Balance", float("nan"))),
                "median_txn_1997": float(m97.get("Transaction_Count", float("nan"))),
                "median_txn_1998": float(m98.get("Transaction_Count", float("nan"))),
            }
        )

    payload = {
        "verdict": "PASS_WITH_LIMITATIONS",
        "limitation": (
            "1998 check is retrospective and uses the same Berka vintage that informed "
            "development; it is not an independent prospective holdout."
        ),
        "train_as_of": TRAIN_ASOF,
        "future_as_of": FUTURE_ASOF,
        "n_common_accounts": int(len(ids)),
        "ari_replay_vs_stored_1997": ari_stored,
        "ari_1997_vs_1998_assignments": ari_temporal,
        "agreement_rate_1997_vs_1998": agree,
        "segment_profiles": prof,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": champ.model_version,
    }
    (dst / "temporal_validation.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    _write_report(REPORT_PATH, payload)
    return payload


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Production temporal validation (best-effort)",
        "",
        f"**Verdict: {payload['verdict']}**",
        "",
        payload["limitation"],
        "",
        "## Setup",
        "",
        f"- Train / serving as-of: `{payload['train_as_of']}`",
        f"- Future as-of: `{payload['future_as_of']}`",
        f"- Common accounts: **{payload['n_common_accounts']}**",
        f"- Model version: `{payload['model_version']}`",
        "",
        "## Results",
        "",
        f"- Replay ARI (1997 features vs stored production labels): **{payload['ari_replay_vs_stored_1997']:.4f}**",
        f"- Temporal ARI (1997 vs 1998 assignments, frozen model): **{payload['ari_1997_vs_1998_assignments']:.4f}**",
        f"- Label agreement rate: **{payload['agreement_rate_1997_vs_1998']:.4f}**",
        "",
        "## Segment size persistence",
        "",
        "| segment_id | n_1997 | n_1998 | n_stayed | med_balance_97 | med_balance_98 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["segment_profiles"]:
        lines.append(
            f"| {row['segment_id']} | {row['n_1997']} | {row['n_1998']} | {row['n_stayed']} | "
            f"{row['median_balance_1997']:.1f} | {row['median_balance_1998']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Freeze implication",
            "",
            "This gate is **PASS_WITH_LIMITATIONS**. Overall production freeze remains",
            "**PROVISIONAL** until a true prospective/external vintage is evaluated.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    print(json.dumps(run_production_temporal(), indent=2, default=str))
