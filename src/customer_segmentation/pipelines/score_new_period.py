"""Score a later Berka period with the frozen champion: the "new data arrives" path.

The champion was fit on accounts as of 1997-12-31. This rebuilds the same features as of a
later date (default 1998-12-31) and scores them with the frozen bundle, without refitting:

- returning accounts: a full transition matrix (1997 segment -> new segment) and stay rate
- new accounts (eligible at the new date, never seen in training): segment mix and how
  confidently they fit the existing segments, compared with returning accounts
- drift: segment-mix shift and per-feature PSI against the training snapshot

Outputs go to artifacts/monitoring/berka/asof_<date>/ and never touch the champion bundle.
The later period is retrospective Berka data, not a prospective holdout.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
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
from customer_segmentation.pipelines.production_temporal import (
    FUTURE_ASOF,
    PRODUCTION_DIR,
    TRAIN_ASOF,
)

MONITORING_ROOT = Path("artifacts/monitoring/berka")
PSI_REVIEW = 0.25  # common rule of thumb: <0.1 stable, 0.1-0.25 moderate, >0.25 major shift
BINARY_COLUMNS = {"has_loan", "has_card"}


def psi(expected: np.ndarray, actual: np.ndarray, *, bins: int = 10, categorical: bool = False) -> float:
    """Population stability index of `actual` against `expected` (quantile bins from expected)."""
    expected, actual = np.asarray(expected, dtype=float), np.asarray(actual, dtype=float)
    if categorical:
        cats = np.union1d(np.unique(expected), np.unique(actual))
        e = np.array([(expected == c).mean() for c in cats])
        a = np.array([(actual == c).mean() for c in cats])
    else:
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
        if len(edges) < 2:
            return 0.0
        edges[0], edges[-1] = -np.inf, np.inf
        e = np.histogram(expected, edges)[0] / len(expected)
        a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def transition_summary(before: pd.Series, after: pd.Series) -> dict[str, Any]:
    """Counts and row shares of segment moves for accounts present in both periods."""
    counts = pd.crosstab(before, after, rownames=["segment_before"], colnames=["segment_after"])
    shares = counts.div(counts.sum(axis=1), axis=0)
    stay = float((before.to_numpy() == after.to_numpy()).mean())
    return {"counts": counts, "shares": shares, "stay_rate": stay,
            "ari": float(adjusted_rand_score(before, after))}


def score_new_period(
    *,
    as_of_date: str = FUTURE_ASOF,
    production_dir: str | Path | None = None,
    berka_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    from customer_segmentation.models.champion import ChampionModel  # needs PyTorch

    out = Path(output_dir) if output_dir else MONITORING_ROOT / f"asof_{as_of_date}"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing monitoring output: {out}")

    champ = ChampionModel.load(Path(production_dir) if production_dir else PRODUCTION_DIR)
    if champ.scaler is None:
        raise RuntimeError("Production champion is missing its scaler")
    min_tenure = int((champ.metrics or {}).get("minimum_tenure_days", 365))

    path = resolve_berka_dir(berka_dir)
    trans, account, _ = load_berka_raw_data(path)
    loan, card, disp = load_berka_product_tables(path)
    periods = {}
    for label, cutoff in (("train", TRAIN_ASOF), ("new", as_of_date)):
        features = build_berka_segmentation_features_asof(
            trans, account, loan, card, disp, as_of_date=cutoff, minimum_tenure_days=min_tenure
        ).sort_values("account_id").reset_index(drop=True)
        scored = champ.score(transform_with_train_scaler(features, champ.scaler), ids=features["account_id"])
        periods[label] = features.merge(scored, on="account_id", validate="one_to_one")

    train, new = periods["train"], periods["new"]
    returning = new["account_id"].isin(train["account_id"])
    both = train.merge(new[["account_id", "segment_id", "confidence"]], on="account_id", suffixes=("_before", "_after"))
    moves = transition_summary(both["segment_id_before"], both["segment_id_after"])

    segments = sorted(set(train["segment_id"]) | set(new["segment_id"]))
    mix = pd.DataFrame({
        "segment_id": segments,
        "train_share": [float((train["segment_id"] == s).mean()) for s in segments],
        "new_period_share": [float((new["segment_id"] == s).mean()) for s in segments],
        "new_accounts_share": [float((new.loc[~returning, "segment_id"] == s).mean()) for s in segments],
    })
    feature_psi = pd.DataFrame([
        {"feature": f, "psi": psi(train[f], new[f], categorical=f in BINARY_COLUMNS)} for f in MODEL_COLUMNS
    ]).assign(review=lambda d: d["psi"] > PSI_REVIEW)
    mix_psi = psi(train["segment_id"], new["segment_id"], categorical=True)

    summary = {
        "model_version": champ.model_version,
        "train_as_of": TRAIN_ASOF,
        "new_as_of": as_of_date,
        "n_train_accounts": len(train),
        "n_new_period_accounts": len(new),
        "n_returning_accounts": int(returning.sum()),
        "n_new_accounts": int((~returning).sum()),
        "returning": {"stay_rate": round(moves["stay_rate"], 4), "ari": round(moves["ari"], 4)},
        "confidence_median": {
            "returning_accounts": round(float(new.loc[returning, "confidence"].median()), 4),
            "new_accounts": round(float(new.loc[~returning, "confidence"].median()), 4) if (~returning).any() else None,
        },
        "segment_mix_psi": round(mix_psi, 4),
        "features_needing_review": feature_psi.loc[feature_psi["review"], "feature"].tolist(),
        "status": "REVIEW" if mix_psi > PSI_REVIEW or feature_psi["review"].any() else "OK",
        "psi_review_threshold": PSI_REVIEW,
        "limitation": "Retrospective Berka data that informed development; not a prospective holdout.",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }

    out.mkdir(parents=True, exist_ok=False)
    moves["counts"].to_csv(out / "transition_counts.csv")
    moves["shares"].round(4).to_csv(out / "transition_shares.csv")
    mix.round(4).to_csv(out / "segment_mix.csv", index=False)
    feature_psi.round(4).to_csv(out / "feature_psi.csv", index=False)
    new.loc[~returning, ["account_id", "segment_id", "confidence"]].to_csv(out / "new_accounts_scored.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_report(out / "report.md", summary, moves["shares"], mix, feature_psi)
    print(f"New-period scoring {summary['status']}: {out}")
    return summary


def _write_report(path: Path, s: dict[str, Any], shares: pd.DataFrame, mix: pd.DataFrame, feature_psi: pd.DataFrame) -> None:
    def table(df: pd.DataFrame) -> str:
        df = df.reset_index() if df.index.name else df
        rows = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
        rows += ["| " + " | ".join(str(round(v, 3)) if isinstance(v, float) else str(v) for v in r) + " |"
                 for r in df.itertuples(index=False)]
        return "\n".join(rows)

    conf = s["confidence_median"]
    lines = [
        f"# Berka new-period scoring: {s['new_as_of']} (frozen {s['model_version']})", "",
        f"**Status: {s['status']}** (PSI review threshold {s['psi_review_threshold']}). {s['limitation']}", "",
        (f"- {s['n_returning_accounts']} returning accounts: {s['returning']['stay_rate']:.1%} stayed in their "
        f"{s['train_as_of']} segment (ARI {s['returning']['ari']})."),
        (f"- {s['n_new_accounts']} accounts are new at {s['new_as_of']} and were never seen in training; "
        f"median fit confidence {conf['new_accounts']} vs {conf['returning_accounts']} for returning accounts."),
        (f"- Segment-mix PSI {s['segment_mix_psi']}; features needing review: "
        f"{', '.join(s['features_needing_review']) or 'none'}."), "",
        "## Where returning accounts moved (row = earlier segment, share of that row)", "", table(shares), "",
        "## Segment mix", "", table(mix), "",
        "## Feature drift (PSI vs training snapshot)", "", table(feature_psi), "",
    ]
    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--as-of-date", default=FUTURE_ASOF)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    score_new_period(as_of_date=args.as_of_date, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
