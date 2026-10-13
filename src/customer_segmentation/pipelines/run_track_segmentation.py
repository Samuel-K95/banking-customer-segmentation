"""Config-driven segmentation for one feature track: compare, evaluate, select, explain.

Built to onboard a new dataset (Track D, BankSim) without track-specific code. Inputs come
from the data layer's per-track matrices and the feature contract; every choice lives in
the track's segmentation config (default: configs/models/track_d/segmentation_v1.yaml).

All candidates are fit on the same training customers and scored on the same holdout in
the same standardized feature space, so traditional and autoencoder candidates are
compared like for like (latent-space scores are not comparable and are not used).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import yaml
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.tree import DecisionTreeClassifier, export_text

from customer_segmentation.features.io import track_outputs
from customer_segmentation.models.metrics import evaluate_clustering
from customer_segmentation.models.tabular_ae import (
    TORCH_OK,
    encode_latent,
    train_autoencoder,
)

DEFAULT_CONFIG = Path("configs/models/track_d/segmentation_v1.yaml")


# ── inputs ────────────────────────────────────────────────────────────────────


def load_inputs(config: dict[str, Any], processed_dir: Path | None = None) -> dict[str, Any]:
    """Model matrix + interpretable frame for the configured track, aligned by id."""
    track = config["track"]
    with open(config["contract"]) as f:
        spec = yaml.safe_load(f)["tracks"][track]
    (id_column,) = spec["identifier_columns"]
    features = list(spec["model_features"])
    interpretable_path, model_path = track_outputs(track, processed_dir)
    model = pd.read_csv(model_path, dtype={id_column: str})
    interpretable = pd.read_csv(interpretable_path, dtype={id_column: str})
    if not model[id_column].equals(interpretable[id_column]):
        raise ValueError(f"{track}: model and interpretable matrices are not row-aligned on {id_column}")
    return {
        "track": track,
        "id_column": id_column,
        "features": features,
        "ids": model[id_column].to_numpy(),
        "X": model[features].to_numpy(dtype=np.float64),
        "interpretable": interpretable,
        "input_sha256": {p.name: _sha256(p) for p in (model_path, interpretable_path)},
    }


def split_indices(n: int, holdout_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.random.default_rng(seed).permutation(n)
    n_holdout = max(round(n * holdout_fraction), 1)
    return np.sort(order[n_holdout:]), np.sort(order[:n_holdout])


# ── candidates ────────────────────────────────────────────────────────────────


class Candidate:
    """A fitted method/k pair that can label any rows of the standardized matrix."""

    def __init__(self, method: str, k: int, model: Any, encoder: Any = None):
        self.method, self.k, self.model, self.encoder = method, k, model, encoder

    @property
    def name(self) -> str:
        return f"{self.method}_k{self.k}"

    def predict(self, X: np.ndarray) -> np.ndarray:
        space = encode_latent(self.encoder, X) if self.encoder is not None else X
        return self.model.predict(space).astype(int)


def fit_candidate(method: str, k: int, X: np.ndarray, seed: int, cfg: dict[str, Any], encoder: Any = None) -> Candidate:
    if method == "kmeans":
        return Candidate(method, k, KMeans(k, n_init=cfg["kmeans"]["n_init"], random_state=seed).fit(X))
    if method == "gmm":
        gmm = GaussianMixture(
            k, covariance_type=cfg["gmm"]["covariance_type"], n_init=cfg["gmm"]["n_init"], random_state=seed
        )
        return Candidate(method, k, gmm.fit(X))
    if method == "ae_kmeans":
        z = encode_latent(encoder, X)
        return Candidate(method, k, KMeans(k, n_init=cfg["kmeans"]["n_init"], random_state=seed).fit(z), encoder)
    raise ValueError(f"Unknown method {method}")


def train_encoder(X: np.ndarray, seed: int, ae_cfg: dict[str, Any]) -> Any:
    return train_autoencoder(
        X,
        latent_dim=ae_cfg["latent_dim"],
        epochs=ae_cfg["epochs"],
        batch_size=ae_cfg["batch_size"],
        lr=ae_cfg["lr"],
        random_state=seed,
    ).model


def segment_fractions(labels: np.ndarray, k: int) -> np.ndarray:
    return np.bincount(labels, minlength=k) / max(len(labels), 1)


def compare_candidates(data: dict[str, Any], config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Candidate], dict]:
    """Fit every method/k on train, score on holdout, measure stability; return the board."""
    X, cfg = data["X"], config["candidates"]
    seed = config["split"]["random_seed"]
    train, holdout = split_indices(len(X), config["split"]["holdout_fraction"], seed)
    X_train, X_hold = X[train], X[holdout]
    n_seeds = config["stability"]["seeds"]
    n_rounds = config["stability"]["subsample_rounds"]

    methods = ["kmeans", "gmm"]
    encoders: dict[int, Any] = {}
    notes: dict[str, str] = {}
    if TORCH_OK:
        methods.append("ae_kmeans")
        encoders = {s: train_encoder(X_train, seed + s, cfg["autoencoder"]) for s in range(n_seeds + 1)}
    else:
        notes["ae_kmeans"] = "skipped: PyTorch is not installed (install the torch-cpu extra)"

    rows, fitted = [], {}
    for method in methods:
        for k in cfg["k_values"]:
            base = fit_candidate(method, k, X_train, seed, cfg, encoders.get(0))
            labels = base.predict(X_hold)
            # Seed stability: refit (for the AE: retrain the encoder too) and compare holdout labels.
            seed_aris = [
                adjusted_rand_score(labels, fit_candidate(method, k, X_train, seed + s, cfg, encoders.get(s)).predict(X_hold))
                for s in range(1, n_seeds + 1)
            ]
            # Subsample stability: refit on 75% of train (same encoder for the AE).
            rng = np.random.default_rng(seed)
            sub_aris = []
            for r in range(n_rounds):
                idx = rng.choice(len(X_train), size=int(0.75 * len(X_train)), replace=False)
                sub = fit_candidate(method, k, X_train[idx], seed + 100 + r, cfg, encoders.get(0))
                sub_aris.append(adjusted_rand_score(labels, sub.predict(X_hold)))
            fractions = segment_fractions(labels, k)
            metrics = evaluate_clustering(X_hold, labels, sample_size=None)
            rows.append({
                "candidate": base.name, "method": method, "k": k,
                "holdout_silhouette": metrics["silhouette"],
                "holdout_davies_bouldin": metrics["db_index"],
                "holdout_calinski_harabasz": metrics["ch_index"],
                "seed_ari_mean": float(np.mean(seed_aris)), "seed_ari_min": float(np.min(seed_aris)),
                "subsample_ari_mean": float(np.mean(sub_aris)),
                "min_segment_fraction": float(fractions.min()), "max_segment_fraction": float(fractions.max()),
            })
            fitted[base.name] = base
    split = {"n_train": len(train), "n_holdout": len(holdout), "train_idx": train, "holdout_idx": holdout}
    return pd.DataFrame(rows), fitted, {"split": split, "notes": notes}


def select_candidate(board: pd.DataFrame, rules: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Gate on segment sizes and seed stability, then rank; ties go to the smaller k."""
    board = board.copy()
    board["size_ok"] = (board["min_segment_fraction"] >= rules["min_segment_fraction"]) & (
        board["max_segment_fraction"] <= rules["max_segment_fraction"]
    )
    board["stable"] = board["seed_ari_mean"] >= rules["min_seed_ari"]
    board["eligible"] = board["size_ok"] & board["stable"]
    pool = board[board["eligible"]] if board["eligible"].any() else board
    ranks = pd.concat(
        [
            pool["holdout_silhouette"].rank(ascending=False),
            pool["holdout_davies_bouldin"].rank(ascending=True),
            pool["seed_ari_mean"].rank(ascending=False),
        ],
        axis=1,
    ).mean(axis=1)
    pool = pool.assign(mean_rank=ranks).sort_values(["mean_rank", "k", "candidate"])
    chosen = pool.iloc[0]
    return chosen["candidate"], {
        "selected": chosen["candidate"],
        "all_gates_met": bool(board["eligible"].any()),
        "eligible_candidates": board.loc[board["eligible"], "candidate"].tolist(),
        "mean_rank": float(chosen["mean_rank"]),
        "runner_up": pool.iloc[1]["candidate"] if len(pool) > 1 else None,
        "rules": rules,
    }


# ── intelligence ──────────────────────────────────────────────────────────────


def profile_segments(data: dict[str, Any], labels: np.ndarray, top_n: int = 3) -> tuple[pd.DataFrame, list[dict]]:
    """Per-segment means (original units), lift vs population and top distinguishing features."""
    features, X, frame = data["features"], data["X"], data["interpretable"]
    population = frame[features].mean()
    rows, catalog = [], []
    for segment in sorted(np.unique(labels)):
        mask = labels == segment
        means = frame.loc[mask, features].mean()
        # X is standardized, so the segment mean is an effect size in population-sd units.
        effect = pd.Series(X[mask].mean(axis=0), index=features)
        top = effect.abs().sort_values(ascending=False).head(top_n).index
        drivers = [{"feature": f, "direction": "high" if effect[f] > 0 else "low", "effect_sd": round(float(effect[f]), 3)} for f in top]
        centroid = X[mask].mean(axis=0)
        nearest = np.argsort(((X[mask] - centroid) ** 2).sum(axis=1))[:3]
        catalog.append({
            "segment": int(segment),
            "name": " / ".join(f"{d['direction']} {d['feature']}" for d in drivers[:2]),
            "size": int(mask.sum()),
            "share": round(float(mask.mean()), 4),
            "top_drivers": drivers,
            "representative_customers": data["ids"][mask][nearest].tolist(),
        })
        for f in features:
            rows.append({"segment": int(segment), "feature": f, "segment_mean": float(means[f]),
                         "population_mean": float(population[f]),
                         "lift": float(means[f] / population[f]) if population[f] else float("nan"),
                         "effect_sd": float(effect[f])})
    return pd.DataFrame(rows), catalog


def explain_with_tree(data: dict[str, Any], labels: np.ndarray, split: dict, depth: int) -> dict[str, Any]:
    """A shallow tree on readable features: rules a business user can follow, plus fidelity."""
    features, frame = data["features"], data["interpretable"]
    tree = DecisionTreeClassifier(max_depth=depth, random_state=0)
    tree.fit(frame.iloc[split["train_idx"]][features], labels[split["train_idx"]])
    fidelity = float((tree.predict(frame.iloc[split["holdout_idx"]][features]) == labels[split["holdout_idx"]]).mean())
    importance = sorted(zip(features, tree.feature_importances_), key=lambda x: -x[1])
    return {
        "surrogate_tree_depth": depth,
        "holdout_fidelity": round(fidelity, 4),
        "feature_importance": {f: round(float(v), 4) for f, v in importance if v > 0},
        "rules": export_text(tree, feature_names=features, decimals=2),
    }


def cramers_v(table: pd.DataFrame) -> float:
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    r, c = table.shape
    return float(np.sqrt(chi2 / (n * max(min(r, c) - 1, 1))))


def fairness_check(data: dict[str, Any], labels: np.ndarray, split: dict, columns: list[str], depth: int, min_group: int = 30) -> dict[str, Any]:
    """Representation by group, association strength, and demographic vs behavioral drivers."""
    frame = data["interpretable"]
    out: dict[str, Any] = {"columns": {}, "min_group_size_for_ratios": min_group}
    for column in columns:
        groups = frame[column].astype(str)
        table = pd.crosstab(labels, groups)
        overall = groups.value_counts(normalize=True)
        large = [g for g in table.columns if (groups == g).sum() >= min_group]
        within = table.div(table.sum(axis=1), axis=0)
        ratios = within[large].div(overall[large], axis=1)
        out["columns"][column] = {
            "cramers_v": round(cramers_v(table[large]), 4),
            "group_sizes": groups.value_counts().to_dict(),
            "representation_ratio": {int(s): {g: round(float(v), 3) for g, v in r.items()} for s, r in ratios.iterrows()},
            "max_ratio": round(float(ratios.to_numpy().max()), 3),
            "min_ratio": round(float(ratios.to_numpy().min()), 3),
        }
    # How well do demographics alone predict the segment, vs always guessing the biggest one?
    demo = pd.get_dummies(frame[columns].astype(str))
    tree = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(demo.iloc[split["train_idx"]], labels[split["train_idx"]])
    hold = labels[split["holdout_idx"]]
    out["demographics_only_accuracy"] = round(float((tree.predict(demo.iloc[split["holdout_idx"]]) == hold).mean()), 4)
    out["majority_class_accuracy"] = round(float(np.bincount(hold).max() / len(hold)), 4)
    return out


def outcome_validation(data: dict[str, Any], labels: np.ndarray, columns: list[str]) -> dict[str, Any]:
    """Rates of a binary outcome the model never saw, per segment (Wilson 95% CI, chi-square)."""
    frame, result = data["interpretable"], {}
    for column in columns:
        y = frame[column].astype(int).to_numpy()
        segments = []
        for segment in sorted(np.unique(labels)):
            mask = labels == segment
            n, hits = int(mask.sum()), int(y[mask].sum())
            low, high = _wilson(hits, n)
            segments.append({"segment": int(segment), "n": n, "rate": round(hits / n, 4), "ci95": [round(low, 4), round(high, 4)]})
        table = pd.crosstab(labels, y)
        result[column] = {
            "overall_rate": round(float(y.mean()), 4),
            "segments": segments,
            "chi2_p_value": float(chi2_contingency(table)[1]) if table.shape[1] > 1 else None,
            "cramers_v": round(cramers_v(table), 4) if table.shape[1] > 1 else None,
        }
    return result


# ── run ───────────────────────────────────────────────────────────────────────


def run_track_segmentation(config_path: str | Path = DEFAULT_CONFIG, processed_dir: Path | None = None) -> dict[str, Any]:
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text())
    out = Path(config["artifacts"]["output_directory"])
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing segmentation output: {out}")

    started = datetime.now(UTC)
    data = load_inputs(config, processed_dir)
    board, fitted, extra = compare_candidates(data, config)
    chosen_name, selection = select_candidate(board, config["selection"])
    chosen = fitted[chosen_name]
    labels = chosen.predict(data["X"])
    split = extra["split"]

    profiles, catalog = profile_segments(data, labels)
    depth = config["explainability"]["surrogate_tree_depth"]
    explain = explain_with_tree(data, labels, split, depth)
    fairness = fairness_check(data, labels, split, config["fairness"]["columns"], depth)
    outcomes = outcome_validation(data, labels, config["outcomes"]["columns"])

    out.mkdir(parents=True, exist_ok=False)
    board.to_csv(out / "comparison_board.csv", index=False)
    profiles.to_csv(out / "segment_profiles.csv", index=False)
    in_holdout = np.zeros(len(labels), dtype=bool)
    in_holdout[split["holdout_idx"]] = True
    pd.DataFrame({data["id_column"]: data["ids"], "segment": labels,
                  "split": np.where(in_holdout, "holdout", "train")}).to_csv(out / "assignments.csv", index=False)
    summary = {
        "experiment": config["experiment"],
        "track": data["track"],
        "selection": selection,
        "notes": extra["notes"],
        "segments": catalog,
        "explainability": {k: v for k, v in explain.items() if k != "rules"},
        "fairness": fairness,
        "outcome_validation": outcomes,
        "outcome_caveat": config["outcomes"].get("caveat"),
    }
    _write_json(out / "summary.json", summary)
    (out / "surrogate_tree_rules.txt").write_text(explain["rules"])
    _write_json(out / "run_metadata.json", {
        "config": str(config_path), "config_sha256": _sha256(config_path),
        "inputs_sha256": data["input_sha256"], "n_customers": len(labels),
        "n_train": split["n_train"], "n_holdout": split["n_holdout"],
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__, "torch_available": TORCH_OK,
    })
    _write_report(out / "report.md", summary, board)
    print(f"Selected {chosen_name}; artifacts in {out}")
    return summary


def _write_report(path: Path, s: dict[str, Any], board: pd.DataFrame) -> None:
    sel, fair, ex = s["selection"], s["fairness"], s["explainability"]
    cols = ["candidate", "holdout_silhouette", "holdout_davies_bouldin", "seed_ari_mean", "subsample_ari_mean",
            "min_segment_fraction", "max_segment_fraction"]
    lines = [
        f"# {s['experiment']} ({s['track']})", "",
        f"**Selected:** `{sel['selected']}` (runner-up `{sel['runner_up']}`). "
        + ("All gates met." if sel["all_gates_met"] else "No candidate met every gate; best-ranked overall was used."),
        "",
        ("Every candidate was fit on the same training customers and scored on the same holdout in the same "
        "standardized feature space."),
        "",
    ]
    lines += [f"- {k}: {v}" for k, v in s["notes"].items()]
    lines += ["", "## Candidates", "", _markdown_table(board[cols].round(4)), "", "## Segments", ""]
    for seg in s["segments"]:
        drivers = ", ".join(f"{d['direction']} {d['feature']} ({d['effect_sd']:+.2f} sd)" for d in seg["top_drivers"])
        lines.append(f"- **{seg['segment']} — {seg['name']}**: {seg['size']} customers ({seg['share']:.1%}); {drivers}")
    lines += ["", "## Explainability", "",
              (f"A depth-{ex['surrogate_tree_depth']} decision tree on readable features reproduces the segment for "
              f"{ex['holdout_fidelity']:.1%} of holdout customers (rules in `surrogate_tree_rules.txt`)."), "",
              "## Fairness", ""]
    for col, f in fair["columns"].items():
        lines.append(f"- `{col}`: Cramér's V {f['cramers_v']}, representation ratio {f['min_ratio']}–{f['max_ratio']} "
                     f"(groups with ≥{fair['min_group_size_for_ratios']} customers)")
    lines += [(f"- Demographics alone predict the segment for {fair['demographics_only_accuracy']:.1%} of holdout customers "
              f"vs {fair['majority_class_accuracy']:.1%} by always guessing the largest segment."), "",
              "## Outcome validation (not used in fitting)", ""]
    for col, o in s["outcome_validation"].items():
        rates = ", ".join(f"{x['segment']}: {x['rate']:.1%} [{x['ci95'][0]:.1%}–{x['ci95'][1]:.1%}]" for x in o["segments"])
        lines.append(f"- `{col}` overall {o['overall_rate']:.1%}; by segment {rates}; chi-square p={o['chi2_p_value']:.2g}, "
                     f"Cramér's V {o['cramers_v']}.")
    lines += ["", "Outcome rates describe the segments; they are not predictions."]
    if s.get("outcome_caveat"):
        lines += ["", f"**Caveat:** {s['outcome_caveat'].strip()}"]
    path.write_text("\n".join(lines) + "\n")


def _markdown_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, rule, *body])


def _wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args(argv)
    run_track_segmentation(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
