import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from customer_segmentation.features.build_banksim_features import MODEL_FEATURES
from customer_segmentation.features.io import save_matrices
from customer_segmentation.pipelines.run_track_segmentation import (
    run_track_segmentation,
    select_candidate,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def write_three_groups(processed_dir: Path, n_per_group: int = 120) -> None:
    """Three well-separated behavior groups; demographics random; outcome tied to group 2."""
    rng = np.random.default_rng(7)
    centers = rng.normal(0, 4, size=(3, len(MODEL_FEATURES)))
    group = np.repeat([0, 1, 2], n_per_group)
    X = centers[group] + rng.normal(0, 0.5, size=(len(group), len(MODEL_FEATURES)))
    ids = [f"C{i}" for i in range(len(group))]
    model = pd.DataFrame(X, columns=MODEL_FEATURES).assign(customer_id=ids)[["customer_id", *MODEL_FEATURES]]
    interpretable = model.copy()
    interpretable["age_band"] = rng.choice(["2", "3", "4"], size=len(group))
    interpretable["gender"] = rng.choice(["F", "M"], size=len(group))
    interpretable["any_fraud"] = (group == 2).astype(int)
    interpretable["fraud_txn_count"] = interpretable["any_fraud"]
    save_matrices("track_d", interpretable, model, processed_dir)


def write_config(tmp_path: Path) -> Path:
    config = yaml.safe_load((REPO_ROOT / "configs/models/track_d/segmentation_v1.yaml").read_text())
    config["contract"] = str(REPO_ROOT / config["contract"])
    config["candidates"]["k_values"] = [2, 3, 4]
    config["candidates"]["autoencoder"]["epochs"] = 3
    config["stability"] = {"seeds": 2, "subsample_rounds": 2}
    config["artifacts"]["output_directory"] = str(tmp_path / "out")
    path = tmp_path / "segmentation.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_segmentation_finds_the_groups_and_writes_every_artifact(tmp_path):
    write_three_groups(tmp_path / "processed")
    config_path = write_config(tmp_path)

    summary = run_track_segmentation(config_path, processed_dir=tmp_path / "processed")

    assert summary["selection"]["selected"].endswith("_k3")
    assert summary["selection"]["all_gates_met"]
    assert summary["explainability"]["holdout_fidelity"] >= 0.95
    for column in ["age_band", "gender"]:
        assert summary["fairness"]["columns"][column]["cramers_v"] < 0.3  # demographics are random here
    fraud = summary["outcome_validation"]["any_fraud"]
    assert sorted(s["rate"] for s in fraud["segments"]) == [0.0, 0.0, 1.0]
    assert fraud["chi2_p_value"] < 1e-6

    out = tmp_path / "out"
    for name in ["comparison_board.csv", "segment_profiles.csv", "assignments.csv", "summary.json",
                 "run_metadata.json", "surrogate_tree_rules.txt", "report.md"]:
        assert (out / name).is_file(), name
    assignments = pd.read_csv(out / "assignments.csv")
    assert len(assignments) == 360 and set(assignments["split"]) == {"train", "holdout"}
    assert json.loads((out / "run_metadata.json").read_text())["n_customers"] == 360


def test_existing_output_is_never_overwritten(tmp_path):
    write_three_groups(tmp_path / "processed")
    config_path = write_config(tmp_path)
    (tmp_path / "out").mkdir()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_track_segmentation(config_path, processed_dir=tmp_path / "processed")


def test_selection_skips_candidates_that_fail_the_size_gate():
    board = pd.DataFrame([
        # best metrics, but one segment holds 80% of customers
        {"candidate": "kmeans_k2", "k": 2, "holdout_silhouette": 0.6, "holdout_davies_bouldin": 1.0,
         "seed_ari_mean": 1.0, "min_segment_fraction": 0.2, "max_segment_fraction": 0.8},
        {"candidate": "gmm_k3", "k": 3, "holdout_silhouette": 0.3, "holdout_davies_bouldin": 1.5,
         "seed_ari_mean": 0.95, "min_segment_fraction": 0.1, "max_segment_fraction": 0.5},
    ])
    rules = {"min_segment_fraction": 0.03, "max_segment_fraction": 0.6, "min_seed_ari": 0.8}
    chosen, selection = select_candidate(board, rules)
    assert chosen == "gmm_k3" and selection["eligible_candidates"] == ["gmm_k3"]
