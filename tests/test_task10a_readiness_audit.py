from pathlib import Path

import numpy as np
import pandas as pd

from scripts.task10a_champion_readiness_audit import (
    _manifest_roles,
    _population_demographics,
    _task_label,
    demographic_groups,
)


def test_demographic_groups_are_descriptive_and_handle_missing_values():
    frame = pd.DataFrame(
        {
            "Age": [18, 34, 65, None],
            "Gender_M": [0, 1, None, 1],
            "Is_Tier_1_City": [1, 0, 1, None],
        }
    )

    grouped = demographic_groups(frame)

    assert grouped["Age_Group"].tolist() == ["18-24", "25-34", "65-100", "Missing_or_out_of_range"]
    assert grouped["Gender_Group"].tolist() == [
        "Gender_M_indicator_0_non_M",
        "Gender_M_indicator_1_Male",
        "Missing_or_unknown",
        "Gender_M_indicator_1_Male",
    ]
    assert grouped["City_Tier_Group"].tolist() == [
        "Tier_1_city",
        "Non_tier_1_city",
        "Tier_1_city",
        "Missing_or_unknown",
    ]


def test_manifest_task_and_internal_split_roles_are_classified_explicitly():
    path = Path(
        "artifacts/experiments/samples/task09_autoencoder/"
        "task09_track_a_final_ae_20261011T160711218914Z_97ecb040/seed42_internal_split.csv"
    )
    frame = pd.DataFrame({"internal_split": ["internal_train", "internal_validation"]})

    assert _task_label(path) == ("Task09", "completed")
    assert _manifest_roles(frame, path) == ["internal_train", "internal_validation"]
    assert _task_label(
        Path("artifacts/experiments/samples/task08_track_a_p2_20261011T135659725728Z_47cae105/two_plus_train.csv")
    ) == ("Task08", "failed_manifest_only")


def test_population_demographic_distribution_uses_age_bands_not_single_years():
    frame = demographic_groups(
        pd.DataFrame(
            {
                "CustomerID": ["a", "b", "c"],
                "Age": [22, 31, 67],
                "Transaction_Count": [1, 2, 3],
                "Gender_M": [0, 1, 1],
                "Is_Tier_1_City": [0, 1, 0],
            }
        )
    )

    distribution, missingness, scopes = _population_demographics(
        frame, np.array(["a", "b"], dtype=str)
    )

    age_groups = set(
        distribution.loc[
            (distribution.scope == "validated_population")
            & (distribution.demographic_field == "Age"),
            "group",
        ]
    )
    assert age_groups == {"18-24", "25-34", "65-100"}
    assert scopes["two_plus_population"] == 2
    assert missingness["missing_or_null"].sum() == 0
