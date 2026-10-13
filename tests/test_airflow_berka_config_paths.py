"""Check Airflow segmentation config resolution without installing Airflow."""

import ast
import importlib
from pathlib import Path

import yaml


def test_airflow_segmentation_stage_configs_resolve():
    root = Path(__file__).resolve().parents[1]
    dag_path = root / "pipelines/airflow/dags/bcip_data_layer.py"
    tree = ast.parse(dag_path.read_text(encoding="utf-8"))

    stages = next(
        (
            ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "SEGMENTATION_STAGES"
                for target in node.targets
            )
        ),
        None,
    )

    assert stages, "SEGMENTATION_STAGES is missing or empty"

    required_stages = {
        "berka_snapshot",
        "berka_baselines",
        "berka_feature_sensitivity",
        "berka_temporal_validation",
        "track_d_segmentation",
    }
    assert required_stages <= set(stages), (
        f"Missing stages: {required_stages - set(stages)}"
    )

    for stage_name, (_, _, config_owner, keys) in stages.items():
        module = importlib.import_module(
            f"customer_segmentation.{config_owner}"
        )
        config_name = getattr(module, "DEFAULT_CONFIG", None)

        assert config_name is not None, (
            f"{stage_name}: module has no DEFAULT_CONFIG"
        )

        config_path = root / config_name
        assert config_path.is_file(), (
            f"{stage_name}: missing {config_path}"
        )

        value = yaml.safe_load(
            config_path.read_text(encoding="utf-8")
        )

        for key in keys:
            assert isinstance(value, dict) and key in value, (
                f"{stage_name}: missing config key {key}"
            )
            value = value[key]

        assert isinstance(value, str) and value.strip(), (
            f"{stage_name}: invalid output location"
        )

        print(f"PASS {stage_name}: {config_name}")
