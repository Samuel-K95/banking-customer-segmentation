"""Static Airflow config-resolution test; does not require Airflow installation."""

import ast
from pathlib import Path

import yaml


def test_berka_airflow_stage_configs_resolve():
    root = Path(__file__).resolve().parents[1]
    dag_path = root / "pipelines/airflow/dags/bcip_data_layer.py"
    tree = ast.parse(dag_path.read_text(encoding="utf-8"))
    stages = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "BERKA_STAGES"
            for target in node.targets
        )
    )
    assert stages, "No Berka stages discovered"
    for stage_name, (_, config_name, keys) in stages.items():
        config_path = root / "configs/models" / config_name
        assert config_path.is_file(), f"{stage_name}: missing {config_path}"
        value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        for key in keys:
            assert key in value, f"{stage_name}: missing output config key {key}"
            value = value[key]
        assert isinstance(value, str) and value.strip(), stage_name
