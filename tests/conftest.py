"""Shared pytest setup."""

from __future__ import annotations

import importlib.util

import pytest

TORCH_MISSING_REASON = (
    "needs PyTorch: the served champion encoder is a PyTorch model "
    "(install with `uv sync --extra dev --extra torch-cpu`)"
)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", f"requires_torch: {TORCH_MISSING_REASON}")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if importlib.util.find_spec("torch") is not None:
        return
    skip = pytest.mark.skip(reason=TORCH_MISSING_REASON)
    for item in items:
        if "requires_torch" in item.keywords:
            item.add_marker(skip)
