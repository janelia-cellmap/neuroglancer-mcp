"""Shared pytest configuration."""

from __future__ import annotations


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "network: tests that hit live external services (skipped by default)",
    )


def pytest_collection_modifyitems(config, items):
    import pytest

    if config.getoption("-m") or config.getoption("markexpr"):
        return
    skip = pytest.mark.skip(reason="network test; run with `-m network` to enable")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)
