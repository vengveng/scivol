"""
pytest configuration and shared fixtures for volkit tests.
"""
import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (4+ minutes, fit many models)"
    )


def pytest_addoption(parser):
    """Add command-line options."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests (parallel/auto selection tests that take 4+ minutes)"
    )


def pytest_collection_modifyitems(config, items):
    """Skip slow tests unless --run-slow is specified."""
    if config.getoption("--run-slow"):
        # --run-slow given: do not skip slow tests
        return
    
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
