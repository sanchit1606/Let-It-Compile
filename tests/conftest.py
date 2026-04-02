import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Run tests marked as slow.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: validation tests for medium/large sizes (opt-in via --runslow)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return

    slow_items = [item for item in items if "slow" in item.keywords]
    if not slow_items:
        return

    for item in slow_items:
        items.remove(item)

    config.hook.pytest_deselected(items=slow_items)
