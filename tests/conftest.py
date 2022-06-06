"""Provide pytest fixtures for the entire test suite.

These fixtures create data and data modules that can be reused by other tests.
"""

from pathlib import Path
import pytest

from braindelphi.utils_root import load_pickle_data


@pytest.fixture
def data_dict() -> dict:
    """Load example data."""
    pkl_path = Path(__file__).parent.joinpath("test_data.pkl")
    data = load_pickle_data(pkl_path)
    return data
