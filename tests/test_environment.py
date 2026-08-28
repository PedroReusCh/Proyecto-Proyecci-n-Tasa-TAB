import importlib
import pytest

REQUIRED_MODULES = [
    "streamlit",
    "pandas",
    "numpy",
    "openpyxl",
    "requests",
    "plotly",
    "statsmodels",
    "sklearn",
    "scipy",
]

@pytest.mark.parametrize("module_name", REQUIRED_MODULES)
def test_required_modules_importable(module_name: str) -> None:
    mod = importlib.import_module(module_name)
    assert mod is not None
