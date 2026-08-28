import numpy as np
import pandas as pd
import pytest
from src.term_structure import (
    TermStructureCurve,
    compute_term_structure,
    compute_projected_curves,
)
from src.forecasting_engine import ForecastResult

def test_compute_term_structure():
    df = pd.DataFrame({
        "Date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")],
        "TAB_UF_90": [2.5, 2.6],
        "TAB_UF_180": [2.8, 2.9],
        "TAB_UF_360": [3.2, 3.3],
    })
    curve = compute_term_structure(df)
    assert isinstance(curve, TermStructureCurve)
    assert curve.tenors == [90, 180, 360]
    assert curve.rates == [2.6, 2.9, 3.3]
    assert curve.slope_360_90 == pytest.approx(3.3 - 2.6)
    assert curve.butterfly_180 == pytest.approx(2 * 2.9 - (2.6 + 3.3))

def test_compute_projected_curves():
    dates = [pd.Timestamp("2024-02-01")] * 252
    f90 = ForecastResult("Ensemble", 252, dates, np.linspace(2.5, 3.0, 252), np.zeros(252), np.zeros(252), np.zeros(252), np.zeros(252))
    f180 = ForecastResult("Ensemble", 252, dates, np.linspace(2.8, 3.3, 252), np.zeros(252), np.zeros(252), np.zeros(252), np.zeros(252))
    f360 = ForecastResult("Ensemble", 252, dates, np.linspace(3.2, 3.7, 252), np.zeros(252), np.zeros(252), np.zeros(252), np.zeros(252))
    
    current_curve = TermStructureCurve(pd.Timestamp("2024-01-01"), [90, 180, 360], [2.5, 2.8, 3.2], 0.7, 0.0, "Actual")
    proj_curves = compute_projected_curves(current_curve, f90, f180, f360)
    
    assert "+1 Mes" in proj_curves
    assert "+3 Meses" in proj_curves
    assert "+6 Meses" in proj_curves
    assert "+1 Año" in proj_curves
    
    c_1m = proj_curves["+1 Mes"]
    assert len(c_1m.rates) == 3
    assert c_1m.tenors == [90, 180, 360]
