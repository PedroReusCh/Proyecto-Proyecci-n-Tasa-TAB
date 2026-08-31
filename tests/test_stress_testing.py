import numpy as np
import pandas as pd
import pytest
from src.forecasting_engine import ForecastResult
from src.stress_testing import (
    simulate_stress_scenarios,
    StressScenarioResult,
)

def test_simulate_stress_scenarios():
    dates = pd.date_range("2024-01-01", periods=63, freq="B")
    point_pred = np.full(63, 3.0)
    base_res = ForecastResult(
        model_name="AutoARIMA",
        horizon_days=63,
        future_dates=list(dates),
        point_forecast=point_pred,
        lower_80=point_pred - 0.2,
        upper_80=point_pred + 0.2,
        lower_95=point_pred - 0.4,
        upper_95=point_pred + 0.4,
    )
    
    stress_res = simulate_stress_scenarios(base_res, shock_bp=100.0, vol_multiplier=1.5)
    
    assert isinstance(stress_res, StressScenarioResult)
    assert len(stress_res.base_path) == 63
    assert len(stress_res.bull_shock_path) == 63
    assert len(stress_res.bear_shock_path) == 63
    
    # El escenario alcista (+100 pb = +1.0%) debe ser mayor al base al final
    assert stress_res.bull_shock_path[-1] > stress_res.base_path[-1]
    assert stress_res.bear_shock_path[-1] < stress_res.base_path[-1]
    assert stress_res.shock_bp == 100.0
