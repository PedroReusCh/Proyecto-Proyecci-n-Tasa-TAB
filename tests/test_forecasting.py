import numpy as np
import pandas as pd
import pytest
from src.forecasting_engine import (
    ForecastResult,
    AutoARIMAForecaster,
    HoltWintersForecaster,
    VasicekForecaster,
    VARMultivariateForecaster,
    MLLagsForecaster,
    EnsembleForecaster,
    generate_all_forecasts,
    generate_future_business_dates,
)

@pytest.fixture
def sample_tab_data() -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=150, freq="B")
    np.random.seed(42)
    s90 = 3.0 + np.cumsum(np.random.normal(0, 0.03, 150))
    s180 = s90 + 0.3 + np.random.normal(0, 0.01, 150)
    s360 = s180 + 0.4 + np.random.normal(0, 0.01, 150)
    return pd.DataFrame({"Date": dates, "TAB_UF_90": s90, "TAB_UF_180": s180, "TAB_UF_360": s360})

def test_generate_future_business_dates():
    last_date = pd.Timestamp("2024-01-05")  # Viernes
    future_dates = generate_future_business_dates(last_date, n_days=5)
    assert len(future_dates) == 5
    assert future_dates[0] == pd.Timestamp("2024-01-08")  # Lunes siguiente
    assert all(d.weekday() < 5 for d in future_dates)

def test_auto_arima_forecaster(sample_tab_data: pd.DataFrame):
    forecaster = AutoARIMAForecaster()
    res = forecaster.forecast(sample_tab_data, "TAB_UF_90", horizon_days=21)
    assert isinstance(res, ForecastResult)
    assert res.model_name == "AutoARIMA"
    assert len(res.point_forecast) == 21
    assert len(res.lower_80) == 21
    assert len(res.upper_80) == 21
    assert (res.upper_95 >= res.lower_95).all()

def test_holt_winters_forecaster(sample_tab_data: pd.DataFrame):
    forecaster = HoltWintersForecaster()
    res = forecaster.forecast(sample_tab_data, "TAB_UF_90", horizon_days=21)
    assert isinstance(res, ForecastResult)
    assert res.model_name == "Holt-Winters"
    assert len(res.point_forecast) == 21
    assert (res.upper_95 >= res.lower_95).all()

def test_vasicek_forecaster(sample_tab_data: pd.DataFrame):
    forecaster = VasicekForecaster()
    res = forecaster.forecast(sample_tab_data, "TAB_UF_90", horizon_days=21)
    assert isinstance(res, ForecastResult)
    assert res.model_name == "Vasicek"
    assert len(res.point_forecast) == 21
    assert (res.upper_95 >= res.lower_95).all()

def test_var_forecaster(sample_tab_data: pd.DataFrame):
    forecaster = VARMultivariateForecaster()
    res = forecaster.forecast(sample_tab_data, "TAB_UF_90", horizon_days=21)
    assert isinstance(res, ForecastResult)
    assert res.model_name == "VAR"
    assert len(res.point_forecast) == 21
    assert (res.upper_95 >= res.lower_95).all()

def test_ml_lags_forecaster(sample_tab_data: pd.DataFrame):
    forecaster = MLLagsForecaster()
    res = forecaster.forecast(sample_tab_data, "TAB_UF_90", horizon_days=21)
    assert isinstance(res, ForecastResult)
    assert res.model_name == "ML-Lags"
    assert len(res.point_forecast) == 21

def test_generate_all_forecasts_and_ensemble(sample_tab_data: pd.DataFrame):
    all_results = generate_all_forecasts(sample_tab_data, "TAB_UF_90", horizon_days=21)
    assert "AutoARIMA" in all_results
    assert "Holt-Winters" in all_results
    assert "Vasicek" in all_results
    assert "VAR" in all_results
    assert "ML-Lags" in all_results
    assert "Ensemble" in all_results
    
    ensemble_res = all_results["Ensemble"]
    assert len(ensemble_res.point_forecast) == 21
    assert not np.isnan(ensemble_res.point_forecast).any()
