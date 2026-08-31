import numpy as np
import pytest
from src.diagnostics import (
    run_adf_test,
    run_ljung_box_test,
    compute_acf_pacf,
    ADFResult,
    LjungBoxResult,
)

def test_run_adf_test():
    np.random.seed(42)
    # Serie estacionaria (ruido blanco)
    stationary_series = np.random.normal(0, 1, 200)
    res = run_adf_test(stationary_series)
    
    assert isinstance(res, ADFResult)
    assert res.p_value < 0.05
    assert res.is_stationary is True
    assert "1%" in res.critical_values

def test_run_ljung_box_test():
    np.random.seed(42)
    # Residuos ruido blanco
    white_noise = np.random.normal(0, 0.1, 150)
    res = run_ljung_box_test(white_noise, lags=10)
    
    assert isinstance(res, LjungBoxResult)
    assert res.lags == 10
    assert len(res.p_values) == 10
    assert res.is_white_noise is True

def test_compute_acf_pacf():
    np.random.seed(42)
    series = np.cumsum(np.random.normal(0, 0.05, 100))
    acf_vals, pacf_vals, conf_bound = compute_acf_pacf(series, nlags=15)
    
    assert len(acf_vals) == 16  # lag 0 a 15
    assert len(pacf_vals) == 16
    assert conf_bound > 0
