import numpy as np
import pandas as pd
import pytest
from src.monte_carlo import (
    MonteCarloResult,
    run_monte_carlo_simulation,
    calibrate_vasicek_params,
)

def test_calibrate_vasicek_params():
    np.random.seed(42)
    # Serie con reversión a 3.0
    series = np.array([3.5, 3.4, 3.3, 3.2, 3.1, 3.0, 3.05, 2.95, 3.02, 2.98])
    kappa, theta, sigma = calibrate_vasicek_params(series)
    assert kappa > 0
    assert 2.0 <= theta <= 4.0
    assert sigma > 0

def test_monte_carlo_simulation_shape_and_percentiles():
    last_date = pd.Timestamp("2024-01-05")
    res = run_monte_carlo_simulation(
        last_date=last_date,
        initial_rate=3.5,
        kappa=0.5,
        theta=3.0,
        sigma=0.08,
        horizon_days=63,
        n_sims=500,
        seed=42,
    )
    assert isinstance(res, MonteCarloResult)
    assert res.simulated_paths.shape == (500, 63)
    assert len(res.future_dates) == 63
    assert len(res.terminal_distribution) == 500
    
    assert "P10" in res.percentiles
    assert "P25" in res.percentiles
    assert "P50" in res.percentiles
    assert "P75" in res.percentiles
    assert "P90" in res.percentiles
    
    # Verificación de monotonicidad de percentiles
    p10 = res.percentiles["P10"]
    p50 = res.percentiles["P50"]
    p90 = res.percentiles["P90"]
    assert (p90 >= p50).all()
    assert (p50 >= p10).all()
