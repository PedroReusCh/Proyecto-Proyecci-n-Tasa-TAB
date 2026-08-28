"""
Módulo de Simulación Monte Carlo y Análisis Estocástico de Riesgo.
Genera distribuciones de trayectorias y bandas de confianza para proyecciones de tasas TAB UF.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from src.forecasting_engine import generate_future_business_dates

@dataclass(frozen=True)
class MonteCarloResult:
    """Resultados de la simulación estocástica de Monte Carlo."""
    simulated_paths: np.ndarray  # Matriz (n_sims, horizon_days)
    percentiles: Dict[str, np.ndarray]
    future_dates: List[pd.Timestamp]
    terminal_distribution: np.ndarray

def calibrate_vasicek_params(series: np.ndarray) -> Tuple[float, float, float]:
    """
    Calibra los parámetros de reversión a la media kappa, theta y sigma a partir de una serie histórica.
    """
    sub_series = np.asarray(series[-min(len(series), 500):], dtype=float)
    r_prev = sub_series[:-1]
    dr = sub_series[1:] - r_prev
    
    poly = np.polyfit(r_prev, dr, 1)
    b, a = float(poly[0]), float(poly[1])
    
    dt = 1.0 / 252.0  # Base diaria
    if b < 0:
        kappa = -np.log(1.0 + b) / dt if (1.0 + b) > 0 else -b / dt
        theta = -a / b if abs(b) > 1e-7 else float(np.mean(sub_series))
    else:
        kappa = 0.3
        theta = float(np.mean(sub_series))
    
    residuals = dr - (a + b * r_prev)
    sigma = float(np.std(residuals) / np.sqrt(dt))
    
    return max(kappa, 0.05), theta, max(sigma, 0.01)

def run_monte_carlo_simulation(
    last_date: pd.Timestamp,
    initial_rate: float,
    kappa: float,
    theta: float,
    sigma: float,
    horizon_days: int,
    n_sims: int = 1000,
    seed: Optional[int] = None,
) -> MonteCarloResult:
    """
    Genera n_sims trayectorias del proceso Ornstein-Uhlenbeck:
      r_{t+dt} = r_t + kappa * (theta - r_t) * dt + sigma * sqrt(dt) * Z_t
    """
    if seed is not None:
        np.random.seed(seed)
    
    dt = 1.0 / 252.0
    sqrt_dt = np.sqrt(dt)
    
    paths = np.zeros((n_sims, horizon_days), dtype=float)
    current_rates = np.full(n_sims, initial_rate, dtype=float)
    
    for step in range(horizon_days):
        z = np.random.normal(0.0, 1.0, n_sims)
        drift = kappa * (theta - current_rates) * dt
        diffusion = sigma * sqrt_dt * z
        current_rates = current_rates + drift + diffusion
        paths[:, step] = current_rates
    
    future_dates = generate_future_business_dates(last_date, horizon_days)
    
    # Cálculo de percentiles a lo largo de cada paso temporal
    percentiles = {
        "P10": np.percentile(paths, 10, axis=0),
        "P25": np.percentile(paths, 25, axis=0),
        "P50": np.percentile(paths, 50, axis=0),
        "P75": np.percentile(paths, 75, axis=0),
        "P90": np.percentile(paths, 90, axis=0),
    }
    
    terminal_dist = paths[:, -1]
    
    return MonteCarloResult(
        simulated_paths=paths,
        percentiles=percentiles,
        future_dates=future_dates,
        terminal_distribution=terminal_dist,
    )
