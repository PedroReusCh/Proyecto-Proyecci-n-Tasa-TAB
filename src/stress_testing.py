"""
Módulo de Simulación de Escenarios de Estrés (Stress Testing) y Shocks Financieros.
Permite evaluar la resiliencia de las proyecciones frente a cambios en la política monetaria y volatilidad.
"""
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd
from src.forecasting_engine import ForecastResult

@dataclass(frozen=True)
class StressScenarioResult:
    """Contenedor de trayectorias bajo diferentes escenarios de estrés macrofinanciero."""
    shock_bp: float
    vol_multiplier: float
    future_dates: List[pd.Timestamp]
    base_path: np.ndarray
    bull_shock_path: np.ndarray
    bear_shock_path: np.ndarray
    stressed_lower_95: np.ndarray
    stressed_upper_95: np.ndarray
    impact_milestones: pd.DataFrame

def simulate_stress_scenarios(
    base_forecast: ForecastResult,
    shock_bp: float = 100.0,
    vol_multiplier: float = 1.0,
) -> StressScenarioResult:
    """
    Simula el impacto de un shock en puntos base (1 bp = 0.01%) y shock de volatilidad.
    Aplica una función de transmisión gradual (ramp-in) a lo largo del horizonte.
    """
    n_steps = base_forecast.horizon_days
    shock_pct = shock_bp / 100.0  # Convertir puntos base a porcentaje anual
    
    # Rampa de transmisión gradual del shock macroeconómico: 1 - exp(-3 * t / T)
    t_norm = np.linspace(0.1, 1.0, n_steps)
    transmission_curve = 1.0 - np.exp(-3.0 * t_norm)
    transmission_curve = transmission_curve / transmission_curve[-1]  # Normalizar a 1.0 al final
    
    dynamic_shock = shock_pct * transmission_curve
    
    base_path = base_forecast.point_forecast.copy()
    bull_path = base_path + dynamic_shock
    bear_path = base_path - dynamic_shock
    
    # Dispersión estresada por el multiplicador de volatilidad
    delta_lower = (base_path - base_forecast.lower_95) * vol_multiplier
    delta_upper = (base_forecast.upper_95 - base_path) * vol_multiplier
    stressed_lower = np.maximum(bear_path - delta_lower, -2.0)
    stressed_upper = bull_path + delta_upper
    
    # Cuadro de impacto en hitos
    milestones = [
        ("1 Mes (21d)", min(20, n_steps - 1)),
        ("3 Meses (63d)", min(62, n_steps - 1)),
        ("6 Meses (126d)", min(125, n_steps - 1)),
        ("1 Año (252d)", n_steps - 1),
    ]
    
    impact_rows = []
    for m_label, step_idx in milestones:
        if step_idx < n_steps:
            f_date = base_forecast.future_dates[step_idx].strftime("%d-%m-%Y")
            b_val = base_path[step_idx]
            bull_val = bull_path[step_idx]
            bear_val = bear_path[step_idx]
            diff_bp = (bull_val - b_val) * 100.0
            
            impact_rows.append({
                "Hito": m_label,
                "Fecha": f_date,
                "Escenario Base (%)": f"{b_val:.2f}%",
                f"Estrés Alcista (+{shock_bp:.0f} pb)": f"{bull_val:.2f}%",
                f"Estrés Bajista (-{shock_bp:.0f} pb)": f"{bear_val:.2f}%",
                "Impacto Máximo (pb)": f"±{diff_bp:.1f} pb",
            })
            
    impact_df = pd.DataFrame(impact_rows)
    
    return StressScenarioResult(
        shock_bp=shock_bp,
        vol_multiplier=vol_multiplier,
        future_dates=base_forecast.future_dates,
        base_path=base_path,
        bull_shock_path=bull_path,
        bear_shock_path=bear_path,
        stressed_lower_95=stressed_lower,
        stressed_upper_95=stressed_upper,
        impact_milestones=impact_df,
    )
