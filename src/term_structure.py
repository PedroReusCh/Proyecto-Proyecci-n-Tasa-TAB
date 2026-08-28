"""
Módulo de Estructura Temporal y Curva de Rendimiento TAB UF.
Calcula curvas de tasas y métricas de forma (pendiente / slope y curvatura / butterfly).
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
from src.forecasting_engine import ForecastResult

TENORS_DAYS = [90, 180, 360]

@dataclass(frozen=True)
class TermStructureCurve:
    """Representa una curva de rendimiento en una fecha dada."""
    date: pd.Timestamp
    tenors: List[int]
    rates: List[float]
    slope_360_90: float
    butterfly_180: float
    label: str

def compute_term_structure(
    df: pd.DataFrame,
    row_idx: int = -1,
    label: str = "Actual",
) -> TermStructureCurve:
    """Extrae la curva de tasas fijadas para una fila específica del DataFrame."""
    row = df.iloc[row_idx]
    date_val = pd.Timestamp(row["Date"])
    r90 = float(row["TAB_UF_90"])
    r180 = float(row["TAB_UF_180"])
    r360 = float(row["TAB_UF_360"])
    
    slope = r360 - r90
    butterfly = 2.0 * r180 - (r90 + r360)
    
    return TermStructureCurve(
        date=date_val,
        tenors=TENORS_DAYS,
        rates=[r90, r180, r360],
        slope_360_90=slope,
        butterfly_180=butterfly,
        label=label,
    )

def compute_projected_curves(
    current_curve: TermStructureCurve,
    forecast_90: ForecastResult,
    forecast_180: ForecastResult,
    forecast_360: ForecastResult,
) -> Dict[str, TermStructureCurve]:
    """
    Construye las curvas proyectadas para los horizontes clave (+1m, +3m, +6m, +12m).
    """
    horizons_map = {
        "+1 Mes": 21,
        "+3 Meses": 63,
        "+6 Meses": 126,
        "+1 Año": 252,
    }
    
    curves: Dict[str, TermStructureCurve] = {}
    
    for h_label, h_step in horizons_map.items():
        step_idx = min(h_step, len(forecast_90.point_forecast)) - 1
        if step_idx >= 0:
            target_date = forecast_90.future_dates[step_idx]
            r90 = float(forecast_90.point_forecast[step_idx])
            r180 = float(forecast_180.point_forecast[step_idx])
            r360 = float(forecast_360.point_forecast[step_idx])
            
            slope = r360 - r90
            butterfly = 2.0 * r180 - (r90 + r360)
            
            curves[h_label] = TermStructureCurve(
                date=target_date,
                tenors=TENORS_DAYS,
                rates=[r90, r180, r360],
                slope_360_90=slope,
                butterfly_180=butterfly,
                label=f"Proyección {h_label}",
            )
            
    return curves
