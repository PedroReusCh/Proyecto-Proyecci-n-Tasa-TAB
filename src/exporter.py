"""
Módulo de Exportación de Datos y Reportes Financieros.
Genera libros Excel (.xlsx) multishoheet y archivos CSV con formato profesional.
"""
import io
from typing import Optional
import pandas as pd

def generate_excel_report(
    df_hist: pd.DataFrame,
    df_projections: pd.DataFrame,
    df_tournament: pd.DataFrame,
    df_monte_carlo: Optional[pd.DataFrame] = None,
) -> bytes:
    """Genera un archivo Excel (.xlsx) en memoria con múltiples pestañas."""
    buffer = io.BytesIO()
    
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        # Pestaña 1: Proyecciones
        df_projections.to_excel(writer, sheet_name="Proyecciones", index=False)
        
        # Pestaña 2: Torneo de Modelos
        df_tournament.to_excel(writer, sheet_name="Torneo_Modelos", index=False)
        
        # Pestaña 3: Simulación Monte Carlo (opcional)
        if df_monte_carlo is not None and not df_monte_carlo.empty:
            df_monte_carlo.to_excel(writer, sheet_name="Monte_Carlo_Percentiles", index=False)
        
        # Pestaña 4: Datos Históricos
        df_hist_export = df_hist.copy()
        if "Date" in df_hist_export.columns:
            df_hist_export["Date"] = pd.to_datetime(df_hist_export["Date"]).dt.strftime("%Y-%m-%d")
        df_hist_export.to_excel(writer, sheet_name="Datos_Historicos", index=False)
    
    return buffer.getvalue()

def generate_csv_report(df: pd.DataFrame) -> str:
    """Genera el contenido CSV en formato texto con codificación utf-8."""
    return df.to_csv(index=False)
