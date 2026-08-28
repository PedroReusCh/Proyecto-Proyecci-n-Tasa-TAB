import io
import pandas as pd
import pytest
from src.exporter import generate_excel_report, generate_csv_report

def test_generate_excel_report():
    df_hist = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=5),
        "TAB_UF_90": [2.5] * 5,
        "TAB_UF_180": [2.8] * 5,
        "TAB_UF_360": [3.2] * 5,
    })
    df_proj = pd.DataFrame({
        "Horizonte": ["1 Mes", "3 Meses", "6 Meses", "1 Año"],
        "Tasa_Proyectada": [2.6, 2.7, 2.9, 3.1],
        "Limite_Inferior_95": [2.3, 2.4, 2.5, 2.6],
        "Limite_Superior_95": [2.9, 3.0, 3.3, 3.6],
    })
    df_tourn = pd.DataFrame({
        "Ranking": [1, 2],
        "Modelo": ["AutoARIMA", "Vasicek"],
        "RMSE": [0.05, 0.08],
        "MAE": [0.04, 0.06],
    })
    
    excel_bytes = generate_excel_report(df_hist, df_proj, df_tourn)
    assert len(excel_bytes) > 0
    
    wb = pd.ExcelFile(io.BytesIO(excel_bytes), engine="openpyxl")
    assert "Proyecciones" in wb.sheet_names
    assert "Torneo_Modelos" in wb.sheet_names
    assert "Datos_Historicos" in wb.sheet_names

def test_generate_csv_report():
    df_proj = pd.DataFrame({
        "Horizonte": ["1 Mes", "3 Meses"],
        "Tasa_Proyectada": [2.6, 2.7]
    })
    csv_str = generate_csv_report(df_proj)
    assert "Horizonte" in csv_str
    assert "Tasa_Proyectada" in csv_str
