"""
Módulo de cliente e ingesta de datos oficiales de tasas de fijación (Fixing Rates) desde CBF.
"""
from dataclasses import dataclass
from typing import Dict, Optional
import io
import re
import requests
import pandas as pd

CBF_HISTORICAL_PAGE_URL = "https://cbf.cl/chilean-benchmarks/historical-rates/?lang=es"
CBF_FALLBACK_EXCEL_URL = "https://cbf.cl/wp-content/uploads/2026/07/TAB-TADO-Historical-Data-since-17-05-2010-to-30-06-2026.xlsx"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"

@dataclass(frozen=True)
class CBFDataResult:
    """Contenedor inmutable de los datos procesados de CBF."""
    df: pd.DataFrame
    last_date: pd.Timestamp
    records_count: int
    latest_rates: Dict[str, float]

def get_latest_excel_download_url(page_url: str = CBF_HISTORICAL_PAGE_URL) -> str:
    """Busca dinámicamente el enlace actual del archivo Excel en la página web de CBF."""
    try:
        resp = requests.get(page_url, headers={"User-Agent": USER_AGENT}, timeout=10)
        if resp.status_code == 200:
            match = re.search(
                r'href=["\'](https://cbf\.cl/wp-content/uploads/[^"\']+TAB-TADO-Historical-Data[^"\']*\.xlsx)["\']',
                resp.text,
                re.IGNORECASE,
            )
            if match:
                return match.group(1)
    except Exception:
        pass
    return CBF_FALLBACK_EXCEL_URL

def fetch_cbf_historical_excel(url: Optional[str] = None) -> bytes:
    """
    Descarga los bytes del archivo Excel de CBF.
    Si la conexión a la web de CBF falla o devuelve HTML/bloqueos,
    utiliza como respaldo seguro el archivo local empaquetado en data/latest_cbf_rates.xlsx.
    """
    import os
    import pathlib
    
    target_url = url or get_latest_excel_download_url()
    try:
        resp = requests.get(target_url, headers={"User-Agent": USER_AGENT}, timeout=15)
        resp.raise_for_status()
        if not resp.content.startswith(b"PK") or len(resp.content) < 5000:
            raise ValueError("La respuesta recibida no es un archivo Excel válido (no empieza con firma PK).")
        return resp.content
    except Exception:
        local_fallback = pathlib.Path(__file__).parent.parent / "data" / "latest_cbf_rates.xlsx"
        if local_fallback.exists():
            with open(local_fallback, "rb") as f:
                return f.read()
        raise RuntimeError("No se pudo obtener el archivo de CBF y no se encontró el respaldo local.")

def parse_fixing_rates_excel(content: bytes) -> pd.DataFrame:
    """
    Parsea la hoja FixingRates del archivo Excel descargado.
    Extrae:
      - Columna C (índice 2): Fecha
      - Columna H (índice 7): TAB UF 90 Días
      - Columna I (índice 8): TAB UF 180 Días
      - Columna J (índice 9): TAB UF 360 Días
    """
    wb = pd.ExcelFile(io.BytesIO(content), engine="openpyxl")
    if "FixingRates" not in wb.sheet_names:
        raise ValueError("La hoja 'FixingRates' no se encuentra en el archivo Excel proporcionado.")
    
    df_raw = pd.read_excel(wb, sheet_name="FixingRates", header=None)
    
    # Filas con datos a partir del índice 5 (fila 6 de Excel)
    if len(df_raw) <= 5:
        raise ValueError("El archivo Excel no contiene filas de datos suficientes.")
    
    data_slice = df_raw.iloc[5:, [2, 7, 8, 9]].copy()
    data_slice.columns = ["Date", "TAB_UF_90", "TAB_UF_180", "TAB_UF_360"]
    
    return clean_tab_uf_dataframe(data_slice)

def clean_tab_uf_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia tipos, elimina nulos y ordena cronológicamente el DataFrame."""
    clean_df = df.copy()
    clean_df["Date"] = pd.to_datetime(clean_df["Date"], errors="coerce")
    clean_df = clean_df.dropna(subset=["Date"])
    
    for col in ["TAB_UF_90", "TAB_UF_180", "TAB_UF_360"]:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")
    
    clean_df = clean_df.dropna(subset=["TAB_UF_90", "TAB_UF_180", "TAB_UF_360"])
    clean_df = clean_df.sort_values(by="Date").reset_index(drop=True)
    return clean_df

def get_clean_tab_uf_dataset(excel_bytes: Optional[bytes] = None) -> CBFDataResult:
    """Obtiene el dataset estructurado y validado listo para modelado."""
    import pathlib
    
    try:
        content = excel_bytes if excel_bytes is not None else fetch_cbf_historical_excel()
        df = parse_fixing_rates_excel(content)
    except Exception as e:
        local_fallback = pathlib.Path(__file__).parent.parent / "data" / "latest_cbf_rates.xlsx"
        if local_fallback.exists():
            with open(local_fallback, "rb") as f:
                content = f.read()
            df = parse_fixing_rates_excel(content)
        else:
            raise RuntimeError(f"Error procesando datos de CBF y fallo el respaldo local: {e}")
            
    if df.empty:
        raise ValueError("El dataset procesado no contiene registros válidos.")
    
    last_row = df.iloc[-1]
    last_date = pd.Timestamp(last_row["Date"])
    latest_rates = {
        "TAB_UF_90": float(last_row["TAB_UF_90"]),
        "TAB_UF_180": float(last_row["TAB_UF_180"]),
        "TAB_UF_360": float(last_row["TAB_UF_360"]),
    }
    return CBFDataResult(
        df=df,
        last_date=last_date,
        records_count=len(df),
        latest_rates=latest_rates,
    )
