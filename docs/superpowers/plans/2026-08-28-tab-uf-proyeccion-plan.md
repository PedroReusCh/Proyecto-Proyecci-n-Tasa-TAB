# Plan de Implementación: Sistema de Proyección Financiera de Tasas TAB UF (CBF)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Desarrollar una aplicación web interactiva y modular en Python (Streamlit + Plotly + Statsmodels + Scikit-Learn) que descargue automáticamente los datos históricos de tasas de fijación (*fixing*) desde CBF, procese las tasas TAB UF (90, 180 y 360 días) y proyecte sus valores a 1, 3, 6 y 12 meses mediante un torneo competitivo de modelos cuantitativos, simulación Monte Carlo, análisis de estructura temporal y exportación a Excel.

**Architecture:** Arquitectura desacoplada por capas: cliente de ingesta resiliente de CBF (`cbf_client.py`), motor de modelado cuantitativo multimodelo con 6 algoritmos (`forecasting_engine.py`), evaluador de torneo con *Walk-Forward Cross-Validation* (`evaluator.py`), simulador estocástico Monte Carlo (`monte_carlo.py`), modelador de curva de tasas (`term_structure.py`), componentes gráficos interactivos (`ui_components.py`) y panel de control Streamlit (`app.py`).

**Tech Stack:** Python 3.11+, Streamlit, Plotly, Pandas, NumPy, Statsmodels, Scikit-Learn, Scipy, Requests, OpenpyXL, Pytest, uv.

**Spec:** [docs/superpowers/specs/2026-08-28-tab-uf-proyeccion-design.md](file:///C:/Users/preusc/Documents/Proyecto%20Proyecci%C3%B3n%20Tasa%20TAB/docs/superpowers/specs/2026-08-28-tab-uf-proyeccion-design.md)

## Global Constraints
- Todas las comunicaciones, docstrings, comentarios y commits de Git deben ser exclusivamente en **español**.
- Los datos de entrada provienen de la hoja `FixingRates` del archivo Excel de CBF: Fila 4 encabezados, datos desde fila 6, Columna C (Fecha), Columna H (TAB UF 90d), Columna I (TAB UF 180d) y Columna J (TAB UF 360d).
- Tipado estricto en Python (`typing.NamedTuple`, `typing.Dict`, `typing.List`, `typing.Tuple`, `pd.DataFrame`).
- Cobertura de pruebas con `pytest` para cada módulo antes de dar por finalizada una tarea.
- Horizontes de proyección definidos: 21 días (1 mes), 63 días (3 meses), 126 días (6 meses), 252 días (1 año).

---

### Task 1: Configuración de Dependencias y Entorno Base

**Files:**
- Create: `pyproject.toml`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/test_environment.py`

**Interfaces:**
- Produces: Entorno virtual y dependencias sincronizadas (`pandas`, `numpy`, `streamlit`, `plotly`, `statsmodels`, `scikit-learn`, `scipy`, `requests`, `openpyxl`, `pytest`).

- [ ] **Step 1: Crear archivo `pyproject.toml` con configuración de dependencias**
```toml
[project]
name = "tab-uf-forecaster"
version = "0.1.0"
description = "Sistema de Proyección Financiera de Tasas TAB UF con CBF"
authors = [{ name = "Quant Team" }]
requires-python = ">=3.11"
dependencies = [
    "streamlit>=1.35.0",
    "pandas>=2.2.0",
    "numpy>=1.26.0",
    "openpyxl>=3.1.2",
    "requests>=2.31.0",
    "plotly>=5.22.0",
    "statsmodels>=0.14.2",
    "scikit-learn>=1.4.0",
    "scipy>=1.13.0",
    "pytest>=8.2.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

- [ ] **Step 2: Crear archivos `__init__.py` en `src/` y `tests/`**
Crear `src/__init__.py` y `tests/__init__.py`.

- [ ] **Step 3: Escribir prueba unitaria de verificación del entorno en `tests/test_environment.py`**
```python
import importlib
import pytest

REQUIRED_MODULES = [
    "streamlit",
    "pandas",
    "numpy",
    "openpyxl",
    "requests",
    "plotly",
    "statsmodels",
    "sklearn",
    "scipy",
]

@pytest.mark.parametrize("module_name", REQUIRED_MODULES)
def test_required_modules_importable(module_name: str) -> None:
    mod = importlib.import_module(module_name)
    assert mod is not None
```

- [ ] **Step 4: Ejecutar la prueba con uv para verificar el entorno**
Run: `uv run pytest tests/test_environment.py -v`
Expected: PASS para todos los módulos.

- [ ] **Step 5: Commit**
```bash
git add pyproject.toml src/__init__.py tests/__init__.py tests/test_environment.py
git commit -m "build: configuracion de dependencias y pruebas de entorno base"
```

---

### Task 2: Cliente de Ingesta y Procesamiento de Datos CBF (`src/cbf_client.py`)

**Files:**
- Create: `src/cbf_client.py`
- Test: `tests/test_cbf_client.py`

**Interfaces:**
- Produces:
  - `CBFDataResult`: Dataclass con `df: pd.DataFrame`, `last_date: pd.Timestamp`, `records_count: int`, `latest_rates: Dict[str, float]`.
  - `fetch_cbf_historical_excel() -> bytes`
  - `parse_fixing_rates_excel(content: bytes) -> pd.DataFrame`
  - `get_clean_tab_uf_dataset() -> CBFDataResult`

- [ ] **Step 1: Escribir pruebas unitarias en `tests/test_cbf_client.py`**
```python
import pandas as pd
import io
from src.cbf_client import parse_fixing_rates_excel, CBFDataResult, clean_tab_uf_dataframe

def test_parse_fixing_rates_excel_structure():
    # Mock de estructura Excel de CBF
    data = [
        [None] * 15,
        [None] * 15,
        [None, None, "Date", None, None, None, None, "TAB UF\n(percentage, annual basis)", None, None] + [None]*5,
        [None, None, "Tenor", "30 Days", "90 Days", "180 Days", "360 Days", "90 Days", "180 Days", "360 Days"] + [None]*5,
        [None] * 15,
        [None, None, "2024-01-02", 0.5, 1.0, 1.5, 2.0, 2.75, 2.90, 3.10] + [None]*5,
        [None, None, "2024-01-03", 0.5, 1.0, 1.5, 2.0, 2.78, 2.92, 3.12] + [None]*5,
    ]
    df_raw = pd.DataFrame(data)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_raw.to_excel(writer, sheet_name="FixingRates", index=False, header=False)
    
    parsed_df = parse_fixing_rates_excel(buf.getvalue())
    assert not parsed_df.empty
    assert list(parsed_df.columns) == ["Date", "TAB_UF_90", "TAB_UF_180", "TAB_UF_360"]
    assert len(parsed_df) == 2
    assert parsed_df.iloc[0]["TAB_UF_90"] == 2.75
    assert parsed_df.iloc[0]["TAB_UF_180"] == 2.90
    assert parsed_df.iloc[0]["TAB_UF_360"] == 3.10
```

- [ ] **Step 2: Ejecutar prueba para verificar que falla**
Run: `uv run pytest tests/test_cbf_client.py -v`
Expected: FAIL con ImportError/ModuleNotFoundError.

- [ ] **Step 3: Implementar `src/cbf_client.py` con tipado estricto y manejo de errores**
```python
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
    df: pd.DataFrame
    last_date: pd.Timestamp
    records_count: int
    latest_rates: Dict[str, float]

def get_latest_excel_download_url(page_url: str = CBF_HISTORICAL_PAGE_URL) -> str:
    """Busca dinámicamente el enlace actual del archivo Excel en la página de CBF."""
    try:
        resp = requests.get(page_url, headers={"User-Agent": USER_AGENT}, timeout=10)
        if resp.status_code == 200:
            match = re.search(r'href=["\'](https://cbf\.cl/wp-content/uploads/[^"\']+TAB-TADO-Historical-Data[^"\']*\.xlsx)["\']', resp.text, re.IGNORECASE)
            if match:
                return match.group(1)
    except Exception:
        pass
    return CBF_FALLBACK_EXCEL_URL

def fetch_cbf_historical_excel(url: Optional[str] = None) -> bytes:
    """Descarga los bytes del archivo Excel de CBF."""
    target_url = url or get_latest_excel_download_url()
    resp = requests.get(target_url, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    return resp.content

def parse_fixing_rates_excel(content: bytes) -> pd.DataFrame:
    """
    Parsea la hoja FixingRates del archivo Excel descargado.
    Extrae Date (Col C / idx 2), TAB UF 90d (Col H / idx 7), TAB UF 180d (Col I / idx 8), TAB UF 360d (Col J / idx 9).
    """
    wb = pd.ExcelFile(io.BytesIO(content), engine="openpyxl")
    if "FixingRates" not in wb.sheet_names:
        raise ValueError("La hoja 'FixingRates' no se encuentra en el archivo Excel proporcionado.")
    
    df_raw = pd.read_excel(wb, sheet_name="FixingRates", header=None)
    
    # Filas con datos a partir del índice 5 (fila 6 de Excel)
    data_slice = df_raw.iloc[5:, [2, 7, 8, 9]].copy()
    data_slice.columns = ["Date", "TAB_UF_90", "TAB_UF_180", "TAB_UF_360"]
    
    return clean_tab_uf_dataframe(data_slice)

def clean_tab_uf_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia y valida el DataFrame de tasas TAB UF."""
    clean_df = df.copy()
    clean_df["Date"] = pd.to_datetime(clean_df["Date"], errors="coerce")
    clean_df = clean_df.dropna(subset=["Date"])
    
    for col in ["TAB_UF_90", "TAB_UF_180", "TAB_UF_360"]:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")
    
    clean_df = clean_df.dropna(subset=["TAB_UF_90", "TAB_UF_180", "TAB_UF_360"])
    clean_df = clean_df.sort_values(by="Date").reset_index(drop=True)
    return clean_df

def get_clean_tab_uf_dataset(excel_bytes: Optional[bytes] = None) -> CBFDataResult:
    """Obtiene el dataset estructurado y listo para modelado."""
    content = excel_bytes if excel_bytes is not None else fetch_cbf_historical_excel()
    df = parse_fixing_rates_excel(content)
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
```

- [ ] **Step 4: Ejecutar pruebas unitarias de `cbf_client`**
Run: `uv run pytest tests/test_cbf_client.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add src/cbf_client.py tests/test_cbf_client.py
git commit -m "feat: cliente de ingesta y parseo de fixing rates cbf con tipado estricto"
```

---

### Task 3: Motor Cuantitativo y Modelos de Proyección (`src/forecasting_engine.py`)

**Files:**
- Create: `src/forecasting_engine.py`
- Test: `tests/test_forecasting.py`

**Interfaces:**
- Produces:
  - `ForecastResult`: Dataclass con `model_name: str`, `horizon_days: int`, `future_dates: List[pd.Timestamp]`, `point_forecast: np.ndarray`, `lower_80: np.ndarray`, `upper_80: np.ndarray`, `lower_95: np.ndarray`, `upper_95: np.ndarray`.
  - Modelos: `AutoARIMAForecaster`, `HoltWintersForecaster`, `VasicekForecaster`, `VARMultivariateForecaster`, `MLLagsForecaster`, `EnsembleForecaster`.
  - `generate_all_forecasts(series_df: pd.DataFrame, target_col: str, horizon_days: int) -> Dict[str, ForecastResult]`

- [ ] **Step 1: Escribir pruebas unitarias en `tests/test_forecasting.py`**
```python
import numpy as np
import pandas as pd
from src.forecasting_engine import (
    AutoARIMAForecaster,
    HoltWintersForecaster,
    VasicekForecaster,
    VARMultivariateForecaster,
    MLLagsForecaster,
    EnsembleForecaster,
    generate_all_forecasts,
)

def test_models_generate_valid_forecasts():
    dates = pd.date_range("2023-01-01", periods=200, freq="B")
    np.random.seed(42)
    s90 = 3.0 + np.cumsum(np.random.normal(0, 0.05, 200))
    s180 = s90 + 0.3 + np.random.normal(0, 0.02, 200)
    s360 = s180 + 0.4 + np.random.normal(0, 0.02, 200)
    df = pd.DataFrame({"Date": dates, "TAB_UF_90": s90, "TAB_UF_180": s180, "TAB_UF_360": s360})
    
    results = generate_all_forecasts(df, "TAB_UF_90", horizon_days=21)
    assert "AutoARIMA" in results
    assert "Holt-Winters" in results
    assert "Vasicek" in results
    assert "VAR" in results
    assert "ML-Lags" in results
    assert "Ensemble" in results
    
    for name, res in results.items():
        assert len(res.point_forecast) == 21
        assert not np.isnan(res.point_forecast).any()
        assert (res.upper_95 >= res.lower_95).all()
```

- [ ] **Step 2: Ejecutar prueba para verificar que falla**
Run: `uv run pytest tests/test_forecasting.py -v`
Expected: FAIL.

- [ ] **Step 3: Implementar `src/forecasting_engine.py`**
Implementar los 6 modelos cuantitativos:
1. `AutoARIMAForecaster`: Implementado con ARIMA/SARIMAX de `statsmodels` con grid search optimizado por AIC para órdenes $(p \in [0, 3], d \in [0, 1], q \in [0, 3])$.
2. `HoltWintersForecaster`: ExponentialSmoothing con tendencia aditiva amortiguada (`damped_trend=True`).
3. `VasicekForecaster`: Calibración OLS $r_t - r_{t-1} = \alpha + \beta r_{t-1} + \epsilon_t \Rightarrow \kappa = -\ln(1+\beta)/\Delta t, \theta = \alpha/(1-e^{-\kappa \Delta t}), \sigma = \text{std}(\epsilon)/\sqrt{\Delta t}$. Trayectoria analítica esperada $E[r_T | r_0] = r_0 e^{-\kappa T} + \theta(1 - e^{-\kappa T})$ y varianza analítica $\text{Var}(r_T | r_0) = \frac{\sigma^2}{2\kappa}(1 - e^{-2\kappa T})$.
4. `VARMultivariateForecaster`: Vector Autoregression de `statsmodels` estimando conjuntamente `TAB_UF_90`, `TAB_UF_180` y `TAB_UF_360`.
5. `MLLagsForecaster`: Regresión autorregresiva Ridge/BayesianRidge con características de retardos (lags 1, 2, 5, 21), media móvil de 5 y 21 días, y proyección recursiva autorregresiva.
6. `EnsembleForecaster`: Combinación ponderada de los pronósticos.

- [ ] **Step 4: Ejecutar pruebas unitarias de `forecasting_engine`**
Run: `uv run pytest tests/test_forecasting.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add src/forecasting_engine.py tests/test_forecasting.py
git commit -m "feat: motor cuantitativo con modelos ARIMA, Holt-Winters, Vasicek, VAR, ML y Ensemble"
```

---

### Task 4: Evaluador de Torneo y Validación Rodante (`src/evaluator.py`)

**Files:**
- Create: `src/evaluator.py`
- Test: `tests/test_evaluator.py`

**Interfaces:**
- Produces:
  - `ModelScore`: Dataclass con `model_name: str`, `rmse: float`, `mae: float`, `mape: float`, `directional_accuracy: float`, `rank: int`.
  - `TournamentResult`: Dataclass con `target_col: str`, `horizon_days: int`, `scores: List[ModelScore]`, `winner_model: str`, `weights: Dict[str, float]`.
  - `evaluate_tournament(series_df: pd.DataFrame, target_col: str, horizon_days: int, n_splits: int = 3) -> TournamentResult`

- [ ] **Step 1: Escribir pruebas unitarias en `tests/test_evaluator.py`**
```python
import numpy as np
import pandas as pd
from src.evaluator import evaluate_tournament, calculate_metrics

def test_calculate_metrics():
    actual = np.array([3.0, 3.1, 3.2, 3.3])
    predicted = np.array([3.05, 3.15, 3.18, 3.32])
    metrics = calculate_metrics(actual, predicted)
    assert metrics["rmse"] > 0
    assert metrics["mae"] > 0
    assert 0 <= metrics["directional_accuracy"] <= 100

def test_tournament_identifies_winner():
    dates = pd.date_range("2023-01-01", periods=250, freq="B")
    np.random.seed(42)
    s90 = 3.0 + np.cumsum(np.random.normal(0, 0.03, 250))
    s180 = s90 + 0.3
    s360 = s180 + 0.4
    df = pd.DataFrame({"Date": dates, "TAB_UF_90": s90, "TAB_UF_180": s180, "TAB_UF_360": s360})
    
    tourn = evaluate_tournament(df, "TAB_UF_90", horizon_days=21)
    assert tourn.winner_model is not None
    assert len(tourn.scores) >= 5
    assert tourn.scores[0].rank == 1
    assert tourn.scores[0].rmse <= tourn.scores[-1].rmse
```

- [ ] **Step 2: Ejecutar prueba para verificar que falla**
Run: `uv run pytest tests/test_evaluator.py -v`
Expected: FAIL.

- [ ] **Step 3: Implementar `src/evaluator.py`**
Implementar validación rodante *Walk-Forward Cross-Validation*, cálculo de métricas de precisión (RMSE, MAE, MAPE, Precisión Direccional) y ranking de modelos con selección automática del ganador.

- [ ] **Step 4: Ejecutar pruebas unitarias de `evaluator`**
Run: `uv run pytest tests/test_evaluator.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add src/evaluator.py tests/test_evaluator.py
git commit -m "feat: evaluador de torneo con backtesting rodante y seleccion automatica del ganador"
```

---

### Task 5: Simulación Monte Carlo y Estructura Temporal (`src/monte_carlo.py` & `src/term_structure.py`)

**Files:**
- Create: `src/monte_carlo.py`
- Create: `src/term_structure.py`
- Test: `tests/test_monte_carlo.py`
- Test: `tests/test_term_structure.py`

**Interfaces:**
- Produces:
  - `MonteCarloResult`: Dataclass con `simulated_paths: np.ndarray`, `percentiles: Dict[str, np.ndarray]`, `future_dates: List[pd.Timestamp]`.
  - `run_monte_carlo_simulation(initial_rate: float, kappa: float, theta: float, sigma: float, horizon_days: int, n_sims: int = 1000) -> MonteCarloResult`
  - `TermStructureCurve`: Dataclass con `tenors: List[int]`, `rates: List[float]`, `slope_360_90: float`, `butterfly_180: float`.
  - `compute_term_structure(df: pd.DataFrame, row_idx: int = -1) -> TermStructureCurve`

- [ ] **Step 1: Escribir pruebas unitarias en `tests/test_monte_carlo.py` y `tests/test_term_structure.py`**
```python
import numpy as np
import pandas as pd
from src.monte_carlo import run_monte_carlo_simulation
from src.term_structure import compute_term_structure

def test_monte_carlo_simulation_shape():
    res = run_monte_carlo_simulation(
        initial_rate=3.5,
        kappa=0.5,
        theta=3.0,
        sigma=0.08,
        horizon_days=63,
        n_sims=500
    )
    assert res.simulated_paths.shape == (500, 63)
    assert "P10" in res.percentiles
    assert "P50" in res.percentiles
    assert "P90" in res.percentiles
    assert (res.percentiles["P90"] >= res.percentiles["P10"]).all()

def test_term_structure_computation():
    df = pd.DataFrame({
        "Date": [pd.Timestamp("2024-01-02")],
        "TAB_UF_90": [2.5],
        "TAB_UF_180": [2.8],
        "TAB_UF_360": [3.2]
    })
    curve = compute_term_structure(df)
    assert curve.tenors == [90, 180, 360]
    assert curve.slope_360_90 == pytest.approx(0.7)
    assert curve.butterfly_180 == pytest.approx(2 * 2.8 - (2.5 + 3.2))
```

- [ ] **Step 2: Ejecutar pruebas para verificar que fallan**
Run: `uv run pytest tests/test_monte_carlo.py tests/test_term_structure.py -v`
Expected: FAIL.

- [ ] **Step 3: Implementar `src/monte_carlo.py` y `src/term_structure.py`**
- [ ] **Step 4: Ejecutar pruebas para verificar que pasan**
Run: `uv run pytest tests/test_monte_carlo.py tests/test_term_structure.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add src/monte_carlo.py src/term_structure.py tests/test_monte_carlo.py tests/test_term_structure.py
git commit -m "feat: modulo de simulacion monte carlo y estructura temporal de tasas"
```

---

### Task 6: Interfaz Streamlit Interactiva, Componentes Plotly y Exportador de Reportes

**Files:**
- Create: `src/ui_components.py`
- Create: `src/exporter.py`
- Create: `app.py`
- Test: `tests/test_exporter.py`

**Interfaces:**
- Produces:
  - `create_forecast_figure(...) -> go.Figure`: Gráfico histórico + cono de proyección + intervalos de confianza.
  - `create_tournament_comparison_figure(...) -> go.Figure`: Gráfico superpuesto de todos los modelos compitiendo.
  - `create_monte_carlo_fan_figure(...) -> go.Figure`: Gráfico de abanico de percentiles y trayectorias.
  - `create_term_structure_figure(...) -> go.Figure`: Curva de rendimiento actual vs proyectadas.
  - `generate_excel_report(...) -> bytes`: Generación de libro Excel formateado con hojas `Proyecciones`, `Torneo_Modelos`, `Datos_Historicos`.
  - `app.py`: Aplicación Streamlit completa con 5 pestañas interactivas.

- [ ] **Step 1: Escribir pruebas unitarias para el exportador en `tests/test_exporter.py`**
```python
import pandas as pd
import io
from src.exporter import generate_excel_report

def test_generate_excel_report():
    df_hist = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=10),
        "TAB_UF_90": [2.5] * 10,
        "TAB_UF_180": [2.8] * 10,
        "TAB_UF_360": [3.2] * 10,
    })
    df_proj = pd.DataFrame({
        "Horizon": ["1 Mes", "3 Meses", "6 Meses", "1 Año"],
        "Tasa_Proyectada": [2.6, 2.7, 2.9, 3.1]
    })
    df_tourn = pd.DataFrame({
        "Modelo": ["AutoARIMA", "Vasicek"],
        "RMSE": [0.05, 0.08]
    })
    
    excel_bytes = generate_excel_report(df_hist, df_proj, df_tourn)
    assert len(excel_bytes) > 0
    wb = pd.ExcelFile(io.BytesIO(excel_bytes), engine="openpyxl")
    assert "Proyecciones" in wb.sheet_names
    assert "Torneo_Modelos" in wb.sheet_names
    assert "Datos_Historicos" in wb.sheet_names
```

- [ ] **Step 2: Implementar `src/ui_components.py`, `src/exporter.py` y `app.py`**
- [ ] **Step 3: Ejecutar todas las pruebas con pytest**
Run: `uv run pytest -v`
Expected: PASS en toda la suite.

- [ ] **Step 4: Verificar la ejecución de la aplicación Streamlit**
Run: `uv run streamlit run app.py --server.headless true` (o verificación sintáctica / ejecución controlada de prueba).

- [ ] **Step 5: Actualizar CHANGELOG.md y Commit**
```bash
git add src/ui_components.py src/exporter.py app.py tests/test_exporter.py CHANGELOG.md
git commit -m "feat: interfaz interactiva streamlit con plotly, kpis y exportador de reportes"
```
