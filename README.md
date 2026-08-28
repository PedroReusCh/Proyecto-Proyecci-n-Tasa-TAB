# Sistema de Proyección Financiera de Tasas TAB UF (Chilean Financial Benchmarks)

Plataforma analítica e interactiva para la ingesta automatizada, modelado cuantitativo, torneo competitivo de predicción y visualización de tasas **TAB UF** (90, 180 y 360 días) en horizontes de **1 mes, 3 meses, 6 meses y 1 año**.

---

## 🚀 Características Principales

1. **Ingesta Automatizada desde CBF:**
   * Descarga directa y parsing inteligente del archivo Excel oficial de fijaciones (*fixing*) desde la web oficial de CBF (`FixingRates`).
   * Extracción limpia y validada de fechas (Columna C) y tasas TAB UF 90 Días (Columna H), 180 Días (Columna I) y 360 Días (Columna J).

2. **Torneo Cuantitativo de Modelos (6 Algoritmos):**
   * **AutoARIMA / SARIMAX:** Modelado de componentes autorregresivas y medias móviles optimizadas por AIC.
   * **Holt-Winters (ETS):** Suavizado exponencial con tendencia amortiguada (*Damped Trend*).
   * **Vasicek (Ornstein-Uhlenbeck):** Proceso de difusión estocástica con reversión a la media calibrado por OLS.
   * **VAR Multivariado:** Proyección conjunta de 90d, 180d y 360d preservando correlación y estructura de curva.
   * **Machine Learning con Lags:** Regresión Ridge recursiva con rezagos temporales y medias móviles.
   * **Ensemble Óptimo Ponderado:** Combinación convexa ponderada por $1/\text{RMSE}^2$.

3. **Backtesting Rodante (*Walk-Forward Validation*):**
   * Evaluación de precisión con métricas RMSE, MAE, MAPE y Precisión Direccional (%).
   * Ranking en tiempo real y selección automática del modelo ganador.

4. **Simulación Estocástica de Monte Carlo:**
   * 1.000 trayectorias proyectadas con conos de percentiles P10 (bajista), P50 (mediana/base) y P90 (alcista).
   * Histograma de distribución de probabilidad terminal e intervalos al 80% y 95%.

5. **Estructura Temporal y Spreads de Plazo:**
   * Visualización de la curva de rendimiento actual vs. proyectada a +1m, +3m, +6m y +12m.
   * Monitoreo de pendiente (*Slope*: 360d - 90d) y curvatura (*Butterfly*: $2 \times 180d - 90d - 360d$).

6. **Exportación de Reportes Financieros:**
   * Descarga directa de libros Excel (`.xlsx`) multishoheet con datos históricos, proyecciones detalladas, torneo de modelos y percentiles Monte Carlo.
   * Descarga de archivos en formato CSV.

---

## 🛠️ Instalación y Uso

### 1. Requisitos Previos
* Gestor de paquetes `uv` instalado en el sistema.

### 2. Ejecución de la Aplicación Web
Para iniciar la aplicación interactiva de Streamlit, ejecute en la terminal:

```bash
uv run streamlit run app.py
```

La aplicación se abrirá automáticamente en su navegador web (por defecto en `http://localhost:8501`).

---

## 🧪 Ejecución de Pruebas Automatizadas

Para ejecutar la suite completa de pruebas unitarias y de integración:

```bash
uv run pytest -v
```

---

## 📁 Estructura del Código

```
Proyecto Proyección Tasa TAB/
├── GEMINI.md                    # Directivas y contratos del asistente
├── README.md                    # Documentación y manual de usuario
├── CHANGELOG.md                 # Registro histórico de versiones
├── pyproject.toml               # Configuración de dependencias y proyecto
├── app.py                       # Aplicación Web Streamlit
├── src/
│   ├── __init__.py
│   ├── cbf_client.py            # Ingesta y parseo del Excel de CBF
│   ├── forecasting_engine.py    # Modelos ARIMA, ETS, Vasicek, VAR, ML y Ensemble
│   ├── evaluator.py             # Torneo y validación rodante
│   ├── monte_carlo.py           # Simulación estocástica y percentiles
│   ├── term_structure.py        # Curvas de rendimiento y spreads
│   ├── ui_components.py         # Gráficos interactivos Plotly
│   └── exporter.py              # Exportador de reportes en Excel y CSV
└── tests/
    ├── __init__.py
    ├── test_environment.py      # Verificación de dependencias del entorno
    ├── test_cbf_client.py       # Pruebas del ingestor CBF
    ├── test_forecasting.py      # Pruebas de modelos predictivos
    ├── test_evaluator.py        # Pruebas de métricas y torneo
    ├── test_monte_carlo.py      # Pruebas de Monte Carlo
    ├── test_term_structure.py   # Pruebas de estructura temporal
    └── test_exporter.py         # Pruebas de exportación a Excel y CSV
```
