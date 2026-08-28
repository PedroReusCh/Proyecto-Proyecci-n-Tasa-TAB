# Registro de Cambios (CHANGELOG)

Todos los cambios notables en este proyecto serán documentados en este archivo.
El formato se basa en [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/) y este proyecto se adhiere a [Semantic Versioning](https://semver.org/lang/es/).

---

## [1.0.0] - 2026-08-28

### Añadido
* **Cliente de Ingesta CBF (`src/cbf_client.py`):** Descarga dinámica y automática del archivo Excel oficial de fijaciones desde la web de CBF, con parsing de la hoja `FixingRates` (Fecha y tenores TAB UF a 90, 180 y 360 días).
* **Motor Cuantitativo Multimodelo (`src/forecasting_engine.py`):** Implementación de 6 modelos de proyección:
  * AutoARIMA / SARIMAX optimizado por AIC.
  * Suavizado Exponencial de Holt-Winters con tendencia amortiguada (*Damped Trend*).
  * Modelo Financiero Estocástico de Vasicek (Ornstein-Uhlenbeck) con reversión a la media.
  * VAR Multivariado (*Vector Autoregression*) proyectando conjuntamente 90d, 180d y 360d.
  * Regresión Machine Learning con características autorregresivas (Lags 1 a 21 y medias móviles).
  * Ensemble Ponderado Óptimo basado en el inverso del error cuadrático medio ($1/\text{RMSE}^2$).
* **Torneo y Backtesting Rodante (`src/evaluator.py`):** *Walk-Forward Cross-Validation* con métricas de error RMSE, MAE, MAPE y Precisión Direccional (%) para clasificación y selección automática del modelo ganador.
* **Simulación Monte Carlo (`src/monte_carlo.py`):** 1.000 trayectorias estocásticas con percentiles (P10, P25, P50, P75, P90) y distribución de densidad terminal.
* **Estructura Temporal de Tasas (`src/term_structure.py`):** Curva de rendimiento actual vs proyectadas y spreads de pendiente (*slope*) y curvatura (*butterfly*).
* **Componentes Gráficos Plotly (`src/ui_components.py`):** Gráficos interactivos con intervalos de confianza al 80% y 95%, abanico Monte Carlo y comparación de modelos.
* **Exportador de Reportes (`src/exporter.py`):** Generación de libros Excel multipestaña (`.xlsx`) y archivos CSV.
* **Aplicación Web Interactiva (`app.py`):** Dashboard Streamlit completo con panel lateral de filtros y 5 pestañas de análisis.
* **Suite de Pruebas Automatizadas (`tests/`):** 27 pruebas unitarias con cobertura total y verificación de entorno.
