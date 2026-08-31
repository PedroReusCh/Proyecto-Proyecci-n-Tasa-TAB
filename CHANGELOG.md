# Registro de Cambios (CHANGELOG)

Todos los cambios notables en este proyecto serán documentados en este archivo.
El formato se basa en [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/) y este proyecto se adhiere a [Semantic Versioning](https://semver.org/lang/es/).

---

## [1.2.0] - 2026-08-31

### Añadido

* **Despliegue Público en la Nube:** Despliegue en producción gratuito sobre **Streamlit Community Cloud** con URL pública: [`https://proyeccion-tab-uf.streamlit.app`](https://proyeccion-tab-uf.streamlit.app).
* **Mecanismo de Respaldo Resiliente (`data/latest_cbf_rates.xlsx`):** Inclusión de una copia de respaldo local de fijaciones oficiales para garantizar alta disponibilidad en entornos cloud con restricciones de cortafuegos o latencia.
* **Configuración Headless (`.streamlit/config.toml`):** Configuración del servidor para ejecución desatendida en la nube con `headless = true` y desactivación de telemetría.
* **Archivo de Dependencias Cloud (`requirements.txt`):** Generación de archivo de dependencias estándar para el gestor de paquetes de Streamlit Cloud.
* **Actualización Integral de Documentación:** Incorporación en `README.md` de la guía completa de despliegue, arquitectura de datos y manual funcional.

---

## [1.1.0] - 2026-08-31

### Añadido

* **Módulo de Diagnóstico Econométrico (`src/diagnostics.py`):**
  * Test de Dickey-Fuller Aumentado (ADF) para evaluar estacionariedad en nivel ($r_t$) y en primera diferencia ($\Delta r_t$).
  * Cálculo de correlogramas de Autocorrelación (ACF) y Autocorrelación Parcial (PACF) con bandas de significancia de Bartlett al 95%.
  * Test de Ljung-Box para validar formalmente la hipótesis de ruido blanco sobre los residuos del modelo.
* **Módulo de Escenarios de Estrés / Stress Testing (`src/stress_testing.py`):**
  * Simulador de shocks de Política Monetaria y mercado en puntos base ($\pm \text{pbs}$) con curva de transmisión gradual.
  * Selector de multiplicadores de volatilidad para estresar los conos de dispersión.
  * Gráfico comparativo de trayectorias y cuadro de impacto en tasas terminales por hito (1m, 3m, 6m, 12m).
* **Componentes Gráficos Avanzados (`src/ui_components.py`):**
  * `create_acf_pacf_figure`: Subplots interactivos de autocorrelación.
  * `create_residuals_diagnostics_figure`: Serie de residuos e histograma de errores vs distribución normal.
  * `create_stress_testing_figure`: Visualización de escenarios Base vs Alcista vs Bajista con cono de estrés al 95%.
* **Ampliación de la Interfaz Streamlit (`app.py`):**
  * Pestaña 4: *🔬 Diagnóstico Econométrico*.
  * Pestaña 5: *⚡ Escenarios de Estrés (Stress Testing)*.
* **Ampliación de Pruebas Unitarias:** 31 pruebas unitarias cubriendo todas las nuevas funcionalidades.

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
* **Aplicación Web Interactiva (`app.py`):** Dashboard Streamlit completo con panel lateral de filtros y pestañas de análisis.
* **Suite de Pruebas Automatizadas (`tests/`):** 27 pruebas unitarias con cobertura total y verificación de entorno.
