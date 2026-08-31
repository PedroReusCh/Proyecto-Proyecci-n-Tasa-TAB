# Sistema de Proyección Financiera de Tasas TAB UF (Chilean Financial Benchmarks)

Plataforma analítica e interactiva para la ingesta automatizada, modelado cuantitativo, torneo competitivo de predicción, diagnóstico econométrico y pruebas de estrés para tasas **TAB UF** (90, 180 y 360 días) en horizontes de **1 mes, 3 meses, 6 meses y 1 año**.

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

3. **Diagnóstico Econométrico y Verificación de Residuos:**
   * **Test de Dickey-Fuller Aumentado (ADF):** Evaluación formal de raíz unitaria y estacionariedad en nivel ($r_t$) y en primera diferencia ($\Delta r_t$).
   * **Correlogramas ACF y PACF:** Gráficos interactivos con bandas de significancia de Bartlett al 95%.
   * **Test de Ljung-Box:** Verificación empírica de ruido blanco sobre los residuos del modelo ganador.

4. **Simulador de Escenarios de Estrés (*Stress Testing*):**
   * Simulación interactiva de shocks de Política Monetaria / liquidez en puntos base ($\pm 25$ a $\pm 200$ pb) con transmisión gradual.
   * Multiplicador de volatilidad de mercado y cuadro de impacto en tasas terminales por hito (1m, 3m, 6m, 12m).

5. **Simulación Estocástica de Monte Carlo:**
   * 1.000 trayectorias proyectadas con conos de percentiles P10 (bajista), P50 (mediana/base) y P90 (alcista).
   * Histograma de distribución de probabilidad terminal e intervalos al 80% y 95%.

6. **Estructura Temporal y Spreads de Plazo:**
   * Visualización de la curva de rendimiento actual vs. proyectada a +1m, +3m, +6m y +12m.
   * Monitoreo de pendiente (*Slope*: 360d - 90d) y curvatura (*Butterfly*: $2 \times 180d - 90d - 360d$).

7. **Exportación de Reportes Financieros:**
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

La aplicación se abrirá automáticamente en su navegador web en `http://localhost:8501`.

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
├── app.py                       # Aplicación Web Streamlit (7 pestañas)
├── src/
│   ├── __init__.py
│   ├── cbf_client.py            # Ingesta y parseo del Excel de CBF
│   ├── forecasting_engine.py    # Modelos ARIMA, ETS, Vasicek, VAR, ML y Ensemble
│   ├── evaluator.py             # Torneo y validación rodante
│   ├── diagnostics.py           # Test ADF, Ljung-Box y ACF/PACF
│   ├── stress_testing.py        # Simulación de shocks de tasa y volatilidad
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
    ├── test_diagnostics.py      # Pruebas de test ADF, Ljung-Box y ACF/PACF
    ├── test_stress_testing.py   # Pruebas de simulación de escenarios de estrés
    ├── test_monte_carlo.py      # Pruebas de Monte Carlo
    ├── test_term_structure.py   # Pruebas de estructura temporal
    └── test_exporter.py         # Pruebas de exportación a Excel y CSV
```
