# Especificación de Diseño Técnico: Sistema de Proyección Financiera de Tasas TAB UF (CBF)

## 1. Visión General del Proyecto

El proyecto **Sistema de Proyección Financiera de Tasas TAB UF** es una aplicación analítica e interactiva desarrollada en Python con Streamlit y Plotly. Su objetivo es automatizar la ingesta de datos históricos oficiales de fijaciones (*fixing*) desde la Asociación de Bancos e Instituciones Financieras de Chile / Chilean Financial Benchmarks (CBF), y aplicar un motor cuantitativo multimodelo con torneo competitivo para proyectar las tasas **TAB UF a 90 días, 180 días y 360 días** en horizontes temporales de **1 mes (30d), 3 meses (90d), 6 meses (180d) y 1 año (360d)**.

---

## 2. Origen de Datos e Ingesta Automatizada (`cbf_client`)

### 2.1 Fuente Oficial
* **URL Fuente:** `https://cbf.cl/chilean-benchmarks/historical-rates/?lang=es`
* **Archivo de Datos:** Archivo Excel dinámico publicado por CBF (ej. `TAB-TADO-Historical-Data-since-17-05-2010-to-*.xlsx`).
* **Mecanismo de Ingesta:**
  1. Consulta HTTP GET con cabeceras estándar de navegador para resolver dinámicamente la URL más reciente del archivo.
  2. Descarga del flujo binario en memoria sin persistencia temporal innecesaria.
  3. Carga en caché mediante `@st.cache_data` con opción de invalidación manual mediante botón de actualización en la interfaz.

### 2.2 Estructura y Limpieza de la Hoja `FixingRates`
* **Hoja Objetivo:** `FixingRates`
* **Fila de Encabezados:** Fila 4 (índice 3 en base 0) para tenores y Fila 3 (índice 2) para nombres de tasa.
* **Inicio de Datos:** Fila 6 (índice 5 en base 0) en adelante.
* **Columnas Extraídas:**
  * **Columna C (Índice 2):** `Date` (Fecha de fijación). Formateada como `datetime64[ns]`.
  * **Columna H (Índice 7):** `TAB UF 90 Days` (Tasa anualizada en %).
  * **Columna I (Índice 8):** `TAB UF 180 Days` (Tasa anualizada en %).
  * **Columna J (Índice 9):** `TAB UF 360 Days` (Tasa anualizada en %).
* **Transformaciones y Calidad de Datos:**
  * Filtrado de filas nulas o no estructuradas.
  * Verificación de monotonicidad cronológica (orden ascendente de fechas).
  * Imputación controlada de días no bancarios / festivos para asegurar continuidad en series de tiempo sin introducir sesgos de nivel.

---

## 3. Motor Cuantitativo y Modelos de Proyección (`forecasting_engine`)

### 3.1 Modelos Implementados
El sistema integra 6 enfoques predictivos independientes:

1. **AutoARIMA / SARIMAX:**
   * Selección óptima de hiperparámetros $(p, d, q)$ minimizando el Criterio de Información de Akaike (AIC).
   * Modela la autocorrelación serial y la integración de orden 1 comúnmente observada en series de tasas de interés.
2. **Holt-Winters / Suavizado Exponencial (ETS):**
   * Modelos de nivel y tendencia amortiguada (*damped trend*) para mitigar proyecciones divergentes a 12 meses.
3. **Modelo Financiero de Vasicek (Ornstein-Uhlenbeck):**
   * Proceso de difusión estocástico de reversión a la media:
     $$dr_t = \kappa(\theta - r_t)dt + \sigma dW_t$$
   * Calibración de velocidad de reversión ($\kappa$), media de largo plazo ($\theta$) y volatilidad ($\sigma$) vía máxima verosimilitud (MLE) y regresión OLS discreta.
4. **VAR (Vector Autoregression Multivariado):**
   * Modela conjuntamente el vector $Y_t = [\text{TAB\_UF\_90d}_t, \text{TAB\_UF\_180d}_t, \text{TAB\_UF\_360d}_t]'$.
   * Preserva el co-movimiento y la consistencia cruzada de la curva de rendimiento.
5. **Machine Learning Autoregresivo con Lags (Ridge / GBDT):**
   * Matriz de características rezagadas ($t-1, t-2, t-5, t-21, t-63$), medias móviles y volatilidad rodante.
6. **Ensemble Ponderado Óptimo:**
   * Combinación lineal convexa: $\hat{y}_t = \sum w_i \hat{y}_{i,t}$, donde los pesos son proporcionales al inverso del error cuadrático medio de validación:
     $$w_i = \frac{1/\text{RMSE}_i^2}{\sum_j (1/\text{RMSE}_j^2)}$$

### 3.2 Horizontes de Proyección
* **1 Mes:** 21 días hábiles bursátiles (~30 días calendario).
* **3 Meses:** 63 días hábiles bursátiles (~90 días calendario).
* **6 Meses:** 126 días hábiles bursátiles (~180 días calendario).
* **1 Año:** 252 días hábiles bursátiles (~360 días calendario).

### 3.3 Torneo de Modelos y Evaluación Rodante (`evaluator`)
* **Metodología:** *Walk-Forward Cross-Validation* en los últimos $K$ periodos históricos para evitar sesgo de anticipación (*look-ahead bias*).
* **Métricas Evaluadas:**
  * **RMSE** (*Root Mean Squared Error*): Métrica principal de clasificación.
  * **MAE** (*Mean Absolute Error*): Error medio absoluto en puntos porcentuales/base.
  * **MAPE** (*Mean Absolute Percentage Error*): Error porcentual relativo.
  * **Directional Accuracy (%)**: Porcentaje de acierto en la dirección del cambio.
* **Selección del Modelo Ganador:**
  * Identificación automática del modelo con menor RMSE para el horizonte seleccionado.
  * Generación de tabla comparativa con ranking de rendimiento.

---

## 4. Simulación Monte Carlo y Estructura Temporal

### 4.1 Simulación Monte Carlo (`monte_carlo`)
* Generación de $N = 1.000$ caminos estocásticos a partir del último fixing y parámetros calibrados.
* Cálculo de percentiles temporales: P10 (Escenario Bajista), P25, P50 (Mediana/Base), P75, P90 (Escenario Alcista).
* Bandas de confianza sombreadas al 80% y 95%.
* Histograma de densidades de probabilidad para cada fecha horizonte.

### 4.2 Análisis de Curva de Estructura Temporal (`term_structure`)
* Curva de Rendimiento TAB UF en los plazos 90d, 180d y 360d.
* Visualización comparativa: Curva Actual vs. Curvas Proyectadas (+1m, +3m, +6m, +12m).
* Monitoreo de spreads de plazo:
  * **Slope (Pendiente):** $\text{TAB UF 360d} - \text{TAB UF 90d}$.
  * **Butterfly (Curvatura):** $2 \times \text{TAB UF 180d} - (\text{TAB UF 90d} + \text{TAB UF 360d})$.

---

## 5. Arquitectura de Software e Interfaz de Usuario

### 5.1 Estructura del Código
```
Proyecto Proyección Tasa TAB/
├── GEMINI.md                    # Directivas y contratos del asistente
├── README.md                    # Documentación y manual de usuario
├── CHANGELOG.md                 # Registro histórico de versiones
├── pyproject.toml               # Configuración de dependencias y proyecto
├── src/
│   ├── __init__.py
│   ├── cbf_client.py            # Descarga y parsing de datos CBF
│   ├── forecasting_engine.py    # Modelos ARIMA, ETS, Vasicek, VAR, ML, Ensemble
│   ├── evaluator.py             # Torneo y validación rodante
│   ├── monte_carlo.py           # Simulación estocástica y percentiles
│   ├── term_structure.py        # Curvas de tasas y análisis de spreads
│   └── ui_components.py         # Gráficos Plotly y tarjetas KPI
├── app.py                       # Punto de entrada de la aplicación Streamlit
└── tests/
    ├── __init__.py
    ├── test_cbf_client.py
    ├── test_forecasting.py
    ├── test_evaluator.py
    └── test_monte_carlo.py
```

### 5.2 Interfaz de Usuario (Streamlit)
* **Sidebar:**
  * Botón de actualización en tiempo real desde CBF.
  * Selector de serie: TAB UF 90d, 180d, 360d, o Todas.
  * Selector de modelo: "Ganador Automático del Torneo", "Ensemble Ponderado" o modelos individuales.
  * Selector de horizonte y rango histórico de entrenamiento.
* **Pestañas Principales:**
  1. 📊 **Resumen Ejecutivo & Proyecciones:** KPIs de último fixing, variación y valores proyectados con intervalos de confianza al 80% y 95%.
  2. 🏆 **Torneo de Modelos:** Tabla de ranking, métricas de error y superposición gráfica de trayectorias.
  3. 📈 **Estructura Temporal:** Curva de tasas actual vs proyectadas y spreads de pendiente.
  4. 🎲 **Simulación Monte Carlo:** Cono de incertidumbre estocástico y distribuciones de densidad.
  5. 💾 **Exportador de Datos:** Generación de libro Excel multishoheet (`.xlsx`) y archivos `.csv`.

---

## 6. Estrategia de Pruebas y Validación

1. **Pruebas Unitarias (`pytest`):**
   * Descarga y parseo correcto del Excel de CBF.
   * Ejecución y convergencia matemática de todos los modelos predictivos.
   * Precisión de las funciones de cálculo de métricas de backtesting.
   * Generación correcta de trayectorias Monte Carlo y ajuste de curvas.
2. **Pruebas de Integración y Rendimiento:**
   * Tiempos de ejecución de proyecciones inferiores a 3 segundos gracias a computación vectorial con NumPy y Pandas.
   * Validación de tipado y linting libre de errores.
