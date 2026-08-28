# Sistema de Proyección Financiera de Tasas TAB UF (Chilean Financial Benchmarks)

Plataforma analítica e interactiva para la ingesta automatizada, modelado cuantitativo, torneo competitivo de predicción y visualización de tasas **TAB UF** (90, 180 y 360 días) en horizontes de **1 mes, 3 meses, 6 meses y 1 año**.

---

## 🚀 Características Principales
* **Ingesta Automatizada desde CBF:** Descarga directa y parsing inteligente del archivo Excel oficial de fijaciones históricas desde mayo de 2010.
* **Torneo Cuantitativo de Modelos:** Comparación en tiempo real de 6 modelos (AutoARIMA, Holt-Winters ETS, Vasicek/Reversión a la media, VAR Multivariado, Machine Learning autoregresivo y Ensemble Óptimo Ponderado).
* **Backtesting Rodante (*Walk-Forward Validation*):** Evaluación rigurosa mediante métricas de error (RMSE, MAE, MAPE, Precisión Direccional) y selección automática del modelo ganador.
* **Simulación Monte Carlo:** Generación de más de 1.000 trayectorias estocásticas con intervalos de confianza al 80% y 95% y percentiles P10, P50 y P90.
* **Estructura Temporal de Tasas:** Modelado de la curva de rendimiento y análisis de spreads de pendiente (*slope*) y curvatura (*butterfly*).
* **Exportación Completa:** Descarga de reportes en Excel (`.xlsx`) y CSV con datos históricos, métricas y proyecciones.

---

## 🛠️ Requisitos e Instalación

### Requisitos:
* Python 3.11+ o gestor `uv`.

### Instalación y Ejecución con `uv`:
```bash
# Sincronizar dependencias
uv sync

# Ejecutar la aplicación interactiva
uv run streamlit run app.py
```

---

## 🧪 Pruebas Unitarias
```bash
uv run pytest
```
