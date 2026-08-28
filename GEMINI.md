# Reglas Operativas y Directivas Técnicas del Proyecto

Este archivo define las directivas técnicas, lineamientos de calidad y estándares de desarrollo para el asistente de IA en el proyecto **Sistema de Proyección Financiera de Tasas TAB UF**.

---

## 1. Directivas Obligatorias
* **Idioma:** Todas las respuestas, explicaciones, documentación y commits de Git deben ser exclusivamente en **español**.
* **Gestión de Entorno:** Verificar e instalar automáticamente las dependencias antes de ejecutar cualquier código (`uv`, `pandas`, `openpyxl`, `streamlit`, `plotly`, `statsmodels`, `scikit-learn`, `scipy`, `pytest`, `requests`).
* **Archivos Raíz Obligatorios:** Mantener actualizados `GEMINI.md`, `README.md` y `CHANGELOG.md`.
* **Control de Versiones:** Realizar commits descriptivos en español tras completar cada tarea o funcionalidad.

---

## 2. Contratos Técnicos de Datos (CBF)
* **Fuente:** Web Oficial CBF (`https://cbf.cl/chilean-benchmarks/historical-rates/?lang=es`).
* **Hoja:** `FixingRates`.
* **Encabezados:** Fila 4 (nombres de tenores), datos a partir de fila 6.
* **Columnas Clave:**
  * Fecha: Columna C (`Date`).
  * TAB UF 90 Días: Columna H (`TAB UF 90 Days`).
  * TAB UF 180 Días: Columna I (`TAB UF 180 Days`).
  * TAB UF 360 Días: Columna J (`TAB UF 360 Days`).

---

## 3. Estándares Cuantitativos y de Código
* Modelos estadísticos y financieros rigurosos (AutoARIMA, Holt-Winters ETS, Vasicek/Ornstein-Uhlenbeck, VAR multivariado, ML Lags, Ensemble óptimo).
* Validación sin fuga de datos futuros (*Walk-Forward Validation*).
* Tipado estricto en Python (`typing`) y cobertura de pruebas con `pytest`.
