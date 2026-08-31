"""
Módulo de Diagnóstico Econométrico y Verificación de Residuos.
Implementa el Test ADF de estacionariedad, cálculo de ACF/PACF y el Test de Ljung-Box para ruido blanco.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

@dataclass(frozen=True)
class ADFResult:
    """Resultados del Test de Dickey-Fuller Aumentado."""
    test_statistic: float
    p_value: float
    used_lag: int
    n_obs: int
    critical_values: Dict[str, float]
    is_stationary: bool
    conclusion: str

@dataclass(frozen=True)
class LjungBoxResult:
    """Resultados del Test de Ljung-Box para autocorrelación de residuos."""
    lags: int
    test_statistics: List[float]
    p_values: List[float]
    is_white_noise: bool
    conclusion: str

def run_adf_test(series: np.ndarray, alpha: float = 0.05) -> ADFResult:
    """
    Ejecuta el Test de Dickey-Fuller Aumentado (ADF) sobre una serie temporal.
    H0: La serie posee raíz unitaria (es no estacionaria).
    H1: La serie es estacionaria.
    """
    clean_series = np.asarray(series[~np.isnan(series)], dtype=float)
    if len(clean_series) < 20:
        raise ValueError("Se requieren al menos 20 observaciones para ejecutar el test ADF.")
    
    result = adfuller(clean_series, autolag="AIC")
    stat = float(result[0])
    p_val = float(result[1])
    used_lags = int(result[2])
    nobs = int(result[3])
    crit_vals = {k: float(v) for k, v in result[4].items()}
    
    is_stationary = p_val < alpha
    conclusion = (
        f"La serie es estacionaria (p-valor = {p_val:.4f} < {alpha}). Se rechaza la hipótesis de raíz unitaria."
        if is_stationary
        else f"La serie NO es estacionaria en nivel (p-valor = {p_val:.4f} >= {alpha}). Posee raíz unitaria."
    )
    
    return ADFResult(
        test_statistic=stat,
        p_value=p_val,
        used_lag=used_lags,
        n_obs=nobs,
        critical_values=crit_vals,
        is_stationary=is_stationary,
        conclusion=conclusion,
    )

def run_ljung_box_test(residuals: np.ndarray, lags: int = 10, alpha: float = 0.05) -> LjungBoxResult:
    """
    Ejecuta el Test de Ljung-Box sobre los residuos del modelo predictivo.
    H0: Los residuos son ruido blanco (no hay autocorrelación serial).
    H1: Existe autocorrelación serial remanente.
    """
    clean_resid = np.asarray(residuals[~np.isnan(residuals)], dtype=float)
    if len(clean_resid) <= lags:
        lags = max(1, len(clean_resid) // 3)
    
    lb_df = acorr_ljungbox(clean_resid, lags=lags, return_df=True)
    stats = [float(x) for x in lb_df["lb_stat"].values]
    p_vals = [float(x) for x in lb_df["lb_pvalue"].values]
    
    # Es ruido blanco si todos los p-valores son mayores a alpha
    is_white_noise = all(p >= alpha for p in p_vals)
    conclusion = (
        f"Los residuos se comportan como Ruido Blanco (p-valor mínimo = {min(p_vals):.4f} >= {alpha}). No hay autocorrelación residual."
        if is_white_noise
        else f"Existe autocorrelación serial residual en algunos retardos (p-valor mínimo = {min(p_vals):.4f} < {alpha})."
    )
    
    return LjungBoxResult(
        lags=lags,
        test_statistics=stats,
        p_values=p_vals,
        is_white_noise=is_white_noise,
        conclusion=conclusion,
    )

def compute_acf_pacf(series: np.ndarray, nlags: int = 20) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calcula los coeficientes de autocorrelación (ACF) y autocorrelación parcial (PACF)
    junto con la banda de significancia de Bartlett al 95%.
    """
    clean_series = np.asarray(series[~np.isnan(series)], dtype=float)
    max_lags = min(nlags, len(clean_series) // 3)
    
    acf_vals = acf(clean_series, nlags=max_lags, fft=True)
    pacf_vals = pacf(clean_series, nlags=max_lags, method="ywm")
    
    # Banda de confianza al 95% = 1.96 / sqrt(N)
    conf_bound = 1.96 / np.sqrt(len(clean_series))
    
    return acf_vals, pacf_vals, float(conf_bound)
