"""
Motor Cuantitativo de Proyección de Tasas de Interés TAB UF.
Implementa modelos estadísticos, financieros y de machine learning para horizontes de 1m, 3m, 6m y 12m.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import VAR
from sklearn.linear_model import Ridge

# Silenciar advertencias numéricas no críticas de statsmodels
warnings.filterwarnings("ignore")

HORIZONS_BUSINESS_DAYS = {
    "1 Mes (30d)": 21,
    "3 Meses (90d)": 63,
    "6 Meses (180d)": 126,
    "1 Año (360d)": 252,
}

@dataclass(frozen=True)
class ForecastResult:
    """Contenedor de resultados de pronóstico para un modelo e intervalo."""
    model_name: str
    horizon_days: int
    future_dates: List[pd.Timestamp]
    point_forecast: np.ndarray
    lower_80: np.ndarray
    upper_80: np.ndarray
    lower_95: np.ndarray
    upper_95: np.ndarray

def generate_future_business_dates(last_date: pd.Timestamp, n_days: int) -> List[pd.Timestamp]:
    """Genera n fechas hábiles bursátiles posteriores a last_date."""
    # Usar date_range con frecuencia B (business days) iniciando en el siguiente día hábil
    dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days * 2, freq="B")
    return list(dates[:n_days])

class BaseForecaster:
    """Clase base para todos los modelos cuantitativos."""
    name: str = "Base"

    def forecast(
        self,
        df: pd.DataFrame,
        target_col: str,
        horizon_days: int,
    ) -> ForecastResult:
        raise NotImplementedError

class AutoARIMAForecaster(BaseForecaster):
    """Modelo autorregresivo integrado de media móvil optimizado por AIC."""
    name: str = "AutoARIMA"

    def forecast(
        self,
        df: pd.DataFrame,
        target_col: str,
        horizon_days: int,
    ) -> ForecastResult:
        series = df[target_col].values
        last_date = pd.Timestamp(df["Date"].iloc[-1])
        future_dates = generate_future_business_dates(last_date, horizon_days)
        
        # Grid search rápido de órdenes (p, d, q) para minimizar AIC
        # Usar muestra representativa reciente si la serie es muy larga para velocidad y relevancia
        sub_series = series[-min(len(series), 500):]
        
        best_aic = float("inf")
        best_order = (1, 1, 1)
        best_model_fit = None
        
        orders_to_try = [
            (1, 1, 1),
            (2, 1, 1),
            (1, 1, 2),
            (2, 1, 2),
            (1, 1, 0),
            (0, 1, 1),
        ]
        
        for order in orders_to_try:
            try:
                mod = ARIMA(sub_series, order=order)
                res = mod.fit()
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_order = order
                    best_model_fit = res
            except Exception:
                continue
        
        if best_model_fit is None:
            # Fallback simple
            mod = ARIMA(sub_series, order=(1, 1, 0))
            best_model_fit = mod.fit()
        
        forecast_obj = best_model_fit.get_forecast(steps=horizon_days)
        point_pred = forecast_obj.predicted_mean
        
        conf_80 = forecast_obj.conf_int(alpha=0.20)
        conf_95 = forecast_obj.conf_int(alpha=0.05)
        
        lower_80 = conf_80[:, 0] if hasattr(conf_80, "shape") and conf_80.ndim > 1 else conf_80.iloc[:, 0].values
        upper_80 = conf_80[:, 1] if hasattr(conf_80, "shape") and conf_80.ndim > 1 else conf_80.iloc[:, 1].values
        lower_95 = conf_95[:, 0] if hasattr(conf_95, "shape") and conf_95.ndim > 1 else conf_95.iloc[:, 0].values
        upper_95 = conf_95[:, 1] if hasattr(conf_95, "shape") and conf_95.ndim > 1 else conf_95.iloc[:, 1].values
        
        return ForecastResult(
            model_name=self.name,
            horizon_days=horizon_days,
            future_dates=future_dates,
            point_forecast=np.asarray(point_pred, dtype=float),
            lower_80=np.asarray(lower_80, dtype=float),
            upper_80=np.asarray(upper_80, dtype=float),
            lower_95=np.asarray(lower_95, dtype=float),
            upper_95=np.asarray(upper_95, dtype=float),
        )

class HoltWintersForecaster(BaseForecaster):
    """Suavizado Exponencial de Holt con tendencia amortiguada (Damped Trend)."""
    name: str = "Holt-Winters"

    def forecast(
        self,
        df: pd.DataFrame,
        target_col: str,
        horizon_days: int,
    ) -> ForecastResult:
        series = df[target_col].values
        last_date = pd.Timestamp(df["Date"].iloc[-1])
        future_dates = generate_future_business_dates(last_date, horizon_days)
        
        sub_series = series[-min(len(series), 500):]
        
        try:
            model = ExponentialSmoothing(
                sub_series,
                trend="add",
                damped_trend=True,
                initialization_method="estimated",
            )
            fitted = model.fit(damping_slope=0.95, optimized=True)
            point_pred = fitted.forecast(horizon_days)
        except Exception:
            # Fallback simple sin amortiguación
            model = ExponentialSmoothing(sub_series, trend="add")
            fitted = model.fit()
            point_pred = fitted.forecast(horizon_days)
        
        # Intervalos calculados a partir de los residuales empíricos
        residuals = fitted.resid
        sigma = np.std(residuals) if len(residuals) > 0 else 0.05
        
        # Cono de dispersión proporcional a sqrt(t)
        time_steps = np.sqrt(np.arange(1, horizon_days + 1))
        lower_80 = point_pred - 1.282 * sigma * time_steps
        upper_80 = point_pred + 1.282 * sigma * time_steps
        lower_95 = point_pred - 1.960 * sigma * time_steps
        upper_95 = point_pred + 1.960 * sigma * time_steps
        
        return ForecastResult(
            model_name=self.name,
            horizon_days=horizon_days,
            future_dates=future_dates,
            point_forecast=np.asarray(point_pred, dtype=float),
            lower_80=np.asarray(lower_80, dtype=float),
            upper_80=np.asarray(upper_80, dtype=float),
            lower_95=np.asarray(lower_95, dtype=float),
            upper_95=np.asarray(upper_95, dtype=float),
        )

class VasicekForecaster(BaseForecaster):
    """Modelo estocástico de tasas de interés con reversión a la media (Ornstein-Uhlenbeck)."""
    name: str = "Vasicek"

    def forecast(
        self,
        df: pd.DataFrame,
        target_col: str,
        horizon_days: int,
    ) -> ForecastResult:
        series = df[target_col].values
        last_date = pd.Timestamp(df["Date"].iloc[-1])
        future_dates = generate_future_business_dates(last_date, horizon_days)
        
        # Estimación de parámetros discretos OLS: r_t - r_{t-1} = a + b * r_{t-1} + e_t
        sub_series = series[-min(len(series), 500):]
        r_prev = sub_series[:-1]
        dr = sub_series[1:] - r_prev
        
        # Regresión lineal
        poly = np.polyfit(r_prev, dr, 1)
        b, a = poly[0], poly[1]
        
        dt = 1.0 / 252.0  # Paso diario en base anual
        
        # kappa (velocidad de reversión a la media) y theta (media de largo plazo)
        if b < 0:
            kappa = -np.log(1.0 + b) / dt if (1.0 + b) > 0 else -b / dt
            theta = -a / b if abs(b) > 1e-7 else float(np.mean(sub_series))
        else:
            kappa = 0.3
            theta = float(np.mean(sub_series))
        
        # Volatilidad sigma
        residuals = dr - (a + b * r_prev)
        sigma = float(np.std(residuals) / np.sqrt(dt))
        
        r0 = float(sub_series[-1])
        t_years = np.arange(1, horizon_days + 1) * dt
        
        # Valor esperado analítico de Vasicek: E[r_t] = r_0 * e^(-kappa*t) + theta * (1 - e^(-kappa*t))
        exp_kt = np.exp(-kappa * t_years)
        point_pred = r0 * exp_kt + theta * (1.0 - exp_kt)
        
        # Varianza analítica de Vasicek: Var(r_t) = (sigma^2 / (2*kappa)) * (1 - e^(-2*kappa*t))
        var_pred = (sigma**2 / (2.0 * max(kappa, 1e-4))) * (1.0 - np.exp(-2.0 * kappa * t_years))
        std_pred = np.sqrt(np.maximum(var_pred, 1e-6))
        
        lower_80 = point_pred - 1.282 * std_pred
        upper_80 = point_pred + 1.282 * std_pred
        lower_95 = point_pred - 1.960 * std_pred
        upper_95 = point_pred + 1.960 * std_pred
        
        return ForecastResult(
            model_name=self.name,
            horizon_days=horizon_days,
            future_dates=future_dates,
            point_forecast=np.asarray(point_pred, dtype=float),
            lower_80=np.asarray(lower_80, dtype=float),
            upper_80=np.asarray(upper_80, dtype=float),
            lower_95=np.asarray(lower_95, dtype=float),
            upper_95=np.asarray(upper_95, dtype=float),
        )

class VARMultivariateForecaster(BaseForecaster):
    """Vector Autoregression multivariado proyectando conjuntamente 90d, 180d y 360d."""
    name: str = "VAR"

    def forecast(
        self,
        df: pd.DataFrame,
        target_col: str,
        horizon_days: int,
    ) -> ForecastResult:
        rates_cols = ["TAB_UF_90", "TAB_UF_180", "TAB_UF_360"]
        data = df[rates_cols].values
        last_date = pd.Timestamp(df["Date"].iloc[-1])
        future_dates = generate_future_business_dates(last_date, horizon_days)
        
        sub_data = data[-min(len(data), 500):]
        
        target_idx = rates_cols.index(target_col) if target_col in rates_cols else 0
        
        try:
            model = VAR(sub_data)
            fitted = model.fit(maxlags=5, ic="aic")
            k_ar = fitted.k_ar
            pred = fitted.forecast(sub_data[-k_ar:], steps=horizon_days)
            point_pred = pred[:, target_idx]
            
            # Intervalos basados en covarianza de pronóstico
            forecast_cov = fitted.forecast_cov(steps=horizon_days)
            std_pred = np.sqrt(np.array([cov[target_idx, target_idx] for cov in forecast_cov]))
        except Exception:
            # Fallback a proyección simple con persistencia
            point_pred = np.full(horizon_days, sub_data[-1, target_idx])
            std_pred = np.sqrt(np.arange(1, horizon_days + 1)) * 0.03
        
        lower_80 = point_pred - 1.282 * std_pred
        upper_80 = point_pred + 1.282 * std_pred
        lower_95 = point_pred - 1.960 * std_pred
        upper_95 = point_pred + 1.960 * std_pred
        
        return ForecastResult(
            model_name=self.name,
            horizon_days=horizon_days,
            future_dates=future_dates,
            point_forecast=np.asarray(point_pred, dtype=float),
            lower_80=np.asarray(lower_80, dtype=float),
            upper_80=np.asarray(upper_80, dtype=float),
            lower_95=np.asarray(lower_95, dtype=float),
            upper_95=np.asarray(upper_95, dtype=float),
        )

class MLLagsForecaster(BaseForecaster):
    """Regresión autorregresiva con características de retardos (Lags 1, 2, 5, 21) y medias móviles."""
    name: str = "ML-Lags"

    def forecast(
        self,
        df: pd.DataFrame,
        target_col: str,
        horizon_days: int,
    ) -> ForecastResult:
        series = df[target_col].values
        last_date = pd.Timestamp(df["Date"].iloc[-1])
        future_dates = generate_future_business_dates(last_date, horizon_days)
        
        # Construcción de características
        lags = [1, 2, 3, 5, 10, 21]
        max_lag = max(lags)
        
        X_list = []
        y_list = []
        for i in range(max_lag, len(series)):
            features = [series[i - lag] for lag in lags]
            # Media móvil reciente de 5 y 21
            features.append(np.mean(series[i - 5:i]))
            features.append(np.mean(series[i - 21:i]))
            X_list.append(features)
            y_list.append(series[i])
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        
        # Proyección recursiva multi-paso
        current_history = list(series)
        preds = []
        for step in range(horizon_days):
            feats = [current_history[-lag] for lag in lags]
            feats.append(np.mean(current_history[-5:]))
            feats.append(np.mean(current_history[-21:]))
            next_val = float(model.predict([feats])[0])
            preds.append(next_val)
            current_history.append(next_val)
        
        point_pred = np.array(preds)
        
        # Error residual para intervalos
        resid = y - model.predict(X)
        sigma = np.std(resid) if len(resid) > 0 else 0.05
        time_steps = np.sqrt(np.arange(1, horizon_days + 1))
        
        lower_80 = point_pred - 1.282 * sigma * time_steps
        upper_80 = point_pred + 1.282 * sigma * time_steps
        lower_95 = point_pred - 1.960 * sigma * time_steps
        upper_95 = point_pred + 1.960 * sigma * time_steps
        
        return ForecastResult(
            model_name=self.name,
            horizon_days=horizon_days,
            future_dates=future_dates,
            point_forecast=np.asarray(point_pred, dtype=float),
            lower_80=np.asarray(lower_80, dtype=float),
            upper_80=np.asarray(upper_80, dtype=float),
            lower_95=np.asarray(lower_95, dtype=float),
            upper_95=np.asarray(upper_95, dtype=float),
        )

class EnsembleForecaster(BaseForecaster):
    """Combinación lineal óptima ponderada de múltiples pronósticos."""
    name: str = "Ensemble"

    def combine(
        self,
        individual_results: Dict[str, ForecastResult],
        weights: Optional[Dict[str, float]] = None,
    ) -> ForecastResult:
        if not individual_results:
            raise ValueError("No se proporcionaron modelos para el ensemble.")
        
        first_res = next(iter(individual_results.values()))
        horizon_days = first_res.horizon_days
        future_dates = first_res.future_dates
        
        model_names = [k for k in individual_results.keys() if k != "Ensemble"]
        if not model_names:
            return first_res
        
        # Normalizar pesos
        if weights is None:
            w_vals = np.ones(len(model_names)) / len(model_names)
        else:
            raw_w = np.array([max(weights.get(m, 1.0), 1e-4) for m in model_names])
            w_vals = raw_w / np.sum(raw_w)
        
        weighted_points = np.zeros(horizon_days)
        weighted_lower_80 = np.zeros(horizon_days)
        weighted_upper_80 = np.zeros(horizon_days)
        weighted_lower_95 = np.zeros(horizon_days)
        weighted_upper_95 = np.zeros(horizon_days)
        
        for idx, m_name in enumerate(model_names):
            m_res = individual_results[m_name]
            w = w_vals[idx]
            weighted_points += w * m_res.point_forecast
            weighted_lower_80 += w * m_res.lower_80
            weighted_upper_80 += w * m_res.upper_80
            weighted_lower_95 += w * m_res.lower_95
            weighted_upper_95 += w * m_res.upper_95
        
        return ForecastResult(
            model_name=self.name,
            horizon_days=horizon_days,
            future_dates=future_dates,
            point_forecast=weighted_points,
            lower_80=weighted_lower_80,
            upper_80=weighted_upper_80,
            lower_95=weighted_lower_95,
            upper_95=weighted_upper_95,
        )

def generate_all_forecasts(
    df: pd.DataFrame,
    target_col: str,
    horizon_days: int,
    custom_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, ForecastResult]:
    """Ejecuta todos los modelos individuales y genera el ensamble óptimo."""
    forecasters = [
        AutoARIMAForecaster(),
        HoltWintersForecaster(),
        VasicekForecaster(),
        VARMultivariateForecaster(),
        MLLagsForecaster(),
    ]
    
    results: Dict[str, ForecastResult] = {}
    for f in forecasters:
        try:
            results[f.name] = f.forecast(df, target_col, horizon_days)
        except Exception:
            pass
    
    if results:
        ensemble = EnsembleForecaster()
        results["Ensemble"] = ensemble.combine(results, custom_weights)
    
    return results
