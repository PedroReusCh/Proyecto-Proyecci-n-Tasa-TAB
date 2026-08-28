"""
Módulo de Evaluación y Torneo de Modelos Cuantitativos.
Implementa validación rodante (Walk-Forward Cross-Validation) y cálculo de métricas de precisión.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple
import warnings
import numpy as np
import pandas as pd
from src.forecasting_engine import (
    AutoARIMAForecaster,
    HoltWintersForecaster,
    VasicekForecaster,
    VARMultivariateForecaster,
    MLLagsForecaster,
    EnsembleForecaster,
    ForecastResult,
)

warnings.filterwarnings("ignore")

@dataclass(frozen=True)
class ModelScore:
    """Métricas y ranking para un modelo evaluado."""
    model_name: str
    rmse: float
    mae: float
    mape: float
    directional_accuracy: float
    rank: int

@dataclass(frozen=True)
class TournamentResult:
    """Resultado global del torneo para un tenor y horizonte."""
    target_col: str
    horizon_days: int
    scores: List[ModelScore]
    winner_model: str
    weights: Dict[str, float]

def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """Calcula RMSE, MAE, MAPE y Precisión Direccional entre valores reales y predichos."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    
    diff = predicted - actual
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    
    # Evitar división por cero en MAPE
    nonzero_mask = np.abs(actual) > 1e-4
    if np.any(nonzero_mask):
        mape = float(np.mean(np.abs(diff[nonzero_mask] / actual[nonzero_mask])) * 100.0)
    else:
        mape = 0.0
    
    # Precisión Direccional (%)
    if len(actual) > 1:
        actual_dir = np.sign(np.diff(actual))
        pred_dir = np.sign(np.diff(predicted))
        dir_acc = float(np.mean(actual_dir == pred_dir) * 100.0)
    else:
        dir_acc = 100.0 if np.sign(predicted[0]) == np.sign(actual[0]) else 0.0
    
    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "directional_accuracy": dir_acc,
    }

def evaluate_tournament(
    series_df: pd.DataFrame,
    target_col: str,
    horizon_days: int,
    n_splits: int = 3,
) -> TournamentResult:
    """
    Ejecuta el torneo competitivo de modelos mediante Walk-Forward Validation en ventanas históricas.
    """
    total_len = len(series_df)
    eval_window = min(horizon_days, 63)
    
    # Verificar longitud mínima
    min_required = eval_window * (n_splits + 1) + 30
    if total_len < min_required:
        n_splits = max(1, (total_len - 30) // eval_window)
    
    forecasters = [
        AutoARIMAForecaster(),
        HoltWintersForecaster(),
        VasicekForecaster(),
        VARMultivariateForecaster(),
        MLLagsForecaster(),
    ]
    
    model_errors: Dict[str, List[Dict[str, float]]] = {f.name: [] for f in forecasters}
    model_errors["Ensemble"] = []
    
    # Generar splits rodantes
    split_indices = []
    for s in range(n_splits, 0, -1):
        cutoff = total_len - s * eval_window
        if cutoff > 30:
            split_indices.append((cutoff, cutoff + eval_window))
    
    if not split_indices:
        split_indices = [(total_len - eval_window, total_len)]
    
    for train_end, test_end in split_indices:
        train_df = series_df.iloc[:train_end].copy()
        test_df = series_df.iloc[train_end:test_end].copy()
        actual_vals = test_df[target_col].values
        k_steps = len(actual_vals)
        
        individual_preds: Dict[str, ForecastResult] = {}
        for f in forecasters:
            try:
                res = f.forecast(train_df, target_col, k_steps)
                individual_preds[f.name] = res
                metrics = calculate_metrics(actual_vals, res.point_forecast[:k_steps])
                model_errors[f.name].append(metrics)
            except Exception:
                pass
        
        if individual_preds:
            ens = EnsembleForecaster()
            ens_res = ens.combine(individual_preds)
            ens_metrics = calculate_metrics(actual_vals, ens_res.point_forecast[:k_steps])
            model_errors["Ensemble"].append(ens_metrics)
    
    # Promediar métricas por modelo
    scores_list: List[ModelScore] = []
    for model_name, metrics_list in model_errors.items():
        if metrics_list:
            avg_rmse = float(np.mean([m["rmse"] for m in metrics_list]))
            avg_mae = float(np.mean([m["mae"] for m in metrics_list]))
            avg_mape = float(np.mean([m["mape"] for m in metrics_list]))
            avg_dir = float(np.mean([m["directional_accuracy"] for m in metrics_list]))
        else:
            avg_rmse = 999.0
            avg_mae = 999.0
            avg_mape = 999.0
            avg_dir = 0.0
        
        scores_list.append(
            ModelScore(
                model_name=model_name,
                rmse=avg_rmse,
                mae=avg_mae,
                mape=avg_mape,
                directional_accuracy=avg_dir,
                rank=0,
            )
        )
    
    # Ordenar por menor RMSE
    scores_list.sort(key=lambda x: x.rmse)
    ranked_scores = [
        ModelScore(
            model_name=s.model_name,
            rmse=s.rmse,
            mae=s.mae,
            mape=s.mape,
            directional_accuracy=s.directional_accuracy,
            rank=idx + 1,
        )
        for idx, s in enumerate(scores_list)
    ]
    
    winner = ranked_scores[0].model_name
    
    # Calcular pesos inversos de error para el ensamble
    weights = {}
    valid_scores = [s for s in ranked_scores if s.model_name != "Ensemble" and s.rmse < 900.0]
    if valid_scores:
        inv_sq = [1.0 / max(s.rmse ** 2, 1e-6) for s in valid_scores]
        total_inv = sum(inv_sq)
        for s, inv in zip(valid_scores, inv_sq):
            weights[s.model_name] = float(inv / total_inv)
    
    return TournamentResult(
        target_col=target_col,
        horizon_days=horizon_days,
        scores=ranked_scores,
        winner_model=winner,
        weights=weights,
    )
