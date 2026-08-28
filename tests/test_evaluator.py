import numpy as np
import pandas as pd
import pytest
from src.evaluator import (
    ModelScore,
    TournamentResult,
    calculate_metrics,
    evaluate_tournament,
)

def test_calculate_metrics():
    actual = np.array([3.0, 3.1, 3.2, 3.3, 3.4])
    predicted = np.array([3.02, 3.09, 3.22, 3.31, 3.38])
    metrics = calculate_metrics(actual, predicted)
    
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "mape" in metrics
    assert "directional_accuracy" in metrics
    assert metrics["rmse"] > 0
    assert metrics["mae"] > 0
    assert 0 <= metrics["directional_accuracy"] <= 100

def test_evaluate_tournament():
    dates = pd.date_range("2023-01-01", periods=180, freq="B")
    np.random.seed(42)
    s90 = 3.0 + np.cumsum(np.random.normal(0, 0.02, 180))
    s180 = s90 + 0.3
    s360 = s180 + 0.4
    df = pd.DataFrame({"Date": dates, "TAB_UF_90": s90, "TAB_UF_180": s180, "TAB_UF_360": s360})
    
    res = evaluate_tournament(df, "TAB_UF_90", horizon_days=10, n_splits=2)
    assert isinstance(res, TournamentResult)
    assert res.target_col == "TAB_UF_90"
    assert res.horizon_days == 10
    assert len(res.scores) >= 5
    assert res.scores[0].rank == 1
    assert res.winner_model in [s.model_name for s in res.scores]
    assert res.scores[0].model_name == res.winner_model
    # Ranking ordenado por menor RMSE
    for i in range(len(res.scores) - 1):
        assert res.scores[i].rmse <= res.scores[i+1].rmse
