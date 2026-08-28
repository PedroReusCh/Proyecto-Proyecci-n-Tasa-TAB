"""
Componentes Gráficos Interactivos con Plotly para el Dashboard de Tasas TAB UF.
"""
from typing import Dict, List
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from src.forecasting_engine import ForecastResult
from src.monte_carlo import MonteCarloResult
from src.term_structure import TermStructureCurve

COLOR_PALETTE = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "success": "#2ca02c",
    "danger": "#d62728",
    "purple": "#9467bd",
    "brown": "#8c564b",
    "pink": "#e377c2",
    "gray": "#7f7f7f",
    "dark": "#2b3e50",
}

def create_forecast_figure(
    historical_df: pd.DataFrame,
    target_col: str,
    forecast_res: ForecastResult,
    target_name: str = "TAB UF 90 Días",
) -> go.Figure:
    """Crea gráfico interactivo de la serie histórica y el cono de proyección con bandas de confianza."""
    fig = go.Figure()
    
    # 1. Serie Histórica (Últimos N registros para no sobrecargar el renderizado inicial)
    hist_subset = historical_df.iloc[-min(len(historical_df), 300):]
    fig.add_trace(
        go.Scatter(
            x=hist_subset["Date"],
            y=hist_subset[target_col],
            mode="lines",
            name=f"Histórico ({target_name})",
            line=dict(color="#1f77b4", width=2),
            hovertemplate="<b>Fecha:</b> %{x|%d-%m-%Y}<br><b>Tasa Histórica:</b> %{y:.2f}%<extra></extra>",
        )
    )
    
    # Unir el último punto histórico con el primer punto proyectado
    last_hist_date = historical_df["Date"].iloc[-1]
    last_hist_val = historical_df[target_col].iloc[-1]
    
    proj_dates = [last_hist_date] + list(forecast_res.future_dates)
    proj_vals = [last_hist_val] + list(forecast_res.point_forecast)
    lower_95 = [last_hist_val] + list(forecast_res.lower_95)
    upper_95 = [last_hist_val] + list(forecast_res.upper_95)
    lower_80 = [last_hist_val] + list(forecast_res.lower_80)
    upper_80 = [last_hist_val] + list(forecast_res.upper_80)
    
    # 2. Banda de Confianza 95%
    fig.add_trace(
        go.Scatter(
            x=proj_dates + proj_dates[::-1],
            y=upper_95 + lower_95[::-1],
            fill="toself",
            fillcolor="rgba(255, 127, 14, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="Intervalo 95%",
        )
    )
    
    # 3. Banda de Confianza 80%
    fig.add_trace(
        go.Scatter(
            x=proj_dates + proj_dates[::-1],
            y=upper_80 + lower_80[::-1],
            fill="toself",
            fillcolor="rgba(255, 127, 14, 0.25)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="Intervalo 80%",
        )
    )
    
    # 4. Línea Central de Proyección
    fig.add_trace(
        go.Scatter(
            x=proj_dates,
            y=proj_vals,
            mode="lines+markers",
            name=f"Proyección ({forecast_res.model_name})",
            line=dict(color="#d62728", width=3, dash="dash"),
            marker=dict(size=4),
            hovertemplate="<b>Fecha:</b> %{x|%d-%m-%Y}<br><b>Tasa Proyectada:</b> %{y:.2f}%<extra></extra>",
        )
    )
    
    fig.update_layout(
        title=f"<b>Trayectoria Histórica y Proyección de {target_name}</b>",
        xaxis_title="Fecha",
        yaxis_title="Tasa de Interés Anual (%)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig

def create_tournament_comparison_figure(
    all_forecasts: Dict[str, ForecastResult],
    target_name: str = "TAB UF 90 Días",
) -> go.Figure:
    """Compara las trayectorias proyectadas por cada modelo en el torneo."""
    fig = go.Figure()
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    
    for idx, (m_name, res) in enumerate(all_forecasts.items()):
        color = colors[idx % len(colors)]
        width = 3.5 if m_name in ["Ensemble", "Ganador"] else 2.0
        dash = "solid" if m_name == "Ensemble" else "dash"
        
        fig.add_trace(
            go.Scatter(
                x=res.future_dates,
                y=res.point_forecast,
                mode="lines",
                name=m_name,
                line=dict(color=color, width=width, dash=dash),
                hovertemplate=f"<b>{m_name}:</b> %{{y:.2f}}%<extra></extra>",
            )
        )
    
    fig.update_layout(
        title=f"<b>Comparación de Trayectorias de Modelos en el Torneo ({target_name})</b>",
        xaxis_title="Fecha Proyectada",
        yaxis_title="Tasa Anual (%)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig

def create_monte_carlo_fan_figure(
    mc_result: MonteCarloResult,
    target_name: str = "TAB UF 90 Días",
    sample_paths_to_show: int = 30,
) -> go.Figure:
    """Crea gráfico de abanico con percentiles y trayectorias muestreadas de Monte Carlo."""
    fig = go.Figure()
    
    dates = mc_result.future_dates
    
    # Muestra de caminos individuales en gris suave
    n_sims = mc_result.simulated_paths.shape[0]
    sample_indices = np.random.choice(n_sims, size=min(sample_paths_to_show, n_sims), replace=False)
    for s_idx in sample_indices:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=mc_result.simulated_paths[s_idx],
                mode="lines",
                line=dict(color="rgba(128, 128, 128, 0.15)", width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    
    # Banda P10 - P90
    fig.add_trace(
        go.Scatter(
            x=dates + dates[::-1],
            y=list(mc_result.percentiles["P90"]) + list(mc_result.percentiles["P10"][::-1]),
            fill="toself",
            fillcolor="rgba(31, 119, 180, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Rango P10 - P90 (80% Prob)",
            hoverinfo="skip",
        )
    )
    
    # Banda P25 - P75
    fig.add_trace(
        go.Scatter(
            x=dates + dates[::-1],
            y=list(mc_result.percentiles["P75"]) + list(mc_result.percentiles["P25"][::-1]),
            fill="toself",
            fillcolor="rgba(31, 119, 180, 0.35)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Rango P25 - P75 (50% Prob)",
            hoverinfo="skip",
        )
    )
    
    # Mediana P50
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=mc_result.percentiles["P50"],
            mode="lines",
            name="Mediana (P50 - Escenario Base)",
            line=dict(color="#1f77b4", width=3),
            hovertemplate="<b>Mediana (P50):</b> %{y:.2f}%<extra></extra>",
        )
    )
    
    # Líneas P10 y P90
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=mc_result.percentiles["P90"],
            mode="lines",
            name="P90 (Escenario Alcista)",
            line=dict(color="#d62728", width=1.5, dash="dot"),
            hovertemplate="<b>P90:</b> %{y:.2f}%<extra></extra>",
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=mc_result.percentiles["P10"],
            mode="lines",
            name="P10 (Escenario Bajista)",
            line=dict(color="#2ca02c", width=1.5, dash="dot"),
            hovertemplate="<b>P10:</b> %{y:.2f}%<extra></extra>",
        )
    )
    
    fig.update_layout(
        title=f"<b>Simulación Monte Carlo (1.000 Trayectorias) - {target_name}</b>",
        xaxis_title="Fecha",
        yaxis_title="Tasa Anual (%)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig

def create_monte_carlo_hist_figure(
    terminal_dist: np.ndarray,
    horizon_label: str = "1 Año",
) -> go.Figure:
    """Histograma y distribución de densidad de la tasa terminal."""
    fig = go.Figure()
    
    fig.add_trace(
        go.Histogram(
            x=terminal_dist,
            nbinsx=35,
            name="Densidad",
            marker=dict(color="#1f77b4", opacity=0.75),
            hovertemplate="Rango de Tasa: %{x:.2f}%<br>Frecuencia: %{y}<extra></extra>",
        )
    )
    
    mean_val = float(np.mean(terminal_dist))
    p10_val = float(np.percentile(terminal_dist, 10))
    p90_val = float(np.percentile(terminal_dist, 90))
    
    fig.add_vline(x=mean_val, line_width=2, line_dash="dash", line_color="black", annotation_text=f"Media: {mean_val:.2f}%")
    fig.add_vline(x=p10_val, line_width=1.5, line_dash="dot", line_color="green", annotation_text=f"P10: {p10_val:.2f}%")
    fig.add_vline(x=p90_val, line_width=1.5, line_dash="dot", line_color="red", annotation_text=f"P90: {p90_val:.2f}%")
    
    fig.update_layout(
        title=f"<b>Distribución Terminal de Tasa a {horizon_label}</b>",
        xaxis_title="Tasa de Interés (%)",
        yaxis_title="Número de Simulaciones",
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig

def create_term_structure_figure(
    current_curve: TermStructureCurve,
    projected_curves: Dict[str, TermStructureCurve],
) -> go.Figure:
    """Curva de Rendimiento de Tasas TAB UF (90d, 180d, 360d): Actual vs Proyectadas."""
    fig = go.Figure()
    
    # Curva Actual
    fig.add_trace(
        go.Scatter(
            x=[f"{t}D" for t in current_curve.tenors],
            y=current_curve.rates,
            mode="lines+markers",
            name=f"Fixing Actual ({current_curve.date.strftime('%d-%m-%Y')})",
            line=dict(color="#000000", width=3.5),
            marker=dict(size=9, symbol="diamond"),
            hovertemplate="<b>Plazo:</b> %{x}<br><b>Tasa Actual:</b> %{y:.2f}%<extra></extra>",
        )
    )
    
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    for idx, (h_name, p_curve) in enumerate(projected_curves.items()):
        color = colors[idx % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=[f"{t}D" for t in p_curve.tenors],
                y=p_curve.rates,
                mode="lines+markers",
                name=f"{h_name} ({p_curve.date.strftime('%d-%m-%Y')})",
                line=dict(color=color, width=2, dash="dash"),
                marker=dict(size=6),
                hovertemplate=f"<b>{h_name}:</b> %{{y:.2f}}%<extra></extra>",
            )
        )
    
    fig.update_layout(
        title="<b>Estructura Temporal de Tasas TAB UF (Curva de Rendimiento)</b>",
        xaxis_title="Plazo (Tenor)",
        yaxis_title="Tasa de Interés Anual (%)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig

def create_spreads_history_figure(historical_df: pd.DataFrame) -> go.Figure:
    """Evolución histórica de los spreads de pendiente (360d - 90d) y curvatura."""
    fig = go.Figure()
    
    sub_df = historical_df.iloc[-min(len(historical_df), 500):].copy()
    slope = sub_df["TAB_UF_360"] - sub_df["TAB_UF_90"]
    butterfly = 2.0 * sub_df["TAB_UF_180"] - (sub_df["TAB_UF_90"] + sub_df["TAB_UF_360"])
    
    fig.add_trace(
        go.Scatter(
            x=sub_df["Date"],
            y=slope,
            mode="lines",
            name="Pendiente (Slope: 360d - 90d)",
            line=dict(color="#1f77b4", width=2),
            hovertemplate="<b>Fecha:</b> %{x|%d-%m-%Y}<br><b>Pendiente:</b> %{y:.2f}%<extra></extra>",
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=sub_df["Date"],
            y=butterfly,
            mode="lines",
            name="Curvatura (Butterfly: 2*180d - 90d - 360d)",
            line=dict(color="#ff7f0e", width=1.5, dash="dot"),
            hovertemplate="<b>Fecha:</b> %{x|%d-%m-%Y}<br><b>Curvatura:</b> %{y:.2f}%<extra></extra>",
        )
    )
    
    fig.add_hline(y=0.0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title="<b>Evolución Histórica de Spreads de Curva (Slope y Butterfly)</b>",
        xaxis_title="Fecha",
        yaxis_title="Diferencial (Puntos %)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig
