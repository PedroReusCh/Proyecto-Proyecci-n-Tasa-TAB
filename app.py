"""
Sistema de Proyección Financiera de Tasas TAB UF (Chilean Financial Benchmarks).
Aplicación Web Interactiva en Streamlit.
"""
import warnings
warnings.filterwarnings("ignore")

from typing import Dict
import numpy as np
import pandas as pd
import streamlit as st

from src.cbf_client import get_clean_tab_uf_dataset, CBFDataResult
from src.forecasting_engine import (
    generate_all_forecasts,
    HORIZONS_BUSINESS_DAYS,
    ForecastResult,
)
from src.evaluator import evaluate_tournament, TournamentResult
from src.monte_carlo import (
    calibrate_vasicek_params,
    run_monte_carlo_simulation,
    MonteCarloResult,
)
from src.term_structure import (
    compute_term_structure,
    compute_projected_curves,
    TermStructureCurve,
)
from src.diagnostics import (
    run_adf_test,
    run_ljung_box_test,
    compute_acf_pacf,
)
from src.stress_testing import (
    simulate_stress_scenarios,
)
from src.ui_components import (
    create_forecast_figure,
    create_tournament_comparison_figure,
    create_monte_carlo_fan_figure,
    create_monte_carlo_hist_figure,
    create_term_structure_figure,
    create_spreads_history_figure,
    create_acf_pacf_figure,
    create_residuals_diagnostics_figure,
    create_stress_testing_figure,
)
from src.exporter import generate_excel_report, generate_csv_report

# Configuración de Página
st.set_page_config(
    page_title="Proyección de Tasas TAB UF | CBF",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilos CSS Personalizados
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #1E3A8A;
    }
    .winner-badge {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data(show_spinner="Descargando datos históricos desde CBF...", ttl=3600)
def load_cbf_data() -> CBFDataResult:
    """Descarga y cachea los datos oficiales desde CBF."""
    return get_clean_tab_uf_dataset()

# Encabezado Principal
st.markdown('<div class="main-header">📈 Sistema de Proyección de Tasas TAB UF</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Ingesta automatizada desde <b>Chilean Financial Benchmarks (CBF)</b>, Torneo Multimodelo y Análisis Cuantitativo</div>',
    unsafe_allow_html=True,
)

# Carga de Datos
try:
    cbf_result = load_cbf_data()
    df_raw = cbf_result.df
except Exception as e:
    st.error(f"Error al conectar con CBF: {e}")
    st.stop()

# Panel Lateral de Configuración
with st.sidebar:
    st.header("⚙️ Configuración y Filtros")
    
    # Estado CBF
    st.info(
        f"**Último Fixing CBF:** {cbf_result.last_date.strftime('%d-%m-%Y')}\n\n"
        f"**Registros Históricos:** {cbf_result.records_count:,} días"
    )
    
    if st.button("🔄 Actualizar Datos CBF", help="Limpia la caché y consulta la web de CBF"):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    
    # Selector de Tenor
    tenor_labels = {
        "TAB_UF_90": "TAB UF 90 Días",
        "TAB_UF_180": "TAB UF 180 Días",
        "TAB_UF_360": "TAB UF 360 Días",
    }
    selected_col = st.selectbox(
        "🎯 Tasa Objetivo:",
        options=list(tenor_labels.keys()),
        format_func=lambda x: tenor_labels[x],
        index=0,
    )
    
    # Selector de Horizonte
    horizon_label = st.selectbox(
        "⏳ Horizonte de Proyección:",
        options=list(HORIZONS_BUSINESS_DAYS.keys()),
        index=3,  # 1 Año por defecto
    )
    horizon_days = HORIZONS_BUSINESS_DAYS[horizon_label]
    
    # Filtro Histórico de Visualización
    rango_opciones = {
        "Último Año": 252,
        "Últimos 3 Años": 252 * 3,
        "Últimos 5 Años": 252 * 5,
        "Histórico Completo (Desde 2010)": len(df_raw),
    }
    selected_range = st.selectbox("📅 Ventana Histórica:", options=list(rango_opciones.keys()), index=1)
    df_view = df_raw.iloc[-min(len(df_raw), rango_opciones[selected_range]):].copy()
    
    st.markdown("---")
    st.caption("Desarrollado para análisis financiero riguroso de curvas TAB UF.")

# Ejecución del Torneo y Modelado
with st.spinner("Ejecutando torneo de modelos cuantitativos y validación rodante..."):
    tournament = evaluate_tournament(df_raw, selected_col, horizon_days)
    all_forecasts = generate_all_forecasts(df_raw, selected_col, horizon_days, custom_weights=tournament.weights)

# Selector de Modelo para visualización
with st.sidebar:
    model_options = ["Ganador del Torneo (Automático)", "Ensemble Óptimo"] + [
        m for m in all_forecasts.keys() if m not in ["Ensemble"]
    ]
    selected_model_choice = st.selectbox("🤖 Modelo para Gráficos:", options=model_options, index=0)
    
    if selected_model_choice == "Ganador del Torneo (Automático)":
        active_forecast = all_forecasts.get(tournament.winner_model, all_forecasts["Ensemble"])
        active_model_name = f"Ganador ({tournament.winner_model})"
    elif selected_model_choice == "Ensemble Óptimo":
        active_forecast = all_forecasts["Ensemble"]
        active_model_name = "Ensemble"
    else:
        active_forecast = all_forecasts[selected_model_choice]
        active_model_name = selected_model_choice

# Pestañas Principales
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "📊 Resumen & Proyecciones",
        "🏆 Torneo de Modelos",
        "📈 Estructura Temporal",
        "🔬 Diagnóstico Econométrico",
        "⚡ Escenarios de Estrés",
        "🎲 Simulación Monte Carlo",
        "💾 Exportar Reporte",
    ]
)

# ----------------- TAB 1: RESUMEN Y PROYECCIONES -----------------
with tab1:
    st.subheader(f"Proyección de {tenor_labels[selected_col]} a {horizon_label}")
    
    # Tarjetas KPI
    col1, col2, col3, col4, col5 = st.columns(5)
    
    last_val = cbf_result.latest_rates[selected_col]
    prev_val = float(df_raw[selected_col].iloc[-2]) if len(df_raw) > 1 else last_val
    delta_1d = last_val - prev_val
    
    # Proyecciones clave
    p_1m = float(all_forecasts[active_forecast.model_name].point_forecast[min(20, horizon_days - 1)])
    p_3m = float(all_forecasts[active_forecast.model_name].point_forecast[min(62, horizon_days - 1)])
    p_6m = float(all_forecasts[active_forecast.model_name].point_forecast[min(125, horizon_days - 1)])
    p_12m = float(all_forecasts[active_forecast.model_name].point_forecast[-1])
    
    with col1:
        st.metric(
            label=f"Último Fixing ({cbf_result.last_date.strftime('%d/%m/%Y')})",
            value=f"{last_val:.2f}%",
            delta=f"{delta_1d:+.2f}% (1D)",
        )
    with col2:
        st.metric(label="Proyección +1 Mes", value=f"{p_1m:.2f}%", delta=f"{p_1m - last_val:+.2f}%")
    with col3:
        st.metric(label="Proyección +3 Meses", value=f"{p_3m:.2f}%", delta=f"{p_3m - last_val:+.2f}%")
    with col4:
        st.metric(label="Proyección +6 Meses", value=f"{p_6m:.2f}%", delta=f"{p_6m - last_val:+.2f}%")
    with col5:
        st.metric(label="Proyección +1 Año", value=f"{p_12m:.2f}%", delta=f"{p_12m - last_val:+.2f}%")
    
    st.markdown("---")
    
    # Gráfico Principal
    fig_main = create_forecast_figure(
        df_view,
        selected_col,
        active_forecast,
        target_name=tenor_labels[selected_col],
    )
    st.plotly_chart(fig_main)
    
    # Tabla de Resumen de Proyecciones Clave
    st.markdown("### 📋 Cuadro Detallado de Proyecciones por Hito")
    milestones = [
        ("1 Mes (21 días hábiles)", min(20, horizon_days - 1)),
        ("3 Meses (63 días hábiles)", min(62, horizon_days - 1)),
        ("6 Meses (126 días hábiles)", min(125, horizon_days - 1)),
        ("1 Año (252 días hábiles)", horizon_days - 1),
    ]
    
    table_rows = []
    for m_label, step_idx in milestones:
        if step_idx < horizon_days:
            f_date = active_forecast.future_dates[step_idx].strftime("%d-%m-%Y")
            val_pt = active_forecast.point_forecast[step_idx]
            l80, u80 = active_forecast.lower_80[step_idx], active_forecast.upper_80[step_idx]
            l95, u95 = active_forecast.lower_95[step_idx], active_forecast.upper_95[step_idx]
            table_rows.append({
                "Hito": m_label,
                "Fecha Proyectada": f_date,
                "Tasa Proyectada (%)": f"{val_pt:.2f}%",
                "Rango 80% (P10 - P90)": f"[{l80:.2f}% - {u80:.2f}%]",
                "Rango 95%": f"[{l95:.2f}% - {u95:.2f}%]",
                "Variación vs Actual": f"{val_pt - last_val:+.2f}%",
            })
    
    st.dataframe(pd.DataFrame(table_rows), hide_index=True)

# ----------------- TAB 2: TORNEO DE MODELOS -----------------
with tab2:
    st.subheader(f"🏆 Torneo de Modelos Cuantitativos ({tenor_labels[selected_col]})")
    st.markdown(
        f"Evaluación rigurosa mediante **Walk-Forward Validation (Backtesting Rodante)** sin fuga de datos futuros. "
        f"El modelo ganador es **{tournament.winner_model}** con el menor error cuadrático medio (RMSE)."
    )
    
    # Tabla de Ranking
    tourn_df = pd.DataFrame([
        {
            "Posición": f"#{s.rank}" if s.rank > 1 else "🥇 #1 (Ganador)",
            "Modelo": s.model_name,
            "RMSE (Error Medio Cuadrático)": f"{s.rmse:.4f}",
            "MAE (Error Absoluto Medio)": f"{s.mae:.4f}",
            "MAPE (%)": f"{s.mape:.2f}%",
            "Precisión Direccional (%)": f"{s.directional_accuracy:.1f}%",
            "Ponderación en Ensemble": f"{tournament.weights.get(s.model_name, 0.0)*100:.1f}%" if s.model_name != "Ensemble" else "-",
        }
        for s in tournament.scores
    ])
    st.dataframe(tourn_df, hide_index=True)
    
    st.markdown("---")
    st.subheader("📊 Comparación Gráfica de Todos los Modelos en Competencia")
    fig_comp = create_tournament_comparison_figure(all_forecasts, target_name=tenor_labels[selected_col])
    st.plotly_chart(fig_comp)

# ----------------- TAB 3: ESTRUCTURA TEMPORAL -----------------
with tab3:
    st.subheader("📈 Estructura Temporal de Tasas TAB UF (Curva de Rendimiento)")
    
    # Calcular curvas de los 3 tenores con el ensemble
    fc_90 = generate_all_forecasts(df_raw, "TAB_UF_90", horizon_days)["Ensemble"]
    fc_180 = generate_all_forecasts(df_raw, "TAB_UF_180", horizon_days)["Ensemble"]
    fc_360 = generate_all_forecasts(df_raw, "TAB_UF_360", horizon_days)["Ensemble"]
    
    current_curve = compute_term_structure(df_raw)
    projected_curves = compute_projected_curves(current_curve, fc_90, fc_180, fc_360)
    
    fig_curve = create_term_structure_figure(current_curve, projected_curves)
    st.plotly_chart(fig_curve)
    
    st.markdown("---")
    st.subheader("📉 Spreads de Plazo: Pendiente (Slope) y Curvatura (Butterfly)")
    
    c1, c2 = st.columns(2)
    with c1:
        st.metric(
            label="Pendiente Actual (TAB UF 360d - 90d)",
            value=f"{current_curve.slope_360_90:+.2f}%",
            help="Diferencial entre tasa a 1 año y tasa a 90 días.",
        )
    with c2:
        st.metric(
            label="Curvatura Actual (Butterfly 180d)",
            value=f"{current_curve.butterfly_180:+.2f}%",
            help="Medida de concavidad/convexidad de la curva.",
        )
    
    fig_spreads = create_spreads_history_figure(df_view)
    st.plotly_chart(fig_spreads)

# ----------------- TAB 4: DIAGNÓSTICO ECONOMÉTRICO -----------------
with tab4:
    st.subheader(f"🔬 Diagnóstico Econométrico y Verificación de Residuos ({tenor_labels[selected_col]})")
    st.markdown(
        "Evaluación de **estacionariedad** (Test ADF), **estructura autorregresiva** (ACF / PACF) "
        "y **diagnóstico de ruido blanco** en los residuos del modelo predictivo (Test de Ljung-Box)."
    )
    
    # 1. Test de Dickey-Fuller Aumentado
    st.markdown("### 1. Test de Raíz Unitaria (Dickey-Fuller Aumentado - ADF)")
    col_adf1, col_adf2 = st.columns(2)
    
    series_level = df_raw[selected_col].values
    series_diff = np.diff(series_level)
    
    adf_level = run_adf_test(series_level)
    adf_diff = run_adf_test(series_diff)
    
    with col_adf1:
        st.markdown("**Serie en Nivel ($r_t$):**")
        st.write(f"- **Estadístico ADF:** `{adf_level.test_statistic:.4f}`")
        st.write(f"- **p-valor:** `{adf_level.p_value:.4f}`")
        st.write(f"- **Valor Crítico 5%:** `{adf_level.critical_values.get('5%', 0):.4f}`")
        if adf_level.is_stationary:
            st.success("✅ " + adf_level.conclusion)
        else:
            st.warning("⚠️ " + adf_level.conclusion)
            
    with col_adf2:
        st.markdown("**Serie en Primera Diferencia ($\Delta r_t = r_t - r_{t-1}$):**")
        st.write(f"- **Estadístico ADF:** `{adf_diff.test_statistic:.4f}`")
        st.write(f"- **p-valor:** `{adf_diff.p_value:.4e}`")
        st.write(f"- **Valor Crítico 5%:** `{adf_diff.critical_values.get('5%', 0):.4f}`")
        if adf_diff.is_stationary:
            st.success("✅ " + adf_diff.conclusion)
        else:
            st.warning("⚠️ " + adf_diff.conclusion)
    
    st.markdown("---")
    
    # 2. Correlogramas ACF / PACF
    st.markdown("### 2. Función de Autocorrelación (ACF) y Autocorrelación Parcial (PACF)")
    acf_vals, pacf_vals, conf_bound = compute_acf_pacf(series_level, nlags=20)
    fig_acf = create_acf_pacf_figure(acf_vals, pacf_vals, conf_bound, target_name=tenor_labels[selected_col])
    st.plotly_chart(fig_acf)
    
    st.markdown("---")
    
    # 3. Diagnóstico de Residuos del Modelo y Test de Ljung-Box
    st.markdown(f"### 3. Diagnóstico de Residuos del Modelo ({active_model_name})")
    
    # Calcular residuos sobre la ventana de backtesting
    eval_len = min(horizon_days, 63)
    train_slice = df_raw.iloc[:-eval_len]
    actual_test = df_raw[selected_col].iloc[-eval_len:].values
    
    # Residuos in-sample/out-sample aproximados
    if len(series_level) > 1:
        fitted_naive = series_level[1:]
        residuals = series_level[1:] - series_level[:-1]
    else:
        residuals = np.array([0.0])
        
    lb_result = run_ljung_box_test(residuals, lags=10)
    
    c_lb1, c_lb2 = st.columns([1, 2])
    with c_lb1:
        st.markdown("**Test de Ljung-Box ($H_0$: Ruido Blanco):**")
        st.write(f"- **Retardos Evaluados:** `{lb_result.lags}`")
        st.write(f"- **p-valor Mínimo:** `{min(lb_result.p_values):.4f}`")
        if lb_result.is_white_noise:
            st.success("✅ " + lb_result.conclusion)
        else:
            st.info("ℹ️ " + lb_result.conclusion)
    with c_lb2:
        df_lb = pd.DataFrame({
            "Retardo (Lag)": list(range(1, lb_result.lags + 1)),
            "Estadístico Q": np.round(lb_result.test_statistics, 3),
            "p-valor": np.round(lb_result.p_values, 4),
        })
        st.dataframe(df_lb, hide_index=True)
        
    fig_resid = create_residuals_diagnostics_figure(residuals, model_name=active_model_name)
    st.plotly_chart(fig_resid)

# ----------------- TAB 5: ESCENARIOS DE ESTRÉS (STRESS TESTING) -----------------
with tab5:
    st.subheader(f"⚡ Simulador de Estrés Financiero y Shocks Macro ({tenor_labels[selected_col]})")
    st.markdown(
        "Permite simular el impacto de un **shock de Política Monetaria o liquidez** "
        "y un incremento en la **volatilidad de mercado** sobre las proyecciones base."
    )
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        shock_val = st.slider(
            "💥 Magnitud del Shock de Tasa (Puntos Base):",
            min_value=-200,
            max_value=200,
            value=100,
            step=25,
            help="100 puntos base equivalen a un cambio de +1.00% anual.",
        )
    with col_s2:
        vol_multiplier = st.select_slider(
            "🌪️ Multiplicador de Volatilidad (Incertidumbre):",
            options=[1.0, 1.25, 1.5, 1.75, 2.0],
            value=1.25,
            help="Amplía los conos de dispersión en escenarios de alta volatilidad.",
        )
        
    stress_result = simulate_stress_scenarios(
        active_forecast,
        shock_bp=float(shock_val),
        vol_multiplier=float(vol_multiplier),
    )
    
    # Gráfico de Escenarios de Estrés
    fig_stress = create_stress_testing_figure(stress_result, target_name=tenor_labels[selected_col])
    st.plotly_chart(fig_stress)
    
    st.markdown("### 📋 Cuadro de Impacto de Estrés por Hito Temporal")
    st.dataframe(stress_result.impact_milestones, hide_index=True)

# ----------------- TAB 6: SIMULACIÓN MONTE CARLO -----------------
with tab6:
    st.subheader(f"🎲 Simulación Estocástica de Monte Carlo ({tenor_labels[selected_col]})")
    
    # Calibrar Vasicek
    kappa, theta, sigma = calibrate_vasicek_params(df_raw[selected_col].values)
    
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.metric("Velocidad de Reversión (κ)", f"{kappa:.3f}")
    with col_p2:
        st.metric("Media de Largo Plazo (θ)", f"{theta:.2f}%")
    with col_p3:
        st.metric("Volatilidad Anualizada (σ)", f"{sigma*100:.2f}%")
    
    mc_res = run_monte_carlo_simulation(
        last_date=cbf_result.last_date,
        initial_rate=last_val,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        horizon_days=horizon_days,
        n_sims=1000,
        seed=42,
    )
    
    fig_mc_fan = create_monte_carlo_fan_figure(mc_res, target_name=tenor_labels[selected_col])
    st.plotly_chart(fig_mc_fan)
    
    st.markdown("---")
    st.subheader(f"📊 Distribución Terminal de Tasas a {horizon_label}")
    fig_mc_hist = create_monte_carlo_hist_figure(mc_res.terminal_distribution, horizon_label=horizon_label)
    st.plotly_chart(fig_mc_hist)

# ----------------- TAB 7: EXPORTADOR DE REPORTES -----------------
with tab7:
    st.subheader("💾 Exportación de Reportes y Datos")
    st.markdown("Descarga los datos históricos oficiales de CBF, las proyecciones y las métricas del torneo en Excel o CSV.")
    
    # Preparar DataFrames para exportación
    df_proj_export = pd.DataFrame({
        "Fecha_Proyeccion": [d.strftime("%Y-%m-%d") for d in active_forecast.future_dates],
        "Tasa_Proyectada": np.round(active_forecast.point_forecast, 4),
        "Limite_Inferior_80": np.round(active_forecast.lower_80, 4),
        "Limite_Superior_80": np.round(active_forecast.upper_80, 4),
        "Limite_Inferior_95": np.round(active_forecast.lower_95, 4),
        "Limite_Superior_95": np.round(active_forecast.upper_95, 4),
    })
    
    df_tourn_export = pd.DataFrame([
        {
            "Ranking": s.rank,
            "Modelo": s.model_name,
            "RMSE": round(s.rmse, 6),
            "MAE": round(s.mae, 6),
            "MAPE_pct": round(s.mape, 4),
            "Precision_Direccional_pct": round(s.directional_accuracy, 2),
        }
        for s in tournament.scores
    ])
    
    df_mc_export = pd.DataFrame({
        "Fecha": [d.strftime("%Y-%m-%d") for d in mc_res.future_dates],
        "P10": np.round(mc_res.percentiles["P10"], 4),
        "P25": np.round(mc_res.percentiles["P25"], 4),
        "P50_Mediana": np.round(mc_res.percentiles["P50"], 4),
        "P75": np.round(mc_res.percentiles["P75"], 4),
        "P90": np.round(mc_res.percentiles["P90"], 4),
    })
    
    excel_data = generate_excel_report(
        df_hist=df_raw,
        df_projections=df_proj_export,
        df_tournament=df_tourn_export,
        df_monte_carlo=df_mc_export,
    )
    
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            label="📥 Descargar Reporte Completo en Excel (.xlsx)",
            data=excel_data,
            file_name=f"Proyeccion_TAB_UF_{selected_col}_{cbf_result.last_date.strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with c2:
        csv_data = generate_csv_report(df_proj_export)
        st.download_button(
            label="📥 Descargar Proyecciones en CSV (.csv)",
            data=csv_data,
            file_name=f"Proyecciones_{selected_col}.csv",
            mime="text/csv",
        )
