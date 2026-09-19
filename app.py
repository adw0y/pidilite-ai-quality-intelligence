"""
Pidilite AI Quality Intelligence Platform — Streamlit Dashboard
Integrates Projects 1, 2, 3, 8, 9, 10, 11, 13, 15.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pidilite AI — Quality Intelligence",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports (lazy, with error handling) ──────────────────────────────────
import config as cfg
import data_pipeline as dp

# ── Sidebar ──────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/test-tube.png", width=60)
st.sidebar.title("Pidilite AI")
st.sidebar.caption("Quality Intelligence Platform")
st.sidebar.divider()

# Gemini API key (for AI Explainer tab)
gemini_key = st.sidebar.text_input("Gemini API Key (optional)", type="password",
                                    help="Required for the AI Explainer tab")
st.sidebar.divider()
st.sidebar.markdown("**Data:** 200 batches (100 A + 100 B)")
st.sidebar.markdown("**Plant:** Pidilite Vizag-1")
st.sidebar.markdown("**Products:** PVAc Emulsion Adhesives")

# ── Load data (cached) ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    dp.load_feature_table.cache_clear()
    ft_df = dp.load_feature_table()
    ft_df["reactor_date"] = pd.to_datetime(ft_df["reactor_date"], errors="coerce")
    return {
        "ft":      ft_df,
        "reactor": dp.load_reactor_bct(),
        "qc":      dp.load_qc_results(),
    }

data = load_data()
ft      = data["ft"]
reactor = data["reactor"]
qc      = data["qc"]

# ── Tabs ─────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Overview & EDA",
    "📈 BCT Variance",
    "🏆 Golden Batch",
    "🤖 Quality Prediction",
    "🔄 Transfer Learning",
    "⏱️ Transfer Time",
    "💬 AI Explainer",
])

# ═══════════════════════════════════════════════════════════════════════
# TAB 1: Overview & EDA
# ═══════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.header("📊 Overview & Exploratory Data Analysis")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Batches", len(ft))
    col2.metric("Product-A", len(ft[ft["product"] == "Product-A"]))
    col3.metric("Product-B", len(ft[ft["product"] == "Product-B"]))
    col4.metric("Features", len(reactor.columns) - 4)  # minus meta cols

    st.subheader("Quality Target Distributions")
    prod_filter = st.selectbox("Filter by Product", ["All", "Product-A", "Product-B"], key="eda_prod")
    df_plot = ft if prod_filter == "All" else ft[ft["product"] == prod_filter]

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df_plot, x="reactor_ipqc_solid_pct", color="product",
                           nbins=25, title="Reactor IPQC Solid Content %",
                           barmode="overlay", opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(df_plot, x="reactor_ipqc_viscosity", color="product",
                           nbins=25, title="Reactor IPQC Viscosity (cP)",
                           barmode="overlay", opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Product A vs B — Box Plots")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(ft, x="product", y="reactor_ipqc_solid_pct",
                     color="product", title="Solid Content % by Product", points="all")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.box(ft, x="product", y="reactor_ipqc_viscosity",
                     color="product", title="Viscosity (cP) by Product", points="all")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Batch Timeline & Quality Trajectory")
    st.caption("Viscosity over chronological batch production date. Bubble size reflects Total BCT (min). Hover over points to view exact Batch # and metrics.")
    
    # Ensure clean chronological sorting
    ft["reactor_date"] = pd.to_datetime(ft["reactor_date"], errors="coerce")
    ft_sorted = ft.dropna(subset=["reactor_date", "reactor_ipqc_viscosity"]).sort_values(by=["reactor_date", "batch_no"]).copy()
    ft_sorted["date_display"] = ft_sorted["reactor_date"].dt.strftime("%d %b %Y")
    
    fig_timeline = px.scatter(
        ft_sorted,
        x="reactor_date",
        y="reactor_ipqc_viscosity",
        color="product",
        size="reactor_total_bct_min",
        size_max=16,
        hover_name="product",
        hover_data={
            "batch_no": True,
            "date_display": True,
            "reactor_date": False,
            "reactor_ipqc_viscosity": ":.1f",
            "reactor_total_bct_min": ":.0f min",
            "reactor_ipqc_solid_pct": ":.2f %",
        },
        labels={
            "reactor_date": "Batch Date",
            "reactor_ipqc_viscosity": "Reactor IPQC Viscosity (cP)",
            "product": "Product",
            "reactor_total_bct_min": "Total BCT (min)",
            "batch_no": "Batch #",
            "date_display": "Date",
            "reactor_ipqc_solid_pct": "Solid Content %"
        },
        title="Reactor Viscosity Over Time (Bubble Size = Total BCT min)"
    )
    fig_timeline.update_xaxes(
        tickformat="%b %Y",
        dtick="M1",
        showgrid=True,
        title_text="Manufacturing Date (2026)"
    )
    fig_timeline.update_yaxes(
        showgrid=True,
        title_text="Viscosity (cP)"
    )
    fig_timeline.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    st.plotly_chart(fig_timeline, use_container_width=True)

    st.subheader("Statistical Variance & Inferential Metrics")
    st.caption("Key dispersion, shape, and stability indicators across products. Variance of Product-B viscosity is 20× higher than Product-A.")
    
    from scipy import stats as sp_stats
    metrics_summary = []
    for prod in ["Product-A", "Product-B"]:
        sub = ft[ft["product"] == prod]
        for col_name, label in [
            ("reactor_ipqc_viscosity", "Viscosity (cP)"),
            ("reactor_ipqc_solid_pct", "Solid Content %"),
            ("reactor_total_bct_min", "Total Cycle Time (min)"),
            ("reactor_act_batch_size", "Actual Yield (kg)")
        ]:
            ser = sub[col_name].dropna()
            if len(ser) > 0:
                mean_val = ser.mean()
                std_val = ser.std()
                var_val = ser.var()
                cv_val = (std_val / mean_val) * 100 if mean_val != 0 else 0
                q75, q25 = ser.quantile(0.75), ser.quantile(0.25)
                iqr_val = q75 - q25
                skew_val = sp_stats.skew(ser)
                kurt_val = sp_stats.kurtosis(ser)
                
                metrics_summary.append({
                    "Product": prod,
                    "Parameter": label,
                    "Count": len(ser),
                    "Mean": round(mean_val, 2),
                    "Std Dev": round(std_val, 2),
                    "Variance": round(var_val, 2),
                    "CV (%)": round(cv_val, 2),
                    "IQR": round(iqr_val, 2),
                    "Skewness": round(skew_val, 3),
                    "Kurtosis": round(kurt_val, 3),
                    "Min": round(ser.min(), 2),
                    "Max": round(ser.max(), 2),
                })
    df_var_table = pd.DataFrame(metrics_summary)
    st.dataframe(df_var_table, use_container_width=True)

    st.subheader("Raw Data Summary")
    st.dataframe(ft.describe().round(2), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════
# TAB 2: BCT Variance Analysis (Project 2)
# ═══════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("📈 BCT Variance Analysis")
    import bct_analysis as bct

    prod2 = st.selectbox("Product", ["Product-A", "Product-B", None],
                         format_func=lambda x: x if x else "Both Products", key="bct_prod")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Step Variance Ranking (CV%)")
        try:
            fig = bct.plot_variance_ranking(prod2)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")

    with c2:
        st.subheader("Variance Decomposition")
        try:
            fig = bct.plot_variance_decomposition(prod2)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")

    st.subheader("Step–Quality Correlation Analysis")
    st.caption("Linear relationship (Pearson r) between individual reactor step durations and the resulting lab quality targets. Values close to +1 or -1 indicate strong predictive influence.")
    
    col_heat, col_table = st.columns([1.1, 1.0])
    with col_heat:
        try:
            fig = bct.plot_correlation_heatmap(prod2)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")
            
    with col_table:
        st.markdown("**Correlation Breakdown Table**")
        try:
            df_corr = bct.compute_quality_correlations(prod2)
            if df_corr.empty:
                st.info("No correlation data available for the selected view.")
            else:
                df_corr_disp = df_corr.copy()
                
                # Determine solids and viscosity column names dynamically
                visc_col = next((c for c in df_corr.columns if "viscos" in str(c).lower()), None)
                solid_col = next((c for c in df_corr.columns if "solid" in str(c).lower()), None)
                
                if visc_col:
                    df_corr_disp["Viscosity Impact"] = df_corr_disp[visc_col].apply(
                        lambda r: "🔺 Increases Viscosity" if r > 0.15 else ("🔻 Decreases Viscosity" if r < -0.15 else "Neutral")
                    )
                if solid_col:
                    df_corr_disp["Solids Impact"] = df_corr_disp[solid_col].apply(
                        lambda r: "🔺 Increases Solids" if r > 0.15 else ("🔻 Decreases Solids" if r < -0.15 else "Neutral")
                    )
                
                # Format numeric subset
                num_cols = [c for c in [solid_col, visc_col] if c is not None]
                style_fmt = {c: "{:+.3f}" for c in num_cols}
                
                st.dataframe(
                    df_corr_disp.style.format(style_fmt).background_gradient(subset=num_cols, cmap="RdBu_r", vmin=-0.4, vmax=0.4),
                    use_container_width=True,
                    height=520
                )
        except Exception as e:
            st.error(f"Table error: {e}")

    st.subheader("Process Step Variance Table")
    st.caption("Step durations sorted by Coefficient of Variation (CV%). Higher CV indicates greater operational inconsistency.")
    try:
        df_var = bct.compute_variance_ranking(prod2)
        st.dataframe(df_var.round(3), use_container_width=True)
    except Exception as e:
        st.error(f"Error: {e}")

# ═══════════════════════════════════════════════════════════════════════
# TAB 3: Golden Batch (Project 3)
# ═══════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("🏆 Golden Batch Profiling")
    import golden_batch as gb

    prod3 = st.selectbox("Product", ["Product-A", "Product-B"], key="gb_prod")
    top_pct = st.slider("Golden Batch Percentile", 0.05, 0.30, 0.15, 0.05, key="gb_pct")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Golden Profile vs Overall")
        try:
            fig = gb.plot_golden_profile(prod3, top_pct=top_pct)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")

    with c2:
        st.subheader("Similarity Score Distribution")
        try:
            fig = gb.plot_similarity_distribution(prod3, top_pct=top_pct)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")

    st.subheader("Batch Clusters — PCA")
    try:
        fig = gb.plot_pca_scatter(prod3, top_pct=top_pct)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error: {e}")

    st.subheader("Batch Clusters — t-SNE")
    try:
        fig = gb.plot_tsne_scatter(prod3, top_pct=top_pct)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error: {e}")

# ═══════════════════════════════════════════════════════════════════════
# TAB 4: Quality Prediction (Project 1)
# ═══════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("🤖 Quality Prediction Models")
    import quality_models as qm

    c1, c2, c3 = st.columns(3)
    prod4 = c1.selectbox("Product", ["Product-A", "Product-B"], key="qm_prod")
    target4 = c2.selectbox("Target", ["reactor_ipqc_viscosity", "reactor_ipqc_solid_pct"],
                           format_func=lambda x: "Viscosity (cP)" if "viscosity" in x else "Solid Content %",
                           key="qm_target")
    model4 = c3.selectbox("Algorithm", ["xgboost", "lightgbm", "random_forest", "ridge"], key="qm_model")

    if st.button("🚀 Train Model", key="qm_train"):
        with st.spinner("Training model..."):
            try:
                result = qm.train_model(prod4, target4, model4)
                st.session_state["model_result"] = result

                # Metrics
                st.subheader("Model Performance")
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("MAPE", f"{result['metrics']['MAPE']:.4f}")
                mc2.metric("RMSE", f"{result['metrics']['RMSE']:.2f}")
                mc3.metric("R²", f"{result['metrics']['R2']:.4f}")
                mc4.metric("MAE", f"{result['metrics']['MAE']:.2f}")

                # Plots
                col_a, col_b = st.columns(2)
                with col_a:
                    fig = qm.plot_predicted_vs_actual(result)
                    st.plotly_chart(fig, use_container_width=True)
                with col_b:
                    fig = qm.plot_feature_importance(result)
                    st.plotly_chart(fig, use_container_width=True)

                # Cross-validation
                st.subheader("5-Fold Cross-Validation")
                cv_result = qm.cross_validate_model(prod4, target4, model4)
                cv_df = pd.DataFrame(cv_result["fold_metrics"])
                st.dataframe(cv_df.round(4), use_container_width=True)
                st.info(f"**Mean MAPE:** {cv_result['mean_metrics']['MAPE']:.4f} ± {cv_result['std_metrics']['MAPE']:.4f}")

                # MLflow logging
                qm.log_to_mlflow(result, cfg.MLFLOW_EXPERIMENT)
                st.success("✅ Model logged to MLflow")
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())

    # Interactive predictor
    st.divider()
    st.subheader("🔮 Single Batch Predictor")
    if "model_result" in st.session_state:
        result = st.session_state["model_result"]
        st.caption("Adjust step durations and see the predicted quality:")
        feat_cols = result["feature_names"]
        # Use median values as defaults
        features_df = dp.build_reactor_features(prod4)
        defaults = features_df[feat_cols].median()

        input_features = {}
        cols = st.columns(3)
        for i, feat in enumerate(feat_cols[:15]):  # show top 15
            with cols[i % 3]:
                val = st.number_input(feat, value=float(defaults.get(feat, 0)),
                                      key=f"pred_{feat}")
                input_features[feat] = val
        # Fill remaining features with defaults
        for feat in feat_cols[15:]:
            input_features[feat] = float(defaults.get(feat, 0))

        if st.button("Predict", key="qm_predict_btn"):
            try:
                pred = qm.predict_single(result, input_features)
                pc1, pc2, pc3 = st.columns(3)
                pc1.metric("Predicted Value", f"{pred['prediction']:.2f}")
                pc2.metric("Risk Level", pred['risk_level'].upper(),
                          delta="⚠️" if pred['risk_level'] != 'low' else "✅")
                st.subheader("Top Drivers")
                for d in pred["top_drivers"]:
                    icon = "🔴" if d["direction"] == "increase" else "🟢"
                    st.write(f"{icon} **{d['feature']}** = {d['value']:.1f} → {d['direction']}s risk")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.info("Train a model first to use the predictor.")

# ═══════════════════════════════════════════════════════════════════════
# TAB 5: Transfer Learning (Project 9)
# ═══════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.header("🔄 Cross-Product Transfer Learning")
    import transfer_learning as tl

    target5 = st.selectbox("Target", ["reactor_ipqc_viscosity", "reactor_ipqc_solid_pct"],
                           format_func=lambda x: "Viscosity (cP)" if "viscosity" in x else "Solid Content %",
                           key="tl_target")

    if st.button("🚀 Run Transfer Learning Experiment", key="tl_run"):
        with st.spinner("Running all approaches (this may take a minute)..."):
            try:
                comparison = tl.compare_all_approaches(target5)
                st.subheader("Results Comparison")
                st.dataframe(comparison.round(4), use_container_width=True)

                fig = tl.plot_comparison(target5)
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Feature Importance: Product-A vs Product-B")
                fig2 = tl.plot_feature_importance_comparison(target5)
                st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())

    st.divider()
    st.markdown("""
    **What this tests:**
    - Can a model trained on Product-A predict Product-B quality?
    - Does fine-tuning an A-trained model with a small B sample beat training on B alone?
    - If yes → new products can bootstrap from existing models with minimal data.
    """)

# ═══════════════════════════════════════════════════════════════════════
# TAB 6: Transfer Time Analysis (Project 8)
# ═══════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.header("⏱️ Transfer Time Root-Cause Analysis")
    import transfer_analysis as ta

    prod6 = st.selectbox("Product", ["Product-A", "Product-B", None],
                         format_func=lambda x: x if x else "Both Products", key="ta_prod")

    # Distribution stats
    st.subheader("Transfer Time Distribution")
    try:
        stats = ta.analyze_transfer_distribution(prod6)
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Mean", f"{stats['mean']:.1f} min")
        mc2.metric("Median", f"{stats['median']:.1f} min")
        mc3.metric("Std Dev", f"{stats['std']:.1f} min")
        mc4.metric("Range", f"{stats['min']:.0f}–{stats['max']:.0f} min")
    except Exception as e:
        st.error(f"Stats error: {e}")

    c1, c2 = st.columns(2)
    with c1:
        try:
            fig = ta.plot_transfer_distribution(prod6)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")

    with c2:
        try:
            fig = ta.plot_transfer_correlations(prod6)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")

# ═══════════════════════════════════════════════════════════════════════
# TAB 7: AI Explainer (Project 13)
# ═══════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.header("💬 AI Batch Explanation Agent")
    import explanation_agent as ea

    if not gemini_key:
        st.warning("⚠️ Enter your Gemini API key in the sidebar to enable AI explanations. "
                   "Without it, rule-based explanations will be used.")

    if "model_result" not in st.session_state:
        st.info("👈 First go to the **Quality Prediction** tab and train a model.")
    else:
        result = st.session_state["model_result"]
        prod8 = result["product"]
        target8 = result["target"]

        st.subheader(f"Explain a {prod8} batch ({target8})")

        # Select a test batch
        n_test = len(result["X_test"])
        batch_idx = st.slider("Select test batch index", 0, n_test - 1, 0, key="ai_batch")

        # Get batch data
        batch_features = result["X_test"].iloc[batch_idx].to_dict()
        actual = result["y_test"].iloc[batch_idx]
        predicted = result["y_pred"][batch_idx]

        # Show batch info
        c1, c2, c3 = st.columns(3)
        c1.metric("Actual", f"{actual:.2f}")
        c2.metric("Predicted", f"{predicted:.2f}")
        c3.metric("Error", f"{abs(actual - predicted):.2f}")

        # Get SHAP drivers
        try:
            pred_result = qm.predict_single(result, batch_features)
            shap_drivers = pred_result["top_drivers"]
            risk = pred_result["risk_level"]

            # Build explanation inputs
            prediction_info = {
                "predicted_value": predicted,
                "target_name": target8,
                "risk_level": risk,
            }

            if st.button("🤖 Generate Explanation", key="ai_explain"):
                with st.spinner("Generating explanation..."):
                    try:
                        explanation = ea.explain_batch(
                            batch_data=batch_features,
                            prediction=prediction_info,
                            shap_drivers=shap_drivers,
                            api_key=gemini_key if gemini_key else None,
                        )
                        st.markdown("### 📝 Explanation")
                        st.markdown(explanation)
                        if not gemini_key:
                            st.caption("ℹ️ Rule-based explanation (add Gemini API key for AI-generated)")
                    except Exception as e:
                        st.error(f"Error: {e}")
        except Exception as e:
            st.error(f"Error getting SHAP drivers: {e}")
