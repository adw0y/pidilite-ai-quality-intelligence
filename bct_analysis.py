"""
Module 1: BCT Variance Analysis (Project 2)
Functions for analyzing Batch Cycle Time variance and correlation with quality.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import config as cfg
import data_pipeline as dp

# Columns to ignore as per instructions
DROP_COLS = ["Start of Batch", "Holding-3", "Sampling"]
ID_COLS = ["product", "batch_no", "Total BCT", "Act Production"]

def _get_step_cols(df):
    cols_to_exclude = set(DROP_COLS + ID_COLS)
    return [col for col in df.columns if col not in cols_to_exclude and df[col].dtype in [np.float64, np.int64]]

def compute_variance_ranking(product=None):
    """
    Computes ranking of process steps by Coefficient of Variation (CV)
    Returns DataFrame with [step, mean, std, cv_pct, min, max]
    """
    df = dp.load_reactor_bct()
    if product:
        df = df[df['product'] == product]
    
    # Drop unwanted columns if they exist
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors='ignore')
    
    step_cols = _get_step_cols(df)
    
    stats = []
    for step in step_cols:
        s = df[step].dropna()
        if len(s) == 0:
            continue
        mean = s.mean()
        std = s.std()
        cv_pct = (std / mean * 100) if mean != 0 else np.nan
        stats.append({
            'step': step,
            'mean': mean,
            'std': std,
            'cv_pct': cv_pct,
            'min': s.min(),
            'max': s.max()
        })
    
    if not stats:
        return pd.DataFrame(columns=['step', 'mean', 'std', 'cv_pct', 'min', 'max'])
        
    stats_df = pd.DataFrame(stats).sort_values('cv_pct', ascending=False).reset_index(drop=True)
    return stats_df

def compute_quality_correlations(product=None):
    """
    Computes correlation matrix between physical reactor step durations and IPQC quality targets.
    """
    df = dp.build_reactor_features(product)
    
    quality_cols = ['reactor_ipqc_solid_pct', 'reactor_ipqc_viscosity']
    quality_cols = [c for c in quality_cols if c in df.columns]
    
    if not quality_cols:
        return pd.DataFrame()
        
    candidate_steps = [
        "Charge Pre Intermediate", "Temprature adjustment",
        "RM-1 (L) charge in reactor", "RM-2 (S) charge in reactor",
        "RM-3 (S) charge in reactor", "RM-4 (S) charge in reactor",
        "Seeding", "Holding-1", "RM-5 charge", "Continous feed -mono",
        "Wait for set  temperature reach", "RM-6 charge", "Holding-2",
        "RM-7 Charge", "RM-8 Charge", "Transfer to blender",
        "Total BCT", "Act Production", "yield_ratio", "holding_total"
    ]
    step_cols = [c for c in candidate_steps if c in df.columns]
    
    corr_matrix = pd.DataFrame(index=step_cols, columns=["Solids % (IPQC)", "Viscosity cP (IPQC)"])
    for step in step_cols:
        for qcol, qlabel in [("reactor_ipqc_solid_pct", "Solids % (IPQC)"), ("reactor_ipqc_viscosity", "Viscosity cP (IPQC)")]:
            if qcol in df.columns:
                corr_val = df[step].corr(df[qcol])
                corr_matrix.loc[step, qlabel] = corr_val
                
    return corr_matrix.astype(float).dropna(how='all')

def compute_variance_decomposition(product=None):
    """
    Shows each step's contribution to Total BCT variance.
    Returns DataFrame with [step, variance, variance_pct]
    """
    df = dp.load_reactor_bct()
    if product:
        df = df[df['product'] == product]
        
    step_cols = _get_step_cols(df)
    
    variances = df[step_cols].var().dropna()
    total_var = variances.sum()
    
    var_df = variances.reset_index()
    var_df.columns = ['step', 'variance']
    var_df['variance_pct'] = (var_df['variance'] / total_var) * 100
    
    return var_df.sort_values('variance_pct', ascending=False).reset_index(drop=True)

def plot_variance_ranking(product=None):
    """
    Returns plotly horizontal bar chart of CV by step.
    """
    df = compute_variance_ranking(product)
    if df.empty:
        return go.Figure()
        
    fig = px.bar(
        df,
        x='cv_pct',
        y='step',
        orientation='h',
        title=f"Step Duration Variability (CV %) {f'- {product}' if product else ''}",
        labels={'cv_pct': 'Coefficient of Variation (%)', 'step': 'Process Step'},
        color='cv_pct',
        color_continuous_scale='Reds'
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    return fig

def plot_correlation_heatmap(product=None):
    """
    Returns an annotated, readable plotly heatmap of step-quality correlations.
    """
    corr_df = compute_quality_correlations(product)
    if corr_df.empty:
        return go.Figure()
        
    fig = px.imshow(
        corr_df,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale='RdBu_r',
        zmin=-0.5, zmax=0.5,
        title=f"Correlation Matrix: Process Steps vs Quality Targets {f'({product})' if product else ''}"
    )
    fig.update_layout(
        height=550,
        margin=dict(l=160, r=40, t=50, b=40),
        xaxis_title="Quality Target",
        yaxis_title="Process Step"
    )
    return fig

def plot_variance_decomposition(product=None):
    """
    Returns plotly pie chart of variance decomposition.
    """
    df = compute_variance_decomposition(product)
    if df.empty:
        return go.Figure()
        
    fig = px.pie(
        df,
        values='variance_pct',
        names='step',
        title=f"Contribution to Total BCT Variance {f'- {product}' if product else ''}",
        hole=0.4
    )
    return fig
