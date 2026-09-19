"""
Module 2: Transfer Time Root-Cause Analysis (Project 8)
Functions for analyzing the 'Transfer to blender' step.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import config as cfg
import data_pipeline as dp

try:
    from lifelines import CoxPHFitter
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False

def get_transfer_col(df):
    target = 'Transfer to blender'
    return target if target in df.columns else None

def analyze_transfer_distribution(product=None):
    """
    Returns basic stats for the transfer time column.
    """
    df = dp.load_reactor_bct()
    if product:
        df = df[df['product'] == product]
        
    col = get_transfer_col(df)
    if not col:
        return {}
        
    s = df[col].dropna()
    if len(s) == 0:
        return {}
        
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    
    return {
        'mean': s.mean(),
        'std': s.std(),
        'median': s.median(),
        'skew': s.skew(),
        'Q1': q1,
        'Q3': q3,
        'IQR': q3 - q1
    }

def compute_transfer_correlations(product=None):
    """
    Returns DataFrame correlating Transfer time with all other step durations and quality.
    """
    df = dp.build_reactor_features(product)
    col = get_transfer_col(df)
    if not col:
        return pd.DataFrame()
        
    num_cols = df.select_dtypes(include=[np.number]).columns
    num_cols = [c for c in num_cols if c != col and c not in ["Start of Batch", "Holding-3", "Sampling"]]
    
    corrs = []
    for c in num_cols:
        valid_df = df[[col, c]].dropna()
        if len(valid_df) > 1:
            r = valid_df[col].corr(valid_df[c])
            corrs.append({'Feature': c, 'Correlation': r})
            
    corr_df = pd.DataFrame(corrs).dropna().sort_values('Correlation', key=abs, ascending=False).reset_index(drop=True)
    return corr_df

def segment_fast_slow(product=None, n_clusters=2):
    """
    Uses KMeans on step features to segment into fast vs slow clusters.
    """
    df = dp.build_reactor_features(product)
    
    # Select numerical step columns
    step_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                 if c not in ['batch_no', 'Start of Batch', 'Holding-3', 'Sampling'] 
                 and 'ipqc' not in c.lower()]
                 
    cluster_df = df[step_cols].dropna().copy()
    if len(cluster_df) < n_clusters:
        return pd.DataFrame(), pd.DataFrame()
        
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_df)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    cluster_df['cluster'] = kmeans.fit_predict(scaled_data)
    
    # Ensure cluster 0 is always the "faster" one on average for Transfer to blender if it exists
    transfer_col = get_transfer_col(cluster_df)
    if transfer_col:
        means = cluster_df.groupby('cluster')[transfer_col].mean()
        if len(means) == 2 and means[0] > means[1]:
            cluster_df['cluster'] = 1 - cluster_df['cluster']
    
    summary = cluster_df.groupby('cluster').mean()
    return cluster_df, summary

def plot_transfer_distribution(product=None):
    """
    Returns plotly histogram with KDE overlay for transfer time.
    """
    df = dp.load_reactor_bct()
    if product:
        df = df[df['product'] == product]
        
    col = get_transfer_col(df)
    if not col:
        return go.Figure()
        
    data = df[col].dropna()
    if len(data) < 2:
        return go.Figure()
        
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data, histnorm='probability density', name='Histogram', opacity=0.7))
    
    # Try adding KDE if scipy is available implicitly via FF
    try:
        kde_fig = ff.create_distplot([data], ['Transfer Time'], show_hist=False)
        fig.add_trace(kde_fig.data[0])
    except:
        pass
        
    fig.update_layout(
        title=f"Transfer to Blender Duration Distribution {f'- {product}' if product else ''}",
        xaxis_title="Duration",
        yaxis_title="Density",
        barmode='overlay'
    )
    return fig

def plot_transfer_correlations(product=None):
    """
    Returns plotly bar chart of top correlations with transfer time.
    """
    df = compute_transfer_correlations(product)
    if df.empty:
        return go.Figure()
        
    # Top 15 correlated
    plot_df = df.head(15).copy()
    plot_df['Color'] = np.where(plot_df['Correlation'] > 0, 'Positive', 'Negative')
    
    fig = px.bar(
        plot_df,
        x='Correlation',
        y='Feature',
        color='Color',
        color_discrete_map={'Positive': 'blue', 'Negative': 'red'},
        title=f"Top Features Correlated with Transfer Time {f'- {product}' if product else ''}",
        orientation='h'
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    return fig

def plot_cluster_comparison(product=None):
    """
    Returns plotly grouped bar chart comparing fast vs slow clusters.
    """
    _, summary = segment_fast_slow(product, n_clusters=2)
    if summary.empty:
        return go.Figure()
        
    # Pick a few key steps to show
    plot_cols = []
    if 'Transfer to blender' in summary.columns:
        plot_cols.append('Transfer to blender')
    if 'Total BCT' in summary.columns:
        plot_cols.append('Total BCT')
        
    # Add top 3 other highest variance steps
    other_cols = [c for c in summary.columns if c not in plot_cols]
    if other_cols:
        vars = summary[other_cols].var()
        plot_cols.extend(vars.nlargest(3).index.tolist())
        
    plot_df = summary[plot_cols].reset_index().melt(id_vars='cluster', var_name='Step', value_name='Average Duration')
    plot_df['cluster'] = plot_df['cluster'].map({0: 'Cluster 0 (Fast Transfer)', 1: 'Cluster 1 (Slow Transfer)'})
    
    fig = px.bar(
        plot_df,
        x='Step',
        y='Average Duration',
        color='cluster',
        barmode='group',
        title=f"Average Step Durations by Cluster {f'- {product}' if product else ''}"
    )
    return fig

def survival_analysis(product=None):
    """
    Cox PH model on transfer time, returns dict with summary DataFrame and plotly figure.
    """
    if not HAS_LIFELINES:
        return {"error": "lifelines package is not installed"}
        
    df = dp.build_reactor_features(product)
    col = get_transfer_col(df)
    if not col:
        return {"error": "Transfer time column not found"}
        
    # Select a few potential regressors to avoid convergence issues
    features = ['Total BCT']
    if 'reactor_ipqc_viscosity' in df.columns:
        features.append('reactor_ipqc_viscosity')
        
    analysis_df = df[[col] + features].dropna()
    if len(analysis_df) < 10:
        return {"error": "Not enough data for survival analysis"}
        
    # Dummy event column (assume all events occurred)
    analysis_df['event'] = 1
    
    try:
        cph = CoxPHFitter()
        cph.fit(analysis_df, duration_col=col, event_col='event')
        
        summary_df = cph.summary.reset_index()
        
        # Plot coefficients
        fig = px.bar(
            summary_df,
            x='coef',
            y='covariate',
            error_x='se(coef)',
            orientation='h',
            title=f"Cox PH Model Coefficients for Transfer Time {f'- {product}' if product else ''}"
        )
        
        return {
            "summary": summary_df,
            "figure": fig
        }
    except Exception as e:
        return {"error": f"Model fitting failed: {str(e)}"}
