import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean

try:
    import config as cfg
    import data_pipeline as dp
except ImportError:
    pass # Assume they exist at runtime

def identify_golden_batches(product, top_pct=0.15):
    """
    Finds batches closest to center of spec range for BOTH solid% and viscosity.
    Returns DataFrame of golden batches with all columns.
    """
    df = dp.build_reactor_features(product)
    if df is None or df.empty:
        return pd.DataFrame()
        
    spec_limits = cfg.SPEC_LIMITS.get(product, {})
    
    solid_center = (spec_limits.get('solid_pct', {}).get('min', 0) + spec_limits.get('solid_pct', {}).get('max', 100)) / 2.0
    viscosity_center = (spec_limits.get('viscosity', {}).get('min', 0) + spec_limits.get('viscosity', {}).get('max', 100)) / 2.0
    
    # Check if spec limits are properly defined, if not fallback to median
    if spec_limits.get('solid_pct', {}).get('max') is None:
        if 'reactor_ipqc_solid_pct' in df.columns:
            solid_center = df['reactor_ipqc_solid_pct'].median()
    if spec_limits.get('viscosity', {}).get('max') is None:
        if 'reactor_ipqc_viscosity' in df.columns:
            viscosity_center = df['reactor_ipqc_viscosity'].median()
        
    if 'reactor_ipqc_solid_pct' not in df.columns or 'reactor_ipqc_viscosity' not in df.columns:
        return pd.DataFrame()
        
    df = df.dropna(subset=['reactor_ipqc_solid_pct', 'reactor_ipqc_viscosity']).copy()
    
    # Normalize distances
    solid_std = df['reactor_ipqc_solid_pct'].std()
    visc_std = df['reactor_ipqc_viscosity'].std()
    
    if solid_std == 0: solid_std = 1
    if visc_std == 0: visc_std = 1
    
    dist_solid = ((df['reactor_ipqc_solid_pct'] - solid_center) / solid_std) ** 2
    dist_visc = ((df['reactor_ipqc_viscosity'] - viscosity_center) / visc_std) ** 2
    
    df['dist_to_center'] = np.sqrt(dist_solid + dist_visc)
    
    threshold = df['dist_to_center'].quantile(top_pct)
    golden = df[df['dist_to_center'] <= threshold].copy()
    
    return golden

def _get_step_cols(df):
    exclude = (
        set(cfg.IPQC_TARGETS)
        | set(cfg.SFG_TARGETS)
        | set(cfg.DROP_COLS)
        | set(cfg.SUMMARY_COLS)
        | {
            "batch_no", "product", "dist_to_center", "is_golden", "date", "product_name",
            "reactor_appearance", "yield_ratio", "holding_total", "charge_total",
            "reaction_phase_pct", "transfer_to_bct_ratio", "bct_z_score"
        }
    )
    # Only keep numeric step columns with duration in minutes that have valid variance
    valid_cols = []
    for c in df.columns:
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c]):
            if df[c].notna().sum() > 0:
                valid_cols.append(c)
    return valid_cols

def compute_golden_profile(product, top_pct=0.15):
    """
    Returns Series of median step durations for golden batches (the reference profile).
    """
    golden = identify_golden_batches(product, top_pct=top_pct)
    if golden.empty:
        return pd.Series(dtype=float)
        
    step_cols = _get_step_cols(golden)
    return golden[step_cols].median().dropna()

def compute_similarity_scores(product, top_pct=0.15):
    """
    Returns DataFrame with batch_no, similarity_score, is_golden flag, quality targets.
    """
    df = dp.build_reactor_features(product)
    if df is None or df.empty:
        return pd.DataFrame()
        
    golden = identify_golden_batches(product, top_pct=top_pct)
    if golden.empty:
        return pd.DataFrame()
        
    golden_batch_nos = set(golden['batch_no'])
    df['is_golden'] = df['batch_no'].isin(golden_batch_nos)
    
    golden_profile = compute_golden_profile(product, top_pct=top_pct)
    step_cols = [c for c in golden_profile.index if c in df.columns]
    
    # Fill any isolated missing feature values with 0
    X_steps = df[step_cols].fillna(0).to_numpy(dtype=float)
    g_vec = golden_profile[step_cols].to_numpy(dtype=float)
    
    # Vectorized Euclidean distance across all batches
    distances = np.linalg.norm(X_steps - g_vec, axis=1)
    df['similarity_score'] = distances
    
    cols = ['batch_no', 'similarity_score', 'is_golden'] + list(cfg.IPQC_TARGETS)
    available_cols = [c for c in cols if c in df.columns]
    
    return df[available_cols]

def compute_pca_embedding(product=None, top_pct=0.15):
    df = dp.build_reactor_features(product)
    if df is None or df.empty:
        return pd.DataFrame()
        
    golden = identify_golden_batches(product, top_pct=top_pct)
    golden_batch_nos = set(golden['batch_no']) if not golden.empty else set()
    df['is_golden'] = df['batch_no'].isin(golden_batch_nos)
    
    step_cols = _get_step_cols(df)
    df = df.dropna(subset=step_cols).copy()
    
    if len(df) < 2:
        return pd.DataFrame()
        
    X = df[step_cols]
    X_scaled = StandardScaler().fit_transform(X)
    
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)
    
    res = df[['batch_no', 'product', 'is_golden']].copy()
    for t in cfg.IPQC_TARGETS:
        if t in df.columns:
            res[t] = df[t]
            
    res['PC1'] = pcs[:, 0]
    res['PC2'] = pcs[:, 1]
    
    return res

def compute_tsne_embedding(product=None, perplexity=15, top_pct=0.15):
    df = dp.build_reactor_features(product)
    if df is None or df.empty:
        return pd.DataFrame()
        
    golden = identify_golden_batches(product, top_pct=top_pct)
    golden_batch_nos = set(golden['batch_no']) if not golden.empty else set()
    df['is_golden'] = df['batch_no'].isin(golden_batch_nos)
    
    step_cols = _get_step_cols(df)
    df = df.dropna(subset=step_cols).copy()
    
    if len(df) < 2:
        return pd.DataFrame()
        
    X = df[step_cols]
    X_scaled = StandardScaler().fit_transform(X)
    
    tsne = TSNE(n_components=2, perplexity=min(perplexity, len(df)-1))
    tsne_res = tsne.fit_transform(X_scaled)
    
    res = df[['batch_no', 'product', 'is_golden']].copy()
    for t in cfg.IPQC_TARGETS:
        if t in df.columns:
            res[t] = df[t]
            
    res['PC1'] = tsne_res[:, 0]
    res['PC2'] = tsne_res[:, 1]
    
    return res

def plot_similarity_distribution(product, top_pct=0.15):
    df = compute_similarity_scores(product, top_pct=top_pct)
    if df.empty:
        return go.Figure()
        
    fig = px.histogram(df, x='similarity_score', color='is_golden', barmode='overlay',
                       title=f"Similarity Score Distribution - {product} (Top {int(top_pct*100)}% Golden)",
                       labels={'similarity_score': 'Distance from Golden Profile', 'is_golden': 'Golden Batch'})
    return fig

def plot_golden_profile(product, top_pct=0.15):
    df = dp.build_reactor_features(product)
    if df is None or df.empty:
        return go.Figure()
        
    golden_profile = compute_golden_profile(product, top_pct=top_pct)
    if golden_profile.empty:
        return go.Figure()
        
    step_cols = list(golden_profile.index)
    overall_median = df[step_cols].median()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=step_cols, y=golden_profile.values, name=f'Golden Profile (Top {int(top_pct*100)}% Median)'))
    fig.add_trace(go.Bar(x=step_cols, y=overall_median.values, name='Overall Fleet Median'))
    
    fig.update_layout(
        title=f"Golden Batch Profile vs Overall (Step Durations in Minutes) - {product}",
        barmode='group',
        height=520,
        margin=dict(l=60, r=40, t=50, b=120),
        yaxis_title="Duration (Minutes)",
        xaxis_title="Process Step",
        legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.99)
    )
    return fig

def plot_pca_scatter(product=None, top_pct=0.15):
    df = compute_pca_embedding(product, top_pct=top_pct)
    if df.empty:
        return go.Figure()
        
    color_col = 'reactor_ipqc_viscosity' if 'reactor_ipqc_viscosity' in df.columns else None
    
    fig = px.scatter(df, x='PC1', y='PC2', color=color_col, symbol='is_golden',
                     hover_data=['batch_no'], title=f"PCA of Step Durations - {product if product else 'All'} (Top {int(top_pct*100)}% Golden)")
    return fig

def plot_tsne_scatter(product=None, top_pct=0.15):
    df = compute_tsne_embedding(product, top_pct=top_pct)
    if df.empty:
        return go.Figure()
        
    color_col = 'reactor_ipqc_viscosity' if 'reactor_ipqc_viscosity' in df.columns else None
    
    fig = px.scatter(df, x='PC1', y='PC2', color=color_col, symbol='is_golden',
                     hover_data=['batch_no'], title=f"t-SNE of Step Durations - {product if product else 'All'} (Top {int(top_pct*100)}% Golden)")
    return fig
