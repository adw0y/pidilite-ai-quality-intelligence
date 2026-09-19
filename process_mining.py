import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

try:
    import config as cfg
    import data_pipeline as dp
except ImportError:
    pass

def _get_step_cols(df):
    exclude = list(cfg.IPQC_TARGETS) + ['batch_no', 'product']
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude]

def compute_step_statistics(product=None):
    """
    Returns DataFrame with [step, mean_duration, std_duration, min, max, pct_of_total_bct] for each reactor step.
    """
    df = dp.build_reactor_features(product)
    if df is None or df.empty:
        return pd.DataFrame()
        
    step_cols = _get_step_cols(df)
    
    stats = []
    total_bct = df[step_cols].sum(axis=1).mean()
    
    for c in step_cols:
        s = df[c]
        mean_val = s.mean()
        stats.append({
            'step': c,
            'mean_duration': mean_val,
            'std_duration': s.std(),
            'min': s.min(),
            'max': s.max(),
            'pct_of_total_bct': (mean_val / total_bct) * 100 if total_bct > 0 else 0
        })
        
    return pd.DataFrame(stats)

def compute_bottleneck_ranking(product=None):
    """
    Returns DataFrame ranking steps by bottleneck impact = mean_duration * cv (steps that are both long AND variable).
    Columns: step, mean, cv, bottleneck_score, rank.
    """
    stats = compute_step_statistics(product)
    if stats.empty:
        return pd.DataFrame()
        
    stats['cv'] = stats['std_duration'] / stats['mean_duration']
    stats['cv'] = stats['cv'].fillna(0)
    stats['bottleneck_score'] = stats['mean_duration'] * stats['cv']
    
    res = stats[['step', 'mean_duration', 'cv', 'bottleneck_score']].copy()
    res.rename(columns={'mean_duration': 'mean'}, inplace=True)
    res = res.sort_values('bottleneck_score', ascending=False).reset_index(drop=True)
    res['rank'] = res.index + 1
    
    return res

def compute_critical_path(product=None):
    """
    Returns DataFrame identifying which steps form the longest chain.
    Since this is batch (sequential), all steps are on the critical path — so rank by absolute duration.
    """
    stats = compute_step_statistics(product)
    if stats.empty:
        return pd.DataFrame()
        
    res = stats[['step', 'mean_duration']].copy()
    res = res.sort_values('mean_duration', ascending=False).reset_index(drop=True)
    res['rank'] = res.index + 1
    
    return res

def simulate_what_if(product, step, reduction_pct):
    """
    Returns dict: original_mean_bct, new_mean_bct, savings_min, savings_pct, batches_per_month_gain
    """
    df = dp.build_reactor_features(product)
    if df is None or df.empty:
        return {}
        
    step_cols = _get_step_cols(df)
    
    if step not in step_cols:
        return {}
        
    bct_series = df[step_cols].sum(axis=1)
    orig_mean = bct_series.mean()
    
    new_bct_series = df[step_cols].drop(columns=[step] if step in df.columns else []).sum(axis=1) + (df[step] * (1 - reduction_pct / 100.0))
    new_mean = new_bct_series.mean()
    
    savings_min = orig_mean - new_mean
    savings_pct = (savings_min / orig_mean) * 100 if orig_mean > 0 else 0
    
    # Assumes 24/7 operation on 1 reactor, 30 days a month = 30 * 24 * 60 minutes
    mins_per_month = 30 * 24 * 60
    orig_batches = mins_per_month / orig_mean if orig_mean > 0 else 0
    new_batches = mins_per_month / new_mean if new_mean > 0 else 0
    
    batches_gain = new_batches - orig_batches
    
    return {
        'original_mean_bct': orig_mean,
        'new_mean_bct': new_mean,
        'savings_min': savings_min,
        'savings_pct': savings_pct,
        'batches_per_month_gain': batches_gain
    }

def plot_step_gantt(product, batch_no=1):
    """
    Returns plotly Gantt-style horizontal bar chart showing step sequence and durations for a specific batch.
    """
    df = dp.build_reactor_features(product)
    if df is None or df.empty:
        return go.Figure()
        
    batch_df = df[df['batch_no'] == batch_no]
    if batch_df.empty:
        return go.Figure()
        
    step_cols = _get_step_cols(batch_df)
    
    data = []
    current_time = 0
    for c in step_cols:
        duration = batch_df.iloc[0][c]
        data.append({
            'Task': c,
            'Start': current_time,
            'Finish': current_time + duration,
            'Duration': duration
        })
        current_time += duration
        
    gantt_df = pd.DataFrame(data)
    
    fig = go.Figure()
    for i, row in gantt_df.iterrows():
        fig.add_trace(go.Bar(
            y=['Batch Sequence'],
            x=[row['Duration']],
            name=row['Task'],
            orientation='h',
            text=row['Task'],
            textposition='inside'
        ))
        
    fig.update_layout(title=f"Step Sequence Gantt - Product {product}, Batch {batch_no}", barmode='stack', showlegend=False)
    return fig

def plot_bottleneck_ranking(product=None):
    """
    Returns plotly bar chart of bottleneck scores.
    """
    df = compute_bottleneck_ranking(product)
    if df.empty:
        return go.Figure()
        
    fig = px.bar(df, x='step', y='bottleneck_score', title=f"Bottleneck Ranking - {product if product else 'All'}",
                 hover_data=['mean', 'cv'])
    return fig

def plot_what_if_comparison(product, scenarios):
    """
    scenarios is list of (step, reduction_pct)
    Returns plotly bar chart comparing original vs each scenario.
    """
    results = []
    orig_mean = 0
    
    for step, pct in scenarios:
        res = simulate_what_if(product, step, pct)
        if res:
            if orig_mean == 0:
                orig_mean = res['original_mean_bct']
                results.append({'Scenario': 'Original', 'Mean BCT': orig_mean})
            
            results.append({
                'Scenario': f"{step} (-{pct}%)",
                'Mean BCT': res['new_mean_bct']
            })
            
    if not results:
        return go.Figure()
        
    res_df = pd.DataFrame(results)
    fig = px.bar(res_df, x='Scenario', y='Mean BCT', title=f"What-If Comparison - {product}")
    return fig
