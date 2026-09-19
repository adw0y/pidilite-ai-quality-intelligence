"""
Cross-Product Transfer Learning - Project 9
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import config as cfg
import data_pipeline as dp

def _calculate_metrics(y_true, y_pred):
    return {
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred)
    }

def train_on_a_test_on_b(target='reactor_ipqc_viscosity', model_type='xgboost'):
    """
    Train on ALL Product-A data, test on ALL Product-B.
    """
    X_a, y_a, _ = dp.get_modeling_xy('Product-A', target)
    X_b, y_b, _ = dp.get_modeling_xy('Product-B', target)
    
    if model_type == 'xgboost':
        model = xgb.XGBRegressor(random_state=42)
    elif model_type == 'random_forest':
        model = RandomForestRegressor(random_state=42)
    else:
        model = xgb.XGBRegressor(random_state=42)
        
    model.fit(X_a, y_a)
    y_pred_b = model.predict(X_b)
    
    metrics = _calculate_metrics(y_b, y_pred_b)
    
    return {
        'metrics_raw': metrics,
        'predictions': y_pred_b,
        'model': model
    }

def fine_tune_on_b(target, model_type='xgboost', n_finetune=20, random_state=42):
    """
    Train on A, fine-tune on B sample. Compare with B-only model.
    """
    X_a, y_a, _ = dp.get_modeling_xy('Product-A', target)
    X_b, y_b, _ = dp.get_modeling_xy('Product-B', target)
    
    # Split B into finetune set and test set
    if len(X_b) <= n_finetune:
        X_b_ft, X_b_test, y_b_ft, y_b_test = X_b, X_b, y_b, y_b
    else:
        X_b_ft, X_b_test, y_b_ft, y_b_test = train_test_split(X_b, y_b, train_size=n_finetune, random_state=random_state)
        
    if model_type == 'xgboost':
        # Train A model
        model_a = xgb.XGBRegressor(random_state=random_state)
        model_a.fit(X_a, y_a)
        
        # Finetune on B
        model_ft = xgb.XGBRegressor(random_state=random_state)
        model_ft.fit(X_b_ft, y_b_ft, xgb_model=model_a.get_booster())
        
        # B-only model
        model_b_only = xgb.XGBRegressor(random_state=random_state)
        model_b_only.fit(X_b_ft, y_b_ft)
    else:
        # Fallback for sklearn models: train on combined with sample weights (A lower)
        X_comb = pd.concat([X_a, X_b_ft])
        y_comb = pd.concat([y_a, y_b_ft])
        weights = np.concatenate([np.ones(len(X_a))*0.1, np.ones(len(X_b_ft))])
        
        model_ft = RandomForestRegressor(random_state=random_state)
        model_ft.fit(X_comb, y_comb, sample_weight=weights)
        
        model_b_only = RandomForestRegressor(random_state=random_state)
        model_b_only.fit(X_b_ft, y_b_ft)
        
    y_pred_ft = model_ft.predict(X_b_test)
    y_pred_b = model_b_only.predict(X_b_test)
    
    metrics_ft = _calculate_metrics(y_b_test, y_pred_ft)
    metrics_b = _calculate_metrics(y_b_test, y_pred_b)
    
    improvement = {k: (metrics_b[k] - metrics_ft[k]) / abs(metrics_b[k]) if metrics_b[k] != 0 else 0 for k in metrics_ft}
    
    return {
        'metrics_a_finetuned': metrics_ft,
        'metrics_b_only': metrics_b,
        'improvement': improvement,
        'model_ft': model_ft,
        'model_b_only': model_b_only
    }

def compare_all_approaches(target):
    """
    Returns DataFrame comparing different approaches.
    """
    res_raw = train_on_a_test_on_b(target, model_type='xgboost')
    res_ft = fine_tune_on_b(target, model_type='xgboost', n_finetune=20)
    
    # Get B-only full metrics (mocked as 100 samples)
    X_b, y_b, _ = dp.get_modeling_xy('Product-B', target)
    if len(X_b) > 100:
        X_b_train, X_b_test, y_b_train, y_b_test = train_test_split(X_b, y_b, train_size=100, random_state=42)
    else:
        X_b_train, X_b_test, y_b_train, y_b_test = train_test_split(X_b, y_b, train_size=0.8, random_state=42)
        
    model_full = xgb.XGBRegressor(random_state=42)
    model_full.fit(X_b_train, y_b_train)
    y_pred_full = model_full.predict(X_b_test)
    metrics_full = _calculate_metrics(y_b_test, y_pred_full)
    
    data = [
        {'approach': 'A-pretrained raw', **res_raw['metrics_raw']},
        {'approach': 'A-pretrained fine-tuned', **res_ft['metrics_a_finetuned']},
        {'approach': 'B-only-small (20 samples)', **res_ft['metrics_b_only']},
        {'approach': 'B-only-full (100 samples)', **metrics_full}
    ]
    return pd.DataFrame(data)

def plot_comparison(target):
    """
    Plots comparison of approaches.
    """
    df = compare_all_approaches(target)
    df_melted = pd.melt(df, id_vars=['approach'], value_vars=['MAPE', 'RMSE', 'R2'], var_name='Metric', value_name='Value')
    
    fig = px.bar(df_melted, x='approach', y='Value', color='Metric', barmode='group',
                 title=f'Model Approach Comparison: {target}')
    return fig

def plot_feature_importance_comparison(target):
    """
    Compares feature importance between A and B models.
    """
    # Simple feature importance based on xgboost weight
    X_a, y_a, features = dp.get_modeling_xy('Product-A', target)
    X_b, y_b, _ = dp.get_modeling_xy('Product-B', target)
    
    model_a = xgb.XGBRegressor(random_state=42).fit(X_a, y_a)
    model_b = xgb.XGBRegressor(random_state=42).fit(X_b, y_b)
    
    imp_a = model_a.feature_importances_
    imp_b = model_b.feature_importances_
    
    df_imp = pd.DataFrame({
        'Feature': features,
        'Importance_A': imp_a,
        'Importance_B': imp_b
    })
    
    df_imp_melt = pd.melt(df_imp, id_vars=['Feature'], value_vars=['Importance_A', 'Importance_B'], 
                          var_name='Model', value_name='Importance')
                          
    fig = px.bar(df_imp_melt, x='Importance', y='Feature', color='Model', barmode='group', orientation='h',
                 title=f'Feature Importance Comparison: {target}')
    return fig
