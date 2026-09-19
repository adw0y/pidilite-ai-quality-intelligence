"""
Quality Prediction Models - Project 1
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import shap
import mlflow
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
import lightgbm as lgb
import config as cfg
import data_pipeline as dp
import warnings

def _get_model(model_type, random_state=42):
    if model_type == 'xgboost':
        return xgb.XGBRegressor(random_state=random_state)
    elif model_type == 'lightgbm':
        return lgb.LGBMRegressor(random_state=random_state)
    elif model_type == 'random_forest':
        return RandomForestRegressor(random_state=random_state)
    elif model_type == 'ridge':
        return Ridge(random_state=random_state)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

def _calculate_metrics(y_true, y_pred):
    return {
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred)
    }

def train_model(product, target, model_type='xgboost', test_size=0.2, random_state=42):
    """
    Trains a model and returns a dictionary with model details and metrics.
    """
    X, y, feature_cols = dp.get_modeling_xy(product, target)
    
    if X.empty or y.empty:
        raise ValueError(f"No data available for {product} - {target}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    model = _get_model(model_type, random_state)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    metrics = _calculate_metrics(y_test, y_pred)
    
    return {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'metrics': metrics,
        'feature_names': feature_cols,
        'model_type': model_type,
        'product': product,
        'target': target
    }

def cross_validate_model(product, target, model_type='xgboost', cv=5):
    """
    Performs cross-validation.
    """
    X, y, _ = dp.get_modeling_xy(product, target)
    if X.empty or y.empty:
        raise ValueError("Empty data.")
        
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    fold_metrics = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model = _get_model(model_type, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        fold_metrics.append(_calculate_metrics(y_test, y_pred))
        
    df_metrics = pd.DataFrame(fold_metrics)
    return {
        'fold_metrics': fold_metrics,
        'mean_metrics': df_metrics.mean().to_dict(),
        'std_metrics': df_metrics.std().to_dict()
    }

def get_shap_values(model_result):
    """
    Calculates SHAP values for the test set.
    """
    model = model_result['model']
    X_test = model_result['X_test']
    model_type = model_result['model_type']
    
    if model_type in ['xgboost', 'lightgbm', 'random_forest']:
        explainer = shap.TreeExplainer(model)
    elif model_type == 'ridge':
        explainer = shap.LinearExplainer(model, model_result['X_train'])
    else:
        explainer = shap.Explainer(model, X_test)
        
    shap_values = explainer(X_test)
    return shap_values

def plot_predicted_vs_actual(model_result):
    """
    Plots predicted vs actual values.
    """
    df = pd.DataFrame({
        'Actual': model_result['y_test'],
        'Predicted': model_result['y_pred']
    })
    
    fig = px.scatter(df, x='Actual', y='Predicted', title=f"Predicted vs Actual: {model_result['target']}")
    
    # Add reference line y=x
    min_val = min(df['Actual'].min(), df['Predicted'].min())
    max_val = max(df['Actual'].max(), df['Predicted'].max())
    fig.add_shape(
        type="line", line=dict(dash='dash', color='red'),
        x0=min_val, y0=min_val, x1=max_val, y1=max_val
    )
    return fig

def plot_feature_importance(model_result):
    """
    Plots SHAP-based feature importance.
    """
    shap_values = get_shap_values(model_result)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    
    df_importance = pd.DataFrame({
        'Feature': model_result['feature_names'],
        'Importance': mean_abs_shap
    }).sort_values(by='Importance', ascending=True).tail(15)
    
    fig = px.bar(df_importance, x='Importance', y='Feature', orientation='h', title='Feature Importance (SHAP)')
    return fig

def plot_shap_waterfall(model_result, sample_idx=0):
    """
    Plots SHAP waterfall for a single sample.
    """
    shap_values = get_shap_values(model_result)
    
    fig = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[sample_idx], show=False)
    plt.tight_layout()
    return fig

def predict_single(model_result, features_dict):
    """
    Predicts for a single instance and returns analysis.
    """
    X_single = pd.DataFrame([features_dict], columns=model_result['feature_names'])
    X_single.fillna(0, inplace=True) # Handle missing
    
    pred = model_result['model'].predict(X_single)[0]
    
    # Determine risk level based on spec limits
    target = model_result['target']
    product = model_result['product']
    risk_level = 'medium'
    
    try:
        limits = cfg.SPEC_LIMITS.get(product, {}).get(target, None)
        if limits:
            min_lim, max_lim = limits
            if pred < min_lim or pred > max_lim:
                risk_level = 'high'
            elif (pred - min_lim) / (max_lim - min_lim) > 0.8 or (pred - min_lim) / (max_lim - min_lim) < 0.2:
                risk_level = 'medium'
            else:
                risk_level = 'low'
    except Exception:
        pass
        
    # Get top drivers for this instance using SHAP
    model = model_result['model']
    model_type = model_result['model_type']
    
    if model_type in ['xgboost', 'lightgbm', 'random_forest']:
        explainer = shap.TreeExplainer(model)
    elif model_type == 'ridge':
        explainer = shap.LinearExplainer(model, model_result['X_train'])
    else:
        explainer = shap.Explainer(model, X_single)
        
    sv = explainer(X_single)
    vals = sv.values[0]
    
    drivers = []
    for i, feature in enumerate(model_result['feature_names']):
        drivers.append({
            'feature': feature,
            'value': features_dict.get(feature, 0),
            'direction': 'increase' if vals[i] > 0 else 'decrease',
            'contribution': abs(vals[i])
        })
        
    drivers.sort(key=lambda x: x['contribution'], reverse=True)
    top_drivers = drivers[:3]
    
    return {
        'prediction': pred,
        'risk_level': risk_level,
        'top_drivers': [{'feature': d['feature'], 'value': d['value'], 'direction': d['direction']} for d in top_drivers]
    }

def log_to_mlflow(model_result, experiment_name=None):
    """
    Logs model results to MLflow.
    """
    try:
        mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
        if experiment_name:
            mlflow.set_experiment(experiment_name)
            
        with mlflow.start_run():
            mlflow.log_param("model_type", model_result['model_type'])
            mlflow.log_param("product", model_result['product'])
            mlflow.log_param("target", model_result['target'])
            
            for metric_name, metric_val in model_result['metrics'].items():
                mlflow.log_metric(metric_name, metric_val)
                
            # Log model
            mlflow.sklearn.log_model(model_result['model'], "model")
            return True
    except Exception as e:
        warnings.warn(f"Failed to log to MLflow: {str(e)}")
        return False
