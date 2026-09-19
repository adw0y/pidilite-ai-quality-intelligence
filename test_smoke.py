import sys
sys.path.insert(0, '.')
import data_pipeline as dp
import bct_analysis as bct
import golden_batch as gb
import quality_models as qm
import transfer_analysis as ta
import transfer_learning as tl
import process_mining as pm
import explanation_agent as ea

print('1. Testing Data Pipeline...')
ft = dp.load_feature_table()
print(f'   Feature table loaded: {ft.shape}')

print('2. Testing BCT Analysis...')
var_df = bct.compute_variance_ranking('Product-A')
print(f'   Variance ranking computed: {var_df.shape}')

print('3. Testing Golden Batch...')
goldens = gb.identify_golden_batches('Product-A')
print(f'   Golden batches identified: {len(goldens)}')

print('4. Testing Quality Models...')
res = qm.train_model('Product-A', 'reactor_ipqc_viscosity', 'xgboost')
mape = res['metrics']['MAPE']
r2 = res['metrics']['R2']
print(f'   Model trained! Test MAPE: {mape:.4f}, R2: {r2:.4f}')

print('5. Testing Transfer Analysis...')
t_stats = ta.analyze_transfer_distribution('Product-A')
print(f"   Transfer stats mean: {t_stats['mean']:.1f} min")

print('6. Testing Process Mining...')
p_stats = pm.compute_step_statistics('Product-A')
print(f'   Step stats count: {len(p_stats)}')

print('7. Testing Explanation Agent (Fallback)...')
expl = ea.explain_batch(
    res['X_test'].iloc[0].to_dict(),
    {'predicted_value': res['y_pred'][0], 'target_name': 'reactor_ipqc_viscosity', 'risk_level': 'low'},
    [{'feature': 'Holding-1', 'value': 16, 'direction': 'increase'}]
)
print('   Explanation length:', len(expl))

print('ALL MODULE TESTS PASSED!')
