import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
assert len(feature_cols) == 31
assert target_col == 'TARGET_10'
assert len(development_folds) == 6
assert len(train_oos_folds) == 4
assert not {target_col, 'TARGET_5', 'Split'}.intersection(feature_cols)

def standardize_fold(fold):
    train_idx = fold['train_fit_idx']
    signal_idx = fold['oos_signal_idx']
    eval_idx = fold['oos_eval_idx']
    X_train_raw = data.loc[train_idx, feature_cols]
    X_oos_raw = data.loc[signal_idx, feature_cols]
    y_train = data.loc[train_idx, target_col].copy()
    assert data.loc[train_idx, 'DlyCalDt'].max() < data.loc[signal_idx, 'DlyCalDt'].min()
    assert data.loc[train_idx, 'Split'].eq('Train').all()
    assert not data.loc[signal_idx, 'Split'].eq('Test').any()
    assert train_idx.intersection(signal_idx).empty
    assert eval_idx.isin(signal_idx).all()
    assert X_train_raw.index.equals(y_train.index)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_raw), index=train_idx, columns=feature_cols)
    X_oos_signal_scaled = pd.DataFrame(scaler.transform(X_oos_raw), index=signal_idx, columns=feature_cols)
    X_oos_eval_scaled = X_oos_signal_scaled.loc[eval_idx].copy()
    train_mean_error = float(X_train_scaled.mean().abs().max())
    expected_variances = np.where(scaler.var_ > 0, 1.0, 0.0)
    train_var_error = float(np.abs(X_train_scaled.var(ddof=0).to_numpy() - expected_variances).max())
    constant_features = [feature_cols[i] for i in range(len(feature_cols)) if scaler.var_[i] == 0]
    assert train_mean_error < 1e-08
    assert train_var_error < 1e-08
    assert np.allclose(scaler.mean_, X_train_raw.mean().to_numpy(), rtol=1e-10, atol=1e-10)
    for X in (X_train_scaled, X_oos_signal_scaled, X_oos_eval_scaled):
        assert X.shape[1] == 31
        assert np.isfinite(X.to_numpy()).all()
    assert np.isfinite(y_train.to_numpy()).all()
    assert len(X_train_scaled) == len(train_idx)
    assert len(X_oos_signal_scaled) == len(signal_idx)
    assert len(X_oos_eval_scaled) == len(eval_idx)
    return {'scaler': scaler, 'X_train': X_train_scaled, 'y_train': y_train, 'X_oos_signal': X_oos_signal_scaled, 'X_oos_eval': X_oos_eval_scaled, 'train_fit_idx': train_idx.copy(), 'oos_signal_idx': signal_idx.copy(), 'oos_eval_idx': eval_idx.copy(), 'train_mean_error': train_mean_error, 'train_var_error': train_var_error, 'constant_features': constant_features, 'train_cutoff': fold['train_cutoff'], 'oos_start': fold['oos_start'], 'oos_end': fold['oos_end']}

def scaling_record(stage, window, oos_year, result):
    return {'Stage': stage, 'Window': window, 'OOS_Year': oos_year, 'Train_Rows': len(result['X_train']), 'OOS_Signal_Rows': len(result['X_oos_signal']), 'OOS_Eval_Rows': len(result['X_oos_eval']), 'Features': len(feature_cols), 'Constant_Features': len(result['constant_features']), 'Train_Mean_Error': result['train_mean_error'], 'Train_Var_Error': result['train_var_error'], 'Status': 'PASSED'}
scaled_development_folds = {}
scaling_rows = []
for (window, year), fold in sorted(development_folds.items()):
    result = standardize_fold(fold)
    scaled_development_folds[window, year] = result
    scaling_rows.append(scaling_record('Development', window, year, result))
scaled_train_oos_folds = {}
for year, fold in sorted(train_oos_folds.items()):
    result = standardize_fold(fold)
    scaled_train_oos_folds[year] = result
    scaling_rows.append(scaling_record('Train_OOS', year - TRAIN_OOS_START_YEAR, year, result))
scaled_validation_fold = standardize_fold(validation_fold)
result = scaled_validation_fold
scaling_rows.append(scaling_record('Validation', 7, '2022-2023', result))
scaling_summary_df = pd.DataFrame(scaling_rows)
assert len(scaling_summary_df) == 11
assert scaling_summary_df['Features'].eq(31).all()
assert scaling_summary_df['Status'].eq('PASSED').all()
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
