import time
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.exceptions import ConvergenceWarning
ENET_PARAMS = {'alpha': 0.0001, 'l1_ratio': 0.5, 'fit_intercept': True, 'max_iter': 3000, 'tol': 0.0001, 'selection': 'cyclic', 'precompute': True}
assert len(scaled_development_folds) == 6
assert len(scaled_train_oos_folds) == 4
assert len(feature_cols) == 31
assert target_col == 'TARGET_10'

def calculate_daily_rank_ic(frame):
    dates = frame['DlyCalDt']
    actual_rank = frame.groupby('DlyCalDt')['Actual'].rank()
    predicted_rank = frame.groupby('DlyCalDt')['Predicted'].rank()
    x = actual_rank - actual_rank.groupby(dates).transform('mean')
    y = predicted_rank - predicted_rank.groupby(dates).transform('mean')
    numerator = (x * y).groupby(dates).sum()
    denominator = np.sqrt(x.pow(2).groupby(dates).sum() * y.pow(2).groupby(dates).sum())
    return numerator / denominator.replace(0, np.nan)

def fit_elastic_net_fold(scaled_fold, stage, window, oos_period, evaluate=True):
    X_train = scaled_fold['X_train']
    y_train = scaled_fold['y_train']
    X_oos = scaled_fold['X_oos_signal']
    train_idx = scaled_fold['train_fit_idx']
    signal_idx = scaled_fold['oos_signal_idx']
    eval_idx = scaled_fold['oos_eval_idx']
    assert X_train.index.equals(train_idx)
    assert X_train.index.equals(y_train.index)
    assert X_oos.index.equals(signal_idx)
    assert eval_idx.isin(signal_idx).all()
    assert data.loc[train_idx, 'DlyCalDt'].max() < data.loc[signal_idx, 'DlyCalDt'].min()
    assert data.loc[train_idx, 'Split'].eq('Train').all()
    assert X_train.columns.tolist() == X_oos.columns.tolist() == feature_cols
    X_fit = np.asfortranarray(X_train.to_numpy(dtype=np.float64))
    y_fit = y_train.to_numpy(dtype=np.float64)
    X_predict = X_oos.to_numpy(dtype=np.float64)
    assert all((np.isfinite(x).all() for x in (X_fit, y_fit, X_predict)))
    start = time.perf_counter()
    model = ElasticNet(**ENET_PARAMS)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(X_fit, y_fit)
    train_seconds = time.perf_counter() - start
    assert not any((issubclass(w.category, ConvergenceWarning) for w in caught)), f'Elastic Net did not converge: {stage}, {window}, {oos_period}'
    predicted = model.predict(X_predict)
    assert len(predicted) == len(signal_idx) and np.isfinite(predicted).all()
    prediction_df = data.loc[signal_idx, ['PERMNO', 'Ticker', 'DlyCalDt', 'Split']].copy()
    prediction_df['Predicted'] = predicted
    coefficient_df = pd.DataFrame({'Stage': stage, 'Window': window, 'OOS_Period': str(oos_period), 'Feature': feature_cols, 'Coefficient': model.coef_})
    coefficient_df['Abs_Coefficient'] = coefficient_df['Coefficient'].abs()
    metrics = {'Stage': stage, 'Window': window, 'OOS_Period': str(oos_period), 'Train_Rows': len(X_train), 'OOS_Signal_Rows': len(X_oos), 'OOS_Eval_Rows': len(eval_idx), 'Nonzero_Features': int((np.abs(model.coef_) > 1e-12).sum()), 'Iterations': int(model.n_iter_), 'Train_Seconds': train_seconds, 'MAE': np.nan, 'OOS_R2': np.nan, 'Rank_IC': np.nan, 'IC_Days': np.nan, 'Status': 'PASSED'}
    if evaluate:
        assert data.loc[eval_idx, 'Split'].eq('Train').all(), 'Independent Validation targets must not be evaluated in Cell 4.'
        actual = data.loc[eval_idx, target_col].to_numpy(dtype=np.float64)
        predicted_eval = prediction_df.loc[eval_idx, 'Predicted'].to_numpy(dtype=np.float64)
        assert np.isfinite(actual).all() and np.isfinite(predicted_eval).all()
        errors = actual - predicted_eval
        mae = float(np.abs(errors).mean())
        mse = float(np.square(errors).mean())
        benchmark_mse = float(np.square(actual - y_train.mean()).mean())
        assert benchmark_mse > 0
        eval_frame = prediction_df.loc[eval_idx, ['DlyCalDt', 'Predicted']].copy()
        eval_frame['Actual'] = actual
        assert eval_frame.groupby('DlyCalDt').size().eq(9).all()
        daily_ic = calculate_daily_rank_ic(eval_frame)
        metrics.update({'MAE': mae, 'OOS_R2': 1.0 - mse / benchmark_mse, 'Rank_IC': float(daily_ic.mean()), 'IC_Days': int(daily_ic.notna().sum())})
    return {'model': model, 'predictions': prediction_df, 'coefficients': coefficient_df, 'metrics': metrics}
enet_development = {}
all_results, all_coefficients = ([], [])
total_start = time.perf_counter()
for (window, year), fold in sorted(scaled_development_folds.items()):
    output = fit_elastic_net_fold(fold, 'Development', window, year)
    enet_development[window, year] = output
    all_results.append(output['metrics'])
    all_coefficients.append(output['coefficients'])
    m = output['metrics']
    pass
enet_train_oos = {}
for year, fold in sorted(scaled_train_oos_folds.items()):
    output = fit_elastic_net_fold(fold, 'Train_OOS', year - TRAIN_OOS_START_YEAR, year)
    enet_train_oos[year] = output
    all_results.append(output['metrics'])
    all_coefficients.append(output['coefficients'])
    m = output['metrics']
    pass
enet_validation = fit_elastic_net_fold(scaled_validation_fold, 'Validation', 7, '2022-2023', evaluate=False)
all_results.append(enet_validation['metrics'])
all_coefficients.append(enet_validation['coefficients'])
pass
enet_results_df = pd.DataFrame(all_results)
enet_coefficients_df = pd.concat(all_coefficients, ignore_index=True)
enet_train_oos_predictions = pd.concat([enet_train_oos[year]['predictions'] for year in sorted(enet_train_oos)]).sort_values(['DlyCalDt', 'Ticker'])
enet_validation_predictions = enet_validation['predictions'].sort_values(['DlyCalDt', 'Ticker']).copy()
assert len(enet_development) == 6
assert len(enet_train_oos) == 4
assert len(enet_results_df) == 11
assert len(enet_coefficients_df) == 11 * len(feature_cols)
assert enet_results_df['Status'].eq('PASSED').all()
assert enet_results_df['Nonzero_Features'].between(0, 31).all()
assert len(enet_train_oos_predictions) == sum((len(train_oos_folds[y]['oos_signal_idx']) for y in TRAIN_OOS_YEARS))
assert len(enet_validation_predictions) == 4509
assert enet_train_oos_predictions['Split'].eq('Train').all()
assert enet_validation_predictions['Split'].eq('Validation').all()
assert enet_results_df.loc[enet_results_df['Stage'].eq('Validation'), ['MAE', 'OOS_R2', 'Rank_IC', 'IC_Days']].isna().all().all()
total_seconds = time.perf_counter() - total_start
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
