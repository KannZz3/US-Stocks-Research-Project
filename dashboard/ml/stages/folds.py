import numpy as np
import pandas as pd
HORIZON = 10
DEV_WINDOWS = [3, 4, 5]
DEV_YEARS = [2020, 2021]
TRAIN_OOS_YEARS = [2018, 2019, 2020, 2021]
TRAIN_OOS_START_YEAR = 2015
assert target_col == 'TARGET_10'
assert len(feature_cols) == 31
assert data['Ticker'].nunique() == 9
assert len(train_pool) == 15516

def get_period(start_year, end_year, split):
    period = data.loc[data['Split'].eq(split) & data['DlyCalDt'].dt.year.between(start_year, end_year)].copy()
    assert not period.empty
    assert period['Ticker'].nunique() == 9
    assert period.groupby('DlyCalDt')['PERMNO'].nunique().eq(9).all()
    return period

def get_realized_indices(period, horizon=HORIZON):
    target_end = period.groupby('PERMNO', sort=False)['DlyCalDt'].shift(-horizon)
    realized_idx = period.index[target_end.notna()].copy()
    excluded_idx = period.index[target_end.isna()].copy()
    assert len(excluded_idx) == period['PERMNO'].nunique() * horizon
    assert (target_end.loc[realized_idx] > period.loc[realized_idx, 'DlyCalDt']).all()
    assert (target_end.loc[realized_idx] <= period['DlyCalDt'].max()).all()
    return (realized_idx, excluded_idx)

def build_fold(train_start_year, train_end_year, oos_start_year, oos_end_year, oos_split):
    train_period = get_period(train_start_year, train_end_year, 'Train')
    oos_period = get_period(oos_start_year, oos_end_year, oos_split)
    assert train_period['DlyCalDt'].max() < oos_period['DlyCalDt'].min()
    train_fit_idx, train_excluded_idx = get_realized_indices(train_period, HORIZON)
    oos_signal_idx = oos_period.index.copy()
    oos_eval_idx, oos_excluded_idx = get_realized_indices(oos_period, HORIZON)
    assert train_fit_idx.intersection(oos_signal_idx).empty
    assert len(train_excluded_idx) == 9 * HORIZON
    assert len(oos_excluded_idx) == 9 * HORIZON
    assert len(train_fit_idx) + len(train_excluded_idx) == len(train_period)
    assert len(oos_eval_idx) + len(oos_excluded_idx) == len(oos_period)
    assert data.loc[train_fit_idx, 'Split'].eq('Train').all()
    assert data.loc[oos_signal_idx, 'Split'].eq(oos_split).all()
    assert np.isfinite(data.loc[train_fit_idx, feature_cols + [target_col]].to_numpy(dtype=float)).all()
    return {'train_fit_idx': train_fit_idx, 'train_full_idx': train_period.index.copy(), 'train_excluded_idx': train_excluded_idx, 'oos_signal_idx': oos_signal_idx, 'oos_eval_idx': oos_eval_idx, 'oos_excluded_idx': oos_excluded_idx, 'train_start': train_period['DlyCalDt'].min(), 'train_cutoff': train_period['DlyCalDt'].max(), 'train_fit_end': data.loc[train_fit_idx, 'DlyCalDt'].max(), 'oos_start': oos_period['DlyCalDt'].min(), 'oos_end': oos_period['DlyCalDt'].max(), 'oos_split': oos_split}
development_folds = {}
development_rows = []
for val_year in DEV_YEARS:
    for window in DEV_WINDOWS:
        fold = build_fold(val_year - window, val_year - 1, val_year, val_year, 'Train')
        development_folds[window, val_year] = fold
        development_rows.append({'Window': window, 'OOS_Year': val_year, 'Train_Start': fold['train_start'], 'Train_Fit_End': fold['train_fit_end'], 'Train_Cutoff': fold['train_cutoff'], 'Train_Fit_Rows': len(fold['train_fit_idx']), 'OOS_Signal_Rows': len(fold['oos_signal_idx']), 'OOS_Eval_Rows': len(fold['oos_eval_idx']), 'Train_Purged': len(fold['train_excluded_idx']), 'OOS_Eval_Excluded': len(fold['oos_excluded_idx'])})
development_summary_df = pd.DataFrame(development_rows).sort_values(['OOS_Year', 'Window']).reset_index(drop=True)
train_oos_folds = {}
train_oos_rows = []
for oos_year in TRAIN_OOS_YEARS:
    fold = build_fold(TRAIN_OOS_START_YEAR, oos_year - 1, oos_year, oos_year, 'Train')
    train_oos_folds[oos_year] = fold
    train_oos_rows.append({'OOS_Year': oos_year, 'Train_Start': fold['train_start'], 'Train_Fit_End': fold['train_fit_end'], 'Train_Cutoff': fold['train_cutoff'], 'Train_Fit_Rows': len(fold['train_fit_idx']), 'OOS_Start': fold['oos_start'], 'OOS_End': fold['oos_end'], 'OOS_Signal_Rows': len(fold['oos_signal_idx']), 'OOS_Eval_Rows': len(fold['oos_eval_idx']), 'Train_Purged': len(fold['train_excluded_idx'])})
train_oos_summary_df = pd.DataFrame(train_oos_rows).sort_values('OOS_Year').reset_index(drop=True)
all_oos_indices = np.concatenate([fold['oos_signal_idx'].to_numpy() for fold in train_oos_folds.values()])
assert len(all_oos_indices) == len(np.unique(all_oos_indices))
validation_fold = build_fold(2015, 2021, 2022, 2023, 'Validation')
assert len(validation_fold['train_full_idx']) == 15516
assert len(validation_fold['train_fit_idx']) == 15426
assert len(validation_fold['oos_signal_idx']) == 4509
assert len(validation_fold['oos_eval_idx']) == 4419
assert len(development_folds) == 6
assert len(train_oos_folds) == 4
assert all((fold['oos_split'] == 'Train' for fold in list(development_folds.values()) + list(train_oos_folds.values())))
assert validation_fold['oos_split'] == 'Validation'
all_folds = list(development_folds.values()) + list(train_oos_folds.values()) + [validation_fold]
assert all((not data.loc[fold['train_full_idx'].union(fold['oos_signal_idx']), 'Split'].eq('Test').any() for fold in all_folds))
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
