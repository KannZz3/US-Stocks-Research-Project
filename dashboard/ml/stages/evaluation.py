import numpy as np
import pandas as pd
C7_H = HORIZON
C7_MIN_VOTES = C6_MIN_VOTES
C7_BOOTSTRAPS = 1500
C7_RANDOM_SEED = 20261107
C7_DIRECTIONS = ('Long', 'Short')
assert target_col == 'TARGET_10' and HORIZON == C7_H
assert C6_MIN_VOTES == C7_MIN_VOTES == C51_MIN_VOTES == 3
assert C7_DIRECTIONS == tuple(C6_SIDES)
assert len(tech_tickers) == 9 and data.index.is_unique
assert set(cell6_signal_panel['Period'].unique()) == {'Train_OOS', 'Validation'}
assert set(cell6_signal_panel['Year'].unique()) == set(range(2018, 2024))
assert not cell6_signal_panel.index.duplicated().any()
assert not cell6_signal_panel['DlyCalDt'].isna().any()
C7_FIELDS = ['Period', 'Year', 'DlyCalDt', 'Ticker', 'PERMNO', 'Candidate_Direction', 'Candidate_Votes', 'Candidate_Strength', 'Daily_Rank_All', 'Daily_Conflict', 'Long_Qualified', 'Short_Qualified', 'Long_Candidate', 'Short_Candidate', 'Long_Family_Count', 'Short_Family_Count', 'Long_Matched_Factors', 'Short_Matched_Factors']
assert all((c in cell6_signal_panel.columns for c in C7_FIELDS))
cell7_eval_panel = cell6_signal_panel[C7_FIELDS].copy()
cell7_eval_panel['DlyCalDt'] = pd.to_datetime(cell7_eval_panel['DlyCalDt'])
cell7_eval_panel['Year'] = cell7_eval_panel['Year'].astype(int)
cell7_eval_panel['Fold_ID'] = np.where(cell7_eval_panel['Period'].eq('Train_OOS'), cell7_eval_panel['Year'].astype(str), '2022-2023')
cell7_eval_panel = cell7_eval_panel.sort_values(['Period', 'Fold_ID', 'PERMNO', 'DlyCalDt'], kind='mergesort')
assert cell7_eval_panel.index.isin(data.index).all()
source = data.loc[cell7_eval_panel.index, ['DlyCalDt', 'Ticker', 'PERMNO', 'Split', target_col]]
assert (pd.to_datetime(source['DlyCalDt']).to_numpy() == cell7_eval_panel['DlyCalDt'].to_numpy()).all()
assert (source['Ticker'].to_numpy() == cell7_eval_panel['Ticker'].to_numpy()).all()
assert (source['PERMNO'].to_numpy() == cell7_eval_panel['PERMNO'].to_numpy()).all()
assert source.loc[cell7_eval_panel['Period'].eq('Train_OOS'), 'Split'].eq('Train').all()
assert source.loc[cell7_eval_panel['Period'].eq('Validation'), 'Split'].eq('Validation').all()
cell7_eval_panel['TARGET_10_OOS'] = pd.to_numeric(source[target_col], errors='coerce').to_numpy(dtype=float)
fold_stock = ['Period', 'Fold_ID', 'PERMNO']
g = cell7_eval_panel.groupby(fold_stock, sort=False, observed=True)
assert g['DlyCalDt'].apply(lambda d: d.is_monotonic_increasing).all()
cell7_eval_panel['OOS_Bar'] = g.cumcount()
cell7_eval_panel['Target_Maturity_Date'] = g['DlyCalDt'].shift(-C7_H)
cell7_eval_panel['Matured_10D'] = cell7_eval_panel['Target_Maturity_Date'].notna() & np.isfinite(cell7_eval_panel['TARGET_10_OOS'])
assert cell7_eval_panel.groupby(fold_stock)['Target_Maturity_Date'].apply(lambda s: int(s.notna().sum()) == max(len(s) - C7_H, 0)).all()
assert not cell7_eval_panel.loc[cell7_eval_panel['Daily_Conflict'], 'Candidate_Direction'].ne('NONE').any()
assert not cell7_eval_panel.loc[cell7_eval_panel['Candidate_Direction'].ne('NONE'), 'Daily_Conflict'].any()
rank_check = cell7_eval_panel.loc[cell7_eval_panel['Candidate_Direction'].ne('NONE')]
assert rank_check['Candidate_Votes'].between(C7_MIN_VOTES, C51_N).all()
assert rank_check['Daily_Rank_All'].notna().all()
assert not (cell7_eval_panel['Long_Candidate'] & cell7_eval_panel['Short_Candidate']).any()
cell7_eval_panel['Stock_Year_Benchmark'] = np.nan
cell7_eval_panel['Stock_Year_Base_Hit'] = np.nan
valid_all = cell7_eval_panel['Matured_10D']
base_keys = ['Period', 'Year', 'Ticker']
base_rows = cell7_eval_panel.loc[valid_all].copy()
base_rows['Stock_Year_Benchmark'] = base_rows.groupby(base_keys, observed=True)['TARGET_10_OOS'].transform('mean')
base_rows['Stock_Year_Base_Hit'] = base_rows.groupby(base_keys, observed=True)['TARGET_10_OOS'].transform(lambda x: float((x > 0).mean()))
cell7_eval_panel.loc[base_rows.index, 'Stock_Year_Benchmark'] = base_rows['Stock_Year_Benchmark'].to_numpy()
cell7_eval_panel.loc[base_rows.index, 'Stock_Year_Base_Hit'] = base_rows['Stock_Year_Base_Hit'].to_numpy()
all_candidate_mask = cell7_eval_panel['Candidate_Direction'].ne('NONE')
valid_signal_mask = all_candidate_mask & cell7_eval_panel['Matured_10D']
cell7_signals = cell7_eval_panel.loc[valid_signal_mask].copy()
side_mult = np.where(cell7_signals['Candidate_Direction'].eq('Long'), 1.0, -1.0)
cell7_signals['Signed_Return_10'] = side_mult * cell7_signals['TARGET_10_OOS'].to_numpy(dtype=float)
cell7_signals['Benchmark_Signed_10'] = side_mult * cell7_signals['Stock_Year_Benchmark'].to_numpy(dtype=float)
cell7_signals['Excess_Return_10'] = cell7_signals['Signed_Return_10'] - cell7_signals['Benchmark_Signed_10']
cell7_signals['Win'] = cell7_signals['Signed_Return_10'].gt(0)
base_hit_short = base_rows.assign(Down=base_rows['TARGET_10_OOS'].lt(0)).groupby(base_keys, observed=True)['Down'].mean().rename('Short_Benchmark_Hit')
cell7_signals = cell7_signals.join(base_hit_short, on=base_keys, how='left')
cell7_signals['Benchmark_Win_Rate'] = np.where(side_mult > 0, cell7_signals['Stock_Year_Base_Hit'], cell7_signals['Short_Benchmark_Hit'])
cell7_signals['Hit_Lift_pp'] = 100 * (cell7_signals['Win'].astype(float) - cell7_signals['Benchmark_Win_Rate'])
cell7_signals['Rank'] = cell7_signals['Daily_Rank_All'].astype(int)
cell7_signals['Votes'] = cell7_signals['Candidate_Votes'].astype(int)
cell7_signals['Direction'] = cell7_signals['Candidate_Direction']
cell7_signals['Strength'] = cell7_signals['Candidate_Strength']
cell7_signals['Family_Count'] = np.where(cell7_signals['Direction'].eq('Long'), cell7_signals['Long_Family_Count'], cell7_signals['Short_Family_Count']).astype(int)
cell7_signals['Signal_Month'] = cell7_signals['DlyCalDt'].dt.to_period('M')
assert np.isfinite(cell7_signals[['Signed_Return_10', 'Excess_Return_10', 'Benchmark_Signed_10', 'Strength']]).all().all()
assert cell7_signals['Votes'].between(C7_MIN_VOTES, C51_N).all()
assert not cell7_signals['Daily_Conflict'].any()
assert not cell7_signals['Target_Maturity_Date'].isna().any()
cell7_signals['Thin_10D'] = False
for key, block in cell7_signals.groupby(fold_stock, sort=False, observed=True):
    block = block.sort_values('OOS_Bar', kind='mergesort')
    last_bar = -C7_H
    taken = []
    for row_index, bar_number in zip(block.index, block['OOS_Bar'].to_numpy()):
        if int(bar_number) - last_bar >= C7_H:
            taken.append(row_index)
            last_bar = int(bar_number)
    if taken:
        cell7_signals.loc[taken, 'Thin_10D'] = True
cell7_thin_signals = cell7_signals.loc[cell7_signals['Thin_10D']].copy()
for _, block in cell7_thin_signals.groupby(fold_stock, sort=False, observed=True):
    assert (block.sort_values('OOS_Bar')['OOS_Bar'].diff().dropna() >= C7_H).all()

def c7_summary(frame, by):
    columns = by + ['N', 'Signal_Days', 'Stocks', 'Hit_Rate', 'Mean_Dir_Return', 'Mean_Excess_Return', 'Median_Dir_Return', 'Hit_Lift_pp', 'Mean_Votes', 'Mean_Strength', 'Mean_Families']
    if frame.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for name, block in frame.groupby(by, sort=True, dropna=False, observed=True):
        key = name if isinstance(name, tuple) else (name,)
        rec = dict(zip(by, key))
        rec.update({'N': len(block), 'Signal_Days': block['DlyCalDt'].nunique(), 'Stocks': block['Ticker'].nunique(), 'Hit_Rate': block['Win'].mean(), 'Mean_Dir_Return': block['Signed_Return_10'].mean(), 'Mean_Excess_Return': block['Excess_Return_10'].mean(), 'Median_Dir_Return': block['Signed_Return_10'].median(), 'Hit_Lift_pp': block['Hit_Lift_pp'].mean(), 'Mean_Votes': block['Votes'].mean(), 'Mean_Strength': block['Strength'].mean(), 'Mean_Families': block['Family_Count'].mean()})
        rows.append(rec)
    return pd.DataFrame(rows)[columns]
cell7_annual_df = c7_summary(cell7_signals, ['Period', 'Year', 'Direction'])
cell7_pooled_df = c7_summary(cell7_signals, ['Period', 'Direction'])
cell7_vote_df = c7_summary(cell7_signals, ['Period', 'Year', 'Direction', 'Votes'])
cell7_stock_df = c7_summary(cell7_signals, ['Period', 'Direction', 'Ticker'])
cell7_family_df = c7_summary(cell7_signals, ['Period', 'Direction', 'Family_Count'])
cell7_thin_annual_df = c7_summary(cell7_thin_signals, ['Period', 'Year', 'Direction'])
cell7_thin_pooled_df = c7_summary(cell7_thin_signals, ['Period', 'Direction'])
rank_frames = []
for n in (1, 2, 3):
    temp = cell7_signals.loc[cell7_signals['Rank'] <= n].copy()
    temp['Rank_Group'] = f'Top{n}'
    rank_frames.append(temp)
rank_frames.append(cell7_signals.loc[cell7_signals['Rank'] >= 4].assign(Rank_Group='Rank4plus'))
cell7_rank_rows = pd.concat(rank_frames, ignore_index=True)
cell7_rank_df = c7_summary(cell7_rank_rows, ['Period', 'Year', 'Direction', 'Rank_Group'])
cell7_rank_pooled_df = c7_summary(cell7_rank_rows, ['Period', 'Direction', 'Rank_Group'])

def c7_month_bootstrap(frame, n_boot=C7_BOOTSTRAPS, seed=C7_RANDOM_SEED):
    out = []
    for k, ((period, side), block) in enumerate(frame.groupby(['Period', 'Direction'], sort=True, observed=True)):
        monthly = block.groupby('Signal_Month', observed=True).agg(N=('Signed_Return_10', 'size'), Sum_Return=('Signed_Return_10', 'sum'), Sum_Excess=('Excess_Return_10', 'sum'))
        m = len(monthly)
        if m < 6:
            out.append({'Period': period, 'Direction': side, 'N': len(block), 'Months_With_Signals': m, 'Mean_Dir_Return': block['Signed_Return_10'].mean(), 'Dir_Boot_Lo': np.nan, 'Dir_Boot_Hi': np.nan, 'Mean_Excess_Return': block['Excess_Return_10'].mean(), 'Excess_Boot_Lo': np.nan, 'Excess_Boot_Hi': np.nan})
            continue
        vals = monthly[['N', 'Sum_Return', 'Sum_Excess']].to_numpy(dtype=float)
        rng = np.random.default_rng(seed + k)
        chosen = rng.integers(0, m, size=(n_boot, m))
        totals = vals[chosen].sum(axis=1)
        d = totals[:, 1] / totals[:, 0]
        e = totals[:, 2] / totals[:, 0]
        out.append({'Period': period, 'Direction': side, 'N': len(block), 'Months_With_Signals': m, 'Mean_Dir_Return': block['Signed_Return_10'].mean(), 'Dir_Boot_Lo': np.quantile(d, 0.025), 'Dir_Boot_Hi': np.quantile(d, 0.975), 'Mean_Excess_Return': block['Excess_Return_10'].mean(), 'Excess_Boot_Lo': np.quantile(e, 0.025), 'Excess_Boot_Hi': np.quantile(e, 0.975)})
    columns = ['Period', 'Direction', 'N', 'Months_With_Signals', 'Mean_Dir_Return', 'Dir_Boot_Lo', 'Dir_Boot_Hi', 'Mean_Excess_Return', 'Excess_Boot_Lo', 'Excess_Boot_Hi']
    return pd.DataFrame(out, columns=columns)
cell7_month_bootstrap_df = c7_month_bootstrap(cell7_signals)
cell7_exclusion_df = cell7_eval_panel.assign(Qualifying=all_candidate_mask, Unmatured_Qualifying=all_candidate_mask & ~cell7_eval_panel['Matured_10D'], Matured_Qualifying=valid_signal_mask).groupby(['Period', 'Year'], observed=True).agg(Stock_Days=('Ticker', 'size'), Qualifying=('Qualifying', 'sum'), Unmatured_Qualifying=('Unmatured_Qualifying', 'sum'), Evaluated=('Matured_Qualifying', 'sum'), Unmatured_All=('Matured_10D', lambda x: int((~x).sum()))).reset_index()
assert (cell7_exclusion_df['Qualifying'] == cell7_exclusion_df['Unmatured_Qualifying'] + cell7_exclusion_df['Evaluated']).all()
assert len(cell7_signals) + cell7_exclusion_df['Unmatured_Qualifying'].sum() == len(cell6_candidates)
assert cell7_eval_panel['Year'].max() == 2023
assert cell7_exclusion_df['Evaluated'].sum() == len(cell7_signals)
for year, selection in sorted(factor_six_train_oos.items()):
    year_rows = cell7_eval_panel.loc[cell7_eval_panel['Period'].eq('Train_OOS') & cell7_eval_panel['Year'].eq(year)]
    for side in C7_DIRECTIONS:
        if not selection['enabled'][side]:
            assert not year_rows[f'{side}_Candidate'].any()
for side in C7_DIRECTIONS:
    if not factor_six_validation['enabled'][side]:
        assert not cell7_eval_panel.loc[cell7_eval_panel['Period'].eq('Validation'), f'{side}_Candidate'].any()

def c7_show_inactive(summary, by):
    if by == ['Period', 'Year', 'Direction']:
        periods = cell7_eval_panel[['Period', 'Year']].drop_duplicates()
        universe = periods.merge(pd.DataFrame({'Direction': C7_DIRECTIONS}), how='cross')
    elif by == ['Period', 'Direction']:
        universe = cell7_eval_panel[['Period']].drop_duplicates().merge(pd.DataFrame({'Direction': C7_DIRECTIONS}), how='cross')
    else:
        return summary
    result = universe.merge(summary, how='left', on=by, validate='one_to_one')
    for count_col in ('N', 'Signal_Days', 'Stocks'):
        result[count_col] = pd.to_numeric(result[count_col], errors='coerce').fillna(0).astype(int)
    return result.sort_values(by).reset_index(drop=True)
cell7_annual_df = c7_show_inactive(cell7_annual_df, ['Period', 'Year', 'Direction'])
cell7_pooled_df = c7_show_inactive(cell7_pooled_df, ['Period', 'Direction'])
cell7_thin_annual_df = c7_show_inactive(cell7_thin_annual_df, ['Period', 'Year', 'Direction'])
cell7_thin_pooled_df = c7_show_inactive(cell7_thin_pooled_df, ['Period', 'Direction'])
assert cell7_annual_df.loc[cell7_annual_df['N'].eq(0), ['Hit_Rate', 'Mean_Dir_Return', 'Mean_Excess_Return']].isna().all().all()
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
