import numpy as np
import pandas as pd
C6_MIN_VOTES = C51_MIN_VOTES
C6_FACTOR_Q = C51_FACTOR_Q
C6_SIDES = tuple(C51_SIDES)
C6_TICKERS = list(tech_tickers)
C6_EXPECTED_TRAIN_YEARS = (2018, 2019, 2020, 2021)
C6_FEATURES = list(feature_cols)
assert C6_MIN_VOTES == 3 and C51_MIN_ACTIVE == 4 and (C51_N == 6)
assert len(C6_FEATURES) == 31 and len(C6_TICKERS) == len(set(C6_TICKERS)) == 9
assert 0 < C6_FACTOR_Q < 1
assert sorted(factor_six_train_oos) == list(C6_EXPECTED_TRAIN_YEARS)
assert len(factor_six_development) == 6 and data.index.is_unique
assert all(('oos_signal_idx' in train_oos_folds[y] for y in C6_EXPECTED_TRAIN_YEARS))
assert 'oos_signal_idx' in validation_fold

def c6_favorable_pct(history, observed, orientation):
    """Per-stock favorable ECDF fitted on outer Train observations only."""
    hist = np.sort(np.asarray(history, dtype=np.float64))
    x = np.asarray(observed, dtype=np.float64)
    assert len(hist) >= 30 and np.isfinite(hist).all() and np.isfinite(x).all()
    if hist[0] == hist[-1]:
        return np.zeros(len(x), dtype=np.float64)
    if orientation == 1:
        pct = np.searchsorted(hist, x, side='right') / len(hist)
    elif orientation == -1:
        pct = (len(hist) - np.searchsorted(hist, x, side='left')) / len(hist)
    else:
        raise ValueError('Selected factor orientation must be +1 or -1.')
    return np.clip(pct, 0.0, 1.0)

def c6_factor_family(name):
    """Factor-family labels for diagnostics; never used as voting filters."""
    if name.startswith('RET_'):
        return 'Returns'
    if name.startswith(('MA_DEV_', 'TREND_', 'BREAKOUT_', 'UP_RATIO_', 'EFF_RATIO_')):
        return 'Trend/Price'
    if name == 'CLV_1':
        return 'Intraday_Position'
    if name.startswith(('GAP_', 'OVERNIGHT_RET_')):
        return 'Gap/Overnight'
    if name.startswith(('RV_', 'ATR_', 'RANGE_', 'DOWNSIDE_', 'PARKINSON_', 'OVERNIGHT_VOL_')):
        return 'Volatility'
    if name.startswith('MAX_DRAWDOWN_'):
        return 'Drawdown'
    if name.startswith(('SIGNED_DVOL_', 'REL_DVOL_', 'LOG_DVOL_', 'TURNOVER_', 'LOG_ILLIQ_', 'RET_DVOL_CORR_')):
        return 'Volume/Liquidity'
    if name == 'LOG_MCAP':
        return 'Size'
    return 'Other'

def c6_quality_weights(factor_output, side, factors):
    if not factors:
        return np.empty(0, dtype=np.float64)
    required = {'Direction', 'Feature', 'Direction_Tier', 'Candidate', 'Composite_Score'}
    ranking = factor_output['rankings']
    assert required.issubset(ranking.columns)
    ranked = ranking.loc[ranking['Direction'].eq(side)].set_index('Feature')
    assert ranked.index.is_unique and set(factors).issubset(ranked.index)
    picked = ranked.loc[factors]
    assert picked['Direction_Tier'].eq(3).all() and picked['Candidate'].all()
    scores = picked['Composite_Score'].to_numpy(dtype=np.float64)
    assert np.isfinite(scores).all() and (scores > 0).all()
    weights = scores / scores.sum()
    assert np.isclose(weights.sum(), 1.0)
    return weights

def c6_build_outer(outer, selection, factor_output, stage, expected_years):
    train_idx = outer['train_fit_idx']
    signal_idx = outer['oos_signal_idx']
    assert len(train_idx) > 0 and len(signal_idx) > 0
    assert pd.Index(signal_idx).is_unique
    columns = ['PERMNO', 'Ticker', 'DlyCalDt', 'Split']
    for side in C6_SIDES:
        chosen = list(selection['selected'][side])
        assert len(chosen) == len(set(chosen))
        assert len(chosen) == 0 or C51_MIN_ACTIVE <= len(chosen) <= C51_N
        assert selection['enabled'][side] == bool(chosen)
        assert set(chosen).issubset(C6_FEATURES)
        assert set(selection['orientations'][side]) == set(chosen)
        assert all((selection['orientations'][side][f] in (-1, 1) for f in chosen))
        columns.extend(chosen)
    if selection['force_flat']:
        assert all((not selection['enabled'][side] for side in C6_SIDES))
    columns = list(dict.fromkeys(columns))
    train = data.loc[train_idx, columns].copy()
    oos = data.loc[signal_idx, columns].copy()
    train['DlyCalDt'] = pd.to_datetime(train['DlyCalDt'])
    oos['DlyCalDt'] = pd.to_datetime(oos['DlyCalDt'])
    assert train['DlyCalDt'].max() < oos['DlyCalDt'].min()
    assert set(train['Ticker']) == set(oos['Ticker']) == set(C6_TICKERS)
    assert set(oos['DlyCalDt'].dt.year) == set(expected_years)
    assert train['Split'].eq('Train').all()
    assert oos['Split'].eq('Train' if stage == 'Train_OOS' else 'Validation').all()
    assert train.notna().all().all() and oos.notna().all().all()
    panel = oos[['PERMNO', 'Ticker', 'DlyCalDt', 'Split']].copy()
    panel.insert(0, 'Source_Row_Index', panel.index)
    panel['Period'] = stage
    panel['Year'] = panel['DlyCalDt'].dt.year
    panel['History_Start'] = train['DlyCalDt'].min()
    panel['History_Cutoff'] = train['DlyCalDt'].max()
    train_ticker = train['Ticker'].to_numpy()
    oos_ticker = oos['Ticker'].to_numpy()
    for side in C6_SIDES:
        factors = list(selection['selected'][side])
        signs = selection['orientations'][side]
        weights = c6_quality_weights(factor_output, side, factors)
        scores = np.zeros((len(panel), len(factors)), dtype=np.float64)
        if factors:
            for ticker in C6_TICKERS:
                a, b = (train_ticker == ticker, oos_ticker == ticker)
                assert a.sum() >= 30 and b.any(), (stage, ticker)
                for j, f in enumerate(factors):
                    scores[b, j] = c6_favorable_pct(train.loc[a, f].to_numpy(), oos.loc[b, f].to_numpy(), int(signs[f]))
        confirmed = scores >= C6_FACTOR_Q
        votes = confirmed.sum(axis=1).astype(int)
        strength = np.divide((scores * confirmed).sum(axis=1), votes, out=np.zeros(len(panel), dtype=float), where=votes > 0)
        priority = scores * confirmed @ weights
        panel[f'{side}_Priority'] = priority
        panel[f'{side}_N_Factors'] = len(factors)
        panel[f'{side}_Enabled'] = bool(factors)
        panel[f'{side}_Votes'] = votes
        panel[f'{side}_Strength'] = strength
        panel[f'{side}_Qualified'] = bool(factors) & (votes >= C6_MIN_VOTES)
        panel[f'{side}_Selected_Factors'] = '|'.join(factors)
        matched, families, breadth = ([], [], [])
        for flags in confirmed:
            hits = [f for f, ok in zip(factors, flags) if ok]
            groups = sorted({c6_factor_family(f) for f in hits})
            matched.append('|'.join(hits))
            families.append('|'.join(groups))
            breadth.append(len(groups))
        panel[f'{side}_Matched_Factors'] = matched
        panel[f'{side}_Matched_Families'] = families
        panel[f'{side}_Family_Count'] = breadth
        for j, f in enumerate(factors):
            panel[f'{side}_Pct__{f}'] = scores[:, j]
            panel[f'{side}_OK__{f}'] = confirmed[:, j]
        assert np.array_equal(votes, confirmed.sum(axis=1))
        assert ((strength >= C6_FACTOR_Q - 1e-12) | (votes == 0)).all()
        if not factors:
            assert not panel[f'{side}_Qualified'].any()
    panel['Signal_Conflict'] = panel['Long_Qualified'] & panel['Short_Qualified']
    by_day = panel.groupby('DlyCalDt', sort=False)[['Long_Qualified', 'Short_Qualified']].transform('any')
    panel['Daily_Conflict'] = by_day['Long_Qualified'] & by_day['Short_Qualified']
    panel['New_Entries_Allowed'] = ~panel['Daily_Conflict']
    for side in C6_SIDES:
        panel[f'{side}_Cancelled_By_Daily_Conflict'] = panel[f'{side}_Qualified'] & panel['Daily_Conflict']
        panel[f'{side}_Candidate'] = panel[f'{side}_Qualified'] & panel['New_Entries_Allowed']
    panel['Candidate_Direction'] = np.select([panel['Long_Candidate'], panel['Short_Candidate']], ['Long', 'Short'], default='NONE')
    panel['Candidate_Votes'] = np.select([panel['Long_Candidate'], panel['Short_Candidate']], [panel['Long_Votes'], panel['Short_Votes']], default=0).astype(int)
    panel['Candidate_Strength'] = np.select([panel['Long_Candidate'], panel['Short_Candidate']], [panel['Long_Strength'], panel['Short_Strength']], default=np.nan)
    panel['Candidate_Priority'] = np.select([panel['Long_Candidate'], panel['Short_Candidate']], [panel['Long_Priority'], panel['Short_Priority']], default=np.nan)
    assert np.isfinite(panel.loc[panel['Candidate_Direction'].ne('NONE'), 'Candidate_Priority']).all()
    panel['Trade_Signal_Available'] = 'AFTER_CLOSE'
    panel['Earliest_Execution'] = 'NEXT_TRADING_DAY'
    assert not (panel['Long_Candidate'] & panel['Short_Candidate']).any()
    assert panel.loc[panel['Daily_Conflict'], 'Candidate_Direction'].eq('NONE').all()
    assert not panel[['DlyCalDt', 'Ticker']].duplicated().any()
    if not any(selection['enabled'].values()):
        assert panel['Candidate_Direction'].eq('NONE').all()
        assert not panel['Daily_Conflict'].any()
    if sum((bool(selection['enabled'][s]) for s in C6_SIDES)) < 2:
        assert not panel['Daily_Conflict'].any()
    return panel
cell6_by_year = {}
for year in C6_EXPECTED_TRAIN_YEARS:
    cell6_by_year[year] = c6_build_outer(train_oos_folds[year], factor_six_train_oos[year], factor_train_oos[year], 'Train_OOS', [year])
cell6_train_signals = pd.concat(cell6_by_year.values())
cell6_validation_signals = c6_build_outer(validation_fold, factor_six_validation, factor_validation, 'Validation', [2022, 2023])
cell6_signal_panel = pd.concat([cell6_train_signals, cell6_validation_signals]).sort_values(['DlyCalDt', 'Ticker'], kind='mergesort').copy()
assert cell6_signal_panel.index.is_unique
assert len(cell6_signal_panel) == sum((len(f['oos_signal_idx']) for f in train_oos_folds.values())) + len(validation_fold['oos_signal_idx'])
assert cell6_signal_panel['Year'].between(2018, 2023).all()
cell6_daily_conflicts_df = cell6_signal_panel.groupby(['Period', 'Year', 'DlyCalDt'], sort=True).agg(Original_Long_3plus=('Long_Qualified', 'sum'), Original_Short_3plus=('Short_Qualified', 'sum'), Daily_Conflict=('Daily_Conflict', 'first'), Cancelled_Long=('Long_Cancelled_By_Daily_Conflict', 'sum'), Cancelled_Short=('Short_Cancelled_By_Daily_Conflict', 'sum'), Long_Eligible=('Long_Candidate', 'sum'), Short_Eligible=('Short_Candidate', 'sum')).reset_index()
cell6_daily_conflicts_df['Cancelled_Signals'] = cell6_daily_conflicts_df['Cancelled_Long'] + cell6_daily_conflicts_df['Cancelled_Short']
assert (cell6_daily_conflicts_df['Daily_Conflict'] == cell6_daily_conflicts_df['Original_Long_3plus'].gt(0) & cell6_daily_conflicts_df['Original_Short_3plus'].gt(0)).all()
assert not cell6_daily_conflicts_df.loc[cell6_daily_conflicts_df['Daily_Conflict'], ['Long_Eligible', 'Short_Eligible']].to_numpy().any()
cell6_candidates = cell6_signal_panel.loc[cell6_signal_panel['Candidate_Direction'].ne('NONE')].copy().sort_values(['DlyCalDt', 'Candidate_Priority', 'Candidate_Votes', 'Candidate_Strength', 'Ticker', 'Candidate_Direction'], ascending=[True, False, False, False, True, True], kind='mergesort')
cell6_candidates['Daily_Rank_All'] = cell6_candidates.groupby('DlyCalDt').cumcount() + 1
cell6_candidates['Daily_Rank_Side'] = cell6_candidates.groupby(['DlyCalDt', 'Candidate_Direction']).cumcount() + 1
cell6_candidates['Daily_Top3_If_Flat'] = cell6_candidates['Daily_Rank_All'] <= 3
cell6_signal_panel['Daily_Rank_All'] = pd.Series(pd.NA, index=cell6_signal_panel.index, dtype='Int64')
cell6_signal_panel.loc[cell6_candidates.index, 'Daily_Rank_All'] = cell6_candidates['Daily_Rank_All'].to_numpy()
summary_rows, vote_rows, stock_rows = ([], [], [])
for (period, year), grp in cell6_signal_panel.groupby(['Period', 'Year'], sort=True):
    row = {'Period': period, 'Year': int(year), 'Stock_Days': len(grp), 'Days': grp['DlyCalDt'].nunique(), 'Long_3plus': int(grp['Long_Qualified'].sum()), 'Short_3plus': int(grp['Short_Qualified'].sum()), 'Stock_Conflict_Rows': int(grp['Signal_Conflict'].sum()), 'Daily_Conflict_Days': grp.loc[grp['Daily_Conflict'], 'DlyCalDt'].nunique(), 'Daily_Conflict_StockDays': int(grp['Daily_Conflict'].sum()), 'Cancelled_Long': int(grp['Long_Cancelled_By_Daily_Conflict'].sum()), 'Cancelled_Short': int(grp['Short_Cancelled_By_Daily_Conflict'].sum()), 'Conflicts_Dropped': int(grp['Long_Cancelled_By_Daily_Conflict'].sum() + grp['Short_Cancelled_By_Daily_Conflict'].sum()), 'Long_Eligible': int(grp['Long_Candidate'].sum()), 'Short_Eligible': int(grp['Short_Candidate'].sum()), 'Any_Eligible': int(grp['Candidate_Direction'].ne('NONE').sum()), 'Signal_Days': grp.loc[grp['Candidate_Direction'].ne('NONE'), 'DlyCalDt'].nunique()}
    for side in C6_SIDES:
        row[f'{side}_N_Factors'] = int(grp[f'{side}_N_Factors'].iloc[0])
        for n in (3, 4, 5, 6):
            row[f'{side}_{n}Votes'] = int(grp[f'{side}_Votes'].eq(n).sum())
        row[f'{side}_Unique_Family_Mean_Eligible'] = float(grp.loc[grp[f'{side}_Candidate'], f'{side}_Family_Count'].mean()) if grp[f'{side}_Candidate'].any() else np.nan
        for n in range(C51_N + 1):
            vote_rows.append({'Period': period, 'Year': int(year), 'Direction': side, 'Votes': n, 'Stock_Days': int(grp[f'{side}_Votes'].eq(n).sum()), 'Fraction': float(grp[f'{side}_Votes'].eq(n).mean())})
    summary_rows.append(row)
for (period, ticker), grp in cell6_signal_panel.groupby(['Period', 'Ticker'], sort=True):
    stock_rows.append({'Period': period, 'Ticker': ticker, 'Stock_Days': len(grp), 'Long_Eligible': int(grp['Long_Candidate'].sum()), 'Short_Eligible': int(grp['Short_Candidate'].sum()), 'Stock_Conflicts': int(grp['Signal_Conflict'].sum()), 'Daily_Conflict_StockDays': int(grp['Daily_Conflict'].sum()), 'Cancelled_Long': int(grp['Long_Cancelled_By_Daily_Conflict'].sum()), 'Cancelled_Short': int(grp['Short_Cancelled_By_Daily_Conflict'].sum()), 'Long_5plus': int(grp['Long_Votes'].ge(5).sum()), 'Short_5plus': int(grp['Short_Votes'].ge(5).sum())})
cell6_summary_df = pd.DataFrame(summary_rows).sort_values(['Year', 'Period'])
cell6_vote_distribution_df = pd.DataFrame(vote_rows)
cell6_stock_coverage_df = pd.DataFrame(stock_rows)
configuration = []
for period, year, selection, factor_output in [*[('Train_OOS', y, factor_six_train_oos[y], factor_train_oos[y]) for y in C6_EXPECTED_TRAIN_YEARS], ('Validation', '2022-2023', factor_six_validation, factor_validation)]:
    for side in C6_SIDES:
        factors = selection['selected'][side]
        weights = c6_quality_weights(factor_output, side, factors)
        configuration.append({'Period': period, 'Year': year, 'Direction': side, 'Enabled': bool(factors), 'N_Factors': len(factors), 'Min_Votes': C6_MIN_VOTES if factors else 0, 'Factors': ' | '.join(factors), 'Quality_Weights': ' | '.join((f'{f}:{w:.6f}' for f, w in zip(factors, weights))), 'Orientation': ' | '.join((f"{f}:{selection['orientations'][side][f]:+d}" for f in factors))})
cell6_configuration_df = pd.DataFrame(configuration)
for side in C6_SIDES:
    assert cell6_signal_panel[f'{side}_Qualified'].sum() == cell6_signal_panel[f'{side}_Candidate'].sum() + cell6_signal_panel[f'{side}_Cancelled_By_Daily_Conflict'].sum()
assert not cell6_signal_panel.loc[cell6_signal_panel['Daily_Conflict'], 'Candidate_Direction'].ne('NONE').any()
assert not (cell6_signal_panel['Long_Candidate'] & cell6_signal_panel['Short_Candidate']).any()
for year in C6_EXPECTED_TRAIN_YEARS:
    if not any(factor_six_train_oos[year]['enabled'].values()):
        assert cell6_by_year[year]['Candidate_Direction'].eq('NONE').all()
for side in C6_SIDES:
    if not factor_six_validation['enabled'][side]:
        assert not cell6_validation_signals[f'{side}_Candidate'].any()
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
with pd.option_context('display.max_colwidth', 200):
    pass
pass
pass
pass
conflict_report = cell6_daily_conflicts_df.groupby(['Period', 'Year']).agg(Conflict_Days=('Daily_Conflict', 'sum'), Cancelled_Long=('Cancelled_Long', 'sum'), Cancelled_Short=('Cancelled_Short', 'sum'), Cancelled_Signals=('Cancelled_Signals', 'sum'), Allowed_Long=('Long_Eligible', 'sum'), Allowed_Short=('Short_Eligible', 'sum'))
pass
pass
pass
pass
pass
pass
preview_cols = ['DlyCalDt', 'Ticker', 'Candidate_Direction', 'Candidate_Votes', 'Candidate_Strength', 'Candidate_Priority', 'Daily_Rank_All', 'Daily_Top3_If_Flat', 'Long_Matched_Factors', 'Short_Matched_Factors', 'Long_Family_Count', 'Short_Family_Count']
for side in C6_SIDES:
    pass
    pass
pass
pass
pass
pass
pass
pass
