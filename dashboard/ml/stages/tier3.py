import time
import numpy as np
import pandas as pd
C51_N = 6
C51_MIN_ACTIVE = 4
C51_MIN_VOTES = 3
C51_FACTOR_Q = C5_Q
C51_CORR_ALERT = 0.8
C51_H = HORIZON
C51_MIN_YEAR_N = 5
C51_MIN_TOTAL_N = 15
C51_MIN_STOCKS = 5
C51_SIDES = ('Long', 'Short')
C51_FEATURES = list(feature_cols)
C51_TICKERS = list(tech_tickers)
C51_F2I = {f: j for j, f in enumerate(C51_FEATURES)}
assert C5_TOP == 10 and len(C51_FEATURES) == 31
assert len(C51_TICKERS) == 9 and target_col == 'TARGET_10'
assert len(factor_development) == 6 and len(factor_train_oos) == 4
assert 0 < C51_FACTOR_Q < 1 and C51_MIN_VOTES <= C51_MIN_ACTIVE <= C51_N
C51_SCORE_COLUMNS = ('Composite_Score', 'Mean_LOFO', 'Positive_LOFO_Year_Fraction')
assert all((c in factor_validation['rankings'].columns for c in C51_SCORE_COLUMNS)), 'Run the updated six-metric Cell 5 before Cell 5.1.'

def c51_select_tier3(factor_output):
    """Take up to six Tier-3 factors in Cell 5 composite-score rank order."""
    qualified = {}
    for side in C51_SIDES:
        top10 = list(factor_output['top10'][side])
        rows = factor_output['rankings']
        rows = rows.loc[rows['Direction'].eq(side) & rows['Candidate']]
        ranks = rows.set_index('Feature').loc[top10]
        assert len(top10) == 10 and len(set(top10)) == 10
        assert ranks['Rank'].is_monotonic_increasing
        qualified[side] = [f for f in top10 if int(ranks.loc[f, 'Direction_Tier']) == 3]
        assert all((factor_output['outer_orientations'][side][f] in (-1, 1) for f in qualified[side]))
    counts = {side: len(qualified[side]) for side in C51_SIDES}
    force_flat = any((0 < counts[side] < C51_MIN_ACTIVE for side in C51_SIDES))
    selected = {side: qualified[side][:C51_N] if not force_flat and counts[side] >= C51_MIN_ACTIVE else [] for side in C51_SIDES}
    return (selected, counts, force_flat)

def c51_matured(block):
    shifted = block.groupby('PERMNO', sort=False)['DlyCalDt'].shift(-C51_H)
    idx = block.index[shifted.notna()]
    assert len(block) - len(idx) == C51_H * len(C51_TICKERS)
    return idx

def c51_thin(mask, min_gap=C51_H):
    taken, last = ([], -min_gap)
    for pos in np.flatnonzero(mask):
        if pos - last >= min_gap:
            taken.append(pos)
            last = pos
    return np.asarray(taken, dtype=int)

def c51_percentile(train_x, oos_x, orientation):
    """Favorable per-stock ECDF based solely on historical training features."""
    if orientation == 0:
        return np.zeros(len(oos_x), dtype=np.float32)
    reference = np.sort(np.asarray(train_x, dtype=np.float64))
    values = np.asarray(oos_x, dtype=np.float64)
    assert len(reference) >= 30
    if reference[0] == reference[-1]:
        return np.zeros(len(values), dtype=np.float32)
    if orientation == 1:
        p = np.searchsorted(reference, values, side='right') / len(reference)
    else:
        p = 1 - np.searchsorted(reference, values, side='left') / len(reference)
    return np.clip(p, 0, 1).astype(np.float32)
c51_history_cache = {}

def c51_prepare_inner(outer, enabled_sides):
    key = (str(outer['train_start']), str(outer['train_cutoff']), tuple(enabled_sides))
    if key in c51_history_cache:
        return c51_history_cache[key]
    history = data.loc[outer['train_full_idx']]
    assert history['Split'].eq('Train').all()
    assert history['DlyCalDt'].max() < outer['oos_start']
    years = sorted(history['DlyCalDt'].dt.year.unique())
    prepared = []
    for year in years[2:]:
        train = history.loc[history['DlyCalDt'].dt.year < year]
        oos = history.loc[history['DlyCalDt'].dt.year == year]
        tr_idx, oo_idx = (c51_matured(train), c51_matured(oos))
        assert data.loc[tr_idx, 'DlyCalDt'].max() < data.loc[oo_idx, 'DlyCalDt'].min()
        xtr = data.loc[tr_idx, C51_FEATURES].to_numpy(dtype=float)
        xoo = data.loc[oo_idx, C51_FEATURES].to_numpy(dtype=float)
        rtr = data.loc[tr_idx, target_col].to_numpy(dtype=float)
        roo = data.loc[oo_idx, target_col].to_numpy(dtype=float)
        ttr = data.loc[tr_idx, 'Ticker'].to_numpy()
        too = data.loc[oo_idx, 'Ticker'].to_numpy()
        dates = data.loc[oo_idx, 'DlyCalDt'].to_numpy()
        assert all((np.isfinite(a).all() for a in (xtr, xoo, rtr, roo)))
        orientation = c5_calibrate_orientations(xtr, rtr, ttr)
        by_stock = {}
        for ticker in C51_TICKERS:
            tr_pos = np.flatnonzero(ttr == ticker)
            oo_pos = np.flatnonzero(too == ticker)
            assert len(tr_pos) >= 30 and len(oo_pos) > C51_H
            by_stock[ticker] = (tr_pos, oo_pos)
        by_side = {}
        for side in enabled_sides:
            signs = orientation[side]['orientation']
            percentiles = np.zeros(xoo.shape, dtype=np.float32)
            for tr_pos, oo_pos in by_stock.values():
                for j in np.flatnonzero(signs):
                    percentiles[oo_pos, j] = c51_percentile(xtr[tr_pos, j], xoo[oo_pos, j], int(signs[j]))
            signed = roo * (1 if side == 'Long' else -1)
            baseline = np.empty(len(signed), dtype=float)
            for _, oo_pos in by_stock.values():
                baseline[oo_pos] = signed[oo_pos].mean()
            by_side[side] = {'percentiles': percentiles, 'signed_returns': signed, 'baseline': baseline}
        prepared.append({'Year': int(year), 'Ticker': too, 'Date': dates, 'by_stock': by_stock, 'by_side': by_side})
    c51_history_cache[key] = prepared
    return prepared

def c51_evaluate(prepared, side, selected, include_signals=False):
    assert C51_MIN_ACTIVE <= len(selected) <= C51_N
    indices = [C51_F2I[f] for f in selected]
    annual, stock_records, signal_records = ([], [], [])
    vote_counts = {f'Votes_{v}': 0 for v in range(C51_MIN_VOTES, C51_N + 1)}
    raw_total = 0
    for fold in prepared:
        year = fold['Year']
        info = fold['by_side'][side]
        p = info['percentiles'][:, indices]
        confirmed = p >= C51_FACTOR_Q
        votes = confirmed.sum(axis=1)
        active = votes >= C51_MIN_VOTES
        strength = np.divide((p * confirmed).sum(axis=1), votes, out=np.zeros(len(votes), dtype=float), where=votes > 0)
        rr, bb = (info['signed_returns'], info['baseline'])
        raw_total += int(active.sum())
        record = {'Year': year, 'Raw_Signals': int(active.sum()), 'Thin_Signals': 0, 'Wins': 0, 'Return_Sum': 0.0, 'Lift_Sum': 0.0, 'Strength_Sum': 0.0, 'Active_Stocks': 0}
        for ticker, (_, positions) in fold['by_stock'].items():
            chosen_pos = positions[c51_thin(active[positions])]
            n = len(chosen_pos)
            wins = int((rr[chosen_pos] > 0).sum())
            ret_sum = float(rr[chosen_pos].sum())
            lift_sum = float((rr[chosen_pos] - bb[chosen_pos]).sum())
            record['Thin_Signals'] += n
            record['Wins'] += wins
            record['Return_Sum'] += ret_sum
            record['Lift_Sum'] += lift_sum
            record['Strength_Sum'] += float(strength[chosen_pos].sum())
            record['Active_Stocks'] += int(n > 0)
            stock_records.append({'Year': year, 'Ticker': ticker, 'N': n, 'Wins': wins, 'Return_Sum': ret_sum, 'Lift_Sum': lift_sum})
            for v in votes[chosen_pos]:
                vote_counts[f'Votes_{int(v)}'] += 1
        annual.append(record)
        if include_signals:
            for i in np.flatnonzero(active):
                signal_records.append({'Year': year, 'Date': pd.Timestamp(fold['Date'][i]), 'Ticker': str(fold['Ticker'][i]), 'Direction': side, 'Votes': int(votes[i]), 'Strength': float(strength[i]), 'Signed_TARGET_10': float(rr[i])})
    yearly = pd.DataFrame(annual)
    stocks = pd.DataFrame(stock_records)
    yearly['Mean_Return'] = yearly['Return_Sum'] / yearly['Thin_Signals'].replace(0, np.nan)
    yearly['Mean_Lift'] = yearly['Lift_Sum'] / yearly['Thin_Signals'].replace(0, np.nan)
    yearly['Hit_Rate'] = yearly['Wins'] / yearly['Thin_Signals'].replace(0, np.nan)
    qualified = yearly['Thin_Signals'] >= C51_MIN_YEAR_N
    active_years = yearly.loc[qualified]
    n_total = int(yearly['Thin_Signals'].sum())
    active_stocks = int((stocks.groupby('Ticker')['N'].sum() > 0).sum())
    positive_stocks = int((stocks.groupby('Ticker')['Lift_Sum'].sum() > 0).sum())
    adequate = n_total >= C51_MIN_TOTAL_N and int(qualified.sum()) >= max(1, int(np.ceil(0.6 * len(yearly)))) and (active_stocks >= C51_MIN_STOCKS)
    summary = {'Coverage_OK': bool(adequate), 'Raw_Signals': raw_total, 'Thin_Signals': n_total, 'Qualified_Years': int(qualified.sum()), 'Profit_Years': int((active_years['Mean_Return'] > 0).sum()), 'Lift_Years': int((active_years['Mean_Lift'] > 0).sum()), 'Active_Stocks': active_stocks, 'Positive_Lift_Stocks': positive_stocks, 'Mean_Year_Return': float(active_years['Mean_Return'].mean()), 'Mean_Year_Lift': float(active_years['Mean_Lift'].mean()), 'Worst_Year_Return': float(active_years['Mean_Return'].min()), 'Pooled_Mean_Return': float(yearly['Return_Sum'].sum() / n_total) if n_total else np.nan, 'Pooled_Mean_Lift': float(yearly['Lift_Sum'].sum() / n_total) if n_total else np.nan, 'Pooled_Hit_Rate': float(yearly['Wins'].sum() / n_total) if n_total else np.nan, **vote_counts}
    if not include_signals:
        return summary
    signals = pd.DataFrame(signal_records, columns=['Year', 'Date', 'Ticker', 'Direction', 'Votes', 'Strength', 'Signed_TARGET_10'])
    if not signals.empty:
        signals = signals.sort_values(['Date', 'Votes', 'Strength', 'Ticker'], ascending=[True, False, False, True]).reset_index(drop=True)
        signals['Daily_Rank'] = signals.groupby('Date').cumcount() + 1
        signals['Daily_Top3_Candidate'] = signals['Daily_Rank'] <= 3
    return (summary, yearly, stocks, signals)
c51_start = time.perf_counter()
factor_six_development = {}
factor_six_train_oos = {}
c51_rows = []

def c51_select_outer(outer, scaled, factor_output, stage, fold_name):
    selected, tier3_counts, force_flat = c51_select_tier3(factor_output)
    enabled_sides = [side for side in C51_SIDES if selected[side]]
    prepared = c51_prepare_inner(outer, enabled_sides) if enabled_sides else []
    output = {'selected': selected, 'orientations': {}, 'ranking': {}, 'annual': {}, 'stocks': {}, 'signals': {}, 'summary': {}, 'enabled': {side: bool(selected[side]) for side in C51_SIDES}, 'tier3_counts': tier3_counts, 'force_flat': force_flat}
    for side in C51_SIDES:
        chosen = selected[side]
        signs = factor_output['outer_orientations'][side]
        output['orientations'][side] = {f: int(signs[f]) for f in chosen}
        output['ranking'][side] = pd.DataFrame()
        output['annual'][side] = pd.DataFrame()
        output['stocks'][side] = pd.DataFrame()
        output['signals'][side] = pd.DataFrame()
        base = {'Stage': stage, 'Fold': str(fold_name), 'Direction': side, 'Tier3_Available': tier3_counts[side], 'Selected_N': len(chosen), 'Min_Votes': C51_MIN_VOTES if chosen else 0, 'Selected_Factors': ', '.join(chosen), 'Selected_Orientations': ', '.join((f'{f}:{signs[f]:+d}' for f in chosen))}
        if not chosen:
            status = 'FLAT_WEAK_SIDE' if force_flat else 'DISABLED_NO_TIER3' if tier3_counts[side] == 0 else 'DISABLED_INSUFFICIENT_TIER3'
            output['summary'][side] = {**base, 'Status': status, 'Max_Abs_Corr': np.nan}
            continue
        corr = scaled['X_train'][chosen].corr().abs().to_numpy()
        np.fill_diagonal(corr, 0.0)
        max_corr = float(corr.max())
        metrics, yearly, stocks, signals = c51_evaluate(prepared, side, chosen, include_signals=True)
        output['annual'][side] = yearly
        output['stocks'][side] = stocks
        output['signals'][side] = signals
        output['ranking'][side] = pd.DataFrame([{'Rank': 1, 'Factors': '|'.join(chosen), 'Max_Abs_Corr': max_corr, **metrics}])
        output['summary'][side] = {**base, 'Status': 'ACTIVE', 'Max_Abs_Corr': max_corr, 'Correlation_Alert': max_corr >= C51_CORR_ALERT, **metrics}
    return output
for key, fold in sorted(development_folds.items()):
    window, year = key
    result = c51_select_outer(fold, scaled_development_folds[key], factor_development[key], 'Development', f'{window}Y_{year}')
    factor_six_development[key] = result
    c51_rows.extend(result['summary'].values())
for year, fold in sorted(train_oos_folds.items()):
    result = c51_select_outer(fold, scaled_train_oos_folds[year], factor_train_oos[year], 'Train_OOS', year)
    factor_six_train_oos[year] = result
    c51_rows.extend(result['summary'].values())
factor_six_validation = c51_select_outer(validation_fold, scaled_validation_fold, factor_validation, 'Validation', '2022-2023')
c51_rows.extend(factor_six_validation['summary'].values())
c51_summary_df = pd.DataFrame(c51_rows)
assert len(c51_summary_df) == 22
assert len(factor_six_development) == 6
assert len(factor_six_train_oos) == 4
for result in [*factor_six_development.values(), *factor_six_train_oos.values(), factor_six_validation]:
    for side in C51_SIDES:
        chosen = result['selected'][side]
        assert len(chosen) == 0 or C51_MIN_ACTIVE <= len(chosen) <= C51_N
        assert set(chosen).issubset(C51_FEATURES)
        assert set(chosen) == set(result['orientations'][side])
C51_ACTIVE_SIDES = [side for side in C51_SIDES if factor_six_validation['enabled'][side]]
C51_FLAT_ONLY = len(C51_ACTIVE_SIDES) == 0
pass
pass
pass
pass
pass
pass
pass
pass
pass
with pd.option_context('display.max_colwidth', 160):
    pass
pass
for side in C51_SIDES:
    result = factor_six_validation
    chosen = result['selected'][side]
    pass
    pass
    if not chosen:
        pass
        continue
    ranking = factor_validation['rankings'].set_index(['Direction', 'Feature'])
    report = pd.DataFrame({'Feature': chosen, 'Orientation': [result['orientations'][side][f] for f in chosen], 'Cell5_Rank': [int(ranking.loc[(side, f), 'Rank']) for f in chosen], 'Tier': [int(ranking.loc[(side, f), 'Direction_Tier']) for f in chosen], 'Composite_Score': [float(ranking.loc[(side, f), 'Composite_Score']) for f in chosen], 'Mean_LOFO': [float(ranking.loc[(side, f), 'Mean_LOFO']) for f in chosen], 'Positive_LOFO_Year_Fraction': [float(ranking.loc[(side, f), 'Positive_LOFO_Year_Fraction']) for f in chosen]})
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
