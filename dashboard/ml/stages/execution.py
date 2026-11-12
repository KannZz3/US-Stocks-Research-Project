import numpy as np
import pandas as pd
from pathlib import Path
C8_SINGLE_FACTOR_Q = C6_FACTOR_Q
C8_SCHEDULE = {'Validation': (2022, 2023), 'Test': (2024, 2025)}
C8_SIGNAL_TRAIN_CUTOFF = 2021
C8_SHARED_TRAIN_CUTOFF = C8_SIGNAL_TRAIN_CUTOFF
C8_SIDES = ('Long', 'Short')
C8_COST_RATE = 0.0 if C8_ZERO_COSTS else C8_COST_BPS_PER_LEG / 10000.0
assert target_col == 'TARGET_10' and HORIZON == 10
assert C51_MIN_VOTES == C6_MIN_VOTES == 3
assert type(C8_MIN_VOTES) is int and C8_MIN_VOTES in (3, 4, 5, 6)
assert np.isclose(C8_SINGLE_FACTOR_Q, C51_FACTOR_Q)
assert isinstance(C8_MAX_POSITIONS, int) and C8_MAX_POSITIONS >= 1
assert isinstance(C8_MAX_HOLD_BARS, int) and C8_MAX_HOLD_BARS >= 1
assert np.isfinite(C8_START_CAPITAL) and C8_START_CAPITAL > 0
assert np.isfinite(C8_POSITION_FRACTION) and 0 < C8_POSITION_FRACTION <= 1
assert np.isfinite(C8_MAX_ENTRY_GAP) and 0 <= C8_MAX_ENTRY_GAP < 1
assert np.isfinite(C8_TP_M_MULT) and C8_TP_M_MULT > 0
assert np.isfinite(C8_SL_M_MULT) and C8_SL_M_MULT > 0
assert isinstance(C8_MIN_STOCK_POSITIVE, int) and C8_MIN_STOCK_POSITIVE >= 1
assert isinstance(C8_MIN_SIDE_POSITIVE, int) and C8_MIN_SIDE_POSITIVE >= 1
assert isinstance(C8_M_TRAIN_START_YEAR, int) and 2015 <= C8_M_TRAIN_START_YEAR <= 2021
assert isinstance(C8_ZERO_COSTS, bool)
assert np.isfinite(C8_COST_BPS_PER_LEG) and C8_COST_BPS_PER_LEG >= 0
assert np.isfinite(C8_SHORT_COLLATERAL_RATIO) and C8_SHORT_COLLATERAL_RATIO >= 1
assert len(C8_TRADE_SIDES) == len(set(C8_TRADE_SIDES))
assert set(C8_TRADE_SIDES).issubset({'Long', 'Short'})
assert len(C8_OHLC_COLS) == 4 and len(set(C8_OHLC_COLS)) == 4
assert all((c in data.columns for c in C8_OHLC_COLS))
assert len(tech_tickers) == 9 and data.index.is_unique
assert set(factor_six_validation['selected']) == {'Long', 'Short'}
assert set(factor_six_validation['orientations']) == {'Long', 'Short'}
assert C8_SCHEDULE == {'Validation': (2022, 2023), 'Test': (2024, 2025)}
assert set(C8_TRADE_SIDES).issubset(set(C51_SIDES))
C8_ENABLED_SIDES = tuple((s for s in ('Long', 'Short') if factor_six_validation['enabled'][s] and s in C8_TRADE_SIDES))
C8_FROZEN_FACTORS = {s: list(factor_six_validation['selected'][s]) for s in ('Long', 'Short')}
C8_FROZEN_SIGNS = factor_six_validation['orientations']
C8_FROZEN_WEIGHTS = {side: c6_quality_weights(factor_validation, side, C8_FROZEN_FACTORS[side]) for side in ('Long', 'Short')}
for side in ('Long', 'Short'):
    fs = C8_FROZEN_FACTORS[side]
    assert not fs or 4 <= len(fs) <= C51_N
    assert len(fs) == len(set(fs))
    assert set(C8_FROZEN_SIGNS[side]) == set(fs)
    assert all((C8_FROZEN_SIGNS[side][f] in (-1, 1) for f in fs))
    assert (len(fs) > 0) == bool(factor_six_validation['enabled'][side])
    assert len(C8_FROZEN_WEIGHTS[side]) == len(fs)
    if fs:
        assert np.isfinite(C8_FROZEN_WEIGHTS[side]).all()
        assert np.isclose(C8_FROZEN_WEIGHTS[side].sum(), 1.0)
assert not (factor_six_validation['force_flat'] and C8_ENABLED_SIDES)

def c8_make_frozen_stage_signals(stage):
    assert stage in C8_SCHEDULE
    start_year, end_year = C8_SCHEDULE[stage]
    features = list(dict.fromkeys(sum((C8_FROZEN_FACTORS[s] for s in ('Long', 'Short')), [])))
    assert target_col not in features, 'Targets cannot be used to generate signals.'
    columns = ['PERMNO', 'Ticker', 'DlyCalDt', 'Split'] + features
    tr = data.loc[validation_fold['train_fit_idx'], columns].copy()
    tr['DlyCalDt'] = pd.to_datetime(tr['DlyCalDt'])
    assert tr['DlyCalDt'].dt.year.max() <= C8_SIGNAL_TRAIN_CUTOFF
    mask = pd.to_datetime(data['DlyCalDt']).dt.year.between(start_year, end_year) & data['Ticker'].isin(tech_tickers)
    oo = data.loc[mask, columns].copy()
    oo['DlyCalDt'] = pd.to_datetime(oo['DlyCalDt'])
    assert not oo.empty and set(oo['Split']) == {stage}
    assert set(oo['Ticker']) == set(tr['Ticker']) == set(tech_tickers)
    assert set(oo['DlyCalDt'].dt.year) == set(range(start_year, end_year + 1))
    assert tr['Split'].eq('Train').all() and tr['DlyCalDt'].max() < oo['DlyCalDt'].min()
    assert not oo[['DlyCalDt', 'Ticker']].duplicated().any()
    assert tr[columns].notna().all().all() and oo[columns].notna().all().all()
    if features:
        assert np.isfinite(tr[features].to_numpy(dtype=float)).all()
        assert np.isfinite(oo[features].to_numpy(dtype=float)).all()
    out = oo[['PERMNO', 'Ticker', 'DlyCalDt', 'Split']].copy()
    out['Period'] = stage
    out['Year'] = oo['DlyCalDt'].dt.year
    out['History_Start'] = tr['DlyCalDt'].min()
    out['History_Cutoff'] = tr['DlyCalDt'].max()
    tr_tick, oo_tick = (tr['Ticker'].to_numpy(), oo['Ticker'].to_numpy())
    for side in ('Long', 'Short'):
        fs = C8_FROZEN_FACTORS[side] if side in C8_ENABLED_SIDES else []
        signs = C8_FROZEN_SIGNS[side]
        scores = np.zeros((len(oo), len(fs)), dtype=float)
        if fs:
            for ticker in tech_tickers:
                a = tr_tick == ticker
                b = oo_tick == ticker
                assert a.sum() >= 30 and b.any()
                for j, f in enumerate(fs):
                    scores[b, j] = c6_favorable_pct(tr.loc[a, f].to_numpy(dtype=float), oo.loc[b, f].to_numpy(dtype=float), int(signs[f]))
        hits = scores >= C8_SINGLE_FACTOR_Q
        votes = hits.sum(axis=1).astype(int)
        strength = np.divide((scores * hits).sum(axis=1), votes, out=np.zeros(len(oo), dtype=float), where=votes > 0)
        priority = scores * hits @ C8_FROZEN_WEIGHTS[side] if fs else np.zeros(len(oo))
        out[f'{side}_Priority'] = priority
        out[f'{side}_N_Factors'] = len(fs)
        out[f'{side}_Enabled'] = bool(fs)
        out[f'{side}_Votes'] = votes
        out[f'{side}_Strength'] = strength
        out[f'{side}_Qualified'] = bool(fs) & (votes >= C8_MIN_VOTES)
        out[f'{side}_Selected_Factors'] = '|'.join(fs)
        matches = [[f for f, hit in zip(fs, flags) if hit] for flags in hits]
        out[f'{side}_Matched_Factors'] = ['|'.join(m) for m in matches]
        out[f'{side}_Matched_Families'] = ['|'.join(sorted({c6_factor_family(f) for f in m})) for m in matches]
        out[f'{side}_Family_Count'] = [len({c6_factor_family(f) for f in m}) for m in matches]
        for j, f in enumerate(fs):
            out[f'{side}_Pct__{f}'] = scores[:, j]
            out[f'{side}_OK__{f}'] = hits[:, j]
        assert (votes <= len(fs)).all()
        if not fs:
            assert not out[f'{side}_Qualified'].any()
    out['Signal_Conflict'] = out['Long_Qualified'] & out['Short_Qualified']
    day_sides = out.groupby('DlyCalDt', sort=False)[['Long_Qualified', 'Short_Qualified']].transform('any')
    out['Daily_Conflict'] = day_sides['Long_Qualified'] & day_sides['Short_Qualified']
    out['New_Entries_Allowed'] = ~out['Daily_Conflict']
    for side in ('Long', 'Short'):
        out[f'{side}_Cancelled_By_Daily_Conflict'] = out[f'{side}_Qualified'] & out['Daily_Conflict']
        out[f'{side}_Candidate'] = out[f'{side}_Qualified'] & ~out['Daily_Conflict']
    out['Candidate_Direction'] = np.select([out['Long_Candidate'], out['Short_Candidate']], ['Long', 'Short'], default='NONE')
    out['Candidate_Votes'] = np.select([out['Long_Candidate'], out['Short_Candidate']], [out['Long_Votes'], out['Short_Votes']], default=0).astype(int)
    out['Candidate_Strength'] = np.select([out['Long_Candidate'], out['Short_Candidate']], [out['Long_Strength'], out['Short_Strength']], default=np.nan)
    out['Candidate_Priority'] = np.select([out['Long_Candidate'], out['Short_Candidate']], [out['Long_Priority'], out['Short_Priority']], default=np.nan)
    assert np.isfinite(out.loc[out['Candidate_Direction'].ne('NONE'), 'Candidate_Priority']).all()
    out['Trade_Signal_Available'] = 'AFTER_CLOSE'
    out['Earliest_Execution'] = 'NEXT_TRADING_DAY'
    order = out.loc[out['Candidate_Direction'].ne('NONE')].sort_values(['DlyCalDt', 'Candidate_Priority', 'Candidate_Votes', 'Candidate_Strength', 'Ticker'], ascending=[True, False, False, False, True], kind='mergesort')
    out['Daily_Rank_All'] = pd.Series(pd.NA, index=out.index, dtype='Int64')
    out.loc[order.index, 'Daily_Rank_All'] = order.groupby('DlyCalDt').cumcount().to_numpy() + 1
    assert not out.loc[out['Daily_Conflict'], 'Candidate_Direction'].ne('NONE').any()
    assert not (out['Long_Candidate'] & out['Short_Candidate']).any()
    if len(C8_ENABLED_SIDES) < 2:
        assert not out['Daily_Conflict'].any()
    return out
cell8_validation_signals = c8_make_frozen_stage_signals('Validation')
cell8_test_signals = c8_make_frozen_stage_signals('Test')
assert cell8_validation_signals.index.intersection(cell8_test_signals.index).empty
if set(C8_ENABLED_SIDES) == set((s for s in ('Long', 'Short') if factor_six_validation['enabled'][s])):
    ref = cell6_validation_signals.reindex(cell8_validation_signals.index)
    assert ref['Ticker'].equals(cell8_validation_signals['Ticker'])
    for side in ('Long', 'Short'):
        assert np.array_equal(ref[f'{side}_Votes'].to_numpy(), cell8_validation_signals[f'{side}_Votes'].to_numpy())
        assert np.allclose(ref[f'{side}_Strength'], cell8_validation_signals[f'{side}_Strength'])
        assert np.allclose(ref[f'{side}_Priority'], cell8_validation_signals[f'{side}_Priority'])
        expected_qualified = ref[f'{side}_Enabled'].to_numpy(dtype=bool) & (ref[f'{side}_Votes'].to_numpy(dtype=int) >= C8_MIN_VOTES)
        assert np.array_equal(expected_qualified, cell8_validation_signals[f'{side}_Qualified'].to_numpy())
        if C8_MIN_VOTES == C6_MIN_VOTES:
            assert np.array_equal(ref[f'{side}_Candidate'].to_numpy(), cell8_validation_signals[f'{side}_Candidate'].to_numpy())
    if C8_MIN_VOTES == C6_MIN_VOTES:
        assert ref['Daily_Conflict'].equals(cell8_validation_signals['Daily_Conflict'])
        assert np.allclose(ref['Candidate_Priority'], cell8_validation_signals['Candidate_Priority'], equal_nan=True)
        ref_rank = cell6_signal_panel.loc[cell8_validation_signals.index, 'Daily_Rank_All']
        assert ref_rank.astype('Int64').equals(cell8_validation_signals['Daily_Rank_All'].astype('Int64'))
C8_PANEL_COLS = ['Period', 'Year', 'DlyCalDt', 'Ticker', 'Candidate_Direction', 'Candidate_Votes', 'Candidate_Strength', 'Candidate_Priority', 'Daily_Rank_All', 'Daily_Conflict', 'Long_Matched_Factors', 'Short_Matched_Factors', 'Long_Matched_Families', 'Short_Matched_Families']
c8_panel = pd.concat([cell8_validation_signals[C8_PANEL_COLS], cell8_test_signals[C8_PANEL_COLS]], axis=0)
c8_panel = c8_panel.sort_values(['DlyCalDt', 'Ticker'], kind='mergesort').copy()
assert c8_panel.index.is_unique
c8_source = data.loc[c8_panel.index, ['Ticker', 'DlyCalDt'] + list(C8_OHLC_COLS)]
assert np.array_equal(c8_source['Ticker'].to_numpy(), c8_panel['Ticker'].to_numpy())
assert np.array_equal(pd.to_datetime(c8_source['DlyCalDt']).to_numpy(), pd.to_datetime(c8_panel['DlyCalDt']).to_numpy())
raw_prices = c8_source[list(C8_OHLC_COLS)].to_numpy(dtype=float)
assert np.isfinite(raw_prices).all() and (raw_prices > 0).all()
for output_col, source_col in zip(('DlyOpen', 'DlyHigh', 'DlyLow', 'DlyClose'), C8_OHLC_COLS):
    c8_panel[output_col] = c8_source[source_col].to_numpy(dtype=float)
assert (c8_panel['DlyHigh'] >= c8_panel[['DlyOpen', 'DlyClose']].max(axis=1)).all()
assert (c8_panel['DlyLow'] <= c8_panel[['DlyOpen', 'DlyClose']].min(axis=1)).all()
assert (c8_panel['DlyHigh'] >= c8_panel['DlyLow']).all()
c8_panel['DlyCalDt'] = pd.to_datetime(c8_panel['DlyCalDt'])
assert not c8_panel[['DlyCalDt', 'Ticker']].duplicated().any()

def c8_calibrate_m():
    fs = list(dict.fromkeys(sum((C8_FROZEN_FACTORS[s] for s in C8_ENABLED_SIDES), [])))
    columns = ['Ticker', 'DlyCalDt', target_col] + fs
    tr = data.loc[validation_fold['train_fit_idx'], columns].copy()
    tr['DlyCalDt'] = pd.to_datetime(tr['DlyCalDt'])
    tr = tr.loc[tr['DlyCalDt'].dt.year.between(C8_M_TRAIN_START_YEAR, 2021)].copy()
    assert tr['DlyCalDt'].max() < pd.Timestamp(validation_fold['oos_start'])
    assert set(tr['Ticker']) == set(tech_tickers)
    assert tr[[target_col] + fs].notna().all().all()
    assert np.isfinite(tr[[target_col] + fs].to_numpy(dtype=float)).all()
    lookup, records, details = ({}, [], {})
    for side in ('Long', 'Short'):
        selected = C8_FROZEN_FACTORS[side] if side in C8_ENABLED_SIDES else []
        signs = C8_FROZEN_SIGNS[side]
        event_counts, positive = ({}, {})
        for ticker in tech_tickers:
            block = tr.loc[tr['Ticker'].eq(ticker)].sort_values('DlyCalDt')
            assert len(block) >= 30
            if selected:
                flags = np.column_stack([c6_favorable_pct(block[f].to_numpy(dtype=float), block[f].to_numpy(dtype=float), int(signs[f])) >= C8_SINGLE_FACTOR_Q for f in selected])
                qualifying = flags.sum(axis=1) >= C8_MIN_VOTES
                signed_returns = block[target_col].to_numpy(dtype=float) * (1 if side == 'Long' else -1)
                good = signed_returns[qualifying & (signed_returns > 0)]
                event_counts[ticker] = int(qualifying.sum())
                positive[ticker] = good
            else:
                event_counts[ticker] = 0
                positive[ticker] = np.array([], dtype=float)
        pooled = np.concatenate(list(positive.values()))
        pooled_m = float(pooled.mean()) if len(pooled) >= C8_MIN_SIDE_POSITIVE else np.nan
        details[side] = {'Pooled_Positive_N': len(pooled), 'Pooled_M': pooled_m, 'Total_Qualifying_Train_Signals': sum(event_counts.values())}
        for ticker in tech_tickers:
            good = positive[ticker]
            n = len(good)
            if not selected:
                m, source = (np.nan, 'DISABLED_SIDE')
            elif n >= C8_MIN_STOCK_POSITIVE:
                m, source = (float(good.mean()), 'STOCK_TRAIN')
            elif np.isfinite(pooled_m):
                m, source = (pooled_m, 'SIDE_POOLED_TRAIN')
            else:
                m, source = (np.nan, 'UNAVAILABLE_NO_TRADE')
            if selected and (not np.isfinite(m) or not 0 < m < 1):
                m, source = (np.nan, 'UNAVAILABLE_NO_TRADE')
            lookup[side, ticker] = (m, source, n, event_counts[ticker])
            records.append({'Fold': 'FROZEN_THROUGH_2021', 'Direction': side, 'Ticker': ticker, 'Calibration': source, 'Train_Confluence_Signals': event_counts[ticker], 'Train_Positive_Events': n, 'Pooled_Positive_Events': len(pooled), 'M_Positive_Signed_Return': m, 'Take_Profit_Pct': C8_TP_M_MULT * m if np.isfinite(m) else np.nan, 'Stop_Loss_Pct': C8_SL_M_MULT * m if np.isfinite(m) else np.nan})
    return (lookup, records, details)
c8_calibrated, c8_records, c8_m_details = c8_calibrate_m()
cell8_m_calibration_df = pd.DataFrame(c8_records)
c8_m_by_fold = {(stage, f'{a}-{b}'): c8_calibrated for stage, (a, b) in C8_SCHEDULE.items()}
C8_TRADE_COUNTER = 0

def c8_mark_equity(cash, positions, bars, mark_col='DlyClose'):
    value = float(cash)
    for ticker, p in positions.items():
        mark = float(bars[ticker][mark_col])
        if p['Direction'] == 'Long':
            value += p['Qty'] * mark
        else:
            value += p['Qty'] * (p['Entry_Price'] - mark)
    return value

def c8_available_cash(cash, positions):
    locked = sum((C8_SHORT_COLLATERAL_RATIO * p['Entry_Notional_USD'] for p in positions.values() if p['Direction'] == 'Short'))
    return float(cash) - locked

def c8_exit_price(p, bar, allow_intraday=True):
    op, hi, lo = (float(bar[k]) for k in ('DlyOpen', 'DlyHigh', 'DlyLow'))
    tp, sl = (p['TP_Price'], p['SL_Price'])
    if p['Direction'] == 'Long':
        if op <= sl:
            return (op, 'STOP_GAP_OPEN', False)
        if op >= tp:
            return (op, 'TAKE_PROFIT_GAP_OPEN', False)
        hit_stop, hit_tp = (hi >= sl and lo <= sl, hi >= tp)
    else:
        if op >= sl:
            return (op, 'STOP_GAP_OPEN', False)
        if op <= tp:
            return (op, 'TAKE_PROFIT_GAP_OPEN', False)
        hit_stop, hit_tp = (hi >= sl, lo <= tp)
    if not allow_intraday:
        return (None, None, False)
    if hit_stop and hit_tp:
        return (sl, 'STOP_AND_TP_SAME_BAR_STOP_FIRST', True)
    if hit_stop:
        return (sl, 'STOP_INTRADAY', False)
    if hit_tp:
        return (tp, 'TAKE_PROFIT_INTRADAY', False)
    return (None, None, False)
for _side in ('Long', 'Short'):
    _p = {'Direction': _side, 'TP_Price': 110.0 if _side == 'Long' else 90.0, 'SL_Price': 95.0 if _side == 'Long' else 105.0}
    assert c8_exit_price(_p, {'DlyOpen': 100.0, 'DlyHigh': 112.0, 'DlyLow': 88.0})[1] == 'STOP_AND_TP_SAME_BAR_STOP_FIRST'
    assert c8_exit_price(_p, {'DlyOpen': 100.0, 'DlyHigh': 101.0, 'DlyLow': 99.0})[0] is None

def c8_run_stage(period, begin_capital):
    global C8_TRADE_COUNTER
    assert period in C8_SCHEDULE
    panel = c8_panel.loc[c8_panel['Period'].eq(period)].sort_values(['DlyCalDt', 'Ticker']).copy()
    dates = list(pd.DatetimeIndex(panel['DlyCalDt'].unique()).sort_values())
    assert dates and (not panel[['DlyCalDt', 'Ticker']].duplicated().any())
    bars_by_date = {}
    for d, daily in panel.groupby('DlyCalDt', sort=True):
        bars_by_date[pd.Timestamp(d)] = {r.Ticker: {'DlyOpen': float(r.DlyOpen), 'DlyHigh': float(r.DlyHigh), 'DlyLow': float(r.DlyLow), 'DlyClose': float(r.DlyClose)} for r in daily.itertuples(index=False)}
        assert len(bars_by_date[pd.Timestamp(d)]) == len(tech_tickers)
    qualified = panel.loc[panel['Candidate_Direction'].ne('NONE')].copy()
    qualified = qualified.sort_values(['DlyCalDt', 'Candidate_Priority', 'Candidate_Votes', 'Candidate_Strength', 'Ticker'], ascending=[True, False, False, False, True], kind='mergesort')
    assert not qualified['Daily_Conflict'].any()
    candidate_by_date = {pd.Timestamp(d): g.to_dict('records') for d, g in qualified.groupby('DlyCalDt', sort=True)}
    cash, positions, trades, curve, skips = (float(begin_capital), {}, [], [], [])
    pending = []
    last_date = dates[-1]
    fold = f'{C8_SCHEDULE[period][0]}-{C8_SCHEDULE[period][1]}'
    m_table = c8_m_by_fold[period, fold]

    def close_position(ticker, exit_price, date, reason, ambiguous=False):
        nonlocal cash
        p = positions.pop(ticker)
        qty, price = (p['Qty'], float(exit_price))
        gross = qty * (price - p['Entry_Price']) * (1 if p['Direction'] == 'Long' else -1)
        fee = qty * price * C8_COST_RATE
        if p['Direction'] == 'Long':
            cash += qty * price - fee
        else:
            cash += gross - fee
        result = dict(p)
        result.update({'Exit_Date': pd.Timestamp(date), 'Exit_Price': price, 'Exit_Reason': reason, 'Exit_Notional_USD': qty * price, 'Trade_Bars': p['Bars_Held'], 'Gross_PnL_USD': gross, 'Exit_Cost_USD': fee, 'Transaction_Cost_USD': p['Entry_Cost_USD'] + fee, 'Net_PnL_USD': gross - p['Entry_Cost_USD'] - fee, 'Return_On_Notional': (gross - p['Entry_Cost_USD'] - fee) / p['Entry_Notional_USD'], 'Win': gross - p['Entry_Cost_USD'] - fee > 0, 'OHLC_Both_Levels_Touched_Stop_First': bool(ambiguous)})
        trades.append(result)
    for date in dates:
        date = pd.Timestamp(date)
        bars = bars_by_date[date]
        exited_at_open = set()
        for ticker in list(positions):
            ep, reason, ambiguous = c8_exit_price(positions[ticker], bars[ticker], False)
            if ep is not None:
                close_position(ticker, ep, date, reason, ambiguous)
                exited_at_open.add(ticker)
        if date != last_date:
            for candidate in pending:
                if len(positions) >= C8_MAX_POSITIONS:
                    break
                ticker, side = (candidate['Ticker'], candidate['Candidate_Direction'])
                if ticker in positions or ticker in exited_at_open:
                    continue
                assert side in C8_ENABLED_SIDES
                m, m_source, positive_n, train_n = m_table[side, ticker]
                if not np.isfinite(m):
                    skips.append({'Period': period, 'Date': date, 'Ticker': ticker, 'Direction': side, 'Reason': 'NO_TRAIN_M'})
                    continue
                op = float(bars[ticker]['DlyOpen'])
                signal_close = float(candidate['DlyClose'])
                gap = op / signal_close - 1.0
                if abs(gap) > C8_MAX_ENTRY_GAP + 1e-12:
                    skips.append({'Period': period, 'Date': date, 'Ticker': ticker, 'Direction': side, 'Reason': 'OPEN_OUTSIDE_CONFIGURED_GAP'})
                    continue
                equity_open = c8_mark_equity(cash, positions, bars, 'DlyOpen')
                if not np.isfinite(equity_open) or equity_open <= 0:
                    skips.append({'Period': period, 'Date': date, 'Ticker': ticker, 'Direction': side, 'Reason': 'NONPOSITIVE_EQUITY'})
                    continue
                target_notional = C8_POSITION_FRACTION * equity_open
                qty = int(np.floor(target_notional / op))
                notional = qty * op
                fee = notional * C8_COST_RATE
                if side == 'Long':
                    tp = op * (1 + C8_TP_M_MULT * m)
                    sl = op * (1 - C8_SL_M_MULT * m)
                else:
                    tp = op * (1 - C8_TP_M_MULT * m)
                    sl = op * (1 + C8_SL_M_MULT * m)
                if not (0 < tp < np.inf and 0 < sl < np.inf):
                    skips.append({'Period': period, 'Date': date, 'Ticker': ticker, 'Direction': side, 'Reason': 'INVALID_TP_SL_LEVEL'})
                    continue
                needed = notional + fee if side == 'Long' else C8_SHORT_COLLATERAL_RATIO * notional + fee
                if qty < 1 or needed > c8_available_cash(cash, positions) + 1e-08:
                    skips.append({'Period': period, 'Date': date, 'Ticker': ticker, 'Direction': side, 'Reason': 'INSUFFICIENT_CASH_OR_EQUITY'})
                    continue
                if side == 'Long':
                    cash -= notional + fee
                else:
                    cash -= fee
                C8_TRADE_COUNTER += 1
                positions[ticker] = {'Trade_ID': f'T{C8_TRADE_COUNTER:06d}', 'Period': period, 'Fold': fold, 'Ticker': ticker, 'Direction': side, 'Signal_Date': pd.Timestamp(candidate['DlyCalDt']), 'Entry_Date': date, 'Signal_Votes': int(candidate['Candidate_Votes']), 'Signal_Strength': float(candidate['Candidate_Strength']), 'Signal_Priority': float(candidate['Candidate_Priority']), 'Signal_Rank': int(candidate['Daily_Rank_All']), 'Matched_Factors': candidate[f'{side}_Matched_Factors'], 'Matched_Families': candidate[f'{side}_Matched_Families'], 'Signal_Close': signal_close, 'Entry_Open': op, 'Entry_Price': op, 'Entry_Gap_From_Signal_Close': gap, 'Entry_Equity_USD': equity_open, 'Target_Notional_USD': target_notional, 'Qty': qty, 'Entry_Notional_USD': notional, 'Entry_Cost_USD': fee, 'M_Source': m_source, 'M_Train_Positive_Events': int(positive_n), 'M_Train_Confluence_Events': int(train_n), 'Historical_M': m, 'TP_Pct': C8_TP_M_MULT * m, 'SL_Pct': C8_SL_M_MULT * m, 'TP_Price': tp, 'SL_Price': sl, 'Bars_Held': 0}
        for ticker in list(positions):
            positions[ticker]['Bars_Held'] += 1
            ep, reason, ambiguous = c8_exit_price(positions[ticker], bars[ticker], True)
            if ep is not None:
                close_position(ticker, ep, date, reason, ambiguous)
        for ticker in list(positions):
            if positions[ticker]['Bars_Held'] >= C8_MAX_HOLD_BARS:
                close_position(ticker, bars[ticker]['DlyClose'], date, f'TIME_EXIT_{C8_MAX_HOLD_BARS}_BARS_CLOSE')
        if date == last_date:
            for ticker in list(positions):
                close_position(ticker, bars[ticker]['DlyClose'], date, 'FOLD_END_CLOSE')
        equity = c8_mark_equity(cash, positions, bars)
        reserved = sum((C8_SHORT_COLLATERAL_RATIO * p['Entry_Notional_USD'] for p in positions.values() if p['Direction'] == 'Short'))
        long_mv = sum((p['Qty'] * bars[t]['DlyClose'] for t, p in positions.items() if p['Direction'] == 'Long'))
        short_exposure = sum((p['Qty'] * bars[t]['DlyClose'] for t, p in positions.items() if p['Direction'] == 'Short'))
        assert len(positions) <= C8_MAX_POSITIONS
        assert c8_available_cash(cash, positions) >= -1e-05
        assert np.isfinite(equity) and equity > 0, 'Equity <= 0; margin-call/liquidation is not modeled.'
        curve.append({'Period': period, 'Year': int(date.year), 'Date': date, 'Cash_USD': cash, 'Short_Collateral_Reserved_USD': reserved, 'Available_Cash_USD': cash - reserved, 'Long_Market_Value_USD': long_mv, 'Short_Market_Exposure_USD': short_exposure, 'Equity_USD': equity, 'Open_Positions': len(positions), 'Long_Positions': sum((p['Direction'] == 'Long' for p in positions.values())), 'Short_Positions': sum((p['Direction'] == 'Short' for p in positions.values())), 'Daily_Conflict': bool(panel.loc[panel['DlyCalDt'].eq(date), 'Daily_Conflict'].iloc[0])})
        pending = [] if date == last_date else candidate_by_date.get(date, [])
    assert not positions
    cols = ['Trade_ID', 'Period', 'Fold', 'Ticker', 'Direction', 'Signal_Date', 'Entry_Date', 'Exit_Date', 'Signal_Votes', 'Signal_Strength', 'Signal_Priority', 'Signal_Rank', 'Matched_Factors', 'Matched_Families', 'Signal_Close', 'Entry_Open', 'Entry_Price', 'Entry_Gap_From_Signal_Close', 'Entry_Equity_USD', 'Target_Notional_USD', 'Qty', 'Entry_Notional_USD', 'Entry_Cost_USD', 'M_Source', 'M_Train_Positive_Events', 'M_Train_Confluence_Events', 'Historical_M', 'TP_Pct', 'SL_Pct', 'TP_Price', 'SL_Price', 'Bars_Held', 'Exit_Price', 'Exit_Reason', 'Exit_Notional_USD', 'Trade_Bars', 'Gross_PnL_USD', 'Exit_Cost_USD', 'Transaction_Cost_USD', 'Net_PnL_USD', 'Return_On_Notional', 'Win', 'OHLC_Both_Levels_Touched_Stop_First']
    return (pd.DataFrame(curve), pd.DataFrame(trades, columns=cols), pd.DataFrame(skips, columns=['Period', 'Date', 'Ticker', 'Direction', 'Reason']), cash)
c8_valid_curve, c8_valid_trades, c8_valid_skips, c8_valid_end = c8_run_stage('Validation', C8_START_CAPITAL)
c8_test_curve, c8_test_trades, c8_test_skips, c8_test_end = c8_run_stage('Test', C8_START_CAPITAL)
cell8_trade_log_df = pd.concat([c8_valid_trades, c8_test_trades], ignore_index=True)
cell8_skipped_entries_df = pd.concat([c8_valid_skips, c8_test_skips], ignore_index=True)
cell8_trade_log_df['Exit_Date'] = pd.to_datetime(cell8_trade_log_df['Exit_Date'])

def c8_add_derived_equity(df, initial):
    frame = df.copy().sort_values('Date').reset_index(drop=True)
    frame['Daily_Return'] = frame['Equity_USD'].pct_change()
    if len(frame):
        frame.loc[0, 'Daily_Return'] = frame.loc[0, 'Equity_USD'] / initial - 1
    frame['Cumulative_Return'] = frame['Equity_USD'] / initial - 1
    frame['Peak_Equity'] = np.maximum.accumulate(np.r_[initial, frame['Equity_USD'].to_numpy()])[1:]
    frame['Drawdown'] = frame['Equity_USD'] / frame['Peak_Equity'] - 1
    prices = c8_panel.loc[c8_panel['Period'].eq(frame['Period'].iat[0])].pivot(index='DlyCalDt', columns='Ticker', values='DlyClose').reindex(frame['Date'])
    assert prices.notna().all().all()
    frame['Equal_Weight_BuyHold_Equity'] = initial * prices.div(prices.iloc[0]).mean(axis=1).to_numpy()
    return frame

def c8_add_flat_start(df):
    frame = df.copy().sort_values('Date').reset_index(drop=True)
    assert frame['Open_Positions'].iat[0] == 0
    assert np.isclose(frame['Equity_USD'].iat[0], C8_START_CAPITAL)
    frame['Row_Type'] = 'DAILY_CLOSE'
    anchor = {c: np.nan for c in frame.columns}
    anchor.update({'Period': frame['Period'].iat[0], 'Year': int(frame['Year'].iat[0]), 'Date': frame['Date'].iat[0], 'Row_Type': 'INITIAL', 'Cash_USD': C8_START_CAPITAL, 'Short_Collateral_Reserved_USD': 0.0, 'Available_Cash_USD': C8_START_CAPITAL, 'Long_Market_Value_USD': 0.0, 'Short_Market_Exposure_USD': 0.0, 'Equity_USD': C8_START_CAPITAL, 'Open_Positions': 0, 'Long_Positions': 0, 'Short_Positions': 0, 'Daily_Conflict': False, 'Daily_Return': 0.0, 'Cumulative_Return': 0.0, 'Peak_Equity': C8_START_CAPITAL, 'Drawdown': 0.0, 'Equal_Weight_BuyHold_Equity': C8_START_CAPITAL})
    return pd.concat([pd.DataFrame([anchor]), frame], ignore_index=True)
c8_valid_curve = c8_add_flat_start(c8_add_derived_equity(c8_valid_curve, C8_START_CAPITAL))
c8_test_curve = c8_add_flat_start(c8_add_derived_equity(c8_test_curve, C8_START_CAPITAL))
cell8_equity_df = pd.concat([c8_valid_curve, c8_test_curve], ignore_index=True)

def c8_metrics(frame, trades, init, period, year='ALL'):
    actual = frame.loc[frame['Row_Type'].eq('DAILY_CLOSE')] if 'Row_Type' in frame else frame
    n = len(actual)
    eq = actual['Equity_USD'].to_numpy(dtype=float)
    end = float(eq[-1]) if n else float(init)
    daily = np.diff(np.r_[init, eq]) / np.r_[init, eq][:-1] if n else np.array([], dtype=float)
    vol = float(daily.std(ddof=1)) if n >= 2 else np.nan
    sharpe = float(np.sqrt(252) * daily.mean() / vol) if n >= 3 and vol > 0 else np.nan
    draw = eq / np.maximum.accumulate(np.r_[init, eq])[1:] - 1 if n else np.array([0.0])
    profits = trades['Net_PnL_USD']
    gross_win = float(profits.loc[profits > 0].sum())
    gross_loss = -float(profits.loc[profits < 0].sum())
    pf = gross_win / gross_loss if gross_loss > 0 else np.inf if gross_win > 0 else np.nan
    return {'Period': period, 'Year': year, 'Initial_Equity_USD': float(init), 'Ending_Equity_USD': end, 'Net_Profit_USD': end - init, 'Total_Return': end / init - 1, 'Annualized_Return': (end / init) ** (252 / n) - 1 if n and end > 0 else np.nan, 'Max_Drawdown': float(min(0, draw.min())), 'Sharpe_252': sharpe, 'Trading_Days': n, 'Trades': len(trades), 'Long_Trades': int(trades['Direction'].eq('Long').sum()), 'Short_Trades': int(trades['Direction'].eq('Short').sum()), 'Trade_Win_Rate': float((profits > 0).mean()) if len(profits) else np.nan, 'Avg_Trade_Return': float(trades['Return_On_Notional'].mean()) if len(trades) else np.nan, 'Profit_Factor': pf, 'Invested_Day_Fraction': float(actual['Open_Positions'].gt(0).mean()) if n else 0.0, 'Mean_Open_Positions': float(actual['Open_Positions'].mean()) if n else 0.0}
metrics = []
for period, curve in [('Validation', c8_valid_curve), ('Test', c8_test_curve)]:
    stage_trades = cell8_trade_log_df.loc[cell8_trade_log_df['Period'].eq(period)]
    metrics.append(c8_metrics(curve, stage_trades, C8_START_CAPITAL, period))
    opening = C8_START_CAPITAL
    for yr, days in curve.loc[curve['Row_Type'].eq('DAILY_CLOSE')].groupby('Year'):
        exited = stage_trades.loc[stage_trades['Exit_Date'].dt.year.eq(int(yr))]
        metrics.append(c8_metrics(days, exited, opening, period, int(yr)))
        opening = float(days['Equity_USD'].iloc[-1])
cell8_performance_df = pd.DataFrame(metrics)

def c8_audit_stage(period, curve, trades, closing_cash):
    actual = curve.loc[curve['Row_Type'].eq('DAILY_CLOSE')]
    assert curve['Row_Type'].iat[0] == 'INITIAL'
    assert len(curve) == len(actual) + 1
    assert np.isclose(curve['Equity_USD'].iat[0], C8_START_CAPITAL)
    assert int(curve['Open_Positions'].iat[0]) == 0
    assert int(curve['Open_Positions'].iat[-1]) == 0
    assert np.isfinite(curve['Equity_USD']).all() and (curve['Equity_USD'] > 0).all()
    assert actual['Open_Positions'].le(C8_MAX_POSITIONS).all()
    assert actual['Available_Cash_USD'].ge(-1e-05).all()
    assert np.isclose(trades['Net_PnL_USD'].sum(), closing_cash - C8_START_CAPITAL, atol=1e-05)
    assert np.isclose(curve['Equity_USD'].iat[-1], closing_cash, atol=1e-05)
    if len(trades):
        assert trades['Trade_ID'].is_unique
        assert trades['Direction'].isin(C8_ENABLED_SIDES).all()
        assert trades['Signal_Votes'].between(C8_MIN_VOTES, C51_N).all()
        assert (trades['Entry_Date'] > trades['Signal_Date']).all()
        assert (trades['Exit_Date'] >= trades['Entry_Date']).all()
        assert trades['Trade_Bars'].between(1, C8_MAX_HOLD_BARS).all()
        assert trades['Entry_Gap_From_Signal_Close'].abs().le(C8_MAX_ENTRY_GAP + 1e-12).all()
        assert trades['Qty'].ge(1).all()
        assert np.allclose(trades['Entry_Notional_USD'], trades['Qty'] * trades['Entry_Price'])
        assert np.allclose(trades['Net_PnL_USD'], trades['Gross_PnL_USD'] - trades['Transaction_Cost_USD'])
        assert np.allclose(trades['Return_On_Notional'], trades['Net_PnL_USD'] / trades['Entry_Notional_USD'])
        stage_signals = c8_panel.loc[c8_panel['Period'].eq(period)].set_index(['DlyCalDt', 'Ticker'])
        signal_key = pd.MultiIndex.from_arrays([trades['Signal_Date'], trades['Ticker']])
        src = stage_signals.reindex(signal_key)
        assert src['Candidate_Direction'].tolist() == trades['Direction'].tolist()
        assert np.array_equal(src['Candidate_Votes'].to_numpy(dtype=int), trades['Signal_Votes'].to_numpy(dtype=int))
        assert np.allclose(src['Candidate_Strength'].to_numpy(dtype=float), trades['Signal_Strength'].to_numpy(dtype=float))
        assert np.allclose(src['Candidate_Priority'].to_numpy(dtype=float), trades['Signal_Priority'].to_numpy(dtype=float))
        assert np.array_equal(src['Daily_Rank_All'].to_numpy(dtype=int), trades['Signal_Rank'].to_numpy(dtype=int))
        calendars = pd.DatetimeIndex(stage_signals.index.get_level_values(0).unique()).sort_values()
        signal_positions = calendars.get_indexer(pd.to_datetime(trades['Signal_Date']))
        assert (signal_positions >= 0).all()
        assert (signal_positions + 1 < len(calendars)).all()
        assert (calendars[signal_positions + 1] == pd.to_datetime(trades['Entry_Date']).to_numpy()).all()
        entry_key = pd.MultiIndex.from_arrays([trades['Entry_Date'], trades['Ticker']])
        assert np.allclose(trades['Entry_Price'], stage_signals['DlyOpen'].reindex(entry_key).to_numpy())
        for ticker, stock in trades.groupby('Ticker'):
            by_date = stock.sort_values(['Entry_Date', 'Exit_Date'])
            assert not (by_date['Entry_Date'].iloc[1:].to_numpy() < by_date['Exit_Date'].iloc[:-1].to_numpy()).any(), ticker
    return {'Period': period, 'Trading_Days': len(actual), 'Start_Equity': float(curve['Equity_USD'].iat[0]), 'Start_Cash': float(curve['Cash_USD'].iat[0]), 'Start_Positions': 0, 'Final_Equity': float(curve['Equity_USD'].iat[-1]), 'Trades': len(trades), 'Closed_PnL': float(trades['Net_PnL_USD'].sum()), 'NAV_Change': float(curve['Equity_USD'].iat[-1] - C8_START_CAPITAL), 'Conflict_Days': int(actual['Daily_Conflict'].sum()), 'Stop_Exits': int(trades['Exit_Reason'].str.contains('STOP').sum()), 'TP_Exits': int(trades['Exit_Reason'].str.contains('TAKE_PROFIT').sum()), 'Ambiguous_OHLC': int(trades['OHLC_Both_Levels_Touched_Stop_First'].sum())}
cell8_audit_df = pd.DataFrame([c8_audit_stage('Validation', c8_valid_curve, c8_valid_trades, c8_valid_end), c8_audit_stage('Test', c8_test_curve, c8_test_trades, c8_test_end)])
rows = []
for period, trades in [('Validation', c8_valid_trades), ('Test', c8_test_trades)]:
    total_entry = float(trades['Entry_Notional_USD'].sum())
    gross = float(trades['Gross_PnL_USD'].sum())
    net = float(trades['Net_PnL_USD'].sum())
    row = {'Period': period, 'Trades': len(trades), 'Gross_PnL_USD': gross, 'Actual_Net_PnL_USD': net, 'Entry_Notional_Total_USD': total_entry, 'Breakeven_Roundtrip_Bps': gross / total_entry * 10000 if total_entry else np.nan, 'Average_PnL_USD_Per_Trade': net / len(trades) if len(trades) else np.nan}
    for bps in (1, 2, 5, 10):
        row[f'PnL_if_roundtrip_{bps}bps'] = gross - total_entry * bps / 10000
    rows.append(row)
cell8_cost_sensitivity_df = pd.DataFrame(rows)
