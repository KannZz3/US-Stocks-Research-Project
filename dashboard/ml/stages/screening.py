C5_Q = 0.8
C5_TOP = 10
C5_H = 10
C5_MIN_TRAIN_YEARS = 2
C5_MIN_YEAR_EVENTS = 5
C5_MIN_STOCK_EVENTS = 5
C5_CORR_ALERT = 0.8
C5_SIDES = ('Long', 'Short')
C5_COLS = list(feature_cols)
C5_TICKERS = list(tech_tickers)
C5_RANK_WEIGHTS = {'Mean_LOFO': 0.35, 'Positive_LOFO_Year_Fraction': 0.15, 'Positive_Lift_Year_Fraction': 0.15, 'Positive_Lift_Stocks': 0.1, 'Leave_Best_Year_Lift': 0.15, 'Sign_Consistency': 0.1}
assert np.isclose(sum(C5_RANK_WEIGHTS.values()), 1.0)
assert len(C5_COLS) == 31 and len(C5_TICKERS) == 9
assert target_col == 'TARGET_10' and HORIZON == C5_H
assert len(development_folds) == len(scaled_development_folds) == 6
assert len(train_oos_folds) == len(scaled_train_oos_folds) == 4
assert set(C5_TICKERS) == set(data.loc[data['Split'].eq('Train'), 'Ticker'])

def c5_directional_y(raw_r, side):
    r = np.asarray(raw_r, dtype=float)
    return np.maximum(r if side == 'Long' else -r, 0.0)

def c5_mature_indices(block):
    assert block.groupby('PERMNO')['DlyCalDt'].apply(lambda s: s.is_monotonic_increasing).all()
    end_date = block.groupby('PERMNO', sort=False)['DlyCalDt'].shift(-C5_H)
    idx = block.index[end_date.notna()]
    assert len(block) - len(idx) == len(C5_TICKERS) * C5_H
    return idx

def c5_calibrate_orientations(Xraw, raw_returns, tickers):
    """Choose each side's higher (+1) or lower (-1) tail from Train-only return lift."""
    Xraw = np.asarray(Xraw, dtype=float)
    returns = np.asarray(raw_returns, dtype=float)
    tickers = np.asarray(tickers)
    assert Xraw.shape == (len(returns), len(C5_COLS))
    output = {}
    for side in C5_SIDES:
        signed_r = returns * (1 if side == 'Long' else -1)
        lift_num = np.zeros((2, len(C5_COLS)))
        lift_den = np.zeros_like(lift_num)
        for ticker in C5_TICKERS:
            mask = tickers == ticker
            X, y = (Xraw[mask], signed_r[mask])
            assert len(y) > 30
            baseline = float(y.mean())
            thresholds = (np.quantile(X, 1 - C5_Q, axis=0), np.quantile(X, C5_Q, axis=0))
            for k, cutoff in enumerate(thresholds):
                selected = X <= cutoff if k == 0 else X >= cutoff
                lift_num[k] += selected.T @ (y - baseline)
                lift_den[k] += selected.sum(axis=0)
        lift = np.divide(lift_num, lift_den, out=np.full_like(lift_num, -np.inf), where=lift_den > 0)
        choice = np.argmax(lift, axis=0)
        best = lift[choice, np.arange(len(C5_COLS))]
        orientation = np.where(best > 0, np.where(choice == 1, 1, -1), 0)
        output[side] = {'orientation': orientation.astype(int), 'train_edge': best}
    return output

def c5_thin_events(flags, horizon=C5_H):
    selected = []
    last = -horizon
    for pos in np.flatnonzero(flags):
        if pos - last >= horizon:
            selected.append(pos)
            last = pos
    return np.asarray(selected, dtype=int)

def c5_inner_history(outer_fold):
    history = data.loc[outer_fold['train_full_idx']].copy()
    assert history['Split'].eq('Train').all()
    assert history['DlyCalDt'].max() < outer_fold['oos_start']
    years = sorted(history['DlyCalDt'].dt.year.unique().tolist())
    assert len(years) >= C5_MIN_TRAIN_YEARS + 1
    importance_records, tail_records, sign_records, diag_records = ([], [], [], [])
    for year in years[C5_MIN_TRAIN_YEARS:]:
        train = history.loc[history['DlyCalDt'].dt.year < year]
        oos = history.loc[history['DlyCalDt'].dt.year == year]
        tr_idx, oo_idx = (c5_mature_indices(train), c5_mature_indices(oos))
        assert data.loc[tr_idx, 'DlyCalDt'].max() < data.loc[oo_idx, 'DlyCalDt'].min()
        assert data.loc[oo_idx, 'DlyCalDt'].max() < outer_fold['oos_start']
        assert data.loc[tr_idx, 'Split'].eq('Train').all()
        assert data.loc[oo_idx, 'Split'].eq('Train').all()
        Xtr_raw = data.loc[tr_idx, C5_COLS].to_numpy(dtype=float)
        Xoo_raw = data.loc[oo_idx, C5_COLS].to_numpy(dtype=float)
        rtr = data.loc[tr_idx, target_col].to_numpy(dtype=float)
        roo = data.loc[oo_idx, target_col].to_numpy(dtype=float)
        ttr = data.loc[tr_idx, 'Ticker'].to_numpy()
        too = data.loc[oo_idx, 'Ticker'].to_numpy()
        assert all((np.isfinite(a).all() for a in (Xtr_raw, Xoo_raw, rtr, roo)))
        calibrations = c5_calibrate_orientations(Xtr_raw, rtr, ttr)
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr_raw)
        Xoo = scaler.transform(Xoo_raw)
        ticker_slices = {}
        for ticker in C5_TICKERS:
            a, b = (ttr == ticker, too == ticker)
            assert a.any() and b.any()
            assert data.loc[oo_idx[b], 'DlyCalDt'].is_monotonic_increasing
            ticker_slices[ticker] = (a, b)
        for side in C5_SIDES:
            ytr, yoo = (c5_directional_y(rtr, side), c5_directional_y(roo, side))
            model = c5_model_fit(Xtr, ytr)
            full_err = (yoo - model.predict(Xoo)) ** 2
            benchmark = float(np.mean((yoo - ytr.mean()) ** 2))
            assert benchmark > 0
            diag_records.append({'Inner_Year': year, 'Direction': side, 'Train_N': len(tr_idx), 'OOS_N': len(oo_idx), 'Full_Model_OOS_R2': 1 - full_err.mean() / benchmark})
            orientations = calibrations[side]['orientation']
            train_edges = calibrations[side]['train_edge']
            oriented_r = roo * (1 if side == 'Long' else -1)
            for j, feature in enumerate(C5_COLS):
                keep = np.arange(len(C5_COLS)) != j
                reduced = c5_model_fit(Xtr[:, keep], ytr)
                reduced_err = (yoo - reduced.predict(Xoo[:, keep])) ** 2
                importance_records.append({'Inner_Year': year, 'Direction': side, 'Feature': feature, 'LOFO': float((reduced_err - full_err).mean() / benchmark)})
                orient = int(orientations[j])
                sign_records.append({'Inner_Year': year, 'Direction': side, 'Feature': feature, 'Orientation': orient, 'Train_Tail_Edge': float(train_edges[j])})
                for ticker, (tr_mask, oo_mask) in ticker_slices.items():
                    xtrain, xoos = (Xtr_raw[tr_mask, j], Xoo_raw[oo_mask, j])
                    rr = oriented_r[oo_mask]
                    baseline, base_hit = (float(rr.mean()), float((rr > 0).mean()))
                    if orient == 1:
                        cutoff = float(np.quantile(xtrain, C5_Q))
                        flags = xoos >= cutoff
                    elif orient == -1:
                        cutoff = float(np.quantile(xtrain, 1 - C5_Q))
                        flags = xoos <= cutoff
                    else:
                        cutoff = np.nan
                        flags = np.zeros(len(xoos), dtype=bool)
                    sparse = c5_thin_events(flags)
                    thin, raw = (rr[sparse], rr[flags])
                    tail_records.append({'Inner_Year': year, 'Direction': side, 'Feature': feature, 'Ticker': ticker, 'Orientation': orient, 'Cutoff': cutoff, 'Raw_Signals': int(flags.sum()), 'Thin_Signals': len(sparse), 'Thin_Hits': int((thin > 0).sum()), 'Thin_Return_Sum': float(thin.sum()), 'Thin_Excess_Sum': float((thin - baseline).sum()), 'Thin_Baseline_Hits': len(sparse) * base_hit, 'Raw_Return_Sum': float(raw.sum())})
    return {'importance': pd.DataFrame(importance_records), 'tail': pd.DataFrame(tail_records), 'signs': pd.DataFrame(sign_records), 'diagnostics': pd.DataFrame(diag_records)}
c5_inner_cache = {}

def c5_rank_outer(outer, scaled, stage, fold_name):
    key = (str(outer['train_start']), str(outer['train_cutoff']), len(outer['train_full_idx']))
    if key not in c5_inner_cache:
        c5_inner_cache[key] = c5_inner_history(outer)
    inner = c5_inner_cache[key]
    fit_idx = outer['train_fit_idx']
    assert scaled['X_train'].index.equals(fit_idx)
    assert data.loc[fit_idx, 'Split'].eq('Train').all()
    outer_calib = c5_calibrate_orientations(data.loc[fit_idx, C5_COLS].to_numpy(dtype=float), data.loc[fit_idx, target_col].to_numpy(dtype=float), data.loc[fit_idx, 'Ticker'].to_numpy())
    rankings, top10, correlations = ([], {}, {})
    for side in C5_SIDES:
        imp = inner['importance'].query('Direction == @side')
        tail = inner['tail'].query('Direction == @side')
        signs = inner['signs'].query('Direction == @side')
        lofo = imp.groupby('Feature')['LOFO'].agg(Mean_LOFO='mean', Median_LOFO='median', Positive_LOFO_Years=lambda s: int((s > 0).sum()))
        lofo['Positive_LOFO_Year_Fraction'] = lofo['Positive_LOFO_Years'] / imp['Inner_Year'].nunique()
        agg = tail.groupby('Feature').agg(Raw_Signals=('Raw_Signals', 'sum'), Thin_Signals=('Thin_Signals', 'sum'), Thin_Hits=('Thin_Hits', 'sum'), Thin_Return_Sum=('Thin_Return_Sum', 'sum'), Thin_Excess_Sum=('Thin_Excess_Sum', 'sum'), Thin_Baseline_Hits=('Thin_Baseline_Hits', 'sum'))
        ranking = lofo.join(agg).reindex(C5_COLS)
        count = ranking['Thin_Signals'].replace(0, np.nan)
        ranking['Tail_Hit_Rate'] = ranking['Thin_Hits'] / count
        ranking['Tail_Hit_Lift_pp'] = 100 * (ranking['Thin_Hits'] - ranking['Thin_Baseline_Hits']) / count
        ranking['Mean_Dir_Return'] = ranking['Thin_Return_Sum'] / count
        ranking['Mean_Return_Lift'] = ranking['Thin_Excess_Sum'] / count
        ranking['Mean_Return_Lift_pp'] = 100 * ranking['Mean_Return_Lift']
        annual = tail.groupby(['Feature', 'Inner_Year'], as_index=False).agg(Signals=('Thin_Signals', 'sum'), Return_Sum=('Thin_Return_Sum', 'sum'), Excess_Sum=('Thin_Excess_Sum', 'sum'))
        annual['Annual_Mean_Dir'] = annual['Return_Sum'] / annual['Signals'].replace(0, np.nan)
        annual['Annual_Mean_Lift'] = annual['Excess_Sum'] / annual['Signals'].replace(0, np.nan)
        annual['Year_Qualified'] = annual['Signals'] >= C5_MIN_YEAR_EVENTS
        stock = tail.groupby(['Feature', 'Ticker'], as_index=False).agg(Signals=('Thin_Signals', 'sum'), Return_Sum=('Thin_Return_Sum', 'sum'), Excess_Sum=('Thin_Excess_Sum', 'sum'))
        stock['Stock_Mean_Lift'] = stock['Excess_Sum'] / stock['Signals'].replace(0, np.nan)
        stock['Stock_Qualified'] = stock['Signals'] >= C5_MIN_STOCK_EVENTS
        outer_orientation = pd.Series(outer_calib[side]['orientation'], index=C5_COLS)
        sign_match = signs.merge(outer_orientation.rename('Outer_Orientation'), left_on='Feature', right_index=True)
        sign_match['Matches'] = sign_match['Orientation'].ne(0) & sign_match['Orientation'].eq(sign_match['Outer_Orientation'])
        rows = []
        for feature in C5_COLS:
            a = annual.loc[annual['Feature'].eq(feature) & annual['Year_Qualified']]
            s = stock.loc[stock['Feature'].eq(feature) & stock['Stock_Qualified']]
            lifts = a['Annual_Mean_Lift'].to_numpy(dtype=float)
            n = len(lifts)
            positive = int((lifts > 0).sum())
            rows.append({'Feature': feature, 'Orientation': int(outer_orientation[feature]), 'Sign_Consistency': float(sign_match.loc[sign_match['Feature'].eq(feature), 'Matches'].mean()), 'Active_Years': n, 'Positive_Lift_Years': positive, 'Positive_Return_Years': int((a['Annual_Mean_Dir'] > 0).sum()), 'Positive_Lift_Year_Fraction': positive / n if n else 0, 'Qualified_Stocks': len(s), 'Positive_Lift_Stocks': int((s['Stock_Mean_Lift'] > 0).sum()), 'Median_Year_Lift': float(np.median(lifts)) if n else np.nan, 'Leave_Best_Year_Lift': float((lifts.sum() - lifts.max()) / (n - 1)) if n >= 2 else np.nan})
        r = ranking.reset_index().merge(pd.DataFrame(rows), on='Feature', validate='one_to_one')
        enough = r['Orientation'].ne(0) & r['Thin_Signals'].ge(10) & r['Mean_Dir_Return'].gt(0) & r['Mean_Return_Lift'].gt(0)
        stable = enough & r['Active_Years'].ge(2) & r['Positive_Lift_Year_Fraction'].ge(0.5) & r['Positive_Lift_Stocks'].ge(3) & r['Sign_Consistency'].ge(0.5)
        robust = stable & r['Active_Years'].ge(3) & r['Positive_Lift_Year_Fraction'].ge(0.6) & r['Positive_Return_Years'].ge(3) & r['Positive_Lift_Stocks'].ge(5) & r['Sign_Consistency'].ge(0.6) & r['Leave_Best_Year_Lift'].gt(0) & r['Thin_Signals'].ge(30)
        r['Direction_Tier'] = np.select([robust, stable, enough], [3, 2, 1], default=0).astype(int)
        r['Composite_Score'] = 0.0
        for metric, weight in C5_RANK_WEIGHTS.items():
            rank_pct = r.groupby('Direction_Tier')[metric].rank(pct=True, method='average', ascending=True).fillna(0.0)
            r['Composite_Score'] += weight * rank_pct
        r = r.sort_values(['Direction_Tier', 'Composite_Score', 'Feature'], ascending=[False, False, True], kind='mergesort').reset_index(drop=True)
        r['Rank'] = np.arange(1, len(r) + 1)
        r['Candidate'] = r['Rank'] <= C5_TOP
        r['Stage'], r['Fold'], r['Direction'] = (stage, str(fold_name), side)
        names = r.loc[r['Candidate'], 'Feature'].tolist()
        top10[side] = names
        corr = scaled['X_train'][names].corr()
        correlations[side] = corr
        r['Max_Abs_Corr_Top10'] = np.nan
        for feature in names:
            r.loc[r['Feature'].eq(feature), 'Max_Abs_Corr_Top10'] = float(corr.loc[feature].drop(index=feature).abs().max())
        rankings.append(r)
    return {'top10': top10, 'rankings': pd.concat(rankings, ignore_index=True), 'correlations': correlations, 'inner': inner, 'outer_orientations': {side: dict(zip(C5_COLS, outer_calib[side]['orientation'])) for side in C5_SIDES}}
c5_started = time.perf_counter()
factor_development, factor_train_oos = ({}, {})
c5_all_selection_rows, c5_all_rankings = ([], [])

def c5_register(result, stage, fold_name):
    c5_all_rankings.append(result['rankings'])
    for side in C5_SIDES:
        chosen = result['rankings'].loc[result['rankings']['Direction'].eq(side) & result['rankings']['Candidate']]
        c5_all_selection_rows.append({'Stage': stage, 'Fold': str(fold_name), 'Direction': side, 'Top10': ', '.join(result['top10'][side]), 'Tier3': int(chosen['Direction_Tier'].eq(3).sum()), 'Tier2Plus': int(chosen['Direction_Tier'].ge(2).sum()), 'Positive_Absolute_Return': int(chosen['Mean_Dir_Return'].gt(0).sum()), 'Positive_Excess_Return': int(chosen['Mean_Return_Lift'].gt(0).sum()), 'Inner_Years': int(result['inner']['diagnostics']['Inner_Year'].nunique())})
for (window, year), fold in sorted(development_folds.items()):
    out = c5_rank_outer(fold, scaled_development_folds[window, year], 'Development', f'{window}Y_{year}')
    factor_development[window, year] = out
    c5_register(out, 'Development', f'{window}Y_{year}')
    pass
for year, fold in sorted(train_oos_folds.items()):
    out = c5_rank_outer(fold, scaled_train_oos_folds[year], 'Train_OOS', year)
    factor_train_oos[year] = out
    c5_register(out, 'Train_OOS', year)
    pass
factor_validation = c5_rank_outer(validation_fold, scaled_validation_fold, 'Validation', '2022-2023')
c5_register(factor_validation, 'Validation', '2022-2023')
pass
factor_selection_df = pd.DataFrame(c5_all_selection_rows)
factor_rankings_df = pd.concat(c5_all_rankings, ignore_index=True)
factor_validation_top10_df = factor_validation['rankings'].loc[factor_validation['rankings']['Candidate']].copy()
assert len(factor_selection_df) == 22
assert len(factor_rankings_df) == 11 * 2 * 31
assert len(factor_validation_top10_df) == 2 * C5_TOP
assert factor_rankings_df.groupby(['Stage', 'Fold', 'Direction'])['Candidate'].sum().eq(C5_TOP).all()
assert data.loc[validation_fold['train_fit_idx'], 'Split'].eq('Train').all()
c5_runtime = time.perf_counter() - c5_started
pass
pass
pass
pass
pass
pass
pass
pass
pass
with pd.option_context('display.max_colwidth', 250):
    pass
report_columns = ['Rank', 'Feature', 'Direction_Tier', 'Orientation', 'Sign_Consistency', 'Active_Years', 'Positive_Lift_Years', 'Positive_Return_Years', 'Positive_Lift_Stocks', 'Thin_Signals', 'Raw_Signals', 'Mean_Dir_Return', 'Mean_Return_Lift_pp', 'Tail_Hit_Rate', 'Tail_Hit_Lift_pp', 'Leave_Best_Year_Lift', 'Mean_LOFO', 'Positive_LOFO_Year_Fraction', 'Composite_Score', 'Max_Abs_Corr_Top10']
for side in C5_SIDES:
    selected = factor_validation_top10_df.loc[factor_validation_top10_df['Direction'].eq(side)].sort_values('Rank')
    names = factor_validation['top10'][side]
    tail = factor_validation['inner']['tail']
    tt = tail.loc[tail['Direction'].eq(side) & tail['Feature'].isin(names)]
    pass
    pass
    year_summary = tt.groupby(['Feature', 'Inner_Year']).agg(N=('Thin_Signals', 'sum'), Return_Sum=('Thin_Return_Sum', 'sum'), Excess_Sum=('Thin_Excess_Sum', 'sum')).reset_index()
    year_summary['Mean_Dir_Return'] = year_summary['Return_Sum'] / year_summary['N'].replace(0, np.nan)
    year_summary['Mean_Return_Lift'] = year_summary['Excess_Sum'] / year_summary['N'].replace(0, np.nan)
    pass
    pass
    stock_summary = tt.groupby(['Feature', 'Ticker']).agg(N=('Thin_Signals', 'sum'), Excess=('Thin_Excess_Sum', 'sum')).reset_index()
    stock_summary['Lift'] = stock_summary['Excess'] / stock_summary['N'].replace(0, np.nan)
    matrix = stock_summary.pivot(index='Feature', columns='Ticker', values='Lift')
    pass
    pass
    corr = factor_validation['correlations'][side]
    pass
    pass
    highly = [(names[i], names[j], float(corr.iloc[i, j])) for i in range(len(names)) for j in range(i + 1, len(names)) if abs(corr.iloc[i, j]) >= C5_CORR_ALERT]
    pass
    signs = factor_validation['inner']['signs']
    pass
    pass
    tier2 = int(selected['Direction_Tier'].ge(2).sum())
    viable = int(selected['Direction_Tier'].ge(1).sum())
    pass
    if tier2 < C5_TOP:
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
