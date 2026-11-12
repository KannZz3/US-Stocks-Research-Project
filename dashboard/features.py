"""Retained formulas from data_engineering.ipynb Cells 2–6 and 9.
Only display statements and fixed-size assertions removed. Caller supplies a copy.
"""
import numpy as np
import pandas as pd

def technical_features(df):
    g = df.groupby('PERMNO')['DlyClose']
    for h in [1, 3, 5, 10, 20, 39]:
        df[f'RET_{h}'] = g.pct_change(periods=h, fill_method=None)
    up = (df['RET_1'] > 0).astype(float)
    up = up.mask(df['RET_1'].isna())
    df['UP_RATIO_20'] = up.groupby(df['PERMNO']).transform(lambda s: s.rolling(window=20, min_periods=20).mean())
    features_A = ['RET_1', 'RET_3', 'RET_5', 'RET_10', 'RET_20', 'RET_39', 'UP_RATIO_20']
    assert len(features_A) == 7
    assert df['UP_RATIO_20'].dropna().between(0, 1).all()
    assert np.isfinite(df[features_A].dropna().to_numpy()).all()
    pass
    pass
    pass
    pass
    pass
    pass
    g = df.groupby('PERMNO')
    close = df['DlyClose']
    log_close = np.log(close)
    for h in [10, 20]:
        ma = g['DlyClose'].transform(lambda s: s.rolling(h, min_periods=h).mean())
        df[f'MA_DEV_{h}'] = close / ma - 1
    high_20 = g['DlyHigh'].transform(lambda s: s.rolling(20, min_periods=20).max())
    low_20 = g['DlyLow'].transform(lambda s: s.rolling(20, min_periods=20).min())
    price_range = (high_20 - low_20).replace(0, np.nan)
    df['BREAKOUT_POS_20'] = (close - low_20) / price_range

    def trend_regression(y, return_t=False):
        n = len(y)
        x = np.arange(n, dtype=float)
        x = x - x.mean()
        yc = y - y.mean()
        sxx = np.dot(x, x)
        sxy = np.dot(x, yc)
        syy = np.dot(yc, yc)
        slope = sxy / sxx
        if not return_t:
            return slope
        sse = max(syy - sxy ** 2 / sxx, 0.0)
        if sse <= 0:
            return np.nan
        se = np.sqrt(sse / (n - 2) / sxx)
        return slope / se
    df['TREND_SLOPE_20'] = log_close.groupby(df['PERMNO']).transform(lambda s: s.rolling(20, min_periods=20).apply(trend_regression, raw=True))
    df['TREND_T_40'] = log_close.groupby(df['PERMNO']).transform(lambda s: s.rolling(40, min_periods=40).apply(lambda y: trend_regression(y, return_t=True), raw=True))
    log_ret = log_close.groupby(df['PERMNO']).diff()
    net_move = (log_close - log_close.groupby(df['PERMNO']).shift(20)).abs()
    total_move = log_ret.abs().groupby(df['PERMNO']).transform(lambda s: s.rolling(20, min_periods=20).sum()).replace(0, np.nan)
    df['EFF_RATIO_20'] = net_move / total_move
    features_B = ['MA_DEV_10', 'MA_DEV_20', 'BREAKOUT_POS_20', 'TREND_SLOPE_20', 'TREND_T_40', 'EFF_RATIO_20']
    assert len(features_B) == 6
    assert np.isfinite(df[features_B].dropna().to_numpy()).all()
    assert df['BREAKOUT_POS_20'].dropna().between(-1e-10, 1 + 1e-10).all()
    assert df['EFF_RATIO_20'].dropna().between(-1e-10, 1 + 1e-10).all()
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    prev_close = df.groupby('PERMNO')['DlyClose'].shift(1)
    df['GAP_1'] = df['DlyOpen'] / prev_close - 1
    overnight_log = np.log1p(df['GAP_1'])
    rolling_overnight = overnight_log.groupby(df['PERMNO']).transform(lambda s: s.rolling(5, min_periods=5).sum())
    df['OVERNIGHT_RET_5'] = np.expm1(rolling_overnight)
    daily_range = df['DlyHigh'] - df['DlyLow']
    df['CLV_1'] = (2 * df['DlyClose'] - df['DlyHigh'] - df['DlyLow']) / daily_range.replace(0, np.nan)
    df['RANGE_PCT_1'] = daily_range / prev_close
    features_C = ['GAP_1', 'OVERNIGHT_RET_5', 'CLV_1', 'RANGE_PCT_1']
    assert len(features_C) == 4
    assert df['CLV_1'].dropna().between(-1 - 1e-10, 1 + 1e-10).all()
    assert (df['RANGE_PCT_1'].dropna() >= 0).all()
    values = df[features_C].to_numpy(dtype=float)
    assert np.isfinite(values[~np.isnan(values)]).all()
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    stock = df['PERMNO']
    close = df['DlyClose']
    high = df['DlyHigh']
    low = df['DlyLow']
    ret = df['RET_1']
    prev_close = close.groupby(stock).shift(1)
    for h in [5, 20]:
        df[f'RV_{h}'] = ret.groupby(stock).transform(lambda s: s.rolling(h, min_periods=h).std(ddof=1))
    downside_sq = np.minimum(ret, 0) ** 2
    df['DOWNSIDE_RMS_20'] = np.sqrt(downside_sq.groupby(stock).transform(lambda s: s.rolling(20, min_periods=20).mean()))
    true_range = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    true_range = true_range.mask(prev_close.isna())
    atr_14 = true_range.groupby(stock).transform(lambda s: s.rolling(14, min_periods=14).mean())
    df['ATR_PCT_14'] = atr_14 / close
    hl_sq = np.log(high / low) ** 2
    parkinson_mean = hl_sq.groupby(stock).transform(lambda s: s.rolling(20, min_periods=20).mean())
    df['PARKINSON_VOL_20'] = np.sqrt(parkinson_mean / (4 * np.log(2)))
    overnight_log = np.log1p(df['GAP_1'])
    df['OVERNIGHT_VOL_20'] = overnight_log.groupby(stock).transform(lambda s: s.rolling(20, min_periods=20).std(ddof=1))

    def rolling_max_drawdown(prices):
        running_max = np.maximum.accumulate(prices)
        drawdowns = 1 - prices / running_max
        return np.max(drawdowns)
    df['MAX_DRAWDOWN_20'] = close.groupby(stock).transform(lambda s: s.rolling(20, min_periods=20).apply(rolling_max_drawdown, raw=True))
    features_D = ['RV_5', 'RV_20', 'DOWNSIDE_RMS_20', 'ATR_PCT_14', 'PARKINSON_VOL_20', 'OVERNIGHT_VOL_20', 'MAX_DRAWDOWN_20']
    assert len(features_D) == 7
    values = df[features_D].to_numpy(dtype=float)
    assert np.isfinite(values[~np.isnan(values)]).all()
    assert (df[features_D].dropna() >= 0).all().all()
    assert df['MAX_DRAWDOWN_20'].dropna().between(0, 1).all()
    total_features = sum((len(x) for x in [features_A, features_B, features_C, features_D]))
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    stock = df['PERMNO']
    close = df['DlyClose']
    volume = df['DlyVol']
    ret = df['RET_1']
    dvol = close * volume
    mcap_usd = df['DlyCap'] * 1000
    assert (dvol > 0).all()
    assert (mcap_usd > 0).all()
    dvol_5 = dvol.groupby(stock).transform(lambda s: s.rolling(5, min_periods=5).mean())
    dvol_20 = dvol.groupby(stock).transform(lambda s: s.rolling(20, min_periods=20).mean())
    df['REL_DVOL_5_20'] = dvol_5 / dvol_20 - 1
    df['LOG_DVOL_20'] = np.log(dvol_20)
    daily_turnover = dvol / mcap_usd
    df['TURNOVER_20'] = daily_turnover.groupby(stock).transform(lambda s: s.rolling(20, min_periods=20).mean())
    daily_illiq = ret.abs() / dvol
    illiq_20 = daily_illiq.groupby(stock).transform(lambda s: s.rolling(20, min_periods=20).mean())
    df['LOG_ILLIQ_20'] = np.log(illiq_20.where(illiq_20 > 0))
    log_dvol = np.log(dvol)
    dlog_dvol = log_dvol.groupby(stock).diff()
    df['RET_DVOL_CORR_20'] = ret.groupby(stock).transform(lambda s: s.rolling(20, min_periods=20).corr(dlog_dvol.loc[s.index]))
    signed_dvol = np.sign(ret) * dvol
    signed_sum_20 = signed_dvol.groupby(stock).transform(lambda s: s.rolling(20, min_periods=20).sum())
    dvol_sum_20 = dvol.groupby(stock).transform(lambda s: s.rolling(20, min_periods=20).sum())
    df['SIGNED_DVOL_BAL_20'] = signed_sum_20 / dvol_sum_20
    features_E = ['REL_DVOL_5_20', 'LOG_DVOL_20', 'TURNOVER_20', 'LOG_ILLIQ_20', 'RET_DVOL_CORR_20', 'SIGNED_DVOL_BAL_20']
    assert len(features_E) == 6
    values = df[features_E].to_numpy(dtype=float)
    assert np.isfinite(values[~np.isnan(values)]).all()
    assert (df['TURNOVER_20'].dropna() > 0).all()
    assert df['RET_DVOL_CORR_20'].dropna().between(-1 - 1e-10, 1 + 1e-10).all()
    assert df['SIGNED_DVOL_BAL_20'].dropna().between(-1 - 1e-10, 1 + 1e-10).all()
    total_features = sum((len(x) for x in [features_A, features_B, features_C, features_D, features_E]))
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    mcap_usd = df['DlyCap'] * 1000
    assert (mcap_usd > 0).all()
    df['LOG_MCAP'] = np.log(mcap_usd)
    return df
