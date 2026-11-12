"""Upload validation and reference preprocessing; no writes to repository data."""
from decimal import Decimal, ROUND_HALF_UP
from io import BytesIO
from pathlib import Path
import csv
import hashlib
import numpy as np
import pandas as pd
from features import technical_features

REFERENCE=Path(__file__).resolve().parent.parent/'data/data_dashboard/data_preML_Tech.csv'
SCHEMA=next(csv.reader(REFERENCE.open(encoding='utf-8')))
RAW=SCHEMA[:10]
FEATURES=SCHEMA[10:-2]
OHLC=['DlyOpen','DlyHigh','DlyLow','DlyClose']


def export_csv(frame):
    return frame.loc[:,SCHEMA].to_csv(index=False,date_format='%Y-%m-%d',float_format='%.17g').encode('utf-8')


def digest(frame):
    return hashlib.sha256(export_csv(frame)).hexdigest()


def read_csv(content):
    try:
        header=next(csv.reader(content.decode('utf-8-sig').splitlines()))
        if len(header)!=len(set(header)): raise ValueError('Duplicate column names.')
        return pd.read_csv(BytesIO(content),dtype=str,keep_default_na=False)
    except (UnicodeError,StopIteration,pd.errors.ParserError,pd.errors.EmptyDataError) as e:
        raise ValueError('Use a nonempty UTF-8 CSV with a single header.') from e


def validate(frame,engineered=False):
    required=SCHEMA if engineered else RAW+['DlyCumFacShr']
    missing=set(required)-set(frame)
    if missing: raise ValueError('Missing required columns: '+', '.join(sorted(missing)))
    data=frame.loc[:,required].copy()
    if data.empty: raise ValueError('CSV contains no observations.')
    data.Ticker=data.Ticker.str.strip().str.upper()
    if not data.Ticker.str.fullmatch(r'[A-Z0-9][A-Z0-9.\-^/]{0,19}').all():
        raise ValueError('Ticker identifiers must be nonempty stock symbols (letters, digits, ., -, ^, /).')
    dates=pd.to_datetime(data.DlyCalDt,format='mixed',errors='coerce')
    if dates.isna().any() or not pd.api.types.is_datetime64_ns_dtype(dates.dtype) or not dates.eq(dates.dt.normalize()).all():
        raise ValueError('Dates must be valid daily dates without intraday times or time zones.')
    if not dates.between('1900-01-01','2025-12-31').all(): raise ValueError('The reference split schedule ends 2025-12-31; dates must be within 1900–2025.')
    data.DlyCalDt=dates
    for col in set(required)-{'Ticker','DlyCalDt','Split'}:
        data[col]=pd.to_numeric(data[col],errors='coerce')
        if not np.isfinite(data[col]).all(): raise ValueError(f'{col}: missing, nonnumeric or infinite values are not allowed.')
    for col in ['PERMNO','DlyVol']:
        if data[col].le(0).any() or data[col].gt(np.iinfo(np.int64).max).any() or not data[col].mod(1).eq(0).all(): raise ValueError(f'{col} must contain positive integers.')
        data[col]=data[col].astype('int64')
    if data.groupby('Ticker').PERMNO.nunique().gt(1).any() or data.groupby('PERMNO').Ticker.nunique().gt(1).any():
        raise ValueError('Ticker and PERMNO must have a one-to-one mapping.')
    if data.duplicated(['PERMNO','DlyCalDt']).any(): raise ValueError('Duplicate stock/date observations; resolve them in the input.')
    if data[OHLC+['DlyCap']].le(0).any().any(): raise ValueError('OHLC and DlyCap must be positive; DlyCap units are thousands of USD.')
    if (data.DlyHigh < data[OHLC].max(axis=1)).any() or (data.DlyLow > data[OHLC].min(axis=1)).any():
        raise ValueError('Invalid OHLC: low <= open/close <= high is required.')
    if data.sprtrn.le(-1).any() or data.groupby('DlyCalDt').sprtrn.nunique().gt(1).any():
        raise ValueError('sprtrn must exceed -1 and agree across stocks for each date.')
    if not engineered and data.DlyCumFacShr.le(0).any(): raise ValueError('DlyCumFacShr must be positive cumulative split factors.')
    data=data.sort_values(['PERMNO','DlyCalDt']).reset_index(drop=True)
    if engineered:
        data.Split=data.Split.str.strip().str.title()
        expected=split_labels(data.DlyCalDt)
        if not data.Split.eq(expected).all(): raise ValueError('Split labels must match the original chronological boundaries.')
        # Verify targets and RET_1 where ten/one consecutive daily observations
        # are represented. Incomplete coverage remains model-ineligible.
        expected_return=data.groupby('PERMNO').DlyClose.pct_change(fill_method=None)
        if not np.allclose(data.loc[expected_return.notna(),'RET_1'],expected_return.dropna(),rtol=1e-6,atol=1e-8):
            raise ValueError('RET_1 disagrees with closing-price returns; check adjustment or missing observations.')
        future=data.groupby('PERMNO').DlyClose.shift(-10)/data.DlyClose-1
        if not np.allclose(data.loc[future.notna(),'TARGET_10'],future.dropna(),rtol=1e-6,atol=1e-8):
            raise ValueError('TARGET_10 disagrees with ten-observation forward closing-price returns.')
    return data


def split_labels(dates):
    return np.select([dates.le('2021-12-31'),dates.le('2023-12-31')],['Train','Validation'],default='Test')


def process(content,adjusted_confirmed=False):
    original=read_csv(content)
    engineered=bool(set(original)&set(FEATURES+['TARGET_10','Split']))
    if engineered and not adjusted_confirmed:
        raise ValueError('Confirm engineered OHLC is already split-adjusted. Adjustment cannot be inferred from this schema.')
    data=validate(original,engineered)
    before=len(data)
    if not engineered:
        if data.groupby('PERMNO').size().lt(50).any(): raise ValueError('Every stock needs at least 50 raw observations: 39 warm-up, 10 future target, and one usable row.')
        # Decimal + ROUND_HALF_UP follows data_processing.ipynb exactly.
        indexed=original.assign(Ticker=original.Ticker.str.strip().str.upper())
        indexed['DlyCalDt']=pd.to_datetime(indexed.DlyCalDt,format='mixed')
        indexed=indexed.sort_values(['PERMNO','DlyCalDt'],key=lambda s: pd.to_numeric(s) if s.name=='PERMNO' else s).reset_index(drop=True)
        for col in OHLC:
            data[col]=[float((Decimal(p)/Decimal(f)).quantize(Decimal('0.00000001'),rounding=ROUND_HALF_UP)) if Decimal(f)!=1 else float(p) for p,f in zip(indexed[col],indexed.DlyCumFacShr)]
        volume=[Decimal(v)*Decimal(f) for v,f in zip(indexed.DlyVol,indexed.DlyCumFacShr)]
        if any(v!=v.to_integral_value() for v in volume): raise ValueError('Adjusted volume must be integral under the reference split convention.')
        data.DlyVol=np.array(volume,dtype='int64')
        data=technical_features(data[RAW].copy())
        data['TARGET_10']=data.groupby('PERMNO').DlyClose.shift(-10)/data.DlyClose-1
        data=data.dropna(subset=FEATURES+['TARGET_10']).reset_index(drop=True)
        if data.empty: raise ValueError('Insufficient usable history: at least 50 observations per stock and nondegenerate feature windows are needed.')
        if not np.isfinite(data[FEATURES+['TARGET_10']]).all().all(): raise ValueError('Feature calculation produced infinite values; check the source data.')
        data['Split']=split_labels(data.DlyCalDt)
    if data.Ticker.nunique()!=original.Ticker.str.strip().str.upper().nunique(): raise ValueError('At least one stock has no complete feature/target observations.')
    data=data.loc[:,SCHEMA]
    return data, {'Format':'Engineered' if engineered else 'Raw','Input rows':before,'Export rows':len(data),'Removed incomplete rows':before-len(data),'Stocks':data.Ticker.nunique(),'First':str(data.DlyCalDt.min().date()),'Last':str(data.DlyCalDt.max().date())}


def eligibility(data):
    reasons=[]
    n=data.Ticker.nunique()
    if n<3: reasons.append('At least three stocks are needed for cross-sectional model evaluation.')
    if data.groupby('DlyCalDt').Ticker.nunique().ne(n).any(): reasons.append('Every stock must have matching date coverage; missing observations are never filled.')
    years=data.DlyCalDt.dt.year
    for year in range(2015,2026):
        counts=data.loc[years.eq(year)].groupby('Ticker').size()
        if len(counts)!=n or counts.min()<40:
            reasons.append(f'{year}: each stock needs at least 40 usable observations for the fixed walk-forward schedule.')
    return reasons
