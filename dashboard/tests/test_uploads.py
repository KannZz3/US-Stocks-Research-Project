"""Reference consistency, validation, session isolation and genuine uploaded execution."""
from pathlib import Path
from io import BytesIO
import copy
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
import upload_data as u
from ml import engine

ROOT=Path(__file__).resolve().parents[2]

@pytest.fixture(scope='module')
def raw():
    return pd.read_csv(ROOT/'data/data_processing/45stocks_daily_raw.csv',dtype=str,keep_default_na=False)

@pytest.fixture(scope='module')
def engineered():
    return u.process(u.REFERENCE.read_bytes(),True)[0]


def test_reference_all_45(raw):
    actual,audit=u.process(raw.to_csv(index=False).encode())
    expected=pd.concat([pd.read_csv(p,parse_dates=['DlyCalDt'],float_precision='round_trip') for p in u.REFERENCE.parent.glob('data_preML_*.csv')]).sort_values(['PERMNO','DlyCalDt']).reset_index(drop=True)
    assert list(actual)==u.SCHEMA and actual.shape==(122265,43)
    assert audit['Removed incomplete rows']==2205
    pd.testing.assert_frame_equal(actual,expected,check_dtype=False,rtol=1e-10,atol=1e-10)
    reread=pd.read_csv(BytesIO(u.export_csv(actual)))
    assert list(reread)==u.SCHEMA
    assert pd.api.types.is_integer_dtype(reread.PERMNO)
    assert pd.api.types.is_integer_dtype(reread.DlyVol)


def test_raw_arbitrary_stock_count_and_prefix_invariance(raw):
    subset=raw[raw.Ticker.isin(['AAPL','MSFT','NVDA'])]
    all_data,_=u.process(subset.to_csv(index=False).encode())
    early,_=u.process(subset[subset.DlyCalDt<'2022-01-01'].to_csv(index=False).encode())
    assert all_data.Ticker.nunique()==3
    common=all_data.set_index(['PERMNO','DlyCalDt']).loc[early.set_index(['PERMNO','DlyCalDt']).index]
    np.testing.assert_allclose(common[u.FEATURES+['TARGET_10']],early[u.FEATURES+['TARGET_10']],rtol=1e-10,atol=1e-10)


def test_engineered_no_second_adjustment(engineered):
    once,audit=u.process(u.export_csv(engineered),True)
    assert audit['Format']=='Engineered' and audit['Removed incomplete rows']==0
    pd.testing.assert_frame_equal(once,engineered)
    with pytest.raises(ValueError,match='Confirm'):u.process(u.export_csv(engineered))

@pytest.mark.parametrize('problem',['missing','duplicate','ticker','date','ohlc','volume','factor','nan','identifier','short'])
def test_invalid_raw(raw,problem):
    x=raw[raw.Ticker.eq('AAPL')].copy().reset_index(drop=True)
    if problem=='missing':x=x.drop(columns='DlyCap')
    elif problem=='duplicate':x=pd.concat([x,x.iloc[:1]])
    elif problem=='ticker':x.loc[0,'Ticker']=''
    elif problem=='date':x.loc[0,'DlyCalDt']='2024-13-01'
    elif problem=='ohlc':x.loc[0,'DlyHigh']='0.01'
    elif problem=='volume':x.loc[0,'DlyVol']='-1'
    elif problem=='factor':x.loc[0,'DlyCumFacShr']='0'
    elif problem=='nan':x.loc[0,'DlyClose']='NaN'
    elif problem=='identifier':x.loc[0,'PERMNO']='1'
    else:x=x.iloc[:49]
    with pytest.raises(ValueError):u.process(x.to_csv(index=False).encode())


def test_invalid_engineered(engineered):
    for col,val in [('Split','Test'),('RET_1',999),('TARGET_10',999),('RV_20',np.inf)]:
        x=engineered.copy();x.loc[1,col]=val
        with pytest.raises(ValueError):u.process(u.export_csv(x),True)
    assert u.eligibility(engineered.iloc[:100])


def test_upload_session_ui(engineered):
    def harness():
        import streamlit as st
        import upload_ui
        from io import BytesIO
        from upload_data import REFERENCE
        upload_ui.render(BytesIO(REFERENCE.read_bytes()))
    app=AppTest.from_function(harness,default_timeout=30).run()
    app.checkbox(key='adjusted_upload').check().run()
    app.button(key='process_upload').click().run()
    assert not app.error and not app.exception
    assert app.session_state['upload_ready'][0].shape==engineered.shape
    app.radio(key='active_dataset').set_value('Uploaded dataset').run()
    assert app.session_state['upload_scope']
    other=AppTest.from_function(harness,default_timeout=30).run()
    assert 'upload_ready' not in other.session_state


def test_genuine_uploaded_backtest_and_cache_isolation(engineered,tmp_path,monkeypatch):
    data=engineered[engineered.Ticker.isin(['AAPL','MSFT','NVDA'])].copy().reset_index(drop=True)
    assert not u.eligibility(data)
    monkeypatch.setattr(engine,'CACHE',tmp_path)
    a,b={},{}
    state=engine.train('Tech','Elastic Net',dataset=data,memory=a,scope='session-a')
    assert not list(tmp_path.iterdir())
    assert engine.identity('Tech','Elastic Net',data,'session-a')!=engine.identity('Tech','Elastic Net',data,'session-b')
    assert engine.identity('Tech','Elastic Net',data,'session-a')!=engine.identity('Tech','Elastic Net')
    assert not b
    original_source=engine.source
    monkeypatch.setattr(engine,'source',lambda _: (_ for _ in ()).throw(AssertionError('Unexpected retraining')))
    assert engine.train('Tech','Elastic Net',dataset=data,memory=a,scope='session-a') is state
    monkeypatch.setattr(engine,'source',original_source)
    result=engine.backtest(state,'Elastic Net',engine.defaults('Elastic Net'),export=False)
    assert len(result['cell8_audit_df'])==2
    assert not list(tmp_path.iterdir())
    assert state['data'].Ticker.nunique()==3
    altered=engine.backtest(state,'Elastic Net',engine.defaults('Elastic Net')|{'C8_MIN_VOTES':3},export=False)
    # Vote threshold changes the recomputed qualification conditions, even if
    # a small uploaded universe has no enabled factors/trades.
    assert altered['parameters']['C8_MIN_VOTES']==3
    assert state['data'].equals(data)


def test_active_upload_integrates_with_sector_pages():
    def harness():
        import runpy
        import streamlit as st
        import upload_ui
        import upload_data as u
        data,_=u.process(u.REFERENCE.read_bytes(),True)
        data=data[data.Ticker.isin(['AAPL','MSFT','NVDA'])].reset_index(drop=True)
        st.session_state['upload_scope']='app-test-session'
        original=upload_ui.render
        try:
            upload_ui.render=lambda file: data
            runpy.run_path(str(u.REFERENCE.parents[2]/'dashboard/app.py'))
        finally: upload_ui.render=original
    app=AppTest.from_function(harness,default_timeout=60).run()
    app.radio(key='page').set_value('Tech').run()
    assert not app.exception and not app.error
    assert app.title[0].value=='Tech · Uploaded research'
    assert len([b for b in app.button if b.label=='Run Backtest'])==4
