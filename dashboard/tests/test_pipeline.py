import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from ml import engine

@pytest.fixture(scope='module')
def trained():
    return engine.train('Tech','Elastic Net')


def test_data_contract_all_sectors():
    for sector in engine.SECTORS:
        state=engine.load_data(sector)
        assert len(state['feature_cols'])==31
        assert 'TARGET_10' not in state['feature_cols']
        assert state['train_pool'].DlyCalDt.max() < state['val_pool'].DlyCalDt.min()


def test_c8_parameters_and_model_defaults():
    for model in engine.MODELS:
        defaults=engine.defaults(model)
        assert len(defaults)==17
        assert defaults['C8_ZERO_COSTS'] is True
        assert defaults['C8_MIN_VOTES']==5
        assert defaults['C8_OUTPUT_DIR']=='.'


def test_cached_training_never_executes_stages(trained,monkeypatch):
    def fail(*args): raise AssertionError('Unexpected retraining')
    monkeypatch.setattr(engine,'source',fail)
    again=engine.train('Tech','Elastic Net')
    assert again['factor_six_validation']['selected']==trained['factor_six_validation']['selected']


def test_real_c8_recalculation_and_isolation(trained,tmp_path,monkeypatch):
    monkeypatch.setattr(engine,'CACHE',tmp_path)
    params=engine.defaults('Elastic Net')
    baseline=engine.backtest(trained,'Elastic Net',params)
    revised=engine.backtest(trained,'Elastic Net',params|{'C8_MIN_VOTES':3})
    assert not baseline['cell8_equity_df'].equals(revised['cell8_equity_df'])
    mutated=copy.deepcopy(trained)
    mask=mutated['data'].Split.ne('Train')
    mutated['data'].loc[mask,'TARGET_10']=999.0
    result=engine.backtest(mutated,'Elastic Net',params)
    for name in ['cell8_equity_df','cell8_m_calibration_df','cell8_trade_log_df']:
        pd.testing.assert_frame_equal(baseline[name],result[name])
    trades=result['cell8_trade_log_df']
    assert (trades.Entry_Date>trades.Signal_Date).all()
    initial=result['cell8_equity_df'].query("Row_Type == 'INITIAL'")
    assert len(initial)==2 and initial.Equity_USD.eq(params['C8_START_CAPITAL']).all()
    assert trained['data'].loc[mask,'TARGET_10'].ne(999).all()


def test_disabled_sides_and_failed_audit(trained,tmp_path,monkeypatch):
    monkeypatch.setattr(engine,'CACHE',tmp_path)
    params=engine.defaults('Elastic Net')
    flat=engine.backtest(trained,'Elastic Net',params|{'C8_TRADE_SIDES':[]})
    assert flat['cell8_trade_log_df'].empty
    assert flat['cell8_equity_df'].Equity_USD.eq(100000).all()
    with pytest.raises(AssertionError):
        engine.backtest(trained,'Elastic Net',params|{'C8_MIN_VOTES':2})


def test_future_prices_do_not_change_earlier_execution(trained,tmp_path,monkeypatch):
    monkeypatch.setattr(engine,'CACHE',tmp_path)
    params=engine.defaults('Elastic Net')
    baseline=engine.backtest(trained,'Elastic Net',params)
    altered=copy.deepcopy(trained)
    future=altered['data'].DlyCalDt.ge('2025-01-01')
    altered['data'].loc[future,['DlyOpen','DlyHigh','DlyLow','DlyClose']]*=1.01
    actual=engine.backtest(altered,'Elastic Net',params)
    for name,date in [('cell8_equity_df','Date'),('cell8_trade_log_df','Exit_Date')]:
        a=baseline[name];b=actual[name]
        pd.testing.assert_frame_equal(a[a[date]<'2025-01-01'],b[b[date]<'2025-01-01'])


def test_interactive_submit_and_reset():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(Path(engine.ROOT).parent/'app.py'),default_timeout=120)
    app.run().radio(key='page').set_value('Tech').run()
    app.number_input(key='Tech:Elastic Net:C8_MIN_VOTES').set_value(3)
    button=next(b for b in app.button if b.label=='Run Backtest' and 'Elastic Net' in b.key)
    button.click().run()
    assert not app.exception and not app.error
    assert app.session_state['Tech:Elastic Net:result'][1]['parameters']['C8_MIN_VOTES']==3
    next(b for b in app.button if b.label=='Reset Defaults' and 'Elastic Net' in b.key).click().run()
    assert app.number_input(key='Tech:Elastic Net:C8_MIN_VOTES').value==5


def test_train_only_scaling_and_target_maturity(trained):
    ns=engine.hydrate(copy.deepcopy(trained),'Elastic Net')
    original=ns['scaled_validation_fold']
    raw=ns['data']
    train_idx=original['train_fit_idx']
    np.testing.assert_allclose(original['scaler'].mean_,raw.loc[train_idx,ns['feature_cols']].mean())
    assert raw.loc[train_idx,'Split'].eq('Train').all()
    realized_end=raw.groupby('PERMNO').DlyCalDt.shift(-10)
    assert (realized_end.loc[train_idx]<=pd.Timestamp('2021-12-31')).all()
    ns['data'].loc[raw.Split.ne('Train'),ns['feature_cols']]*=10
    altered=ns['standardize_fold'](ns['validation_fold'])
    pd.testing.assert_frame_equal(original['X_train'],altered['X_train'])
    np.testing.assert_array_equal(original['scaler'].mean_,altered['scaler'].mean_)
