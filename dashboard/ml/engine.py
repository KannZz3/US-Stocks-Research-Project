"""Explicit, isolated pipeline jobs with content-addressed local caches."""
import ast
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import tempfile
import types

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
MODELS = {'Elastic Net':'elastic','XGBoost':'xgb','CatBoost':'cat','LightGBM':'light'}
SECTORS = ('Tech','Consumption','Finance','Utilities','Energy')
CACHE = ROOT / '.cache'
RAW = ['PERMNO','Ticker','DlyCalDt','DlyOpen','DlyHigh','DlyLow','DlyClose','DlyVol','DlyCap','sprtrn']


def defaults(model):
    return json.loads((ROOT/'defaults.json').read_text(encoding='utf-8'))[MODELS[model]] | {'C8_OUTPUT_DIR': '.'}


def stages(model):
    key=MODELS[model]
    return ['folds','scaling',f'fit_{key}',f'screen_fit_{key}','screening','tier3','signals','evaluation']


def source(stage):
    return (ROOT/'stages'/f'{stage}.py').read_text(encoding='utf-8')


def identity(sector,model):
    if sector not in SECTORS: raise ValueError('Unknown sector')
    path=REPO/'data/data_dashboard'/f'data_preML_{sector}.csv'
    h=hashlib.sha256(path.read_bytes())
    h.update((sector+model).encode())
    for file in [Path(__file__),ROOT/'defaults.json',*(ROOT/'stages'/f'{s}.py' for s in stages(model))]:
        h.update(file.read_bytes())
    for package in ['numpy','pandas','scikit-learn','scipy',{'elastic':'scikit-learn','xgb':'xgboost-cpu','cat':'catboost','light':'lightgbm'}[MODELS[model]]]:
        h.update((package+importlib.metadata.version(package)).encode())
    return h.hexdigest()


def cache_path(sector,model):
    return CACHE / (identity(sector,model)+'.pkl')


def load_data(sector):
    data=pd.read_csv(REPO/'data/data_dashboard'/f'data_preML_{sector}.csv',parse_dates=['DlyCalDt'],float_precision='round_trip')
    data=data.sort_values(['PERMNO','DlyCalDt']).reset_index(drop=True)
    features=[c for c in data if c not in RAW+['TARGET_10','Split']]
    assert data.shape==(24453,43) and len(features)==31
    assert data.Ticker.nunique()==data.PERMNO.nunique()==9
    assert data.groupby('Ticker').size().eq(2717).all()
    assert data.groupby('Ticker').PERMNO.nunique().eq(1).all()
    assert not data.duplicated(['PERMNO','DlyCalDt']).any()
    assert np.isfinite(data[features+['TARGET_10']].to_numpy()).all()
    for split,lo,hi in [('Train',2015,2021),('Validation',2022,2023),('Test',2024,2025)]:
        assert data.loc[data.Split.eq(split),'DlyCalDt'].dt.year.between(lo,hi).all()
    assert set(data.Split)=={'Train','Validation','Test'}
    return dict(data=data,feature_cols=features,target_col='TARGET_10',target_cols=['TARGET_10'],
                tech_tickers=sorted(data.Ticker.unique()),
                train_pool=data.loc[data.Split.eq('Train')].copy(),
                val_pool=data.loc[data.Split.eq('Validation')].copy(),
                test_pool=data.loc[data.Split.eq('Test')].copy())


def hydrate(state,model):
    ns=dict(state)
    # Rebind functions to this job's globals without executing stage bodies.
    for stage in stages(model):
        tree=ast.parse(source(stage))
        tree.body=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef))]
        exec(compile(tree,stage,'exec'),ns)
    return ns


def train(sector,model,progress=lambda message: None):
    path=cache_path(sector,model)
    if path.exists():
        with path.open('rb') as f: return pickle.load(f)
    ns=load_data(sector)
    for stage in stages(model):
        progress(stage)
        exec(compile(source(stage),stage,'exec'),ns)
    state={}
    for name,value in ns.items():
        if name.startswith('__') or isinstance(value,types.ModuleType) or callable(value): continue
        try: pickle.dumps(value,protocol=5)
        except (TypeError,pickle.PicklingError): continue
        state[name]=value
    CACHE.mkdir(exist_ok=True)
    fd,temp=tempfile.mkstemp(dir=CACHE,suffix='.tmp')
    try:
        with os.fdopen(fd,'wb') as f: pickle.dump(state,f,protocol=5)
        os.replace(temp,path)
    finally:
        if os.path.exists(temp): os.unlink(temp)
    return state


def backtest(state,model,parameters):
    expected=defaults(model)
    if set(parameters)!=set(expected): raise ValueError('Cell 8 parameter keys must match defaults')
    ns=hydrate(copy.deepcopy(state),model)
    ns.update(parameters)
    exec(compile(source('execution'),'Cell 8','exec'),ns)
    result = {k:v for k,v in ns.items() if k.startswith('cell8_') and isinstance(v,pd.DataFrame)} | {
        'enabled_sides':ns['C8_ENABLED_SIDES'], 'parameters':copy.deepcopy(parameters)}

    destination=(CACHE/'exports'/str(parameters['C8_OUTPUT_DIR'])).resolve()
    base=(CACHE/'exports').resolve()
    if not destination.is_relative_to(base): raise ValueError('Output directory must stay inside the local cache export directory')
    destination.mkdir(parents=True,exist_ok=True)
    for name,frame in result.items():
        if isinstance(frame,pd.DataFrame): frame.to_csv(destination/(name+'.csv'),index=False)
    return result


def result_path(sector,model,parameters):
    key=identity(sector,model)+source('execution')+json.dumps(parameters,sort_keys=True)
    return CACHE/(hashlib.sha256(key.encode()).hexdigest()+'.result')


def save_result(path,result):
    CACHE.mkdir(exist_ok=True)
    fd,temp=tempfile.mkstemp(dir=CACHE,suffix='.tmp')
    with os.fdopen(fd,'wb') as f: pickle.dump(result,f,protocol=5)
    os.replace(temp,path)
