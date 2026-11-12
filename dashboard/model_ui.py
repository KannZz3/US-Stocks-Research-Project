"""Model controls and results, independent of the historical dashboard views."""
import traceback
import pickle
import streamlit as st
import pandas as pd
import charts
from ml import engine


def reset(prefix,model):
    for name,value in engine.defaults(model).items():
        st.session_state[prefix+name] = ','.join(value) if name=='C8_OHLC_COLS' else value


def render(sector,model):
    prefix=f'{sector}:{model}:'
    try:
        path=engine.cache_path(sector,model)
    except Exception as error:
        st.error(f'Model unavailable: {error}'); return
    st.caption('Cells 1–7 cached' if path.exists() else 'Cells 1–7 not computed. Training runs only when requested.')
    st.warning('Test parameter experimentation is exploratory, not an untouched holdout. Dividends, borrowing and financing are not modeled.')
    st.button('Reset Defaults',key=prefix+'reset',on_click=reset,args=(prefix,model))
    with st.form(prefix+'form'):
        with st.expander('Cell 8 parameters'):
            params={}
            for name,value in engine.defaults(model).items():
                key=prefix+name
                if key not in st.session_state:
                    st.session_state[key]=','.join(value) if name=='C8_OHLC_COLS' else value
                if name=='C8_TRADE_SIDES':
                    params[name]=st.multiselect(name,['Long','Short'],key=key)
                elif name=='C8_OHLC_COLS':
                    params[name]=[s.strip() for s in st.text_input(name,key=key).split(',')]
                elif name=='C8_OUTPUT_DIR': params[name]=st.text_input(name,key=key,help='Relative to dashboard/ml/.cache/exports; CSVs only. No notebook PNG exports.')
                elif isinstance(value,bool): params[name]=st.checkbox(name,key=key)
                elif isinstance(value,(int,float)): params[name]=st.number_input(name,key=key,step=1 if isinstance(value,int) else .01,format='%d' if isinstance(value,int) else '%.5f')
            st.caption('Frozen C8 constants are not tuning controls. CSV exports stay in the local cache; interactive charts replace notebook PNGs.')
        run=st.form_submit_button('Run Backtest')
    result_key=prefix+'result'
    if run:
        st.session_state.pop(result_key,None)
        try:
            with st.status('Running pipeline',expanded=True) as status:
                state=engine.train(sector,model,st.write)
                result=engine.backtest(state,model,params)
                engine.save_result(engine.result_path(sector,model,params),result)
                st.session_state[result_key]=(path.name,result)
                status.update(label='Backtest completed; notebook audits passed',state='complete')
        except Exception as error:
            st.error(f'Run failed / audit not passed: {type(error).__name__}: {error}')
            with st.expander('Failure details'):
                st.code(traceback.format_exc())
            return
    if result_key not in st.session_state:
        cached=engine.result_path(sector,model,engine.defaults(model))
        if cached.exists():
            with cached.open('rb') as f: st.session_state[result_key]=(path.name,pickle.load(f))
    saved=st.session_state.get(result_key)
    if not saved or saved[0]!=path.name:
        st.info('Not computed'); return
    result=saved[1]
    st.caption('Displayed results belong to the last submitted settings, shown below. Editing controls does not change these results until Run Backtest.')
    st.json(result['parameters'],expanded=False)
    st.write('Enabled directions: '+(', '.join(result['enabled_sides']) or 'None — all cash'))
    st.write('Disabled directions: '+(', '.join(s for s in ('Long','Short') if s not in result['enabled_sides']) or 'None'))
    for period,tab in zip(('Validation','Test'),st.tabs(['Validation · 2022–2023','Test · 2024–2025 · exploratory'])):
        with tab:
            perf=result['cell8_performance_df']; row=perf[perf.Period.eq(period)&perf.Year.eq('ALL')].iloc[0]
            for col,(name,value) in zip(st.columns(5), [('Return',f'{row.Total_Return:.2%}'),('Sharpe',f'{row.Sharpe_252:.3f}' if pd.notna(row.Sharpe_252) else 'Unavailable'),('Max drawdown',f'{row.Max_Drawdown:.2%}'),('Trades',str(int(row.Trades))),('Win rate',f'{row.Trade_Win_Rate:.2%}' if pd.notna(row.Trade_Win_Rate) else 'Unavailable')]): col.metric(name,value)
            curve=result['cell8_equity_df'];curve=curve[curve.Period.eq(period)].set_index('Date')
            st.plotly_chart(charts.lines(curve[['Equity_USD','Equal_Weight_BuyHold_Equity']],'Account equity'),key=prefix+period+'equity',width='stretch')
            st.plotly_chart(charts.lines(curve.Drawdown,'Drawdown',percent=True),key=prefix+period+'drawdown',width='stretch')
            trades=result['cell8_trade_log_df'];trades=trades[trades.Period.eq(period)]
            st.dataframe(trades,width='stretch')
            if not trades.empty:
                st.dataframe(trades.groupby(['Direction','Exit_Reason']).agg(Trades=('Trade_ID','size'),PnL=('Net_PnL_USD','sum')),width='stretch')
    with st.expander('M calibration, audits and downloads'):
        st.success('All execution assertions passed for this submitted run.')
        for name,frame in result.items():
            if isinstance(frame,pd.DataFrame):
                preview=frame.head(100).copy()
                if 'Year' in preview: preview['Year']=preview['Year'].astype(str)
                st.write(name);st.dataframe(preview,width='stretch')
                st.download_button('Download '+name,frame.to_csv(index=False),name+'.csv','text/csv',key=prefix+name)
