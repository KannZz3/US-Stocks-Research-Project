"""Opt-in full Tech reproduction: fresh reference code vs extracted stages."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ml import engine

def verify(model):
    provenance=json.loads((engine.ROOT/'provenance.json').read_text())[engine.MODELS[model]]
    notebook=json.loads((engine.REPO/provenance['source']).read_text(encoding='utf-8'))
    cells=[''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code']
    cells=[s for s in cells if s.strip() and not s.startswith('%')][:9]
    # Cell 1 consumes the already-prepared, unchanged sector export. Reference
    # Cells 2–8 run freshly; display output is discarded, never read as state.
    reference=engine.load_data('Tech')
    with contextlib.redirect_stdout(io.StringIO()):
        for source in cells[1:8]:
            source=source.replace('from IPython.display import display','display = lambda *args, **kwargs: None')
            exec(compile(source,'reference','exec'),reference)
        source=cells[8].split('# 7. Two CSV files')[0]
        source=source.replace('from IPython.display import display, FileLink','display = lambda *args, **kwargs: None')
        with tempfile.TemporaryDirectory() as directory:
            source=source.replace('C8_OUTPUT_DIR = Path.cwd()',f'C8_OUTPUT_DIR = Path({directory!r})')
            exec(compile(source,'reference Cell 8','exec'),reference)
    state=engine.train('Tech',model,print)
    actual=engine.backtest(state,model,engine.defaults(model))
    for name in ['cell8_equity_df','cell8_trade_log_df','cell8_performance_df','cell8_m_calibration_df','cell8_audit_df','cell8_validation_signals','cell8_test_signals']:
        pd.testing.assert_frame_equal(actual[name],reference[name],check_exact=False,rtol=1e-10,atol=1e-10)
    engine.save_result(engine.result_path('Tech',model,engine.defaults(model)),actual)
    print('REPRODUCED',model,flush=True)
    return actual['cell8_performance_df'].to_dict('records')

if __name__=='__main__':
    report={}
    for model in engine.MODELS:
        print('VERIFY',model,flush=True)
        report[model]=verify(model)
        (engine.ROOT/'.cache'/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
