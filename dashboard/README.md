# Dashboard

Stock charts and team backtest results for FIN 556. Uses the original 12 stocks.
Model pages wait for the team's results.

## Run

Python 3.11+ required. On Mac, run `zsh start.command` inside this folder.
Or, from the repository root on Mac/Linux:

```sh
cd dashboard
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

Windows PowerShell:

```powershell
cd dashboard
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open http://localhost:8501 and keep the terminal open.

## Data

The ZIP has code only. Inside `fin556-project/dashboard`, the app finds the
repository's `data.csv`. Elsewhere, choose your CSV in **Data settings**.
You can also save it as `dashboard/data/stocks.csv`. Datasets stay out of Git.
The newer 45-stock format needs a later update.

## Results

Add daily CSVs to `backtests/` and click **Reload data**. Required columns:

```text
timestamp,ticker,model,equity
```

Use one row per day and `split=test` for test results. The first equity is a
baseline unless matching JSON metadata supplies `initial_capital`.
Include `equity_basis` (`gross` or `net`) and `costs` in that JSON to describe costs.
Calculation notes are in **About the Data**.

Tests: `.venv/bin/python -m pytest -q` (Windows: `.venv\Scripts\python.exe -m pytest -q`).
