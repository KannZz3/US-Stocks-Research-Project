# FIN 556 Stock Data Preparation

A reproducible Python workflow for preparing the supplied stock dataset for return,
risk, and portfolio analysis. The source file, `data.csv`, remains unchanged. All
project code, documentation, output labels, and commit messages use English.

## Run in VS Code

The project environment is `.venv`. Open `fin556-project.code-workspace` in VS Code.
The workspace includes Python interpreter selection, unit-test discovery, a build
task, and a debug configuration.

- Press **Ctrl+Shift+B** to regenerate the prepared datasets.
- Select **Prepare stock data** in Run and Debug, then press **F5** to debug.
- Run **Test data preparation** from **Terminal > Run Task**, or use the Testing panel.
- If VS Code has a previously selected interpreter, run **Python: Select Interpreter**
  and choose `.venv/Scripts/python.exe` on Windows.

On Windows, new project terminals put `.venv/Scripts` first on PATH and set
VIRTUAL_ENV directly. Automatic activation scripts are disabled, so no PowerShell
execution-policy change is required. The workspace also clears inherited PSModulePath
for new terminals, allowing Windows PowerShell to discover its own compatible modules.

If an old terminal shows `Unable to load PSReadLine`, create a new terminal after
loading this workspace. Existing terminals retain their original environment.
The interpreter must point to `.venv/Scripts/python.exe`, not the `.venv` directory.

The Python and Python Debugger extensions are recommended by the workspace.

### Terminal commands

From the project directory in PowerShell:

```powershell
.\.venv\Scripts\python.exe prepare_data.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

For a fresh checkout, install Python 3.11 or newer and create the local environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe prepare_data.py
```

On macOS or Linux, use `.venv/bin/python` instead of `.venv/Scripts/python.exe` and
select that executable with **Python: Select Interpreter**.
The script also accepts custom input and output paths:

```powershell
.\.venv\Scripts\python.exe prepare_data.py --input data.csv --output-dir data/processed
```

## Prepared files

The generated CSVs use UTF-8, English snake_case field names, ISO dates, and decimal
returns. They are committed so the prepared data can be inspected immediately.
Regenerate them after changing the source or preparation logic.

| File in `data/processed/` | Observation unit | Current size | Purpose |
| --- | --- | --- | --- |
| `daily.csv` | Security and observed date | 30,432 rows, 25 columns | Validated daily panel; all source fields retained and renamed |
| `returns_daily.csv` | Observed date | 2,536 rows, 13 columns | Date plus one total-return column per stock |
| `monthly.csv` | Security and observed month | 1,452 rows, 11 columns | Monthly stock returns, volume, and last observed prices/caps |
| `returns_monthly.csv` | Observed month | 121 rows, 13 columns | Month plus one total-return column per stock |
| `market_daily.csv` | Observed date | 2,536 rows, 2 columns | One S&P 500 price return per date |
| `market_monthly.csv` | Observed month | 121 rows, 5 columns | Compounded market price returns and coverage fields |
| `quality_report.json` | Source dataset | One report | Source hash, coverage, identifiers, missing values, and output sizes |

The supplied sample spans **2015-12-01 through 2025-12-31**, with 12 securities:
ADBE, AMD, AMZN, AAPL, COST, GOOGL, MCD, MSFT, NKE, NVDA, SBUX, and WMT.
Matrix columns are sorted alphabetically. Daily and monthly stock panels are sorted
by PERMNO and date/month. The sample contains 121 observed months; no research window
is removed automatically.

## Transformation rules

1. Read the CSV as strings before converting numeric fields. In particular,
   `DlyRetMissFlg = "NA"` means that a return is available; it is not a missing value.
2. Keep PERMNO, PERMCO, and CUSIP as identifiers. Preserve leading zeros in CUSIP.
   Use `(permno, date)` as the daily key and verify that ticker labels map one-to-one
   to PERMNO throughout this sample before creating ticker-based matrices.
3. Validate dates, numeric values, unique keys, return bounds, and matching stock
   coverage. Stop with a clear error for missing required values, ambiguous tickers,
   duplicate keys, or conflicting same-date market returns. Do not delete records,
   winsorize returns, forward-fill observations, or insert zero-return dates.
4. Preserve optional distribution blanks in `DisDivAmt` and `DisFacPr`. Keep flags,
   dividends, and price adjustment factors for inspection.
5. Use supplied `DlyRet` and `DlyRetx`; do not reconstruct returns from raw prices.
   Splits can make unadjusted prices jump. `DlyFacPrc` is a daily adjustment factor,
   not a cumulative price-adjustment series.
6. Deduplicate the benchmark by date after checking all repeated values agree.
   Compound its daily observations only once, even though each source market return
   appears alongside every stock.
7. Compound monthly simple returns as `product(1 + daily_return) - 1`.
   For example, +10% followed by -10% produces -1%, not 0%.
8. Sum volume within a month. Take price and market cap from the last observed date.
   Store `first_trading_date`, `last_trading_date`, and `observations` so coverage is
   visible. Month labels use `YYYY-MM`; they do not invent a calendar month-end trade.

Returns stay in decimal units: `0.016007` means `1.6007%`. Market capitalization
stays in **thousands of US dollars**. Do not rescale a field just for display.

### Interpretation limits

- Stock matrices contain **total returns**, while `sprtrn` is an S&P 500 **price-return**
  benchmark excluding dividends. These are different dividend conventions; use a
  total-return benchmark for like-for-like total-return comparisons.
- Shared dates are checked against the dates present in the source, not an external
  exchange calendar. A date missing for all stocks cannot be detected by this check.
  Monthly values summarize observed source returns; full calendar coverage is not
  independently certified. A shorter input can contain partial boundary months.
- D2/D3/D4 duration flags can reflect elapsed calendar days across weekends and
  holidays. They are retained, not treated automatically as missing trading days.
- No risk-free series is supplied. The pipeline does not label market-adjusted
  returns as excess returns or estimate CAPM without the necessary inputs.
- Field definitions follow CRSP documentation because the column names match that
  format. The file itself does not independently establish its download provenance.

## Validation and reproducibility

Run the unit tests before committing preparation changes. Tests cover compounding,
month boundaries, market deduplication, split-day return preservation, last-date
aggregation, ticker mappings, missing observations, invalid inputs, and output order.

`quality_report.json` records the SHA-256 of the original source. Outputs omit run
timestamps so identical inputs and code produce identical files. Environment folders
and caches are excluded from Git. The source and processed data are not edited by
the unit tests; synthetic fixtures are written only to temporary directories.

## Field reference

See [the data dictionary](docs/data_dictionary.md) for all source-to-output mappings,
monthly field definitions, and units.

Definitions were checked against the [CRSP Stock Database Format 2.0 Guide](https://www.crsp.org/wp-content/uploads/guides/CRSP_US_Stock_%26_Indexes_Database_Guide_Flat_File_Format_2.0.pdf)
and the [CRSP Historical Indexes Guide](https://www.crsp.org/crsp_pdf/crsp-historical-indexes-guide/).
