# Streamlit dashboard — Phases 1, 2 and 3

Run from the repository root (Python 3.11+):

```sh
python -m pip install -r dashboard/requirements.txt
python -m streamlit run dashboard/app.py
```

Home compares Tech, Consumption, Finance, Utilities and Energy. The fixed,
scrollable desktop sidebar occupies 1/7 of the viewport; on narrow screens it
uses Streamlit's collapsible navigation. Five buttons below the Home chart open
sector pages. Each sector has Elastic Net (default), XGBoost, CatBoost and
LightGBM tabs. Uncomputed models show **Not computed**; Phase 2 adds explicit execution controls below.
Individual stock prices/candlesticks, trailing moving averages, volume, returns,
volatility, comparisons, correlations and summary downloads remain available.

## Data and portfolio definition

Only the five explicit `data/data_dashboard/data_preML_<Sector>.csv` files load.
The master `data_preML.csv` is excluded. Each built-in sector must have nine stocks and
complete matching date coverage. The supplied files each contain 24,453 rows
covering 2,717 dates, 2015-03-02 through 2025-12-16.

`DlyClose` is already split-adjusted by the repository's preprocessing notebook
(`data/data_processing/data_processing.ipynb`, OHLC divided by `DlyCumFacShr`).
The dashboard does not apply another split adjustment. `RET_1` is a price return,
not dividend-inclusive `DlyRet`. Neither portfolio growth nor RET_1 charts include
dividends. Legacy DlyRet loading remains supported by the existing loader, but
is not used for the sector portfolios.

At each sector's first observed date t0, invest 100/9 in each constituent:

`V(t) = (100/9) * sum(P_i(t) / P_i(t0))`

Hold those adjusted-share units fixed. This is initial equal weighting, not daily
rebalancing or compounding the mean daily return. Date selection only crops the
chart; it does not change inception or holdings. No future prices, targets, model
features or future-dependent stock selection enter the calculation. Trailing
indicators are calculated before display filters. Invalid/missing portfolio
prices fail explicitly; there is no backfill, forward fill or zero-return fill.

The supplied fixed universe can have selection/survivorship bias. Its preprocessing
and external trading-calendar completeness are not independently certified. This
historical visualization is not an out-of-sample strategy backtest.

## Uploads

Raw and engineered uploads support validation, processing, CSV export and session-only research. See Phase 3 below for the financial schema and eligibility requirements.

## Verification

From the repository root:

```sh
python -m pytest dashboard/tests -q
```

Alternatively, run from `dashboard/` with `python -m pytest tests -q`.
Tests cover the real five-sector files, fixed-share arithmetic, absence of future
price dependence, missing data, all navigation destinations, model-tab contents,
date controls, candles, empty comparisons, upload validation and preview rendering.
Streamlit AppTest cannot set a file uploader; the preview renderer is tested with
an in-memory uploaded file. Browser QA also checks real CSV upload, desktop sidebar proportions and narrow-screen layout. Existing result-reader tests are retained; `results.py` remains a future
integration utility, and remain covered alongside Phase 2 execution tests.

## Phase 2 — explicit model execution and backtesting

All five sector pages retain the Phase 1 layout and stock exploration. Each model
tab now offers **Run Backtest**, **Reset Defaults**, and all 17 user-adjustable
Cell 8 settings. Elastic Net is the first/default tab. The four notebook defaults
currently agree, but are extracted and stored separately per model.

Install the updated requirements before launch. A Run action fits Cells 1–7 only
when no matching local cache exists, then recomputes Cell 8. Refreshing or changing
widgets never starts training. Training can take several minutes, especially the
nested tree-model LOFO screening. Do not interrupt a run merely because a tab has
not yet rendered its results.

### Architecture and reference fidelity

`ml/stages/` holds extracted computational Python: shared fold construction,
Train-only scaling, screening, Tier-3 selection, ECDF voting, conflict filtering,
evaluation and OHLC execution; model-specific fitting modules preserve their
original estimator settings and numeric precision. `ml/engine.py` supplies sector
data and isolated job namespaces. Runtime never executes notebooks or consumes
saved notebook outputs. Notebook SHA-256 references are in `ml/provenance.json`.

Cell 1 uses the existing, already-prepared nine-stock sector CSV instead of
rebuilding or overwriting the master/Tech exports. All five bundled files satisfy the
original dimensions and chronological split contract; uploaded cardinalities are generalized in Phase 3. `tech_tickers` remains an
internal compatibility name for the selected sector's nine tickers. The original
2015–2021 Train, 2022–2023 Validation and 2024–2025 Test boundaries are retained.
Purging, train-only nested LOFO, factor selection and historical ECDF are unchanged.
Validation outcomes are evaluated only after selection, never used to select
factors or calibrate M. Test labels do not generate signals or calibrate M.

One documented execution correction: a user-disabled direction uses zero priority
rather than multiplying an empty score matrix by the frozen nonempty weights.
This fixes disabling previously enabled directions and leaves default execution
unchanged. All notebook execution assertions remain active. Do not use Python -O.

The repeatable extraction tool rejects differences between model notebooks in
shared computational stages:

```sh
python dashboard/tools/extract_pipeline.py
```

### Caches and reproducibility

`dashboard/ml/.cache/` is ignored by Git. Upstream keys include sector, model,
full CSV bytes, computational source/configuration and installed numerical/model
library versions. Cell 8 parameter values are excluded from upstream keys.
Execution results have separate keys including execution source and parameters.
Only locally generated cache files are loaded; never copy untrusted pickle files
into this directory. Writes use temporary files followed by atomic replacement.
Cached functions are rebound into each job's own namespace, and each backtest uses
a deep copy so another submission cannot modify frozen upstream state.

The UI can reuse a genuine cached default result after restart. It labels the
last submitted settings; unsent edits do not silently relabel an old result.
Run Backtest always recomputes signals, M calibration and the full execution ledger,
even if an execution-result cache already exists. Failed runs display the error
and do not publish a partial result.

`C8_OUTPUT_DIR` is relative to the local cache's `exports/` directory (default `.`),
not an arbitrary system path. It exports CSV tables; interactive Plotly charts
replace the notebook PNG side effects. CSV downloads are also available directly.
Frozen/derived C8 values (schedule, cutoffs, factor percentile, enabled directions,
weights and cost rate) are not editable controls. Defaults restore all adjustable
values without rerunning the model or backtest.

### Results and interpretation

Validation and Test have independent initial cash accounts. Tabs show return,
Sharpe, maximum drawdown, trades, win rate, interactive equity/drawdown curves,
trade logs, exit reason/PnL diagnostics, M calibration, signal tables and audits.
Disabled sides and unavailable M calibration are explicit; they are never replaced
with invented trades. Zero transaction costs remain the default. Signal timing,
next-open execution, gap filters, position limits, short collateral, conservative
stop-first ambiguous OHLC fills and terminal liquidation follow Cell 8.

Test parameter exploration is labeled **exploratory**. No dividend, borrow or
financing model is added. M is a Train winner-only signed TARGET_10 mean, not an
estimated next-open profit or volatility. The source ends on 2025-12-16; the Test
account does not represent a complete calendar year. Supplied universe selection
bias and original upstream engineering limitations remain.

### Verification commands

```sh
python -m pytest dashboard/tests -q
python dashboard/tools/verify_baselines.py
```

The second command is intentionally opt-in and computationally expensive. It runs
fresh reference Cells 2–8 using the unchanged sector export for Cell 1, then
compares independently executed extracted code with tight numeric tolerances.
It never treats saved display outputs as baselines. The comparison covers signals,
trade logs, equity, performance, M and audits for Tech under identical settings.
The local verification report is `.cache/verification.json`. This establishes
code reproduction in the installed environment, not an independently certified
investment performance claim. Full historical runs for every model/sector
combination are not part of the fast test suite.

## Phase 3 — raw and engineered uploads

The sidebar's **Validate / Process CSV** action accepts:

- Raw, unadjusted daily records with `PERMNO`, `Ticker`, `DlyCalDt`, `DlyOpen`,
  `DlyHigh`, `DlyLow`, `DlyClose`, `DlyVol`, `DlyCap`, `sprtrn`, and
  `DlyCumFacShr`. Capitalization is in thousands of USD. Cumulative share factors
  must correspond to the supplied unadjusted prices/volume. Never supply already
  adjusted raw prices with non-unit factors. Dropped source fields such as bid,
  ask, DlyRet, DlyRetx and DlyCumFacPr are not required by the retained formulas.
- Complete engineered sector CSVs with all 43 reference columns. Confirm that
  prices are already split-adjusted; this cannot be established from the schema
  alone. Columns are reordered and types/dates/tickers normalized, but features
  and adjustments are not applied again. Available adjacent RET_1 and ten-row
  TARGET_10 values are checked against prices; first-day returns and final-ten-row
  labels cannot be independently reconstructed from a truncated engineered file.

Validation checks unique headers, required fields, finite numbers, one-to-one
Ticker/PERMNO identity, daily dates, duplicate keys, positive/integer volume and
identifiers, OHLC ordering, capitalization, market-return agreement and adjustment
factors. Raw source missing values are rejected as in the notebook. Output shows
row counts, removed incomplete rows, date range, stock count, preview, model
eligibility reasons and a downloadable CSV in the exact 43-column order.

`features.py` contains the retained per-stock formulas from engineering Cells
2–6 and 9; `upload_data.py` implements the Decimal/ROUND_HALF_UP split convention,
TARGET_10 and chronological cleaning/splits. Price rounding is eight decimal
places; adjusted volume must remain integral. No forward fill, interpolation,
winsorization or invented financial fields are used. Market/peer features and
TARGET_5 are omitted because Phase 2 consumes only the retained 31 features and
TARGET_10. Cleaning checks these retained columns, so degenerate uploads are not
required to compute discarded peer-group statistics. This reproduces every row
of the supplied reference output. Undefined retained features are dropped,
including the 39-observation warm-up and final ten target observations. Every
stock needs at least 50 raw observations and at least one complete output row.

Split boundaries remain 2021-12-31 / 2023-12-31 / 2025-12-31. Later dates are rejected
rather than silently classified as holdout. No permanent boundary purge occurs
in the exported CSV: the existing walk-forward engine dynamically purges ten
observations per stock before fitting. Historical features use past/current
observations only; forward values are isolated to TARGET_10.

Choose **Uploaded dataset** after validation to replace the research input on
sector pages. Home continues to show the five bundled sectors and explicitly
indicates when an upload is active. Select **Built-in sectors** to restore bundled
research. A valid export may be ineligible for modeling: at least three stocks,
matching per-date stock coverage, and at least 40 usable observations per stock
in every year 2015–2025 are required by the unchanged walk-forward schedule.
Further notebook assertions can still reject degenerate model inputs. Stock and
row-count assertions are generalized; estimator parameters and calculations are
unchanged. All four model tabs and all 17 Cell 8 controls are retained.

Uploads, processed frames, training caches and execution results live in the
individual Streamlit session. Uploaded cache keys incorporate content, sector,
model, numerical-library versions, source/configuration and a session identifier.
Uploaded runs never read/write built-in persistent caches, never export server-side
CSV files, and never mutate bundled data. CSV downloads use the actual in-memory
result. Reloading into a new session requires uploading again. C8_OUTPUT_DIR is
retained for compatibility but only built-in runs use server-side exports.

Reference extraction can be regenerated with:

```sh
python dashboard/tools/extract_features.py
```

Notebook hashes are recorded in `processing_provenance.json`. The original
notebooks and datasets remain unchanged. No new dependencies are required for
Phase 3. Full non-Tech tree-model baselines (12 combinations) remain unverified;
they are executable on demand and are not labeled verified performance.
