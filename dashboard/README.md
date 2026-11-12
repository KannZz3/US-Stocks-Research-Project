# Streamlit dashboard — Phase 1

Run from the repository root (Python 3.11+):

```sh
python -m pip install -r dashboard/requirements.txt
python -m streamlit run dashboard/app.py
```

Home compares Tech, Consumption, Finance, Utilities and Energy. The fixed,
scrollable desktop sidebar occupies 1/7 of the viewport; on narrow screens it
uses Streamlit's collapsible navigation. Five buttons below the Home chart open
sector pages. Each sector has Elastic Net (default), XGBoost, CatBoost and
LightGBM tabs showing **Not computed**. No models are executed or results invented.
Individual stock prices/candlesticks, trailing moving averages, volume, returns,
volatility, comparisons, correlations and summary downloads remain available.

## Data and portfolio definition

Only the five explicit `data/data_dashboard/data_preML_<Sector>.csv` files load.
The master `data_preML.csv` is excluded. Each sector must have nine stocks and
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

The sidebar accepts UTF-8 CSVs with `Ticker`, `DlyCalDt`, `DlyClose`. Validation
checks nonempty data, dates/tickers, positive finite closes and duplicate stock/date
keys. A preview shows the first 50 rows and row/stock counts. Additional columns
are retained. Uploads do not replace sector data, write files, infer split adjustment,
or run preprocessing. Full preprocessing belongs to Phase 3.

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
integration utility, and no backtesting or ML pipeline was added.
