# US Stocks Quantitative Research

**[Open Dashboard](#quick-start)** — local launch; no live deployment verified.

## Overview

A 45-stock research workflow for evaluating machine-learning factor signals against buy-and-hold price portfolios. It combines reproducible preprocessing, chronological model evaluation, OHLC backtesting and an interactive Streamlit dashboard.

## Data

Five sectors contain nine stocks each:

| Sector | Tickers |
| --- | --- |
| Tech | AAPL, ADBE, AMAT, AMD, AVGO, CRM, MSFT, NVDA, ORCL |
| Consumption | AMZN, COST, HD, KO, MCD, NKE, SBUX, TJX, WMT |
| Finance | AXP, BAC, BLK, CME, JPM, MS, PGR, SCHW, V |
| Utilities | AEP, AWK, D, DUK, EIX, NEE, SO, SRE, XEL |
| Energy | COP, EOG, LNG, OXY, PSX, SLB, VLO, WMB, XOM |

The five `data_preML_<Sector>.csv` files each contain 24,453 observations from **2015-03-02 through 2025-12-16**. Their 43 columns comprise ten source fields, 31 technical features, `TARGET_10` and `Split`. The dashboard excludes the master `data_preML.csv`.

Preprocessing split-adjusts OHLC and volume using `DlyCumFacShr`. Retained features describe returns, trend, intraday/overnight behavior, volatility, liquidity and size. `TARGET_10` is the ten-observation forward close-to-close price return. Incomplete feature/target rows are dropped without filling. Returns exclude dividends; `RET_1` is not a total return.

Training uses 2015–2021, Validation 2022–2023 and Test 2024–2025. Exported boundary rows are retained; ten-observation target purging occurs within model folds.

## Methodology

Elastic Net, XGBoost, CatBoost and LightGBM share the notebook-derived workflow: chronological walk-forward evaluation, Train-only scaling and nested leave-one-feature-out screening, directional ranking, Tier-3 selection and historical empirical-CDF confluence voting. Daily Long/Short conflicts block new entries. Execution factors, weights and signal calibration remain frozen through 2021.

## Backtesting

Cell 8 uses after-close signals and next-session OHLC execution. Train-only M calibration sets take-profit and stop-loss distances. Controls cover capital, position sizing/count, votes, entry gaps, holding periods, calibration thresholds, costs, collateral and permitted directions. The UI exposes all 17 adjustable settings, including OHLC mapping and export location.

Validation and Test use independent accounts. Defaults include $100,000 capital, three positions, 33% allocation per position, five votes, a 2.5% entry-gap limit, 15 holding bars and zero transaction costs. Ambiguous same-bar TP/SL touches use stop-first execution. Results include equity, drawdown, return, Sharpe, trade count, win rate, logs, exit diagnostics, calibration and audits.

## Dashboard

Home compares five initially equal-weight, fixed-holding portfolios normalized to 100. Sector pages retain stock charts, volume, moving averages and risk comparisons alongside the four model tabs.

**Run Backtest** trains only when the dataset/model/configuration cache is missing, then recomputes Cell 8. Parameter edits do not retrain upstream stages. Raw or engineered CSV uploads support validation, exact-schema export and session-only research; uploaded data never overwrite bundled files or share training state across sessions. See [dashboard instructions](dashboard/README.md) for input requirements.

## Structure

```text
data/
  data_processing/   Raw/processed data and preprocessing notebooks
  data_dashboard/    Master and five sector exports
pipeline/            Four authoritative Tech model notebooks
dashboard/
  app.py             Streamlit entry point
  upload_data.py     Validation, adjustment and export
  features.py        Retained feature formulas
  ml/                Extracted stages, configuration and caching
  tests/             Data, execution and UI tests
  tools/             Extraction and baseline verification
```

## Quick start

Python 3.12 is the verified environment. From the repository root:

```sh
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` on macOS/Linux, then:

```sh
python -m pip install -r dashboard/requirements.txt
python -m streamlit run dashboard/app.py
```

Open [localhost:8501](http://localhost:8501). Verify with:

```sh
python -m pytest dashboard/tests -q
```

## Verification and limitations

The suite passes **64 tests**. [Recorded verification](dashboard/PHASE3_VERIFICATION.md) covers reference-data numerical consistency, uploaded execution, cache isolation and leakage checks. [Phase 2 verification](dashboard/VERIFICATION.md) reproduced all four Tech baselines; Elastic Net also ran across all five sectors. The **12 non-Tech tree-model combinations remain unverified historical runs**, available on demand.

Test parameter experimentation is exploratory. Universe selection bias and external trading-calendar completeness are not independently certified; dividends, borrowing and financing are not modeled. The source ends before year-end 2025. Uploaded adjustment provenance requires correct source factors or explicit confirmation; the fixed model schedule requires matching stock coverage and sufficient history. Failed statistical audits remain visible rather than producing substitute results.
