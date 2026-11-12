# Phase 2 verification

Verified on 2026-11-12. The full suite passed: **47 tests**. The final UI-only adjustment also passed the submit/reset test. One unrelated anyio/pytest deprecation warning remains.

## Tech baseline reproduction

Fresh reference Cells 2–8 and the extracted pipeline ran independently under identical settings, consuming the existing Tech sector export for Cell 1. Signals, trades, equity, performance, M calibration and audits matched with rtol/atol 1e-10 for all four models. Notebook display outputs were not used. These are computed historical results, not externally audited investment performance.

| Model | Validation return | Test return | Validation trades | Test trades |
| --- | ---: | ---: | ---: | ---: |
| Elastic Net | 26.957465% | 46.438069% | 79 | 81 |
| XGBoost | 10.018646% | 59.535005% | 98 | 90 |
| CatBoost | 32.305354% | 44.867852% | 83 | 85 |
| LightGBM | 81.379166% | 59.464859% | 101 | 114 |

## Additional checks

- Elastic Net completed genuine runs and execution audits for Consumption, Finance, Utilities and Energy as well as Tech.
- All five sector pages and all four model tabs render; submit/reset behavior was exercised with real Cell 8 execution.
- Changing votes changes actual NAV without upstream retraining. Cache-hit tests fail if a training stage executes.
- Changing held-out targets leaves M, trades and equity unchanged; changing future prices leaves earlier executions unchanged.
- Scaling matches Train-only means; target maturity is checked across the ten-observation purge boundary.
- Empty direction selection yields an all-cash account; invalid vote settings fail rather than display a partial result.
- Browser check confirmed cached XGBoost results, separate Validation/Test tabs, metrics, disabled-direction labels and interactive charts.

## Environment

- numpy: 2.4.4
- pandas: 2.3.3
- scipy: 1.17.1
- scikit-learn: 1.8.0
- xgboost-cpu: 3.4.1
- catboost: 1.2.10
- lightgbm: 4.7.0
- streamlit: 1.67.0

## Limits

The 12 non-Tech tree-model/sector combinations have not received full historical execution checks in this session. Their pages and data loading were checked, and they use the same generalized stages; compute them explicitly with Run Backtest.

Test parameter experimentation is exploratory. Original upstream engineering, universe selection bias and external calendar completeness are not independently certified. Prices exclude dividends; borrow/financing costs are not modeled. Source coverage ends 2025-12-16.

C8_OUTPUT_DIR is scoped to local cache exports. Interactive charts replace notebook PNG output. The documented disabled-direction matrix correction is the only execution-logic change; original notebooks and datasets remain unchanged.

Repeat: `python -m pytest dashboard/tests -q` and `python dashboard/tools/verify_baselines.py`. Genuine caches and run reports remain local and are ignored by Git.
