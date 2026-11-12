# Dashboard reference

For setup and project methodology, see the [root README](../README.md#quick-start).

## Research controls

Home always compares the five bundled sectors. Choose a sector for stock exploration and the Elastic Net, XGBoost, CatBoost or LightGBM tabs. **Run Backtest** fits missing upstream stages and recomputes signals, M calibration and execution. **Reset Defaults** restores controls without executing a run. Displayed results retain their submitted settings until another run succeeds.

Built-in training caches are keyed by sector, model, dataset content, source/configuration and library versions. Cell 8 settings do not enter the upstream cache key. Results have separate keys. Local caches reside in `ml/.cache/` and are ignored by Git; never load untrusted pickle files there. Initial nested screening can take several minutes.

Validation and Test results are separate. Tabs expose performance, equity/drawdown charts, trades, exit diagnostics and downloadable calibration/audit tables. Disabled directions and unavailable results are explicit. Notebook assertions must remain enabled: do not run Python with `-O`.

## CSV uploads

Upload a UTF-8 CSV and select **Validate / Process CSV**.

Raw inputs require:

```text
PERMNO,Ticker,DlyCalDt,DlyOpen,DlyHigh,DlyLow,DlyClose,DlyVol,DlyCap,sprtrn,DlyCumFacShr
```

OHLCV must be unadjusted; cumulative share factors must match those observations. `DlyCap` is in thousands of USD. Prices are divided by the factor and rounded to eight decimals using ROUND_HALF_UP; volume is multiplied by it and must remain integral. Every stock needs at least 50 raw observations and one complete output row.

Engineered inputs require all columns in [the reference schema](../data/data_dashboard/data_preML_Tech.csv). Confirm that prices are already split-adjusted. The loader normalizes column order, types, dates and ticker labels without repeating adjustment or engineering. Reconstructable RET_1 and TARGET_10 values are checked; the first return and final ten labels cannot be reconstructed from a truncated file.

Validation rejects missing/nonfinite financial values, inconsistent identifiers, duplicates, invalid OHLCV, conflicting market returns and invalid factors. Generated incomplete feature/target rows are dropped without filling. The export contains exactly 43 columns. Discarded market/peer features and TARGET_5 are not computed; cleaning uses the retained 31 features and TARGET_10.

Choose **Uploaded dataset** to use the validated frame on sector research pages; choose **Built-in sectors** to restore bundled inputs. Model eligibility requires at least three stocks, matching observed dates and at least 40 usable observations per stock in every year 2015–2025. Further statistical audits may reject degenerate inputs. Later dates require a schedule revision.

Uploaded data, training states and results remain in session memory and require re-upload in a new session. They never use built-in persistent caches or server-side CSV exports. Downloads contain actual in-memory data. `C8_OUTPUT_DIR` applies only to built-in exports and is restricted to `ml/.cache/exports/`; interactive charts replace notebook PNG output.

## Reproducibility

Original notebooks are references, not runtime state. `ml/provenance.json` and `processing_provenance.json` record their hashes. From the repository root:

```sh
python -m pytest dashboard/tests -q
python dashboard/tools/verify_baselines.py
```

The second command is expensive: it independently executes reference and extracted Tech code, comparing signals, trades, equity, metrics, M and audits. Saved notebook display outputs are never executable inputs.

Maintainers can regenerate extracted modules with `dashboard/tools/extract_pipeline.py` and `dashboard/tools/extract_features.py`. The engine generalizes dataset-cardinality assertions without changing estimator settings. One documented Cell 8 correction assigns zero priority to a user-disabled direction instead of multiplying incompatible empty matrices.

See [model verification](VERIFICATION.md) and [upload/integration verification](PHASE3_VERIFICATION.md) for completed checks and unverified combinations.
