# Phase 3 verification

Verified 2026-11-12 in the existing Phase 2 environment; no new dependencies.

- Full suite: **64 passed**, one unrelated anyio/pytest deprecation warning.
  After final display/confirmation fixes, three targeted UI tests passed; the
  regenerated feature module also passed the full-reference and invalid-engineered checks.
- Full raw reference: 124,470 rows / 45 stocks became 122,265 rows with the exact
  43-column reference order. The expected 2,205 incomplete observations were
  removed. Every retained feature and TARGET_10 matched the five bundled sector
  exports; largest observed absolute difference was approximately 1.6e-13.
  Tests use rtol=atol=1e-10 to allow floating-point rolling/serialization noise.
- Engineered reference upload round-trips without adjustment or feature changes.
  The first RET_1 and final ten TARGET_10 values cannot be independently verified
  from a truncated engineered file; reconstructable values are checked.
- Genuine three-stock uploaded Elastic Net training and Cell 8 execution passed.
  Cache-hit tests prevent unnecessary upstream execution, and uploaded runs write
  no disk-cache or export files. Independent session scopes have distinct keys;
  a fresh Streamlit session cannot see another session's processed upload.
- Upload processing has no feature look-ahead: processing a historical prefix
  matches the corresponding full-history feature/target observations for which
  labels have matured. Existing Train-only scaling, purging, future-price,
  held-out-target and Cell 8 parameter-sensitivity tests continue to pass.
- Existing five-sector Home/navigation, all four model tabs, all 17 Cell 8
  controls, separate Validation/Test results and stock exploration remain covered.
- Browser workflow: uploaded the Tech engineered CSV, confirmed adjustment status,
  validated 24,453 rows / nine stocks, activated uploaded research and ran a genuine
  session-only Elastic Net backtest. Validation returned 79 trades with audits
  passed; these were computed results, not notebook display output.
- Built-in Tech Elastic Net results still match the verified Phase 2 baseline at
  rtol=atol=1e-10 for equity, returns, Sharpe, drawdown and trade count.
- Original data and notebooks have no Git diff. Uploads do not mutate them.

## Limits

The 12 non-Tech tree-model combinations remain unverified historical runs, though
available on demand. This phase did not repeat the expensive four-model Tech
reference reproduction; that verification is recorded in VERIFICATION.md.

Raw price adjustment provenance cannot be inferred: raw OHLCV must be unadjusted
and accompanied by valid cumulative share factors. Engineered prices require an
explicit already-adjusted confirmation. Discarded market/peer features are not
computed, and cleaning uses the retained 31 features and TARGET_10. Missing raw
financial fields are never invented; incomplete generated rows are dropped.

Model eligibility requires at least three stocks, common observed dates and at
least 40 usable observations per stock/year for 2015–2025. Degenerate inputs can
still fail the original statistical audits, which are displayed rather than
suppressed. Later dates require a future schedule revision. Test tuning is
exploratory; external calendar completeness, dividends, borrow and financing are
not newly modeled or certified. Uploaded caches are session-memory only and do
not survive a new session.

Run: `python -m pytest dashboard/tests -q`.
