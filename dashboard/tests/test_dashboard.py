"""Small artificial fixtures test calculations and the app. No model results are shipped."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import market
import results

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def stock_csv(tmp_path):
    dates = pd.bdate_range("2022-01-03", periods=300)
    frames = []
    for i, ticker in enumerate(["AAA", "BBB"]):
        close = 100 + np.arange(len(dates)) * (i + 1) / 10
        frames.append(pd.DataFrame({"Ticker": ticker, "DlyCalDt": dates, "DlyClose": close,
                                    "DlyOpen": close, "DlyHigh": close + 1, "DlyLow": close - 1,
                                    "DlyVol": 1000, "DlyRet": np.sin(np.arange(len(dates))) / 100,
                                    "sprtrn": np.cos(np.arange(len(dates))) / 100}))
    path = tmp_path / "stocks.csv"
    pd.concat(frames).to_csv(path, index=False)
    return path


@pytest.fixture
def equity():
    return pd.DataFrame({"timestamp": pd.bdate_range("2023-01-02", periods=5),
                         "ticker": "AAA", "model": "Test model", "equity": [100, 110, 99, 108.9, 110],
                         "split": "test", "timeframe": "daily"})


def test_stock_data_and_trailing_indicators(stock_csv):
    frame, audit = market.load_stocks(stock_csv)
    assert audit["Rows"] == 600 and audit["Stocks"] == 2
    a = frame[frame.ticker.eq("AAA")]
    assert a["SMA 20"].iloc[:19].isna().all()
    assert a["SMA 20"].iloc[19] == pytest.approx(a.close.iloc[:20].mean())
    assert a.Volatility.iloc[19] == pytest.approx(a["return"].iloc[:20].std(ddof=1) * np.sqrt(252))
    # Adding later observations must not change already-computed features.
    raw = pd.read_csv(stock_csv)
    raw = raw[raw.DlyCalDt <= str(a.date.iloc[70].date())]
    raw.to_csv(stock_csv, index=False)
    earlier, _ = market.load_stocks(stock_csv)
    expected = frame[frame.date <= a.date.iloc[70]].reset_index(drop=True)
    pd.testing.assert_frame_equal(earlier, expected)


def test_uploaded_csv_matches_disk_file(stock_csv):
    disk, _ = market.load_stocks(stock_csv)
    uploaded, _ = market.load_stocks(stock_csv.read_bytes())
    pd.testing.assert_frame_equal(uploaded, disk)


def test_stock_file_lookup(tmp_path):
    dashboard = tmp_path / "dashboard"
    dashboard.mkdir()
    assert market.stock_path(dashboard) == dashboard / "data/stocks.csv"
    (tmp_path / "data.csv").write_text("repository data")
    assert market.stock_path(dashboard) == tmp_path / "data.csv"
    (dashboard / "data").mkdir()
    (dashboard / "data/stocks.csv").write_text("local data")
    assert market.stock_path(dashboard) == dashboard / "data/stocks.csv"


@pytest.mark.parametrize("problem", ["duplicate", "date", "return", "benchmark", "coverage"])
def test_invalid_stocks_are_rejected(stock_csv, problem):
    frame = pd.read_csv(stock_csv)
    if problem == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif problem == "date":
        frame.loc[0, "DlyCalDt"] = "bad date"
    elif problem == "return":
        frame.loc[0, "DlyRet"] = np.nan
    elif problem == "benchmark":
        frame.loc[0, "sprtrn"] = .8
    else:
        frame = frame.iloc[1:]
    frame.to_csv(stock_csv, index=False)
    with pytest.raises(ValueError):
        market.load_stocks(stock_csv)


def test_historical_base_date_is_not_an_earned_return():
    assert market.historical_index(pd.Series([.5, .1, -.1])).tolist() == pytest.approx([100, 110, 99])


def test_metrics_compound_and_include_initial_loss():
    stat = results.metrics(pd.Series([-.1, .1]))
    assert stat["Total return"] == pytest.approx(-.01)
    assert stat["Max drawdown"] == pytest.approx(-.1)
    assert np.isnan(stat["Annual return"])
    assert np.isnan(results.metrics(pd.Series([.01] * 10))["Sharpe"])
    r = pd.Series([.01, -.005] * 126)
    stat = results.metrics(r, .04)
    assert stat["Annual return"] == pytest.approx((1 + r).prod() - 1)
    assert stat["Sharpe"] == pytest.approx((r.mean() - (1.04 ** (1 / 252) - 1)) / r.std(ddof=1) * np.sqrt(252))


@pytest.mark.parametrize("values", [[], [.1, np.nan], [-1.5], [float("inf")]])
def test_invalid_metric_inputs(values):
    with pytest.raises(ValueError):
        results.metrics(pd.Series(values, dtype=float))


def test_equity_and_return_schemas(equity):
    frame, _ = results.prepare_result(equity, {})
    assert np.isnan(frame["return"].iloc[0])
    assert frame["return"].iloc[1] == pytest.approx(.1)
    frame, _ = results.prepare_result(equity, {"initial_capital": 125, "equity_convention": "end_of_day"})
    assert frame["return"].iloc[0] == pytest.approx(-.2)
    returns = equity.drop(columns="equity").assign(input_type="returns", strategy_return=.01)
    frame, notes = results.prepare_result(returns, {})
    assert frame.equity.iloc[0] == pytest.approx(101)
    assert any("not a dollar" in note for note in notes)


@pytest.mark.parametrize("problem", ["duplicate", "missing", "training", "intraday", "conflict"])
def test_bad_experiments_do_not_hide_good_ones(tmp_path, equity, problem):
    equity.to_csv(tmp_path / "good.csv", index=False)
    bad = equity.copy()
    if problem == "duplicate":
        bad = pd.concat([bad, bad.iloc[:1]])
    elif problem == "missing":
        bad.loc[0, "equity"] = np.nan
    elif problem == "training":
        bad["split"] = "train"
    elif problem == "intraday":
        bad["timeframe"] = "1m"
    else:
        bad["strategy_return"] = .5
    bad.to_csv(tmp_path / "bad.csv", index=False)
    experiments, errors = results.read_results(tmp_path)
    assert len(experiments) == 1 and errors


def test_comparison_does_not_bridge_missing_dates(tmp_path, equity):
    equity.to_csv(tmp_path / "a.csv", index=False)
    equity.iloc[[0, 1, 3, 4]].to_csv(tmp_path / "b.csv", index=False)
    experiments, _ = results.read_results(tmp_path)
    common, excluded = results.shared_returns(experiments, "2023-01-02", "2023-01-10")
    assert len(common) == 2 and excluded == 3
    assert common.index.min() == pd.Timestamp("2023-01-05")


@pytest.fixture
def app(stock_csv, tmp_path, monkeypatch):
    folder = tmp_path / "backtests"
    folder.mkdir()
    monkeypatch.setenv("FIN556_STOCKS_PATH", str(stock_csv))
    monkeypatch.setenv("FIN556_BACKTEST_DIR", str(folder))
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=20), folder


@pytest.mark.parametrize("page", ["Overview", "Explore Stocks", "Compare Stocks", "Compare Models", "Strategy Results", "About the Data"])
def test_all_pages_without_models(app, page):
    at, _ = app
    at.run().radio(key="page").set_value(page).run()
    assert not at.exception and not at.error


def test_candles_and_empty_selection(app):
    at, _ = app
    at.run().radio(key="page").set_value("Explore Stocks").run()
    at.selectbox(key="chart_type").set_value("Candlestick").run()
    assert not at.exception and len(at.get("plotly_chart")) == 4
    at.radio(key="page").set_value("Compare Stocks").run()
    at.multiselect(key="compare_stocks").set_value([]).run()
    assert not at.exception and at.info


def test_result_pages_with_temporary_fixtures(app, equity):
    at, folder = app
    equity.to_csv(folder / "fixture.csv", index=False)
    (folder / "fixture.json").write_text(json.dumps({"initial_capital": 100, "equity_basis": "net", "costs": "Test fixture only"}))
    for page in ["Compare Models", "Strategy Results"]:
        at.run().radio(key="page").set_value(page).run()
        assert not at.exception and not at.error
        assert len(at.get("plotly_chart")) >= 3


def test_missing_stock_file_keeps_results_page_available(app):
    at, _ = app
    at.run().text_input(key="stock_path").set_value("/no/such/file.csv").run()
    assert not at.error and not at.exception and at.info
    assert len(at.get("file_uploader")) == 1
    at.radio(key="page").set_value("Compare Models").run()
    assert not at.exception
