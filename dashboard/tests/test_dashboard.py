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
def app():
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)


@pytest.mark.parametrize("page", ["Home", *market.SECTORS])
def test_all_pages_without_models(app, page):
    app.run().radio(key="page").set_value(page).run()
    assert not app.exception and not app.error
    if page == "Home":
        assert len(app.get("plotly_chart")) == 1
    else:
        assert [tab.label for tab in app.tabs[:4]] == ["Elastic Net", "XGBoost", "CatBoost", "LightGBM"]
        assert sum(x.value == "Not computed" for x in app.info) == 4
    assert len(app.get("file_uploader")) == 1


def test_home_buttons_and_exploration(app):
    app.run()
    for sector in market.SECTORS:
        app.button(key=f"open_{sector}").click().run()
        assert app.radio(key="page").value == sector
        assert not app.exception
        app.radio(key="page").set_value("Home").run()
    app.button(key="open_Tech").click().run()
    app.selectbox(key="chart_type").set_value("Candlestick").run()
    assert not app.exception
    app.multiselect(key="compare_stocks").set_value([]).run()
    assert not app.exception and app.info


def test_all_sector_portfolios():
    sectors = market.load_sectors(ROOT.parent / "data/data_dashboard")
    assert list(sectors) == list(market.SECTORS)
    for frame in sectors.values():
        assert frame.ticker.nunique() == 9
        curve = market.buy_and_hold(frame)
        assert curve.iloc[0] == pytest.approx(100)
        prices = frame.pivot(index="date", columns="ticker", values="close").sort_index()
        fixed_units = (100 / 9) / prices.iloc[0]
        np.testing.assert_allclose(curve, prices.mul(fixed_units).sum(axis=1))
        earlier = frame[frame.date <= curve.index[100]]
        pd.testing.assert_series_equal(market.buy_and_hold(earlier), curve.iloc[:101])


def test_fixed_holdings_are_not_daily_rebalanced():
    dates = pd.date_range("2024-01-01", periods=3)
    frame = pd.DataFrame({"date": list(dates)*2, "ticker": ["A"]*3+["B"]*3,
                          "close": [100,200,100,100,100,200]})
    assert market.buy_and_hold(frame).tolist() == pytest.approx([100,150,150])
    with pytest.raises(ValueError):
        market.buy_and_hold(frame.iloc[1:])


@pytest.mark.parametrize("content", [b"", b"x,y\n1,2", b"Ticker,DlyCalDt,DlyClose\nA,bad,2",
    b"Ticker,DlyCalDt,DlyClose\nA,2024-01-01,-1",
    b"Ticker,DlyCalDt,DlyClose\nA,2024-01-01,inf",
    b"Ticker,DlyCalDt,DlyClose\nA,2024-01-01,1\nA,2024-01-01,2"])
def test_bad_upload(content):
    with pytest.raises(ValueError):
        market.preview_upload(content)


def test_upload_preview_rendered():
    # AppTest does not provide a file-uploader setter. Exercise the same UI helper
    # with an in-memory uploaded-file object instead of relying on private APIs.
    def upload_harness():
        from io import BytesIO
        import runpy
        from pathlib import Path
        import market
        namespace = runpy.run_path(str(Path(market.__file__).with_name("app.py")))
        namespace["upload_preview"](BytesIO(b"Ticker,DlyCalDt,DlyClose\nAAA,2024-01-01,100"))
    at = AppTest.from_function(upload_harness, default_timeout=30).run()
    assert not at.exception
    assert any("Valid CSV" in s.value for s in at.success)
    assert len(at.dataframe) == 1


def test_date_filter_preserves_inception(app):
    app.run()
    app.date_input(key="home_start").set_value(pd.Timestamp("2020-01-01").date()).run()
    assert not app.exception and len(app.get("plotly_chart")) == 1
    app.date_input(key="home_end").set_value(pd.Timestamp("2019-01-01").date()).run()
    assert app.warning and not app.exception


def test_sector_closes_match_upstream_split_adjustment():
    raw = pd.read_csv(ROOT.parent / 'data/data_processing/45stocks_daily_raw.csv',
                      usecols=['Ticker', 'DlyCalDt', 'DlyClose', 'DlyCumFacShr'])
    raw['date'] = pd.to_datetime(raw.DlyCalDt)
    for frame in market.load_sectors(ROOT.parent / 'data/data_dashboard').values():
        joined = frame.merge(raw, left_on=['ticker', 'date'], right_on=['Ticker', 'date'], validate='one_to_one')
        assert len(joined) == len(frame)
        np.testing.assert_allclose(joined.close, joined.DlyClose / joined.DlyCumFacShr, atol=1e-8, rtol=1e-8)
