"""Run with: python -m streamlit run app.py"""
from pathlib import Path

import pandas as pd
import streamlit as st

import charts
import market
import results
import model_ui

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Dashboard", page_icon="◈", layout="wide")
st.markdown(f"<style>{(ROOT / 'style.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def plot(figure, source="Source: historical stock CSV."):
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})
    st.caption(source)


def number(value, percent=False):
    if value is None or pd.isna(value):
        return "Not available"
    return f"{value:.2%}" if percent else f"{value:,.2f}"


def cards(items):
    for column, (label, value) in zip(st.columns(len(items)), items):
        column.metric(label, value)


def dates(index, key, years=None):
    first, last = index.min().date(), index.max().date()
    start = max(first, (pd.Timestamp(last) - pd.DateOffset(years=years)).date()) if years else first
    a, b = st.columns(2)
    start = a.date_input("Start date", start, min_value=first, max_value=last, key=f"{key}_start")
    end = b.date_input("End date", last, min_value=first, max_value=last, key=f"{key}_end")
    if start > end:
        st.warning("Start date must come before the end date.")
        return None
    return start, end


def table(frame, filename):
    formats = {c: "{:.2%}" for c in ["Total return", "Annual return", "Annual volatility", "Max drawdown"] if c in frame}
    formats.update({"Sharpe": "{:.2f}", "Periods": "{:,.0f}"})
    st.dataframe(frame.style.format(formats, na_rep="—"), width="stretch")
    if allow_downloads:
        st.download_button("Download summary", frame.to_csv().encode(), filename, "text/csv")


def explore():
    st.subheader("Explore stocks")
    st.write("See one stock’s price, trading activity and daily returns.")
    a, b, c = st.columns([1, 1, 2])
    ticker = a.selectbox("Stock", sorted(stocks.ticker.unique()), key="stock")
    kind = b.selectbox("Price chart", ["Line", "Candlestick"], key="chart_type")
    averages = c.multiselect("Moving averages", ["SMA 20", "SMA 50", "SMA 200"], default=["SMA 20", "SMA 50"])
    history = stocks[stocks.ticker.eq(ticker)]
    window = dates(pd.DatetimeIndex(history.date), "explore", years=2)
    if window is None:
        return
    frame = history[history.date.between(pd.Timestamp(window[0]), pd.Timestamp(window[1]))]
    if frame.empty:
        st.info("No trading days in this range. Try a wider date range.")
        return
    st.caption(f"Showing {len(frame):,} trading days. Earlier dates are available in the date selectors; the full dataset starts {history.date.min():%d %b %Y}.")
    last = frame.iloc[-1]
    cards([("Last close", number(last.close)), ("Last daily return", number(last["return"], True)),
           ("20-day annualized volatility", number(last.Volatility, True))])
    plot(charts.price(frame, ticker, kind, averages))
    st.caption("Prices are split-adjusted upstream. Moving averages use past observations only. A candlestick shows the day’s open, high, low and close.")
    if st.toggle("Show volume", value=True):
        plot(charts.bars(frame.set_index("date").volume, "Daily trading volume"))
    left, right = st.columns(2)
    with left:
        plot(charts.bars(frame.set_index("date")["return"], "Daily reported returns", percent=True, height=310))
    with right:
        plot(charts.lines(frame.set_index("date").Volatility, "20-day volatility · annualized", percent=True, height=310))
    st.caption("Returns use RET_1, a close-to-close price return excluding dividends. Volatility measures how much daily returns fluctuate; this chart annualizes it using 252 trading days.")


def compare_stocks():
    st.subheader("Compare stocks")
    st.write("Choose stocks and a date range to compare their growth and risk.")
    tickers = sorted(stocks.ticker.unique())
    chosen = st.multiselect("Stocks", tickers, default=tickers[:3], key="compare_stocks")
    window = dates(pd.DatetimeIndex(stocks.date), "compare", years=3)
    if len(chosen) < 2 or window is None:
        st.info("Choose at least two stocks and a valid date range.")
        return
    include_market = st.checkbox("Include S&P 500", value=True)
    returns, prices = market.comparison(stocks, chosen, *window, benchmark=include_market)
    if len(returns) < 2:
        st.info("Choose a range with at least two trading dates.")
        return
    growth, price_tab, risk = st.tabs(["Reported returns", "Share prices", "Risk and correlation"])
    with growth:
        plot(charts.lines(market.historical_index(returns), "Growth from a starting value of 100", height=420))
        st.caption("Stocks compound RET_1 price returns; the S&P 500 also excludes dividends. This chart shows market history, not a trading strategy.")
        summary = pd.DataFrame({name: results.metrics(returns[name].iloc[1:]) for name in returns}).T
        table(summary, "stock-summary.csv")
    with price_tab:
        plot(charts.lines(prices / prices.iloc[0] * 100, "Share prices · starting value 100", height=420))
        st.caption("Uses supplied closing prices, already split-adjusted upstream, excluding dividends.")
    with risk:
        left, right = st.columns(2)
        with left:
            plot(charts.distribution(returns.iloc[1:]))
        with right:
            plot(charts.correlations(returns.iloc[1:]))
        volatility = stocks[stocks.ticker.isin(chosen)].pivot(index="date", columns="ticker", values="Volatility")
        plot(charts.lines(volatility.loc[str(window[0]):str(window[1])], "20-day volatility · annualized", percent=True))


MODELS = ("Elastic Net", "XGBoost", "CatBoost", "LightGBM")
DATA_DIR = ROOT.parent / "data" / "data_dashboard"
allow_downloads = True


@st.cache_data(show_spinner=False)
def sector_data(folder, stamps):
    return market.load_sectors(folder)


def navigate(sector):
    st.session_state.page = sector


def upload_preview(uploaded):
    if uploaded is None:
        return
    try:
        preview, audit = market.preview_upload(uploaded.getvalue())
    except ValueError as error:
        st.error(str(error))
        return
    st.success(f"Valid CSV · {audit['Rows']:,} rows · {audit['Stocks']} stocks")
    st.dataframe(preview, hide_index=True, width="stretch")
    st.caption("First 50 rows. Preview only; adjustment status is not verified. Preprocessing arrives in Phase 3.")


def portfolio_chart(curves, key):
    window = dates(curves.index, key)
    if window is None:
        return
    selected = curves.loc[str(window[0]):str(window[1])]
    if selected.empty:
        st.info("No trading days in this date range.")
        return
    figure = charts.lines(selected, "Portfolio value · inception = 100", height=520)
    figure.update_xaxes(rangeslider_visible=True)
    figure.update_layout(dragmode="zoom", margin=dict(t=150))
    plot(figure, "Source: data/data_dashboard/ · split-adjusted closing prices · excludes dividends")
    st.caption("Nine stocks per sector, equal capital at inception and fixed holdings thereafter. Date filters do not reset the base or rebalance holdings.")


with st.sidebar:
    st.markdown('<p class="brand">Stock Research</p>', unsafe_allow_html=True)
    page = st.radio("Pages", ["Home", *market.SECTORS], key="page", label_visibility="collapsed")
    st.divider()
    uploaded = st.file_uploader("Upload stock CSV", type=["csv"])
    upload_preview(uploaded)
    st.caption("Phase 2 · Models & history")

try:
    stamps = tuple(market.fingerprint(DATA_DIR / f"data_preML_{s}.csv") for s in market.SECTORS)
    sectors = sector_data(str(DATA_DIR), stamps)
except ValueError as error:
    st.error(str(error))
    st.stop()

if page == "Home":
    st.title("Sector overview")
    st.write("Five sectors. Forty-five stocks. One historical view.")
    curves = pd.concat({name: market.buy_and_hold(frame) for name, frame in sectors.items()}, axis=1)
    portfolio_chart(curves, "home")
    for column, sector in zip(st.columns(5), market.SECTORS):
        column.button(sector, key=f"open_{sector}", on_click=navigate, args=(sector,), width="stretch")
    st.caption("Fixed supplied universe; selection/survivorship bias may remain. These are historical price portfolios, not model results.")
else:
    stocks = sectors[page]
    st.title(page)
    st.caption(" · ".join(sorted(stocks.ticker.unique())))
    for tab, model in zip(st.tabs(MODELS), MODELS):
        with tab:
            model_ui.render(page, model)
    portfolio_chart(market.buy_and_hold(stocks).rename(page).to_frame(), page)
    with st.expander("Explore individual stocks", expanded=True):
        explore()
    with st.expander("Compare stocks and risk"):
        compare_stocks()
