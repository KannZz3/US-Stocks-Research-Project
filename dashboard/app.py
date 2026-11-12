"""Run with: python -m streamlit run app.py"""
from pathlib import Path

import pandas as pd
import streamlit as st

import charts
import market
import results
import model_ui
import upload_ui

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="US Stocks Research", layout="wide", initial_sidebar_state="auto")
st.markdown(f"<style>{(ROOT / 'style.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def plot(figure, source=None):
    charts.render(figure)
    if source:
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
    st.caption(f"{len(frame):,} trading days · split-adjusted prices")
    last = frame.iloc[-1]
    cards([("Last close", number(last.close)), ("Last daily return", number(last["return"], True)),
           ("20-day annualized volatility", number(last.Volatility, True))])
    plot(charts.price(frame, ticker, kind, averages))
    if st.toggle("Show volume", value=True):
        plot(charts.bars(frame.set_index("date").volume, "Daily trading volume"))
    left, right = st.columns(2)
    with left:
        plot(charts.bars(frame.set_index("date")["return"], "Daily reported returns", percent=True, height=310))
    with right:
        plot(charts.lines(frame.set_index("date").Volatility, "20-day volatility · annualized", percent=True, height=310))
    st.caption("Close-to-close returns exclude dividends. Volatility is annualized using 252 trading days.")


def compare_stocks():
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
    first, last = curves.index[[0, -1]]
    with st.popover("Chart settings"):
        window = dates(curves.index, key)
        scale = st.radio("Value scale", ["Linear", "Log"], horizontal=True, key=f"{key}_scale")
    if window is None:
        return
    selected = curves.loc[str(window[0]):str(window[1])]
    if selected.empty:
        st.info("No trading days in this date range.")
        return
    figure = charts.lines(selected, "", height=420, unit="index")
    figure.update_xaxes(rangeslider=dict(visible=True, thickness=.09))
    figure.update_yaxes(type="log" if scale == "Log" else "linear")
    plot(figure)
    st.caption(f"{first:%d %b %Y} – {last:%d %b %Y} · Base 100 at inception · Fixed holdings · Dividends excluded")


with st.sidebar:
    st.markdown('<p class="eyebrow">RESEARCH</p><p class="brand">US Equities</p>', unsafe_allow_html=True)
    page = st.radio("Pages", ["Home", *market.SECTORS], key="page", label_visibility="collapsed")
    st.divider()
    st.markdown('<p class="eyebrow">DATA</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload stock CSV", type=["csv"], label_visibility="collapsed")
    active_upload = upload_ui.render(uploaded)

try:
    stamps = tuple(market.fingerprint(DATA_DIR / f"data_preML_{s}.csv") for s in market.SECTORS)
    sectors = sector_data(str(DATA_DIR), stamps)
except ValueError as error:
    st.error(str(error))
    st.stop()

if page == "Home":
    st.title("Sector overview")
    if active_upload is not None: st.info("Upload active for sector research pages. Home retains the five bundled sector comparison.")
    st.caption("45 stocks across 5 sectors · Split-adjusted price history")
    st.subheader("Sector portfolio performance")
    curves = pd.concat({name: market.buy_and_hold(frame) for name, frame in sectors.items()}, axis=1)
    portfolio_chart(curves, "home")
    with st.container(key="sector_shortcuts"):
        for column, sector in zip(st.columns(5), market.SECTORS):
            column.button(sector, key=f"open_{sector}", on_click=navigate, args=(sector,), width="stretch")
else:
    stocks = sectors[page]
    if active_upload is not None:
        from upload_data import export_csv
        try:
            stocks,_ = market.load_stocks(export_csv(active_upload))
        except ValueError as error:
            st.warning(f'Upload is available for export, but chart/model coverage is insufficient: {error}')
            for tab,model in zip(st.tabs(MODELS),MODELS):
                with tab: model_ui.render(page,model,active_upload,st.session_state['upload_scope'])
            st.stop()
    st.title(page if active_upload is None else f"{page} · Uploaded research")
    st.caption(" · ".join(sorted(stocks.ticker.unique())))
    for tab, model in zip(st.tabs(MODELS), MODELS):
        with tab:
            model_ui.render(page, model, active_upload, st.session_state.get("upload_scope", "") if active_upload is not None else "")
    with st.expander("Sector portfolio"):
        portfolio_chart(market.buy_and_hold(stocks).rename(page).to_frame(), page)
    with st.expander("Explore individual stocks"):
        explore()
    with st.expander("Compare stocks and risk"):
        compare_stocks()
