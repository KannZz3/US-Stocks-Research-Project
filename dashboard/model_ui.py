"""Backtest controls and results; execution is delegated to the research engine."""
import pickle
import traceback

import pandas as pd
import streamlit as st

import charts
from ml import engine

PARAMETER_GROUPS = {
    "Signal & execution": {
        "C8_TRADE_SIDES": "Directions",
        "C8_MIN_VOTES": "Minimum votes",
        "C8_MAX_ENTRY_GAP": "Maximum entry gap (fraction)",
        "C8_MAX_HOLD_BARS": "Maximum holding bars",
    },
    "Capital & exits": {
        "C8_START_CAPITAL": "Starting capital (USD)",
        "C8_MAX_POSITIONS": "Maximum positions",
        "C8_POSITION_FRACTION": "Allocation per position (fraction)",
        "C8_TP_M_MULT": "Take-profit multiple of M",
        "C8_SL_M_MULT": "Stop-loss multiple of M",
        "C8_SHORT_COLLATERAL_RATIO": "Short collateral ratio",
    },
    "Calibration, costs & data": {
        "C8_MIN_STOCK_POSITIVE": "Minimum positive stock observations",
        "C8_MIN_SIDE_POSITIVE": "Minimum positive side observations",
        "C8_M_TRAIN_START_YEAR": "Calibration start year",
        "C8_ZERO_COSTS": "Simulate zero transaction costs",
        "C8_COST_BPS_PER_LEG": "Cost per leg (bps)",
        "C8_OHLC_COLS": "OHLC columns",
        "C8_OUTPUT_DIR": "CSV export directory",
    },
}

TABLE_LABELS = {
    "cell8_audit_df": "Execution audits",
    "cell8_m_calibration_df": "M calibration",
    "cell8_performance_df": "Performance summary",
    "cell8_equity_df": "Daily equity",
    "cell8_trade_log_df": "Trade log",
    "cell8_skipped_entries_df": "Skipped entries",
    "cell8_cost_sensitivity_df": "Cost sensitivity",
}


def reset(prefix, model):
    for name, value in engine.defaults(model).items():
        st.session_state[prefix + name] = ",".join(value) if name == "C8_OHLC_COLS" else value


def parameter_input(name, label, value, key, uploaded):
    if key not in st.session_state:
        st.session_state[key] = ",".join(value) if name == "C8_OHLC_COLS" else value
    if name == "C8_TRADE_SIDES":
        return st.multiselect(label, ["Long", "Short"], key=key)
    if name == "C8_OHLC_COLS":
        columns = st.text_input(label, key=key, help="Comma-separated Open, High, Low, Close columns.")
        return [column.strip() for column in columns.split(",")]
    if name == "C8_OUTPUT_DIR":
        return st.text_input(label, key=key, disabled=uploaded,
                             help="Relative to the local export folder. Uploads use browser downloads.")
    if isinstance(value, bool):
        return st.checkbox(label, key=key)
    return st.number_input(label, key=key, step=1 if isinstance(value, int) else .01,
                           format="%d" if isinstance(value, int) else "%.5f")


def render(sector, model, dataset=None, scope=""):
    from upload_data import digest, eligibility

    uploaded = dataset is not None
    prefix = f"{scope}:{digest(dataset)}:{sector}:{model}:" if uploaded else f"{sector}:{model}:"
    if uploaded:
        reasons = eligibility(dataset)
        if reasons:
            st.warning("This dataset cannot run a backtest.")
            with st.expander("Coverage requirements"):
                st.markdown("\n".join(f"- {reason}" for reason in reasons))
            st.button("Run Backtest", key=prefix + "disabled", disabled=True)
            return

    memory = st.session_state.setdefault("upload_training_cache", {})
    try:
        path = engine.cache_path(sector, model, dataset, scope)
    except Exception as error:
        st.error(f"Model unavailable: {error}")
        return

    cached_training = path.name in memory if uploaded else path.exists()
    cache_status = st.empty()
    cache_status.caption("Training cached · Runs reuse fitted models" if cached_training
                         else "Training required · The first run may take several minutes")
    defaults = engine.defaults(model)
    with st.form(prefix + "form", border=False):
        params = {}
        for column, (group, fields) in zip(st.columns(3, gap="small"), PARAMETER_GROUPS.items()):
            with column, st.expander(group):
                for name, label in fields.items():
                    params[name] = parameter_input(name, label, defaults[name], prefix + name, uploaded)
        with st.container(horizontal=True, gap="small"):
            run = st.form_submit_button("Run Backtest", type="primary", width="content")
            st.form_submit_button("Reset Defaults", on_click=reset, args=(prefix, model), width="content")
    result_key = prefix + "result"
    if run:
        try:
            with st.status("Running backtest", expanded=False) as status:
                state = engine.train(sector, model, st.write, dataset, memory, scope)
                cache_status.caption("Training cached · Runs reuse fitted models")
                result = engine.backtest(state, model, params, export=not uploaded)
                if not uploaded:
                    engine.save_result(engine.result_path(sector, model, params), result)
                st.session_state[result_key] = (path.name, result)
                status.update(label="Backtest complete · Execution audits passed", state="complete")
        except Exception as error:
            st.error(f"Backtest failed: {error or type(error).__name__}")
            with st.expander("Error details"):
                st.code(traceback.format_exc())
    if not uploaded and result_key not in st.session_state:
        cached = engine.result_path(sector, model, defaults)
        if cached.exists():
            with cached.open("rb") as file:
                st.session_state[result_key] = (path.name, pickle.load(file))
    saved = st.session_state.get(result_key)
    if not saved or saved[0] != path.name:
        st.caption("No results yet. Run a backtest to view performance and trades.")
        return

    show_results(saved[1], prefix)


def show_results(result, prefix):
    st.subheader("Backtest performance")
    enabled = result["enabled_sides"]
    disabled = [side for side in ("Long", "Short") if side not in enabled]
    active = ", ".join(enabled) or "None (all cash)"
    inactive = ", ".join(disabled) or "None"
    st.caption(f"Last successful run · Active: {active} · Disabled: {inactive}")
    for period, tab in zip(("Validation", "Test"), st.tabs(["Validation · 2022–23", "Test · 2024–25"])):
        with tab:
            if period == "Test":
                st.caption("Test parameter experiments are exploratory. Dividends, borrowing and financing are not modeled.")
            show_period(result, prefix, period)

    with st.expander("Run settings, audits & downloads"):
        with st.popover("Submitted settings"):
            st.json(result["parameters"], expanded=True)
        tables = {name: frame for name, frame in result.items() if isinstance(frame, pd.DataFrame)}
        if tables:
            names = sorted(tables, key=lambda name: list(TABLE_LABELS).index(name) if name in TABLE_LABELS else len(TABLE_LABELS))
            label = lambda name: TABLE_LABELS.get(name, name.removeprefix("cell8_").removesuffix("_df").replace("_", " ").capitalize())
            selected = st.selectbox("Report", names, format_func=label, key=prefix + "report")
            frame = tables[selected]
            preview = frame.head(100).copy()
            if "Year" in preview:
                preview["Year"] = preview["Year"].astype(str)
            st.dataframe(preview, hide_index=True, width="stretch")
            st.download_button("Download CSV", frame.to_csv(index=False), selected + ".csv",
                               "text/csv", key=prefix + selected)


def show_period(result, prefix, period):
    perf = result["cell8_performance_df"]
    row = perf[perf.Period.eq(period) & perf.Year.eq("ALL")].iloc[0]
    metrics = [
        ("Return", f"{row.Total_Return:.2%}"),
        ("Sharpe", f"{row.Sharpe_252:.3f}" if pd.notna(row.Sharpe_252) else "—"),
        ("Max drawdown", f"{row.Max_Drawdown:.2%}"),
        ("Trades", str(int(row.Trades))),
        ("Win rate", f"{row.Trade_Win_Rate:.2%}" if pd.notna(row.Trade_Win_Rate) else "—"),
    ]
    with st.container(key="result_metrics_" + prefix + period):
        for column, (name, value) in zip(st.columns(5), metrics):
            column.metric(name, value)
    curve = result["cell8_equity_df"]
    curve = curve[curve.Period.eq(period)].set_index("Date")
    equity = curve[["Equity_USD", "Equal_Weight_BuyHold_Equity"]].rename(
        columns={"Equity_USD": "Strategy", "Equal_Weight_BuyHold_Equity": "Buy & hold"}
    )
    charts.render(charts.lines(equity, "Account equity", unit="USD", height=320), key=prefix + period + "equity")
    charts.render(charts.lines(curve.Drawdown, "Drawdown", percent=True, height=210), key=prefix + period + "drawdown")
    trades = result["cell8_trade_log_df"]
    trades = trades[trades.Period.eq(period)]
    with st.expander(f"Trade log & exits · {len(trades):,} trades"):
        if trades.empty:
            st.caption("No trades in this period.")
        else:
            st.dataframe(trades, hide_index=True, width="stretch")
            st.dataframe(trades.groupby(["Direction", "Exit_Reason"]).agg(
                Trades=("Trade_ID", "size"), PnL=("Net_PnL_USD", "sum")
            ), width="stretch")
