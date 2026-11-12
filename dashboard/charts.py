"""Shared Plotly charts."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

COLORS = ["#76C5B5", "#8EADEF", "#D9B77A", "#BD9ACF", "#DC909D", "#8BB8C8"]
FONT = "system-ui, -apple-system, Segoe UI, sans-serif"


def render(figure, **kwargs):
    import streamlit as st

    st.plotly_chart(
        figure, theme=None, width="stretch",
        config={"displaylogo": False, "modeBarButtonsToRemove": ["select2d", "lasso2d"]},
        **kwargs,
    )


def style(figure, title, percent=False, height=360):
    legend = len(figure.data) > 1
    figure.update_layout(
        template="plotly_dark", title=dict(text=title, x=0, xref="paper", y=.98,
                                          yanchor="top", font_size=14),
        paper_bgcolor="#101419", plot_bgcolor="#101419",
        font=dict(color="#AEBBC9", family=FONT, size=12), colorway=COLORS,
        margin=dict(l=12, r=14, t=(80 if title else 60) if legend else (42 if title else 16), b=32),
        height=height, hovermode="x unified", showlegend=legend,
        legend=dict(orientation="h", y=1.02, x=0, yanchor="bottom", font=dict(size=11)),
        hoverlabel=dict(bgcolor="#1B242C", bordercolor="#394953", font=dict(color="#E1E7EE")),
    )
    figure.update_xaxes(showgrid=False, automargin=True, linecolor="#303A44", showline=True,
                        tickfont=dict(size=11), nticks=7)
    figure.update_yaxes(gridcolor="#252E37", zeroline=False, automargin=True,
                        tickfont=dict(size=11), nticks=6)
    if percent:
        figure.update_yaxes(tickformat=".0%")
    return figure


def lines(frame, title, percent=False, height=360, unit=None):
    if isinstance(frame, pd.Series):
        frame = frame.to_frame()
    figure = go.Figure()
    fmt = ".2%" if percent else ",.2f"
    for column in frame:
        figure.add_trace(go.Scatter(
            x=frame.index, y=frame[column], name=str(column), mode="lines",
            line_width=1.7, connectgaps=False,
            hovertemplate=("$" if unit == "USD" else "") + "%{y:" + fmt + "}",
        ))
    style(figure, title, percent, height)
    if isinstance(frame.index, pd.DatetimeIndex):
        figure.update_xaxes(hoverformat="%d %b %Y")
    if unit == "USD":
        figure.update_yaxes(tickprefix="$")
    elif unit == "index":
        figure.update_yaxes(tickformat=",~g")
    return figure


def price(frame, ticker, kind, averages):
    figure = go.Figure()
    if kind == "Candlestick":
        figure.add_trace(go.Candlestick(x=frame.date, open=frame.open, high=frame.high, low=frame.low,
                                       close=frame.close, name="Daily prices",
                                       increasing_line_color=COLORS[0], decreasing_line_color=COLORS[4]))
        figure.update_layout(xaxis_rangeslider_visible=False)
    else:
        figure.add_trace(go.Scatter(x=frame.date, y=frame.close, name="Close", mode="lines", line_width=2))
    for average in averages:
        figure.add_trace(go.Scatter(x=frame.date, y=frame[average], name=average, mode="lines", line_width=1.2))
    style(figure, f"{ticker} · Share price", height=440)
    figure.update_yaxes(tickprefix="$")
    figure.update_xaxes(hoverformat="%d %b %Y")
    return figure


def bars(series, title, percent=False, height=250):
    figure = go.Figure(go.Bar(x=series.index, y=series,
                             marker_color=np.where(series.ge(0), COLORS[0], COLORS[4])))
    return style(figure, title, percent, height)


def distribution(frame):
    if isinstance(frame, pd.Series):
        frame = frame.to_frame()
    figure = go.Figure()
    for column in frame:
        figure.add_trace(go.Histogram(x=frame[column], name=str(column), histnorm="probability", nbinsx=60, opacity=.55))
    style(figure, "Daily return distribution")
    figure.update_layout(barmode="overlay", hovermode="closest")
    figure.update_xaxes(tickformat=".0%", title="Daily return")
    figure.update_yaxes(tickformat=".0%", title="Share of observations")
    return figure


def correlations(returns):
    matrix = returns.corr(min_periods=2)
    figure = go.Figure(go.Heatmap(z=matrix, x=matrix.columns, y=matrix.index, zmin=-1, zmax=1,
                                 colorscale=[[0, "#A55F77"], [.5, "#172434"], [1, COLORS[0]]],
                                 text=matrix.round(2), texttemplate="%{text}", colorbar_thickness=12))
    return style(figure, "How closely the returns move together", height=410)
