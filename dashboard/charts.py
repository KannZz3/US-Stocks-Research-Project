"""Shared Plotly charts."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

COLORS = ["#70D6C3", "#91ABF6", "#E3BE7A", "#BC9CE9", "#E796A6", "#85BBCB"]


def style(figure, title, percent=False, height=360):
    figure.update_layout(
        template="plotly_dark", title=dict(text=title, x=.01, y=.98, yanchor="top", font_size=16),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#CDD9E9", family="Arial", size=12), colorway=COLORS,
        margin=dict(l=15, r=20, t=78, b=30), height=height, hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=0, yanchor="bottom"))
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(gridcolor="rgba(160,180,205,.10)", zeroline=False)
    if percent:
        figure.update_yaxes(tickformat=".0%")
    return figure


def lines(frame, title, percent=False, height=360):
    if isinstance(frame, pd.Series):
        frame = frame.to_frame()
    figure = go.Figure()
    for column in frame:
        figure.add_trace(go.Scatter(x=frame.index, y=frame[column], name=str(column), mode="lines",
                                   line_width=1.8, connectgaps=False))
    return style(figure, title, percent, height)


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
    figure.update_yaxes(title="USD · prices as supplied")
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
