"""Plotly charts for the dashboard."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

BLUE = "#2459a9"
PALETTE = [BLUE, "#4987c6", "#7aaad9", "#183956", "#88a0b8", "#3c697e",
           "#5963a4", "#1b817d", "#b2bce3", "#8b623e", "#6299a2", "#a7546f"]


def style(
    figure: go.Figure, *, x: str = "", y: str = "", height: int = 345
) -> go.Figure:
    return figure.update_layout(
        height=height, margin=dict(l=12, r=12, t=12, b=8),
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font=dict(family="Inter, sans-serif", size=12, color="#334661"),
        xaxis=dict(title=x, gridcolor="#e8eef6", zeroline=False),
        yaxis=dict(title=y, gridcolor="#e8eef6", zeroline=False),
        legend=dict(orientation="h", y=1.02, x=0, yanchor="bottom"),
        hoverlabel=dict(bgcolor="#ffffff"),
    )


def simulation_chart(paths: np.ndarray, confidence: float) -> go.Figure:
    days = np.arange(paths.shape[0])
    tail = (1 - confidence) / 2
    lower, median, upper = np.quantile(paths, [tail, .5, 1 - tail], axis=1)
    fig = go.Figure()
    sample = np.linspace(0, paths.shape[1] - 1, min(36, paths.shape[1]), dtype=int)
    fig.add_trace(go.Scatter(
        x=days, y=lower, line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=days, y=upper, line=dict(width=0), fill="tonexty",
        fillcolor="rgba(73,135,198,0.21)", name=f"{confidence:.0%} outcome band",
        hoverinfo="skip",
    ))
    for index in sample:
        fig.add_trace(go.Scatter(
            x=days, y=paths[:, index], line=dict(color="rgba(36,89,169,0.16)", width=1),
            showlegend=False, hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(
        x=days, y=median, name="Median", line=dict(color=BLUE, width=3),
        customdata=np.column_stack((lower, upper)),
        hovertemplate="Day %{x}<br>Median %{y:,.0f}<br>Band %{customdata[0]:,.0f} to %{customdata[1]:,.0f}<extra></extra>",
    ))
    return style(fig, x="Trading day", y="Portfolio value", height=360).update_yaxes(tickformat=",.0f")


def terminal_chart(terminal_returns: np.ndarray, median_return: float) -> go.Figure:
    fig = go.Figure(go.Histogram(
        x=terminal_returns, nbinsx=44, marker_color=BLUE,
        hovertemplate="Return %{x:.1%}<br>Paths %{y:,}<extra></extra>",
    ))
    fig.add_vline(x=0, line_dash="dot", line_color="#687890", annotation_text="Break-even")
    fig.add_vline(x=median_return, line_color="#2459a9")
    return style(fig, x="Horizon return", y="Simulations", height=360).update_xaxes(tickformat=".0%")


def allocation_chart(tickers: list[str], weights: np.ndarray) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=tickers, values=weights, hole=.66, sort=False, textinfo="none",
        marker=dict(colors=PALETTE[:len(tickers)], line=dict(color="white", width=2)),
        hovertemplate="%{label}<br>%{percent}<extra></extra>",
    ))
    return style(fig, height=250).update_layout(
        legend=dict(orientation="h", x=.5, xanchor="center", y=-.04, yanchor="top"),
        annotations=[dict(text=f"{len(tickers)} assets", x=.5, y=.5, showarrow=False)],
    )


def history_chart(values: pd.Series) -> go.Figure:
    performance = values / values.iloc[0] - 1
    fig = go.Figure(go.Scatter(
        x=values.index, y=performance, line=dict(color=BLUE, width=2),
        hovertemplate="%{x|%d %b %Y}<br>Return %{y:+.2%}<extra></extra>",
    ))
    return style(fig, x="Observation date", y="Cumulative return").update_yaxes(tickformat=".0%")


def correlation_chart(correlation: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=correlation.to_numpy(), x=correlation.columns, y=correlation.index,
        zmin=-1, zmax=1, colorscale=["#ad543f", "#f2f5fa", "#2459a9"],
        text=correlation.fillna(0).map(lambda value: f"{value:.2f}").mask(correlation.isna(), "—"),
        texttemplate="%{text}", hovertemplate="%{y} / %{x}<br>Correlation %{text}<extra></extra>",
        colorbar=dict(tickvals=[-1, 0, 1]),
    ))
    return style(fig).update_yaxes(autorange="reversed")
