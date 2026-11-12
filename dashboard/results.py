"""Load teammates' daily backtests and calculate performance."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Experiment:
    name: str
    ticker: str
    model: str
    frame: pd.DataFrame
    info: dict
    notes: list


def read_results(folder):
    experiments, errors = [], []
    for path in sorted(Path(folder).glob("*.csv")):
        try:
            data = pd.read_csv(path)
            required = {"timestamp", "ticker", "model"}
            if data.empty or not required.issubset(data):
                raise ValueError("Expected timestamp, ticker, model and equity columns.")
            for column in ["ticker", "model"]:
                if data[column].isna().any() or data[column].astype(str).str.strip().eq("").any():
                    raise ValueError(f"Missing {column}.")
                data[column] = data[column].astype(str).str.strip()
            data["ticker"] = data.ticker.str.upper()
            if "experiment_id" not in data:
                data["experiment_id"] = path.stem
            if data.experiment_id.isna().any():
                raise ValueError("Missing experiment_id.")
            sidecar = path.with_suffix(".json")
            info = json.loads(sidecar.read_text()) if sidecar.exists() else {}
            if not isinstance(info, dict):
                raise ValueError("Metadata must be a JSON object.")
            for (ticker, model, run), rows in data.groupby(["ticker", "model", "experiment_id"]):
                name = f"{model} · {run} · {path.name}"
                try:
                    frame, notes = prepare_result(rows, info)
                    experiments.append(Experiment(name, ticker, model, frame, info, notes))
                except (ValueError, TypeError) as error:
                    errors.append(f"{name}: {error}")
        except (OSError, ValueError, TypeError) as error:
            errors.append(f"{path.name}: {error}")
    return experiments, errors


def numeric(frame, column):
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any() or not np.isfinite(values).all():
        raise ValueError(f"{column} must contain finite numbers on every row.")
    return values


def prepare_result(rows, info):
    frame, notes = rows.copy(), []
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, errors="coerce", format="mixed").dt.tz_localize(None).dt.normalize()
    if frame.timestamp.isna().any() or frame.timestamp.duplicated().any():
        raise ValueError("Invalid dates or multiple rows on the same daily date.")
    frame = frame.sort_values("timestamp").set_index("timestamp")
    if len(frame) < 2:
        raise ValueError("At least two daily observations are needed.")
    if "timeframe" in frame and not frame.timeframe.isin(["daily", "1d"]).all():
        raise ValueError("Only daily results are supported.")
    if "split" in frame and not frame.split.isin(["test", "oos", "out_of_sample", "walk_forward"]).all():
        raise ValueError("Export test results only; remove training/validation rows.")
    if "split" not in frame:
        notes.append("The file does not identify these as out-of-sample results.")
    kinds = frame.input_type.unique().tolist() if "input_type" in frame else ["equity"]
    if len(kinds) != 1 or kinds[0] not in ["equity", "returns"]:
        raise ValueError("input_type must be equity or returns for every row.")
    capital = info.get("initial_capital")
    if capital is not None and (not isinstance(capital, (int, float)) or not np.isfinite(capital) or capital <= 0):
        raise ValueError("initial_capital must be a positive number.")
    rf = info.get("risk_free_rate", 0)
    if not isinstance(rf, (int, float)) or not np.isfinite(rf) or rf <= -1:
        raise ValueError("risk_free_rate must be an annual decimal greater than -1.")
    if kinds[0] == "equity":
        if "equity" not in frame:
            raise ValueError("Equity files need an equity column. Return files need input_type=returns.")
        frame["equity"] = numeric(frame, "equity")
        if frame.equity.le(0).any():
            raise ValueError("Equity must stay positive.")
        frame["return"] = frame.equity.pct_change(fill_method=None)
        convention = info.get("equity_convention", "end_of_day" if capital else "first_observation_baseline")
        if convention == "end_of_day" and capital:
            frame.iloc[0, frame.columns.get_loc("return")] = frame.equity.iloc[0] / capital - 1
        elif convention not in ["end_of_day", "first_observation_baseline"]:
            raise ValueError("Unknown equity_convention.")
        else:
            if convention == "first_observation_baseline" and capital and not np.isclose(capital, frame.equity.iloc[0]):
                raise ValueError("Initial capital disagrees with the starting baseline.")
            notes.append("First equity is a baseline; its unknown return is excluded.")
        if "strategy_return" in frame:
            supplied = numeric(frame, "strategy_return")
            known = frame["return"].notna()
            if not np.allclose(supplied[known], frame.loc[known, "return"], atol=1e-6, rtol=1e-4):
                raise ValueError("strategy_return disagrees with the equity curve.")
    else:
        if "strategy_return" not in frame or "equity" in frame:
            raise ValueError("Return files need strategy_return and must omit equity.")
        frame["return"] = numeric(frame, "strategy_return")
        if frame["return"].lt(-1).any():
            raise ValueError("A daily return cannot be below −100%.")
        frame["equity"] = (capital or 100) * (1 + frame["return"]).cumprod()
        if capital is None:
            notes.append("Equity is an index starting from 100, not a dollar amount.")
    for column in ["benchmark_return", "gross_return", "transaction_cost"]:
        if column in frame:
            frame[column] = numeric(frame, column)
            if column.endswith("return") and frame[column].lt(-1).any():
                raise ValueError(f"Invalid values in {column}.")
    if info.get("equity_basis") not in ["gross", "net"]:
        notes.append("The file does not state whether trading costs are included.")
    if not info.get("costs"):
        notes.append("Trading-cost and slippage assumptions were not supplied.")
    if "risk_free_rate" not in info:
        notes.append("Sharpe uses a 0% annual risk-free rate.")
    if frame.index.to_series().diff().dt.days.max() > 7:
        notes.append("There are gaps longer than a week; check the daily date coverage.")
    return frame, notes


def shared_returns(experiments, start, end):
    series = [e.frame["return"].loc[str(start):str(end)].rename(e.name) for e in experiments]
    union = pd.concat(series, axis=1).sort_index()
    complete = union.notna().all(axis=1)
    groups = complete.ne(complete.shift()).cumsum()
    blocks = [rows for _, rows in union[complete].groupby(groups[complete])]
    shared = max(blocks, key=len) if blocks else union.iloc[:0]
    return shared, len(union) - len(shared)


def drawdown(returns):
    wealth = (1 + returns).cumprod()
    return wealth / wealth.cummax().clip(lower=1) - 1


def metrics(returns, risk_free=0):
    if returns.empty or returns.isna().any() or not np.isfinite(returns).all() or returns.lt(-1).any():
        raise ValueError("Metrics need complete daily returns.")
    total = (1 + returns).prod() - 1
    std = returns.std(ddof=1)
    daily_rf = (1 + risk_free) ** (1 / 252) - 1
    return {"Periods": len(returns), "Total return": total,
            "Annual return": (1 + total) ** (252 / len(returns)) - 1 if len(returns) >= 252 else np.nan,
            "Annual volatility": std * np.sqrt(252),
            "Sharpe": (returns.mean() - daily_rf) / std * np.sqrt(252) if std > 1e-12 else np.nan,
            "Max drawdown": drawdown(returns).min()}
