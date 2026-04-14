from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd
from shinybroker import Contract, fetch_historical_data

ETF_UNIVERSE = [
    "SPY",
    "QQQ",
    "SMH",
    "SOXX",
    "ARKK",
    "XLE",
    "XLK",
    "XLF",
    "XBI",
    "GLD",
    "TLT",
]
TRAINING_DAYS = 252
TEST_DAYS = 63
INITIAL_CAPITAL = 100_000.0
POSITION_SIZE = 100
CAPITAL_FRACTION_PER_TRADE = 0.95
RISK_FREE_RATE = 0.0

BREAKOUT_LOOKBACK_OPTIONS = [20, 55]
ATR_WINDOW_OPTIONS = [14]
STOP_ATR_OPTIONS = [2.0, 2.5]
TARGET_ATR_OPTIONS = [2.0, 4.0]
MAX_HOLD_DAYS_OPTIONS = [10, 20]
TREND_FILTER_OPTIONS = ["none", "sma50", "sma200"]
VOLUME_FILTER_OPTIONS = [False]

IBKR_HOST = "127.0.0.1"
IBKR_PORT = 7497
IBKR_CLIENT_ID = 9999
HISTORY_DURATION = "3 Y"
BAR_SIZE = "1 day"


@dataclass(frozen=True)
class StrategyParams:
    breakout_lookback: int
    atr_window: int
    stop_atr: float
    target_atr: float
    max_hold_days: int
    trend_filter: str
    volume_filter: bool


def fetch_etf_history(symbol: str) -> pd.DataFrame:
    contract = Contract(
        {
            "symbol": symbol,
            "secType": "STK",
            "exchange": "SMART",
            "currency": "USD",
        }
    )

    data = fetch_historical_data(
        contract=contract,
        durationStr=HISTORY_DURATION,
        barSizeSetting=BAR_SIZE,
        whatToShow="TRADES",
        useRTH=True,
        host=IBKR_HOST,
        port=IBKR_PORT,
        client_id=IBKR_CLIENT_ID,
        timeout=10,
    )

    if isinstance(data, dict) and "hst_dta" in data:
        df = pd.DataFrame(data["hst_dta"]).copy()
    else:
        df = pd.DataFrame(data).copy()

    if df.empty:
        raise RuntimeError(f"No historical data returned for {symbol}.")

    df = df.rename(columns={"timestamp": "date"})
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[["date", "open", "high", "low", "close", "volume"]].dropna().sort_values("date")
    df["symbol"] = symbol
    return df.reset_index(drop=True)


def average_true_range(df: pd.DataFrame, window: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window=window).mean()


def detect_breakouts(df: pd.DataFrame, breakout_lookback: int, atr_window: int) -> pd.DataFrame:
    """
    Identify long breakouts using a Donchian-style rule.

    A breakout is flagged when today's close is above the highest high from the
    previous `breakout_lookback` sessions. The signal is intended to be acted on
    at the next day's open. ATR is calculated alongside the signal so that the
    stop-loss and profit target use only information available at signal time.
    """
    out = df.copy()
    out["donchian_high"] = out["high"].shift(1).rolling(breakout_lookback).max()
    out["atr"] = average_true_range(out, atr_window)
    out["sma50"] = out["close"].rolling(50).mean()
    out["sma200"] = out["close"].rolling(200).mean()
    out["avg_volume_20"] = out["volume"].rolling(20).mean()
    out["breakout_signal"] = (
        (out["close"] > out["donchian_high"])
        & out["donchian_high"].notna()
        & out["atr"].notna()
    )
    return out


def apply_signal_filters(signal_df: pd.DataFrame, params: StrategyParams) -> pd.Series:
    signal = signal_df["breakout_signal"].copy()

    if params.trend_filter == "sma50":
        signal &= signal_df["close"] > signal_df["sma50"]
    elif params.trend_filter == "sma200":
        signal &= signal_df["close"] > signal_df["sma200"]

    if params.volume_filter:
        signal &= signal_df["volume"] > signal_df["avg_volume_20"]

    return signal.fillna(False)


def build_equity_curve(
    price_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
) -> pd.DataFrame:
    equity = pd.DataFrame({"date": pd.to_datetime(price_df["date"])}).copy()
    equity["cash"] = initial_capital
    equity["position_shares"] = 0
    equity["close"] = price_df["close"].values

    if not trades_df.empty:
        for trade in trades_df.to_dict("records"):
            entry_mask = equity["date"] >= trade["entry_date"]
            exit_mask = equity["date"] > trade["exit_date"]
            equity.loc[entry_mask, "cash"] -= trade["entry_price"] * trade["size"]
            equity.loc[exit_mask, "cash"] += trade["exit_price"] * trade["size"]
            equity.loc[entry_mask, "position_shares"] += trade["size"]
            equity.loc[exit_mask, "position_shares"] -= trade["size"]

    equity["market_value"] = equity["position_shares"] * equity["close"]
    equity["equity"] = equity["cash"] + equity["market_value"]
    equity["daily_return"] = equity["equity"].pct_change().fillna(0.0)
    equity["cum_return"] = equity["equity"] / initial_capital - 1.0
    equity["running_peak"] = equity["equity"].cummax()
    equity["drawdown"] = equity["equity"] / equity["running_peak"] - 1.0
    return equity


def backtest_breakout_strategy(
    df: pd.DataFrame,
    params: StrategyParams,
    symbol: str,
    position_size: int = POSITION_SIZE,
    initial_capital: float = INITIAL_CAPITAL,
    entry_start_date: pd.Timestamp | None = None,
    entry_end_date: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal_df = detect_breakouts(df, params.breakout_lookback, params.atr_window).reset_index(drop=True)
    filtered_signal = apply_signal_filters(signal_df, params)
    trades: list[dict] = []
    i = 0
    current_capital = initial_capital

    while i < len(signal_df) - 1:
        row = signal_df.iloc[i]
        if not bool(filtered_signal.iloc[i]):
            i += 1
            continue

        entry_idx = i + 1
        entry_row = signal_df.iloc[entry_idx]
        entry_date = pd.Timestamp(entry_row["date"])
        if entry_start_date is not None and entry_date < entry_start_date:
            i += 1
            continue
        if entry_end_date is not None and entry_date > entry_end_date:
            i += 1
            continue
        atr_at_signal = float(row["atr"])
        entry_price = float(entry_row["open"])
        size = max(1, int((current_capital * CAPITAL_FRACTION_PER_TRADE) // entry_price))
        stop_price = entry_price - params.stop_atr * atr_at_signal
        target_price = entry_price + params.target_atr * atr_at_signal

        exit_price = float(signal_df.iloc[-1]["close"])
        exit_date = pd.Timestamp(signal_df.iloc[-1]["date"])
        exit_reason = "end_of_sample"

        for j in range(entry_idx, min(len(signal_df), entry_idx + params.max_hold_days + 1)):
            current = signal_df.iloc[j]
            if float(current["low"]) <= stop_price:
                exit_price = stop_price
                exit_date = pd.Timestamp(current["date"])
                exit_reason = "stop_loss"
                break
            if float(current["high"]) >= target_price:
                exit_price = target_price
                exit_date = pd.Timestamp(current["date"])
                exit_reason = "profit_target"
                break
            if j - entry_idx + 1 >= params.max_hold_days:
                exit_price = float(current["close"])
                exit_date = pd.Timestamp(current["date"])
                exit_reason = "timeout"
                break

        gross_return = (exit_price - entry_price) / entry_price
        pnl = (exit_price - entry_price) * size
        if exit_reason == "stop_loss":
            outcome = "Stop-loss triggered"
        elif exit_reason == "profit_target" or gross_return > 0:
            outcome = "Successful"
        else:
            outcome = "Timed out"
        current_capital += pnl

        trades.append(
            {
                "symbol": symbol,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": round(entry_price, 4),
                "exit_price": round(exit_price, 4),
                "size": size,
                "direction": "LONG",
                "gross_return": gross_return,
                "pnl": pnl,
                "holding_days": int((exit_date - pd.Timestamp(entry_row["date"])).days) + 1,
                "stop_price": round(stop_price, 4),
                "profit_target": round(target_price, 4),
                "exit_reason": exit_reason,
                "outcome": outcome,
                "breakout_lookback": params.breakout_lookback,
                "atr_window": params.atr_window,
                "stop_atr": params.stop_atr,
                "target_atr": params.target_atr,
                "max_hold_days": params.max_hold_days,
                "trend_filter": params.trend_filter,
                "volume_filter": params.volume_filter,
            }
        )

        while i < len(signal_df) and pd.Timestamp(signal_df.iloc[i]["date"]) <= exit_date:
            i += 1

    trades_df = pd.DataFrame(trades)
    equity = build_equity_curve(signal_df, trades_df, initial_capital=initial_capital)
    return trades_df, equity


def compute_performance_metrics(
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE,
) -> dict[str, float]:
    if trades.empty or equity.empty:
        return {
            "trade_count": 0,
            "average_return_per_trade": 0.0,
            "annualized_sharpe_ratio": 0.0,
            "annualized_sortino_ratio": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy_dollars": 0.0,
            "max_drawdown": 0.0,
            "total_return": 0.0,
        }

    daily_rf = risk_free_rate / 252
    excess = equity["daily_return"] - daily_rf
    downside = excess[excess < 0]
    excess_std = float(excess.std(ddof=1)) if len(excess) > 1 else 0.0
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    gross_profit = float(trades.loc[trades["pnl"] > 0, "pnl"].sum())
    gross_loss = float(-trades.loc[trades["pnl"] < 0, "pnl"].sum())

    sharpe = 0.0 if excess_std == 0 else float(excess.mean() / excess_std * sqrt(252))
    sortino = 0.0 if downside_std == 0 else float(excess.mean() / downside_std * sqrt(252))

    return {
        "trade_count": int(len(trades)),
        "average_return_per_trade": float(trades["gross_return"].mean()),
        "annualized_sharpe_ratio": sharpe,
        "annualized_sortino_ratio": sortino,
        "win_rate": float((trades["pnl"] > 0).mean()),
        "profit_factor": 0.0 if gross_loss == 0 else gross_profit / gross_loss,
        "expectancy_dollars": float(trades["pnl"].mean()),
        "max_drawdown": float(equity["drawdown"].min()),
        "total_return": float(equity["equity"].iloc[-1] / equity["equity"].iloc[0] - 1.0),
    }


def parameter_grid() -> list[StrategyParams]:
    return [
        StrategyParams(a, b, c, d, e, f, g)
        for a in BREAKOUT_LOOKBACK_OPTIONS
        for b in ATR_WINDOW_OPTIONS
        for c in STOP_ATR_OPTIONS
        for d in TARGET_ATR_OPTIONS
        for e in MAX_HOLD_DAYS_OPTIONS
        for f in TREND_FILTER_OPTIONS
        for g in VOLUME_FILTER_OPTIONS
    ]


def walk_forward_backtest(df: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test_trades_all: list[pd.DataFrame] = []
    test_equity_all: list[pd.DataFrame] = []
    windows: list[dict] = []

    for test_start in range(TRAINING_DAYS, len(df) - TEST_DAYS, TEST_DAYS):
        train = df.iloc[test_start - TRAINING_DAYS : test_start].reset_index(drop=True)
        test = df.iloc[test_start : test_start + TEST_DAYS].reset_index(drop=True)
        context = df.iloc[test_start - TRAINING_DAYS : test_start + TEST_DAYS].reset_index(drop=True)
        test_start_date = pd.Timestamp(test["date"].iloc[0])
        test_end_date = pd.Timestamp(test["date"].iloc[-1])

        best_score = -np.inf
        best_params: StrategyParams | None = None

        for params in parameter_grid():
            train_trades, train_equity = backtest_breakout_strategy(train, params, symbol)
            metrics = compute_performance_metrics(train_trades, train_equity)
            score = (
                metrics["annualized_sharpe_ratio"] * 2.0
                + metrics["average_return_per_trade"] * 100.0
                + metrics["win_rate"]
            )
            if metrics["trade_count"] >= 2 and score > best_score:
                best_score = score
                best_params = params

        if best_params is None:
            continue

        test_trades, test_equity = backtest_breakout_strategy(
            context,
            best_params,
            symbol,
            entry_start_date=test_start_date,
            entry_end_date=test_end_date,
        )
        test_equity = test_equity.loc[
            (pd.to_datetime(test_equity["date"]) >= test_start_date)
            & (pd.to_datetime(test_equity["date"]) <= test_end_date)
        ].reset_index(drop=True)
        if not test_trades.empty:
            test_trades = test_trades.assign(
                train_start=pd.Timestamp(train["date"].iloc[0]),
                train_end=pd.Timestamp(train["date"].iloc[-1]),
                test_start=test_start_date,
                test_end=test_end_date,
            )
        metrics = compute_performance_metrics(test_trades, test_equity)

        windows.append(
            {
                "symbol": symbol,
                "train_start": pd.Timestamp(train["date"].iloc[0]),
                "train_end": pd.Timestamp(train["date"].iloc[-1]),
                "test_start": pd.Timestamp(test["date"].iloc[0]),
                "test_end": pd.Timestamp(test["date"].iloc[-1]),
                "selected_breakout_lookback": best_params.breakout_lookback,
                "selected_stop_atr": best_params.stop_atr,
                "selected_target_atr": best_params.target_atr,
                "selected_max_hold_days": best_params.max_hold_days,
                "selected_trend_filter": best_params.trend_filter,
                "selected_volume_filter": best_params.volume_filter,
                "test_trade_count": metrics["trade_count"],
                "test_sharpe": metrics["annualized_sharpe_ratio"],
                "test_total_return": metrics["total_return"],
            }
        )
        test_trades_all.append(test_trades)
        test_equity_all.append(test)

    all_trades = pd.concat(test_trades_all, ignore_index=True) if test_trades_all else pd.DataFrame()
    stitched_prices = pd.concat(test_equity_all, ignore_index=True) if test_equity_all else pd.DataFrame()
    all_equity = build_equity_curve(stitched_prices, all_trades, initial_capital=INITIAL_CAPITAL)
    window_df = pd.DataFrame(windows)
    return all_trades, all_equity, window_df


def summarize_screen(results: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
    rows: list[dict] = []
    for symbol, (trades, equity, windows) in results.items():
        metrics = compute_performance_metrics(trades, equity)
        rows.append(
            {
                "symbol": symbol,
                "trade_count": metrics["trade_count"],
                "annualized_sharpe_ratio": metrics["annualized_sharpe_ratio"],
                "average_return_per_trade": metrics["average_return_per_trade"],
                "win_rate": metrics["win_rate"],
                "max_drawdown": metrics["max_drawdown"],
                "total_return": metrics["total_return"],
                "walk_forward_windows": int(len(windows)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        by=["annualized_sharpe_ratio", "trade_count", "total_return"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
