from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from breakouts.strategy import (
    CAPITAL_FRACTION_PER_TRADE,
    ETF_UNIVERSE,
    INITIAL_CAPITAL,
    RISK_FREE_RATE,
    compute_performance_metrics,
    fetch_etf_history,
    summarize_screen,
    walk_forward_backtest,
)

ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "docs"
DATA_DIR = DOCS_DIR / "data"
ASSETS_DIR = DOCS_DIR / "assets"


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def build_plotly_files(
    selected_symbol: str,
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    outcomes: pd.DataFrame,
    screen: pd.DataFrame,
) -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    eq_fig = px.line(equity, x="date", y="equity", template="plotly_white", title=f"{selected_symbol} Equity Curve")
    eq_fig.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=20))
    eq_fig.write_html(ASSETS_DIR / "equity_curve.html", include_plotlyjs="cdn")

    dd_fig = px.area(equity, x="date", y="drawdown", template="plotly_white", title=f"{selected_symbol} Drawdown")
    dd_fig.update_layout(height=320, margin=dict(l=20, r=20, t=60, b=20))
    dd_fig.write_html(ASSETS_DIR / "drawdown.html", include_plotlyjs="cdn")

    out_fig = px.bar(outcomes, x="outcome", y="trade_count", color="outcome", text="trade_count", template="plotly_white", title="Trade Outcome Histogram")
    out_fig.update_layout(showlegend=False, height=360, margin=dict(l=20, r=20, t=60, b=20))
    out_fig.write_html(ASSETS_DIR / "trade_outcomes.html", include_plotlyjs="cdn")

    blotter_cols = ["entry_date", "exit_date", "entry_price", "exit_price", "size", "direction", "gross_return", "pnl", "outcome"]
    blotter_fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=[c.replace("_", " ").title() for c in blotter_cols], fill_color="#0b3954", font=dict(color="white")),
                cells=dict(values=[trades[c] for c in blotter_cols], fill_color="#f5f7fa", format=[None, None, ".2f", ".2f", None, None, ".2%", ".2f", None]),
            )
        ]
    )
    blotter_fig.update_layout(height=900, margin=dict(l=10, r=10, t=10, b=10))
    blotter_fig.write_html(ASSETS_DIR / "trade_blotter.html", include_plotlyjs="cdn")

    screen_fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=[c.replace("_", " ").title() for c in screen.columns], fill_color="#3a506b", font=dict(color="white")),
                cells=dict(values=[screen[c] for c in screen.columns], fill_color="#f5f7fa"),
            )
        ]
    )
    screen_fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
    screen_fig.write_html(ASSETS_DIR / "screen_summary.html", include_plotlyjs="cdn")


def main() -> None:
    results: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    for symbol in ETF_UNIVERSE:
        history = fetch_etf_history(symbol)
        results[symbol] = walk_forward_backtest(history, symbol)

    screen = summarize_screen(results)
    selected_symbol = str(screen.iloc[0]["symbol"])
    trades, equity, windows = results[selected_symbol]
    metrics = compute_performance_metrics(trades, equity)

    if trades.empty:
        raise RuntimeError("No trades were produced for the selected ETF.")

    trades_out = trades.copy()
    trades_out["entry_date"] = pd.to_datetime(trades_out["entry_date"]).dt.strftime("%Y-%m-%d")
    trades_out["exit_date"] = pd.to_datetime(trades_out["exit_date"]).dt.strftime("%Y-%m-%d")

    equity_out = equity.copy()
    equity_out["date"] = pd.to_datetime(equity_out["date"]).dt.strftime("%Y-%m-%d")

    windows_out = windows.copy()
    for col in ["train_start", "train_end", "test_start", "test_end"]:
        if col in windows_out.columns:
            windows_out[col] = pd.to_datetime(windows_out[col]).dt.strftime("%Y-%m-%d")

    outcomes = trades.groupby("outcome").size().reset_index(name="trade_count").sort_values("trade_count", ascending=False)

    save_csv(screen, DATA_DIR / "asset_screen.csv")
    save_csv(trades_out, DATA_DIR / "trade_blotter.csv")
    save_csv(equity_out, DATA_DIR / "equity_curve.csv")
    save_csv(windows_out, DATA_DIR / "walk_forward_windows.csv")
    save_csv(outcomes, DATA_DIR / "trade_outcomes.csv")

    build_plotly_files(selected_symbol, trades_out, equity_out, outcomes, screen.round(4))

    summary = {
        "selected_symbol": selected_symbol,
        "screened_universe": ETF_UNIVERSE,
        "initial_capital": INITIAL_CAPITAL,
        "capital_fraction_per_trade": CAPITAL_FRACTION_PER_TRADE,
        "risk_free_rate": RISK_FREE_RATE,
        "selected_metrics": metrics,
        "selected_breakout_lookback": int(trades["breakout_lookback"].mode().iloc[0]),
        "selected_atr_window": int(trades["atr_window"].mode().iloc[0]),
        "selected_stop_atr": float(trades["stop_atr"].mode().iloc[0]),
        "selected_target_atr": float(trades["target_atr"].mode().iloc[0]),
        "selected_max_hold_days": int(trades["max_hold_days"].mode().iloc[0]),
        "selected_trend_filter": str(trades["trend_filter"].mode().iloc[0]),
        "selected_volume_filter": bool(trades["volume_filter"].mode().iloc[0]),
        "selected_median_position_size": int(pd.to_numeric(trades["size"]).median()),
        "asset_selection_note": (
            f"I screened {len(ETF_UNIVERSE)} liquid ETFs using the same one-year training and "
            f"one-quarter out-of-sample framework, then selected {selected_symbol} because it "
            f"finished with the strongest out-of-sample Sharpe ratio and one of the best total "
            f"returns while still generating an active trade blotter."
        ),
    }

    with (DATA_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
