import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from engine import BacktestEngine
from strategies import sma_crossover, vol_target_momentum


def load_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    return df


def walk_forward(data: pd.DataFrame, strategy_fn, n_splits: int = 4):
    """
    Split into N in-sample / out-of-sample chunks.
    Report OOS Sharpe — this is the only number that matters.
    """
    n = len(data)
    chunk = n // (n_splits + 1)
    results = []
    for i in range(1, n_splits + 1):
        # In-sample: first i chunks, out-of-sample: chunk i+1
        is_end = chunk * i
        oos_end = chunk * (i + 1)
        oos_data = data.iloc[is_end:oos_end].copy()
        if len(oos_data) < 100:
            continue
        oos_data["signal"] = strategy_fn(oos_data)
        engine = BacktestEngine(oos_data).run()
        s = engine.stats()
        results.append(s)
    return results


def main():
    print("=== Institutional Backtester ===\n")

    data = load_data("SPY", "2010-01-01", "2024-12-31")
    print(f"Loaded {len(data)} bars of SPY\n")

    # ---- Buy & Hold benchmark ----
    bh_return = data["Close"].iloc[-1] / data["Close"].iloc[0] - 1
    bh_vol = data["Close"].pct_change().std() * np.sqrt(252)
    bh_sharpe = (bh_return / (len(data) / 252) - 0.02) / bh_vol

    print(f"Buy & Hold: Return={bh_return:.2%}, Vol={bh_vol:.2%}, "
          f"Sharpe={bh_sharpe:.2f}\n")

    # ---- Strategy backtests ----
    for name, fn in [
        ("SMA 50/200", sma_crossover),
        ("Vol-Target Momentum", vol_target_momentum),
    ]:
        data_s = data.copy()
        data_s["signal"] = fn(data_s)
        engine = BacktestEngine(data_s).run()
        s = engine.stats()

        print(f"--- {name} ---")
        print(f"  Total return:       {s['total_return']:>8.2%}")
        print(f"  Annualized return:  {s['annualized_return']:>8.2%}")
        print(f"  Annualized vol:     {s['annualized_vol']:>8.2%}")
        print(f"  Sharpe:             {s['sharpe']:>8.2f}")
        print(f"  Sortino:            {s['sortino']:>8.2f}")
        print(f"  Max drawdown:       {s['max_drawdown']:>8.2%}")
        print(f"  # trades:           {s['num_trades']:>8d}")
        print(f"  Win rate:           {s['win_rate']:>8.2%}")
        print(f"  Total commission:  ${s['total_commission']:>8.2f}")
        print()

        # ---- Walk-Forward ----
        wf = walk_forward(data, fn, n_splits=4)
        oos_sharpes = [r["sharpe"] for r in wf if r["num_trades"] > 0]
        if oos_sharpes:
            print(f"  Walk-forward OOS Sharpe (mean): {np.mean(oos_sharpes):.2f}")
            print(f"  Walk-forward OOS Sharpe (std):  {np.std(oos_sharpes):.2f}")
        print()

    # ---- Plot: strategy vs buy & hold ----
    data_s = data.copy()
    data_s["signal"] = sma_crossover(data_s)
    engine = BacktestEngine(data_s).run()

    equity = engine.equity_curve["equity"]
    bh_equity = engine.initial_capital * (data["Close"] / data["Close"].iloc[0])

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(equity.index, equity.values, label="SMA 50/200 (with costs)", lw=1.5)
    ax.plot(bh_equity.index, bh_equity.values, label="Buy & Hold SPY",
            lw=1.5, alpha=0.7)
    ax.set_title("SMA Crossover vs Buy & Hold — SPY 2010-2024 (with costs)")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results.png", dpi=120)
    print("Saved results.png")


if __name__ == "__main__":
    main()