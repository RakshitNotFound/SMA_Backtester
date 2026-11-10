📈 Institutional Backtester
A realistic, bias-aware backtesting framework for evaluating trading strategies on historical market data — with proper handling of trading costs, look-ahead bias, and out-of-sample validation.

Overview
This project simulates how a trading strategy would have performed historically, accounting for real-world frictions (commission, slippage, spread) and statistical traps (look-ahead bias, overfitting) that make most amateur backtests lie.

It answers one question:

"If I had followed this trading rule since 2010, paying real costs, would I have beaten simply buying and holding SPY?"

Features
✅ No look-ahead bias — signals are lagged by one bar (shift(1)) so trades execute on yesterday's signal at today's price

✅ Realistic execution costs — commission per share, slippage (bps), and bid-ask spread all modeled separately

✅ Risk-adjusted metrics — Sharpe, Sortino, max drawdown, annualized return/vol, win rate, trade P&L

✅ Walk-forward validation — out-of-sample testing to catch overfitting

✅ Benchmark comparison — every strategy is compared against Buy & Hold SPY

✅ Equity curve visualization — strategy vs benchmark plotted to results.png

✅ Clean, extensible code — modular engine, strategies, and runner

Project Structure
text
.
├── engine.py         # BacktestEngine — the simulator + performance stats
├── strategies.py     # Trading signal generators (SMA crossover, momentum)
├── main.py           # Runner: loads data, runs backtests, plots results
└── results.png       # Output chart (generated on run)
How It Works
1. engine.py — The Backtest Engine
BacktestEngine replays historical price data bar-by-bar:

Parameter	Default	Meaning
initial_capital	$100,000	Starting cash
commission_per_share	$0.005	Commission per share traded
slippage_bps	2.0	Price slippage in basis points
spread_bps	1.0	Bid-ask spread in basis points
position_fraction	0.95	Fraction of equity deployed per trade
Execution logic:

Signal +1 → enter long (if flat)

Signal -1 → exit to cash (if holding)

All signals lagged one day to prevent look-ahead

Execution price adjusted for half-spread + slippage

Metrics computed (stats()):

Total & annualized return

Annualized volatility

Sharpe ratio (vs. 2% risk-free rate)

Sortino ratio (downside-only volatility)

Max drawdown

Number of round-trip trades, win rate, average trade P&L

Total commissions paid

2. strategies.py — Trading Rules
Strategy	Description
sma_crossover	Classic 50/200 SMA golden cross — long when fast MA > slow MA
vol_target_momentum	Long when 60-day momentum is positive
Both return a pd.Series of signals: +1 (long), -1 (exit), 0 (neutral).

3. main.py — The Runner
Downloads SPY daily data (2010–2024) via yfinance

Computes Buy & Hold benchmark stats

Runs each strategy through the engine and prints results

Performs walk-forward out-of-sample testing (4 splits)

Plots the SMA strategy equity curve vs Buy & Hold → results.png

Installation
bash
pip install numpy pandas yfinance matplotlib
Usage
bash
python main.py
Sample output:

text
=== Institutional Backtester ===

Loaded 3773 bars of SPY

Buy & Hold: Return=...%, Vol=...%, Sharpe=...

--- SMA 50/200 ---
  Total return:       ...
  Annualized return:  ...
  Sharpe:             ...
  Sortino:            ...
  Max drawdown:       ...
  # trades:           ...
  Win rate:           ...

  Walk-forward OOS Sharpe (mean): ...
  Walk-forward OOS Sharpe (std):  ...
The equity curve plot is saved to results.png.

Adding Your Own Strategy
Just write a function that takes a pd.DataFrame (with a Close column) and returns a signal pd.Series:

python
def my_strategy(data: pd.DataFrame) -> pd.Series:
    signal = pd.Series(0, index=data.index, dtype=int)
    # your logic here — set +1 to buy, -1 to sell
    return signal
Then plug it into main.py:

python
from strategies import my_strategy
engine = BacktestEngine(data_s).run()
