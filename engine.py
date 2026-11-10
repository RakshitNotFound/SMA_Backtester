import numpy as np
import pandas as pd


class BacktestEngine:


    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 100_000.0,
        commission_per_share: float = 0.005,
        slippage_bps: float = 2.0,
        spread_bps: float = 1.0,
        position_fraction: float = 0.95,
    ):
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.commission_per_share = commission_per_share
        self.slippage_bps = slippage_bps / 10_000
        self.spread_bps = spread_bps / 10_000
        self.position_fraction = position_fraction

        self.trades = []
        self.equity_curve = None

    def _execute_price(self, price: float, side: str) -> float:
        """Apply half-spread + slippage to the reference price."""
        cost = price * (self.spread_bps / 2 + self.slippage_bps)
        return price + cost if side == "BUY" else price - cost

    def run(self):
        cash = self.initial_capital
        position = 0.0  # shares
        equity_history = []

        self.data["Executed_Signal"] = (
            self.data["signal"].shift(1).fillna(0).astype(int)
        )

        for date, row in self.data.iterrows():
            price = float(row["Close"])
            signal = int(row["Executed_Signal"])

            # Mark-to-market BEFORE the trade
            equity = cash + position * price

            # ---- ENTRY ----
            if signal == 1 and position == 0:
                target_notional = equity * self.position_fraction
                exec_px = self._execute_price(price, "BUY")
                shares = int(target_notional / exec_px)
                if shares > 0:
                    cost = shares * exec_px
                    commission = shares * self.commission_per_share
                    if cost + commission <= cash:
                        cash -= cost + commission
                        position = shares
                        self.trades.append({
                            "date": date, "type": "BUY",
                            "price": exec_px, "shares": shares,
                            "commission": commission,
                        })

            # ---- EXIT ----
            elif signal == -1 and position > 0:
                exec_px = self._execute_price(price, "SELL")
                proceeds = position * exec_px
                commission = position * self.commission_per_share
                cash += proceeds - commission
                self.trades.append({
                    "date": date, "type": "SELL",
                    "price": exec_px, "shares": position,
                    "commission": commission,
                })
                position = 0.0

            # Mark-to-market AFTER the trade
            equity = cash + position * price
            equity_history.append({"date": date, "equity": equity})

        self.equity_curve = pd.DataFrame(equity_history).set_index("date")
        self.data["Portfolio_Value"] = self.equity_curve["equity"].values
        self.data["Returns"] = self.data["Portfolio_Value"].pct_change().fillna(0)
        return self

    def stats(self) -> dict:
        eq = self.equity_curve["equity"]
        rets = self.data["Returns"]

        total_return = eq.iloc[-1] / self.initial_capital - 1
        days = len(eq)
        ann_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
        ann_vol = rets.std() * np.sqrt(252)
        sharpe = (ann_return - 0.02) / ann_vol if ann_vol > 0 else 0

        # Sortino: downside volatility only
        downside = rets[rets < 0].std() * np.sqrt(252)
        sortino = (ann_return - 0.02) / downside if downside > 0 else 0

        # Max drawdown
        cum = (1 + rets).cumprod()
        peak = cum.cummax()
        drawdown = (cum / peak) - 1
        max_dd = drawdown.min()

        # Trade stats
        round_trips = []
        buy = None
        for t in self.trades:
            if t["type"] == "BUY":
                buy = t
            elif t["type"] == "SELL" and buy is not None:
                pnl = (
                    (t["price"] - buy["price"]) * t["shares"]
                    - buy["commission"] - t["commission"]
                )
                round_trips.append(pnl)
                buy = None

        wins = [p for p in round_trips if p > 0]
        total_commission = sum(t["commission"] for t in self.trades)

        return {
            "total_return": total_return,
            "annualized_return": ann_return,
            "annualized_vol": ann_vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_dd,
            "num_trades": len(round_trips),
            "win_rate": len(wins) / len(round_trips) if round_trips else 0,
            "avg_trade_pnl": np.mean(round_trips) if round_trips else 0,
            "total_commission": total_commission,
            "final_equity": eq.iloc[-1],
        }