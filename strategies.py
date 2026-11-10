import numpy as np
import pandas as pd


def sma_crossover(data: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.Series:
    """Classic 50/200 SMA crossover."""
    fast_ma = data["Close"].rolling(fast).mean()
    slow_ma = data["Close"].rolling(slow).mean()
    signal = pd.Series(0, index=data.index, dtype=int)
    valid = fast_ma.notna() & slow_ma.notna()
    signal[valid] = np.where(fast_ma[valid] > slow_ma[valid], 1, -1)
    return signal


def vol_target_momentum(data: pd.DataFrame, lookback: int = 60,
                        target_vol: float = 0.10) -> pd.Series:
    """
    Momentum + volatility targeting.
    Long if 60-day return > 0, sized to target 10% annualized vol.
    This is what real CTAs do.
    """
    rets = data["Close"].pct_change()
    mom = data["Close"].pct_change(lookback)
    vol = rets.rolling(lookback).std() * np.sqrt(252)
    signal = pd.Series(0, index=data.index, dtype=int)
    valid = mom.notna() & vol.notna() & (vol > 0)
    signal[valid] = np.where(mom[valid] > 0, 1, -1)
    return signal