import numpy as np
import pandas as pd
import ta

class EMAATRMeanReversion:
    """EMA crossover with ATR-based mean reversion entries and exits."""
    
    param_grid = {
        "fast_len": [8, 13, 21],
        "slow_len": [100, 150, 200], 
        "atr_mul": [1.0, 1.5]
    }
    
    @staticmethod
    def entries(price: pd.Series, fast_len: int = 13, slow_len: int = 100, atr_mul: float = 1.5) -> pd.Series:
        fast_ema = ta.trend.ema_indicator(price, window=fast_len)
        slow_ema = ta.trend.ema_indicator(price, window=slow_len)
        atr = ta.volatility.average_true_range(price, price, price, window=14)
        
        # Fast EMA crosses above slow EMA
        ema_cross_up = (fast_ema > slow_ema) & (fast_ema.shift(1) <= slow_ema.shift(1))
        
        # Price is more than atr_mul * ATR below slow EMA (oversold)
        oversold = price < (slow_ema - atr_mul * atr)
        
        return ema_cross_up & oversold
    
    @staticmethod
    def exits(price: pd.Series, fast_len: int = 13, slow_len: int = 100, atr_mul: float = 1.5) -> pd.Series:
        fast_ema = ta.trend.ema_indicator(price, window=fast_len)
        slow_ema = ta.trend.ema_indicator(price, window=slow_len)
        atr = ta.volatility.average_true_range(price, price, price, window=14)
        
        # Fast EMA crosses below slow EMA
        ema_cross_down = (fast_ema < slow_ema) & (fast_ema.shift(1) >= slow_ema.shift(1))
        
        # Price hits profit target (slow EMA + atr_mul * ATR)
        profit_target = price > (slow_ema + atr_mul * atr)
        
        return ema_cross_down | profit_target
