import numpy as np
import pandas as pd
import ta

class MACDRSITrendFollower:
    """MACD trend confirmation with RSI oversold filter."""
    
    param_grid = {
        "macd_fast": [8, 12],
        "macd_slow": [21, 26], 
        "macd_signal": [9],
        "rsi_thresh_high": [65]
    }
    
    @staticmethod
    def entries(price: pd.Series, macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9, rsi_thresh_high: int = 65) -> pd.Series:
        # MACD
        macd_line = ta.trend.macd(price, window_fast=macd_fast, window_slow=macd_slow)
        macd_signal_line = ta.trend.macd_signal(price, window_fast=macd_fast, window_slow=macd_slow, window_sign=macd_signal)
        
        # RSI
        rsi = ta.momentum.rsi(price, window=14)
        
        # Entry: MACD line > signal line and RSI < 30 (oversold in uptrend)
        macd_bullish = macd_line > macd_signal_line
        rsi_oversold = rsi < 30
        
        return macd_bullish & rsi_oversold
    
    @staticmethod
    def exits(price: pd.Series, macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9, rsi_thresh_high: int = 65) -> pd.Series:
        # MACD
        macd_line = ta.trend.macd(price, window_fast=macd_fast, window_slow=macd_slow)
        macd_signal_line = ta.trend.macd_signal(price, window_fast=macd_fast, window_slow=macd_slow, window_sign=macd_signal)
        
        # RSI
        rsi = ta.momentum.rsi(price, window=14)
        
        # Exit: MACD line < signal line or RSI > threshold
        macd_bearish = macd_line < macd_signal_line
        rsi_overbought = rsi > rsi_thresh_high
        
        return macd_bearish | rsi_overbought
