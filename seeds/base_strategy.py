#!/usr/bin/env python
"""Base strategy class for inheritance"""
import pandas as pd
import numpy as np
import talib

class BaseStrategy:
    """
    Base class for all strategies.
    Subclasses should implement entries() and exits() methods.
    """
    
    # Will be set by batch_backtest.py
    price = None
    high = None
    low = None
    volume = None
    
    def __init__(self):
        """Initialize strategy"""
        pass
    
    def entries(self):
        """
        Generate entry signals.
        Returns: pd.Series of bool (True = enter position)
        """
        raise NotImplementedError("Subclass must implement entries()")
    
    def exits(self):
        """
        Generate exit signals.
        Returns: pd.Series of bool (True = exit position)
        """
        raise NotImplementedError("Subclass must implement exits()")
    
    def calculate_indicators(self):
        """Helper method to calculate common indicators"""
        indicators = {}
        
        # Price-based indicators
        indicators['sma_20'] = talib.SMA(self.price.values, timeperiod=20)
        indicators['sma_50'] = talib.SMA(self.price.values, timeperiod=50)
        indicators['ema_12'] = talib.EMA(self.price.values, timeperiod=12)
        indicators['ema_26'] = talib.EMA(self.price.values, timeperiod=26)
        
        # Momentum indicators
        indicators['rsi'] = talib.RSI(self.price.values, timeperiod=14)
        indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = talib.MACD(
            self.price.values, fastperiod=12, slowperiod=26, signalperiod=9
        )
        
        # Volatility indicators
        indicators['bb_upper'], indicators['bb_middle'], indicators['bb_lower'] = talib.BBANDS(
            self.price.values, timeperiod=20, nbdevup=2, nbdevdn=2
        )
        indicators['atr'] = talib.ATR(self.high.values, self.low.values, self.price.values, timeperiod=14)
        
        # Volume indicators
        if self.volume is not None:
            indicators['volume_sma'] = talib.SMA(self.volume.values, timeperiod=20)
        
        # Convert to pandas Series with proper index
        for key, value in indicators.items():
            indicators[key] = pd.Series(value, index=self.price.index)
        
        return indicators