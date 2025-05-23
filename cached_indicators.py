#!/usr/bin/env python
"""
Cached indicator wrappers for high-performance backtesting
All indicators are computed once and cached across runs
"""
import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Optional, Union, Dict, Any
from functools import wraps
import talib

from factor_store import get_factor_store
from data_manager import DataManager


# Global data reference (set by backtester)
_current_data = None
_current_symbol = None
_current_timeframe = None


def set_current_data(data: pd.DataFrame, symbol: str, timeframe: str):
    """Set the current data context for indicator computation"""
    global _current_data, _current_symbol, _current_timeframe
    _current_data = data
    _current_symbol = symbol
    _current_timeframe = timeframe


def get_current_data():
    """Get current data context"""
    if _current_data is None:
        raise RuntimeError("No data context set. Call set_current_data() first.")
    return _current_data, _current_symbol, _current_timeframe


def cached_indicator(indicator_type: str):
    """Decorator for creating cached indicators"""
    def decorator(func):
        @wraps(func)
        def wrapper(**kwargs):
            data, symbol, timeframe = get_current_data()
            factor_store = get_factor_store()
            
            # Create compute function that captures current data
            def compute_func():
                return func(data, **kwargs)
            
            # Get from cache or compute
            return factor_store.get_factor(
                indicator_type=indicator_type,
                parameters=kwargs,
                symbol=symbol,
                timeframe=timeframe,
                compute_func=compute_func
            )
        
        # Add metadata
        wrapper.indicator_type = indicator_type
        wrapper.is_cached = True
        
        return wrapper
    return decorator


# -----------------------------------------------------------------------------
# Moving Averages
# -----------------------------------------------------------------------------

@cached_indicator('sma')
def sma(data: pd.DataFrame, period: int) -> pd.Series:
    """Simple Moving Average"""
    if 'close' in data.columns:
        close = data['close']
    else:
        close = data  # Assume it's a Series
    
    # Ensure period is integer
    period = int(period)
    
    return vbt.MA.run(close, period, short_name=f'sma_{period}').ma.squeeze()


@cached_indicator('ema')
def ema(data: pd.DataFrame, period: int) -> pd.Series:
    """Exponential Moving Average"""
    if 'close' in data.columns:
        close = data['close']
    else:
        close = data
    
    # Ensure period is integer
    period = int(period)
    
    return vbt.MA.run(close, period, ewm=True, short_name=f'ema_{period}').ma.squeeze()


@cached_indicator('wma')
def wma(data: pd.DataFrame, period: int) -> pd.Series:
    """Weighted Moving Average"""
    if 'close' in data.columns:
        close = data['close']
    else:
        close = data
    
    return talib.WMA(close.values, timeperiod=period)


# Convenience aliases
def ema_fast(period: int) -> pd.Series:
    """Fast EMA helper"""
    return ema(period=period)


def ema_slow(period: int) -> pd.Series:
    """Slow EMA helper"""
    return ema(period=period)


def sma_fast(period: int) -> pd.Series:
    """Fast SMA helper"""
    return sma(period=period)


def sma_slow(period: int) -> pd.Series:
    """Slow SMA helper"""
    return sma(period=period)


# -----------------------------------------------------------------------------
# Momentum Indicators
# -----------------------------------------------------------------------------

@cached_indicator('rsi')
def rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Relative Strength Index"""
    if 'close' in data.columns:
        close = data['close']
    else:
        close = data
    
    # Ensure period is integer (vectorbt requirement)
    period = int(period)
    
    return vbt.RSI.run(close, period).rsi.squeeze()


@cached_indicator('macd')
def macd(data: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, 
         signal_period: int = 9) -> Dict[str, pd.Series]:
    """MACD - returns dict with 'macd', 'signal', 'histogram' """
    if 'close' in data.columns:
        close = data['close']
    else:
        close = data
    
    macd_result = vbt.MACD.run(
        close, 
        fast_period=fast_period,
        slow_period=slow_period,
        signal_period=signal_period
    )
    
    return {
        'macd': macd_result.macd.squeeze(),
        'signal': macd_result.macd_signal.squeeze(),
        'histogram': macd_result.macd_diff.squeeze()
    }


@cached_indicator('stochastic')
def stochastic(data: pd.DataFrame, k_period: int = 14, d_period: int = 3,
               smooth_k: int = 3) -> Dict[str, pd.Series]:
    """Stochastic Oscillator"""
    high = data['high']
    low = data['low']
    close = data['close']
    
    k, d = talib.STOCH(
        high.values, low.values, close.values,
        fastk_period=k_period,
        slowk_period=smooth_k,
        slowk_matype=0,
        slowd_period=d_period,
        slowd_matype=0
    )
    
    return {
        'k': pd.Series(k, index=close.index),
        'd': pd.Series(d, index=close.index)
    }


@cached_indicator('adx')
def adx(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index"""
    high = data['high']
    low = data['low']
    close = data['close']
    
    result = talib.ADX(high.values, low.values, close.values, timeperiod=period)
    return pd.Series(result, index=close.index)


# -----------------------------------------------------------------------------
# Volatility Indicators
# -----------------------------------------------------------------------------

@cached_indicator('atr')
def atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range"""
    high = data['high']
    low = data['low']
    close = data['close']
    
    return vbt.ATR.run(high, low, close, period).atr.squeeze()


@cached_indicator('bollinger_bands')
def bollinger_bands(data: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
    """Bollinger Bands"""
    if 'close' in data.columns:
        close = data['close']
    else:
        close = data
    
    bb_result = vbt.BBANDS.run(close, period, alpha=std_dev)
    
    return {
        'upper': bb_result.upper.squeeze(),
        'middle': bb_result.middle.squeeze(),
        'lower': bb_result.lower.squeeze(),
        'bandwidth': bb_result.bandwidth.squeeze() if hasattr(bb_result, 'bandwidth') else (bb_result.upper - bb_result.lower).squeeze()
    }


@cached_indicator('keltner_channels')
def keltner_channels(data: pd.DataFrame, period: int = 20, atr_mult: float = 2.0) -> Dict[str, pd.Series]:
    """Keltner Channels"""
    close = data['close']
    
    # Middle line is EMA
    middle = ema(period=period)
    
    # Get ATR
    atr_value = atr(period=period)
    
    return {
        'upper': middle + (atr_mult * atr_value),
        'middle': middle,
        'lower': middle - (atr_mult * atr_value)
    }


# -----------------------------------------------------------------------------
# Volume Indicators
# -----------------------------------------------------------------------------

@cached_indicator('obv')
def obv(data: pd.DataFrame) -> pd.Series:
    """On Balance Volume"""
    close = data['close']
    volume = data['volume']
    
    return vbt.OBV.run(close, volume).obv.squeeze()


@cached_indicator('vwap')
def vwap(data: pd.DataFrame) -> pd.Series:
    """Volume Weighted Average Price"""
    high = data['high']
    low = data['low']
    close = data['close']
    volume = data['volume']
    
    typical_price = (high + low + close) / 3
    return (typical_price * volume).cumsum() / volume.cumsum()


@cached_indicator('mfi')
def mfi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Money Flow Index"""
    high = data['high']
    low = data['low']
    close = data['close']
    volume = data['volume']
    
    result = talib.MFI(high.values, low.values, close.values, volume.values, timeperiod=period)
    return pd.Series(result, index=close.index)


# -----------------------------------------------------------------------------
# Trend Indicators
# -----------------------------------------------------------------------------

@cached_indicator('supertrend')
def supertrend(data: pd.DataFrame, period: int = 7, multiplier: float = 3.0) -> Dict[str, pd.Series]:
    """Supertrend Indicator"""
    high = data['high']
    low = data['low']
    close = data['close']
    
    # Calculate ATR
    atr_value = atr(period=period)
    
    # Calculate basic bands
    hl_avg = (high + low) / 2
    basic_upper = hl_avg + multiplier * atr_value
    basic_lower = hl_avg - multiplier * atr_value
    
    # Initialize
    supertrend = pd.Series(index=close.index, dtype=float)
    direction = pd.Series(index=close.index, dtype=float)
    
    for i in range(period, len(close)):
        # Calculate final bands
        if i == period:
            final_upper = basic_upper.iloc[i]
            final_lower = basic_lower.iloc[i]
        else:
            final_upper = basic_upper.iloc[i] if basic_upper.iloc[i] < supertrend.iloc[i-1] or close.iloc[i-1] > supertrend.iloc[i-1] else supertrend.iloc[i-1]
            final_lower = basic_lower.iloc[i] if basic_lower.iloc[i] > supertrend.iloc[i-1] or close.iloc[i-1] < supertrend.iloc[i-1] else supertrend.iloc[i-1]
        
        # Determine trend
        if i == period:
            if close.iloc[i] <= final_upper:
                supertrend.iloc[i] = final_upper
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = final_lower
                direction.iloc[i] = 1
        else:
            if supertrend.iloc[i-1] == final_upper:
                supertrend.iloc[i] = final_lower if close.iloc[i] > final_upper else final_upper
                direction.iloc[i] = 1 if close.iloc[i] > final_upper else -1
            else:
                supertrend.iloc[i] = final_upper if close.iloc[i] < final_lower else final_lower
                direction.iloc[i] = -1 if close.iloc[i] < final_lower else 1
    
    return {
        'supertrend': supertrend,
        'direction': direction
    }


@cached_indicator('ichimoku')
def ichimoku(data: pd.DataFrame, conversion: int = 9, base: int = 26, 
             span_b: int = 52, displacement: int = 26) -> Dict[str, pd.Series]:
    """Ichimoku Cloud"""
    high = data['high']
    low = data['low']
    
    # Conversion Line (Tenkan-sen)
    high_conversion = high.rolling(window=conversion).max()
    low_conversion = low.rolling(window=conversion).min()
    conversion_line = (high_conversion + low_conversion) / 2
    
    # Base Line (Kijun-sen)
    high_base = high.rolling(window=base).max()
    low_base = low.rolling(window=base).min()
    base_line = (high_base + low_base) / 2
    
    # Leading Span A (Senkou Span A)
    span_a = ((conversion_line + base_line) / 2).shift(displacement)
    
    # Leading Span B (Senkou Span B)
    high_span = high.rolling(window=span_b).max()
    low_span = low.rolling(window=span_b).min()
    span_b_line = ((high_span + low_span) / 2).shift(displacement)
    
    # Lagging Span (Chikou Span)
    lagging_span = data['close'].shift(-displacement)
    
    return {
        'conversion': conversion_line,
        'base': base_line,
        'span_a': span_a,
        'span_b': span_b_line,
        'lagging': lagging_span
    }


# -----------------------------------------------------------------------------
# Pattern Recognition (cached)
# -----------------------------------------------------------------------------

@cached_indicator('pivot_points')
def pivot_points(data: pd.DataFrame) -> Dict[str, pd.Series]:
    """Calculate Pivot Points"""
    high = data['high']
    low = data['low']
    close = data['close']
    
    # Standard pivot calculation
    pivot = (high + low + close) / 3
    
    # Support and resistance levels
    r1 = 2 * pivot - low
    r2 = pivot + (high - low)
    r3 = high + 2 * (pivot - low)
    
    s1 = 2 * pivot - high
    s2 = pivot - (high - low)
    s3 = low - 2 * (high - pivot)
    
    return {
        'pivot': pivot,
        'r1': r1, 'r2': r2, 'r3': r3,
        's1': s1, 's2': s2, 's3': s3
    }


# -----------------------------------------------------------------------------
# Composite Indicators
# -----------------------------------------------------------------------------

@cached_indicator('rsi_divergence')
def rsi_divergence(data: pd.DataFrame, period: int = 14, lookback: int = 20) -> pd.Series:
    """Detect RSI divergence"""
    close = data['close']
    rsi_values = rsi(period=period)
    
    divergence = pd.Series(0, index=close.index)
    
    for i in range(lookback, len(close)):
        # Price making higher highs but RSI making lower highs (bearish divergence)
        price_highs = close.iloc[i-lookback:i].rolling(3).max()
        rsi_highs = rsi_values.iloc[i-lookback:i].rolling(3).max()
        
        if len(price_highs.dropna()) > 1 and len(rsi_highs.dropna()) > 1:
            if price_highs.iloc[-1] > price_highs.iloc[-2] and rsi_highs.iloc[-1] < rsi_highs.iloc[-2]:
                divergence.iloc[i] = -1
            
            # Price making lower lows but RSI making higher lows (bullish divergence)
            price_lows = close.iloc[i-lookback:i].rolling(3).min()
            rsi_lows = rsi_values.iloc[i-lookback:i].rolling(3).min()
            
            if price_lows.iloc[-1] < price_lows.iloc[-2] and rsi_lows.iloc[-1] > rsi_lows.iloc[-2]:
                divergence.iloc[i] = 1
    
    return divergence


# -----------------------------------------------------------------------------
# Multi-timeframe Indicators
# -----------------------------------------------------------------------------

def mtf_indicator(indicator_func, higher_timeframe: str, **kwargs):
    """
    Get indicator from higher timeframe
    Note: This requires data for the higher timeframe to be available
    """
    data, symbol, current_tf = get_current_data()
    
    # This would need integration with data manager to fetch higher TF data
    # For now, we'll use resampling as approximation
    
    # Map timeframes to pandas frequencies
    tf_map = {
        '1m': '1T', '5m': '5T', '15m': '15T', '30m': '30T',
        '1h': '1H', '4h': '4H', '1d': '1D'
    }
    
    if current_tf not in tf_map or higher_timeframe not in tf_map:
        raise ValueError(f"Unsupported timeframe: {current_tf} or {higher_timeframe}")
    
    # Resample data
    resampled = data.resample(tf_map[higher_timeframe]).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    # Set context for higher timeframe
    original_data = _current_data
    original_tf = _current_timeframe
    
    set_current_data(resampled, symbol, higher_timeframe)
    
    try:
        # Calculate indicator on higher timeframe
        result = indicator_func(**kwargs)
        
        # Reindex to original timeframe
        result = result.reindex(data.index, method='ffill')
    finally:
        # Restore original context
        set_current_data(original_data, symbol, original_tf)
    
    return result


# -----------------------------------------------------------------------------
# Cache Management Functions
# -----------------------------------------------------------------------------

def warm_cache(indicators: Dict[str, list], symbol: str, timeframe: str, data: pd.DataFrame):
    """Pre-compute common indicators to warm the cache"""
    set_current_data(data, symbol, timeframe)
    
    results = {}
    
    # Moving averages
    if 'sma' in indicators:
        for period in indicators['sma']:
            results[f'sma_{period}'] = sma(period=period)
    
    if 'ema' in indicators:
        for period in indicators['ema']:
            results[f'ema_{period}'] = ema(period=period)
    
    # Momentum
    if 'rsi' in indicators:
        for period in indicators['rsi']:
            results[f'rsi_{period}'] = rsi(period=period)
    
    # Volatility
    if 'atr' in indicators:
        for period in indicators['atr']:
            results[f'atr_{period}'] = atr(period=period)
    
    if 'bb' in indicators:
        for period, std in indicators['bb']:
            results[f'bb_{period}_{std}'] = bollinger_bands(period=period, std_dev=std)
    
    return results


def get_cache_info():
    """Get information about cached indicators"""
    factor_store = get_factor_store()
    return factor_store.get_cache_stats()


def clear_cache(symbol: Optional[str] = None, timeframe: Optional[str] = None):
    """Clear cached indicators"""
    factor_store = get_factor_store()
    
    if symbol and timeframe:
        factor_store.invalidate_factors(symbol, timeframe)
    else:
        # Clear all - would need to implement in factor_store
        print("Full cache clear not implemented yet")