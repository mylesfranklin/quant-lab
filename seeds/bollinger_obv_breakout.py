import numpy as np
import pandas as pd
import ta

class BollingerOBVBreakout:
    """Bollinger Band breakout filtered by OBV momentum."""
    
    param_grid = {
        "bb_window": [20, 30, 40],
        "std_mul": [2.0, 2.5],
        "obv_slope_window": [5]
    }
    
    @staticmethod
    def entries(price: pd.Series, bb_window: int = 20, std_mul: float = 2.0, obv_slope_window: int = 5) -> pd.Series:
        # Bollinger Bands
        bb_upper = ta.volatility.bollinger_hband(price, window=bb_window, window_dev=std_mul)
        
        # OBV and its slope
        volume = pd.Series(np.ones(len(price)), index=price.index)  # Assume unit volume
        obv = ta.volume.on_balance_volume(price, volume)
        obv_slope = obv.rolling(window=obv_slope_window).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == obv_slope_window else 0)
        
        # Entry: close breaks upper BB and OBV slope is positive
        bb_breakout = price > bb_upper
        obv_momentum = obv_slope > 0
        
        return bb_breakout & obv_momentum
    
    @staticmethod
    def exits(price: pd.Series, bb_window: int = 20, std_mul: float = 2.0, obv_slope_window: int = 5) -> pd.Series:
        # Bollinger Bands
        bb_middle = ta.volatility.bollinger_mavg(price, window=bb_window)
        
        # OBV and its slope
        volume = pd.Series(np.ones(len(price)), index=price.index)
        obv = ta.volume.on_balance_volume(price, volume)
        obv_slope = obv.rolling(window=obv_slope_window).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == obv_slope_window else 0)
        
        # Exit: price re-enters middle band or OBV slope turns negative
        return_to_middle = price <= bb_middle
        obv_negative = obv_slope < 0
        
        return return_to_middle | obv_negative
