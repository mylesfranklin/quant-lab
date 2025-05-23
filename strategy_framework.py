#!/usr/bin/env python
"""
Improved strategy framework with clear separation of concerns
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
import numpy as np
import vectorbt as vbt
from dataclasses import dataclass
from datetime import datetime


@dataclass
class MarketData:
    """Container for market data"""
    symbol: str
    timeframe: str
    open: pd.Series
    high: pd.Series
    low: pd.Series
    close: pd.Series
    volume: pd.Series
    
    @property
    def ohlcv(self) -> pd.DataFrame:
        """Return OHLCV DataFrame"""
        return pd.DataFrame({
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        })
    
    def __len__(self):
        return len(self.close)


@dataclass
class StrategyConfig:
    """Base configuration for strategies"""
    name: str
    version: str = "1.0"
    author: str = ""
    description: str = ""
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'version': self.version,
            'author': self.author,
            'description': self.description
        }


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies
    Enforces clean separation between data, logic, and parameters
    """
    
    # Class-level parameter grid for optimization
    param_grid: Dict[str, List] = {}
    
    # Strategy metadata
    config: StrategyConfig = None
    
    def __init__(self, **params):
        """Initialize strategy with parameters"""
        # Set parameters as attributes
        for key, value in params.items():
            setattr(self, key, value)
        
        # Initialize indicator cache
        self._indicator_cache = {}
        
        # Validate parameters
        self.validate_parameters()
    
    def validate_parameters(self):
        """Override to add parameter validation"""
        pass
    
    @abstractmethod
    def calculate_indicators(self, data: MarketData) -> Dict[str, pd.Series]:
        """
        Calculate all indicators needed for the strategy
        Returns dict of indicator_name -> Series
        """
        pass
    
    @abstractmethod
    def generate_signals(self, data: MarketData, indicators: Dict[str, pd.Series]) -> Tuple[pd.Series, pd.Series]:
        """
        Generate entry and exit signals based on indicators
        Returns (entries, exits) as boolean Series
        """
        pass
    
    def entries(self, data: MarketData) -> pd.Series:
        """Generate entry signals - called by backtester"""
        indicators = self.calculate_indicators(data)
        entries, _ = self.generate_signals(data, indicators)
        return entries
    
    def exits(self, data: MarketData) -> pd.Series:
        """Generate exit signals - called by backtester"""
        indicators = self.calculate_indicators(data)
        _, exits = self.generate_signals(data, indicators)
        return exits
    
    def get_required_history(self) -> int:
        """Override to specify minimum history needed (in bars)"""
        return 200  # Default to 200 bars
    
    def get_indicator_plots(self) -> List[Dict]:
        """
        Override to define which indicators to plot
        Returns list of dicts with plot configuration
        """
        return []


class MACrossoverStrategy(BaseStrategy):
    """Example implementation: Simple MA Crossover"""
    
    config = StrategyConfig(
        name="MA Crossover",
        version="2.0",
        description="Crosses of fast and slow moving averages"
    )
    
    param_grid = {
        'fast_period': [10, 20, 30],
        'slow_period': [50, 100, 200],
        'ma_type': ['SMA', 'EMA']
    }
    
    def validate_parameters(self):
        if self.fast_period >= self.slow_period:
            raise ValueError("Fast period must be less than slow period")
    
    def calculate_indicators(self, data: MarketData) -> Dict[str, pd.Series]:
        """Calculate moving averages"""
        if self.ma_type == 'SMA':
            fast_ma = vbt.MA.run(data.close, self.fast_period, short_name='fast').ma.squeeze()
            slow_ma = vbt.MA.run(data.close, self.slow_period, short_name='slow').ma.squeeze()
        else:  # EMA
            fast_ma = vbt.MA.run(data.close, self.fast_period, ewm=True, short_name='fast').ma.squeeze()
            slow_ma = vbt.MA.run(data.close, self.slow_period, ewm=True, short_name='slow').ma.squeeze()
        
        return {
            'fast_ma': fast_ma,
            'slow_ma': slow_ma
        }
    
    def generate_signals(self, data: MarketData, indicators: Dict[str, pd.Series]) -> Tuple[pd.Series, pd.Series]:
        """Generate crossover signals"""
        fast_ma = indicators['fast_ma']
        slow_ma = indicators['slow_ma']
        
        # Entry when fast crosses above slow
        entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        
        # Exit when fast crosses below slow
        exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
        
        return entries, exits
    
    def get_required_history(self) -> int:
        return self.slow_period + 50  # Extra buffer
    
    def get_indicator_plots(self) -> List[Dict]:
        return [
            {
                'name': 'fast_ma',
                'title': f'{self.ma_type} {self.fast_period}',
                'color': 'blue',
                'overlay': True
            },
            {
                'name': 'slow_ma', 
                'title': f'{self.ma_type} {self.slow_period}',
                'color': 'red',
                'overlay': True
            }
        ]


class RSIMeanReversionStrategy(BaseStrategy):
    """RSI-based mean reversion strategy"""
    
    config = StrategyConfig(
        name="RSI Mean Reversion",
        version="1.0",
        description="Buy oversold, sell overbought conditions"
    )
    
    param_grid = {
        'rsi_period': [14, 21, 28],
        'rsi_oversold': [20, 25, 30],
        'rsi_overbought': [70, 75, 80],
        'hold_bars': [0, 5, 10]  # Minimum bars to hold position
    }
    
    def validate_parameters(self):
        if self.rsi_oversold >= self.rsi_overbought:
            raise ValueError("Oversold level must be less than overbought level")
    
    def calculate_indicators(self, data: MarketData) -> Dict[str, pd.Series]:
        """Calculate RSI"""
        rsi = vbt.RSI.run(data.close, self.rsi_period).rsi.squeeze()
        return {'rsi': rsi}
    
    def generate_signals(self, data: MarketData, indicators: Dict[str, pd.Series]) -> Tuple[pd.Series, pd.Series]:
        """Generate mean reversion signals"""
        rsi = indicators['rsi']
        
        # Entry when RSI is oversold
        entries = rsi < self.rsi_oversold
        
        # Exit when RSI is overbought
        exits = rsi > self.rsi_overbought
        
        # Apply minimum holding period if specified
        if self.hold_bars > 0:
            # Create a series that tracks bars since entry
            in_position = pd.Series(False, index=entries.index)
            bars_held = pd.Series(0, index=entries.index)
            
            for i in range(1, len(entries)):
                if entries.iloc[i-1]:
                    in_position.iloc[i] = True
                    bars_held.iloc[i] = 1
                elif in_position.iloc[i-1] and not exits.iloc[i-1]:
                    in_position.iloc[i] = True
                    bars_held.iloc[i] = bars_held.iloc[i-1] + 1
                
                # Only allow exits after minimum hold period
                if bars_held.iloc[i] < self.hold_bars:
                    exits.iloc[i] = False
        
        return entries, exits
    
    def get_indicator_plots(self) -> List[Dict]:
        return [
            {
                'name': 'rsi',
                'title': f'RSI({self.rsi_period})',
                'color': 'purple',
                'overlay': False,
                'levels': [self.rsi_oversold, self.rsi_overbought]
            }
        ]


# Strategy registry for easy discovery
STRATEGY_REGISTRY = {
    'ma_crossover': MACrossoverStrategy,
    'rsi_mean_reversion': RSIMeanReversionStrategy
}


def load_strategy(name: str, **params) -> BaseStrategy:
    """Load a strategy by name with parameters"""
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_REGISTRY.keys())}")
    
    strategy_class = STRATEGY_REGISTRY[name]
    return strategy_class(**params)


def list_strategies() -> Dict[str, StrategyConfig]:
    """List all available strategies with their metadata"""
    return {
        name: cls.config.to_dict() 
        for name, cls in STRATEGY_REGISTRY.items()
        if cls.config is not None
    }