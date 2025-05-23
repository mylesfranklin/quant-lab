"""
Re-export factor store enhanced functionality for easy importing.
This allows strategies to import from src.backtesting.factor_store_enhanced
"""

# Import from root directory
import sys
from pathlib import Path

# Add parent directory to path
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

# Import all functions from the root factor_store_enhanced
from factor_store_enhanced import (
    get_enhanced_factor_store,
    factor_info,
    find_factors,
    validate_factor,
    invalidate_stale_factors,
    EnhancedFactorStore
)

# Re-export everything
__all__ = [
    'get_enhanced_factor_store',
    'factor_info',
    'find_factors',
    'validate_factor',
    'invalidate_stale_factors',
    'EnhancedFactorStore'
]