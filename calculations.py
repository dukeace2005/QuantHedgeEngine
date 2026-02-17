# calculations.py
"""
Optimized calculations module for options trading
"""
from functools import lru_cache
import numpy as np
import pandas as pd
from scipy.stats import norm
import hashlib
import time
from contextlib import contextmanager
import streamlit as st

# Performance tracking
@contextmanager
def track_time(component_name):
    """Track execution time for debugging"""
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
    
    if st.session_state.debug_mode:
        start = time.time()
        yield
        duration = time.time() - start
        if duration > 0.1:  # Log slow operations
            st.caption(f"⏱️ {component_name}: {duration*1000:.0f}ms")
    else:
        yield

class OptionsCalculator:
    """Singleton calculator with extensive caching"""
    
    _instance = None
    _cache = {}
    _greeks_cache = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @lru_cache(maxsize=256)
    def black_scholes(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
        """
        Cached Black-Scholes option pricing
        T: time to expiry in years
        """
        if T <= 0 or sigma <= 0:
            return max(0, (S - K) if option_type == 'call' else (K - S))
        
        d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type == 'call':
            price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        else:
            price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        
        return max(0, price)
    
    @lru_cache(maxsize=256)
    def calculate_greeks_batch(self, S: float, K: float, T: float, r: float, sigma: float) -> dict:
        """Calculate all Greeks at once (more efficient than separate calls)"""
        if T <= 0 or sigma <= 0:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        
        d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        nd1 = norm.pdf(d1)
        
        return {
            'delta': norm.cdf(d1),
            'gamma': nd1 / (S * sigma * np.sqrt(T)),
            'theta': (-(S * nd1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r*T) * norm.cdf(d2)) / 365,  # Daily theta
            'vega': S * nd1 * np.sqrt(T) / 100,  # Per 1% IV change
            'rho': K * T * np.exp(-r*T) * norm.cdf(d2) / 100  # Per 1% rate change
        }
    
    def calculate_chain_metrics_vectorized(self, chain_df: pd.DataFrame, current_price: float) -> pd.DataFrame:
        """Vectorized calculation for entire option chain"""
        df = chain_df.copy()
        
        # Convert expiry to DTE (Days to Expiry)
        df['expiry_date'] = pd.to_datetime(df['expiry'])
        df['dte'] = (df['expiry_date'] - pd.Timestamp.now()).dt.days
        df['dte_years'] = df['dte'] / 365
        
        # Filter out expired
        df = df[df['dte'] > 0].copy()
        
        # Vectorized probability calculations (simplified)
        df['moneyness'] = (current_price - df['strike']) / df['strike']
        df['prob_itm'] = norm.cdf(df['moneyness'] / 0.3)  # Simplified, uses 30% vol
        df['prob_otm'] = 1 - df['prob_itm']
        
        # Calculate implied vol (simplified)
        df['implied_vol'] = np.sqrt(2*np.pi/df['dte_years'].clip(lower=0.01)) * df['lastPrice'] / current_price
        df['implied_vol'] = df['implied_vol'].clip(0.1, 1.0)  # Sanity check
        
        # Calculate returns and metrics
        df['premium_yield'] = (df['lastPrice'] / df['strike']) * 100
        df['break_even'] = np.where(
            df['option_type'] == 'call',
            df['strike'] + df['lastPrice'],
            df['strike'] - df['lastPrice']
        )
        df['move_to_be'] = ((df['break_even'] - current_price) / current_price) * 100
        
        return df

class DataProcessor:
    """Efficient data processing with caching"""
    
    def __init__(self):
        self._processed_cache = {}
        self.calculator = OptionsCalculator()
    
    def get_cache_key(self, data, params):
        """Generate cache key from data and parameters"""
        content = str(data) + str(params)
        return hashlib.md5(content.encode()).hexdigest()
    
    def process_option_subset(self, chain_data, price, min_prob, max_prob, option_type, risk_tolerance='Moderate'):
        """Process and filter options with caching"""
        cache_key = self.get_cache_key(chain_data, (price, min_prob, max_prob, option_type, risk_tolerance))
        
        if cache_key in self._processed_cache:
            return self._processed_cache[cache_key].copy()
        
        df = pd.DataFrame(chain_data)
        df['option_type'] = option_type
        
        # Calculate metrics
        df = self.calculator.calculate_chain_metrics_vectorized(df, price)
        
        # Apply filters based on risk tolerance
        if risk_tolerance == 'Conservative':
            df = df[df['premium_yield'] < 30]  # Lower yield, safer
        elif risk_tolerance == 'Aggressive':
            df = df[df['premium_yield'] > 10]  # Higher yield, riskier
        
        # Apply probability filter
        prob_col = 'prob_itm' if option_type == 'call' else 'prob_otm'
        df = df[(df[prob_col] >= min_prob) & (df[prob_col] <= max_prob)]
        
        # Sort and reset index
        df = df.sort_values(['dte', 'strike'], ascending=[True, True]).reset_index(drop=True)
        
        # Cache the result
        self._processed_cache[cache_key] = df.copy()
        
        return df