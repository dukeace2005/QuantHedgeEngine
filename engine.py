import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from scipy.stats import norm

# Add to your existing QuantOptionEngine class
from functools import lru_cache
import pandas as pd
import numpy as np
from scipy.stats import norm
import hashlib

class QuantOptionEngine:
    def __init__(self, risk_free_rate=0.045):
        self.r = risk_free_rate
        self._pitm_cache = {}  # Add cache dictionary
        self._chain_cache = {}  # Cache for option chains
    
    @lru_cache(maxsize=1024)
    def calculate_pitm_cached(self, S, K, T_days, iv):
        """Cached version of probability ITM calculation"""
        return self.calculate_pitm(S, K, T_days, iv)
    
    def calculate_pitm_vectorized(self, S, K_array, T_days_array, iv_array):
        """Vectorized PITM calculation for multiple options at once"""
        T = T_days_array / 365.0
        valid = (T > 0) & (iv_array > 0)
        
        d2 = np.where(
            valid,
            (np.log(S / K_array) + (self.r - 0.5 * iv_array**2) * T) / (iv_array * np.sqrt(T)),
            0
        )
        
        prob = norm.cdf(d2)
        prob[~valid] = 0.5
        return prob
    
    def get_option_recommendations(self, symbol, score, capital):
        """Optimized version with caching and vectorization"""
        
        # Cache ticker data to avoid repeated yfinance calls
        cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d')}"
        
        if cache_key in self._chain_cache:
            ticker_data = self._chain_cache[cache_key]
        else:
            ticker = yf.Ticker(symbol)
            price = ticker.info.get('regularMarketPrice', ticker.info.get('previousClose'))
            
            if not ticker.options:
                return {"error": "No options found for this ticker."}
            
            # Cache the ticker data
            self._chain_cache[cache_key] = {
                'ticker': ticker,
                'price': price,
                'options': ticker.options,
                'timestamp': datetime.now()
            }
            ticker_data = self._chain_cache[cache_key]
        
        ticker = ticker_data['ticker']
        price = ticker_data['price']
        
        # 1. Select Expirations (optimized with list comprehension)
        today = datetime.now()
        today_tuple = today.toordinal()  # Faster than datetime objects
        
        valid_expirations = [
            date_str for date_str in ticker.options
            if 15 <= (datetime.strptime(date_str, "%Y-%m-%d").toordinal() - today_tuple) <= 45
        ]
        
        if not valid_expirations:
            # Fallback - find closest to 30 days
            valid_expirations = [min(
                ticker.options,
                key=lambda x: abs((datetime.strptime(x, "%Y-%m-%d").toordinal() - today_tuple) - 30)
            )]
        
        # 2. Strategy targets (unchanged - your logic is good)
        if score > 75:
            target_put_itm, target_call_itm = 0.45, 0.65
        elif score > 55:
            target_put_itm, target_call_itm = 0.25, 0.50
        else:
            target_put_itm, target_call_itm = 0.15, 0.35
        
        # 3. Batch process all expirations
        all_puts = []
        all_calls = []
        
        # Pre-calculate date differences
        exp_days = {
            exp: (datetime.strptime(exp, "%Y-%m-%d").toordinal() - today_tuple)
            for exp in valid_expirations
        }
        
        for exp in valid_expirations:
            try:
                chain = ticker.option_chain(exp)
                days = exp_days[exp]
                
                # Process Puts - vectorized where possible
                puts = chain.puts.copy()
                puts['expiry'] = exp
                puts['days_to_expiry'] = days
                
                # Vectorized PITM calculation
                puts['prob_itm'] = 1 - self.calculate_pitm_vectorized(
                    price,
                    puts['strike'].values,
                    np.full(len(puts), days),
                    puts['impliedVolatility'].fillna(0.3).values
                )
                
                puts['diff'] = (puts['prob_itm'] - target_put_itm).abs()
                puts['annualized_yield'] = (puts['lastPrice'] / puts['strike']) * (365.0 / days)
                
                # Vectorized expected move calculations
                puts['expected_move'] = price * puts['impliedVolatility'].fillna(0.3) * np.sqrt(days / 365.0)
                puts['score_target'] = price + (puts['expected_move'] * ((score - 50) / 40.0))
                puts['downside_target'] = price - puts['expected_move']
                
                # Fill NA for volume/OI
                puts['volume'] = puts.get('volume', 0).fillna(0)
                puts['openInterest'] = puts.get('openInterest', 0).fillna(0)
                
                all_puts.append(puts)
                
                # Process Calls similarly
                calls = chain.calls.copy()
                calls['expiry'] = exp
                calls['days_to_expiry'] = days
                
                calls['prob_itm'] = self.calculate_pitm_vectorized(
                    price,
                    calls['strike'].values,
                    np.full(len(calls), days),
                    calls['impliedVolatility'].fillna(0.3).values
                )
                
                calls['diff'] = (calls['prob_itm'] - target_call_itm).abs()
                calls['expected_move'] = price * calls['impliedVolatility'].fillna(0.3) * np.sqrt(days / 365.0)
                calls['score_target'] = price + (calls['expected_move'] * ((score - 50) / 40.0))
                calls['downside_target'] = price - calls['expected_move']
                
                calls['volume'] = calls.get('volume', 0).fillna(0)
                calls['openInterest'] = calls.get('openInterest', 0).fillna(0)
                
                all_calls.append(calls)
                
            except Exception as e:
                continue
        
        if not all_puts or not all_calls:
            return {"error": "No valid options data parsed."}
        
        # Concatenate efficiently
        all_puts_df = pd.concat(all_puts, ignore_index=True)
        all_calls_df = pd.concat(all_calls, ignore_index=True)
        
        # Optimized selection - use nsmallest instead of full sort
        best_put = all_puts_df.nsmallest(1, 'diff').iloc[0].to_dict()
        best_call = all_calls_df.nsmallest(1, 'diff').iloc[0].to_dict()
        
        return {
            "symbol": symbol,
            "price": price,
            "csp": {
                "strike": best_put['strike'],
                "premium": best_put['lastPrice'],
                "prob_itm": f"{best_put['prob_itm']:.1%}",
                "annualized_yield": f"{best_put['annualized_yield']:.1%}",
                "contracts": int(capital // (best_put['strike'] * 100)),
                "expiry": best_put['expiry'],
                "score_target": best_put['score_target'],
                "downside_target": best_put['downside_target'],
                "volume": int(best_put['volume']),
                "open_interest": int(best_put['openInterest'])
            },
            "long_call": {
                "strike": best_call['strike'],
                "premium": best_call['lastPrice'],
                "prob_itm": f"{best_call['prob_itm']:.1%}",
                "breakeven": best_call['strike'] + best_call['lastPrice'],
                "expiry": best_call['expiry'],
                "score_target": best_call['score_target'],
                "downside_target": best_call['downside_target'],
                "volume": int(best_call['volume']),
                "open_interest": int(best_call['openInterest'])
            },
            "chain_data": {
                "puts": all_puts_df.to_dict('records'),
                "calls": all_calls_df.to_dict('records')
            }
        }
class QuantHedgeEngine:
    def __init__(self, total_capital=100000, log_dir="trade_logs"):
        self.total_capital = total_capital
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Cache for sector benchmarks
        self.sector_benchmarks = {
            'Technology': 0.85, 'Financial Services': 0.95, 
            'Healthcare': 0.90, 'Consumer Cyclical': 1.05,
            'Energy': 1.10, 'Utilities': 0.70, 'Default': 0.90
        }
        
        # Add caching for ticker analysis
        self._analysis_cache = {}
        self._hist_cache = {}
    
    @staticmethod
    @lru_cache(maxsize=128)
    def _get_sector_multiplier(sector):
        """Cached sector multiplier lookup"""
        multipliers = {
            'Technology': 0.85, 'Financial Services': 0.95, 
            'Healthcare': 0.90, 'Consumer Cyclical': 1.05,
            'Energy': 1.10, 'Utilities': 0.70
        }
        return multipliers.get(sector, 0.90)
    
    def _get_cached_history(self, ticker, period="3mo"):
        """Cache historical data to avoid repeated downloads"""
        symbol = ticker.ticker if hasattr(ticker, 'ticker') else str(ticker)
        cache_key = f"{symbol}_{period}"
        
        if cache_key in self._hist_cache:
            hist = self._hist_cache[cache_key]
            # Check if cache is still fresh (e.g., < 1 hour old)
            if (datetime.now() - hist['timestamp']).seconds < 3600:
                return hist['data']
        
        # Download new data
        hist = ticker.history(period=period)
        self._hist_cache[cache_key] = {
            'data': hist,
            'timestamp': datetime.now()
        }
        return hist
    
    def analyze_ticker_optimized(self, symbol):
        """Optimized version with caching"""
        
        # Check cache first (5 minute TTL)
        cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d%H')}"
        if cache_key in self._analysis_cache:
            cache_entry = self._analysis_cache[cache_key]
            if (datetime.now() - cache_entry['timestamp']).seconds < 300:  # 5 minutes
                return cache_entry['data']
        
        ticker = yf.Ticker(symbol)
        price = ticker.info.get('regularMarketPrice', ticker.info.get('previousClose', 0))
        
        # Early exit if no price
        if price is None or price == 0:
            return self._create_error_packet(symbol)
        
        # Parallel execution of independent calculations
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Submit tasks
            market_future = executor.submit(self._get_market_sentiment, ticker)
            insider_future = executor.submit(self._get_weighted_insider_score, ticker)
            atr_future = executor.submit(self._get_atr_targets_optimized, ticker, price)
            
            # Get results
            mkt_score, mkt_reason = market_future.result()
            ins_score, ins_reason = insider_future.result()
            exits = atr_future.result()
        
        total_score = (mkt_score * 0.6) + (ins_score * 0.4)
        
        # Get analyst targets (can be done synchronously as it's fast)
        target_low = ticker.info.get('targetLowPrice', price * 0.8)
        target_mean = ticker.info.get('targetMeanPrice', price * 1.15)
        
        sizing = self._calculate_kelly_sizing_optimized(
            total_score,
            exits['risk_reward'],
            price,
            exits['stop_loss']
        )
        
        analysis_packet = {
            "metadata": {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "engine_version": "2.1-Optimized"
            },
            "analyst_targets": {
                "low": target_low,
                "mean": target_mean
            },
            "verdict": {
                "signal": "BUY" if total_score > 60 else "HOLD" if total_score > 45 else "AVOID",
                "score": round(total_score),
                "is_halted": total_score < 45
            },
            "trade_parameters": {
                "entry_price": price,
                "position_size_usd": sizing['suggested_trade_size'],
                "stop_loss": exits['stop_loss'],
                "take_profit": exits['take_profit'],
                "risk_reward": exits['risk_reward']
            },
            "logic_breakdown": {
                "market_sentiment": mkt_reason,
                "market_score": mkt_score,
                "insider_activity": ins_reason,
                "insider_score": ins_score,
                "atr_volatility": exits['atr']
            }
        }
        
        # Cache the result
        self._analysis_cache[cache_key] = {
            'data': analysis_packet,
            'timestamp': datetime.now()
        }
        
        # Async save (don't block)
        import threading
        threading.Thread(target=self.save_to_json, args=(analysis_packet,)).start()
        
        return analysis_packet
    
    def _get_atr_targets_optimized(self, ticker, current_price, atr_period=14, atr_multiplier=2, risk_reward_ratio=1.5):
        """Optimized ATR calculation with numpy"""
        hist = self._get_cached_history(ticker, "3mo")
        
        if hist.empty or len(hist) < atr_period:
            atr_val = current_price * 0.05
        else:
            # Vectorized true range calculation
            high_low = hist['High'].values - hist['Low'].values
            high_close = np.abs(hist['High'].values - hist['Close'].shift(1).values)
            low_close = np.abs(hist['Low'].values - hist['Close'].shift(1).values)
            
            true_range = np.maximum.reduce([high_low, high_close, low_close])
            
            # Exponential moving average
            alpha = 1.0 / atr_period
            atr_val = pd.Series(true_range).ewm(alpha=alpha, adjust=False).mean().iloc[-1]
        
        if pd.isna(atr_val) or atr_val == 0:
            atr_val = current_price * 0.05
        
        stop_loss_price = current_price - (atr_val * atr_multiplier)
        take_profit_price = current_price + (atr_val * atr_multiplier * risk_reward_ratio)
        
        return {
            'stop_loss': round(stop_loss_price, 2),
            'take_profit': round(take_profit_price, 2),
            'risk_reward': risk_reward_ratio,
            'atr': round(atr_val, 2)
        }
    
    def _calculate_kelly_sizing_optimized(self, total_score, risk_reward_ratio, entry_price, stop_loss_price, max_risk_pct=0.02):
        """Optimized Kelly sizing with numpy"""
        risk_reward = max(risk_reward_ratio, 1.5)
        win_prob = np.clip(total_score / 100.0, 0.01, 0.99)
        
        kelly_fraction = win_prob - ((1 - win_prob) / risk_reward)
        risk_fraction = min(max_risk_pct, max(0, kelly_fraction * 0.5))
        
        if risk_fraction <= 0:
            return {'suggested_trade_size': 0}
        
        dollar_risk = self.total_capital * risk_fraction
        risk_per_share = entry_price - stop_loss_price
        
        if risk_per_share <= 0:
            return {'suggested_trade_size': 0}
        
        num_shares = dollar_risk / risk_per_share
        return {'suggested_trade_size': num_shares * entry_price}
    
    def _create_error_packet(self, symbol):
        """Helper for error packets"""
        return {
            "metadata": {"symbol": symbol, "timestamp": datetime.now().isoformat(), "engine_version": "2.1-Optimized"},
            "verdict": {"signal": "AVOID", "score": 0, "is_halted": True},
            "trade_parameters": {"entry_price": 0, "position_size_usd": 0, "stop_loss": 0, "take_profit": 0, "risk_reward": 0},
            "logic_breakdown": {"market_sentiment": f"Could not retrieve price for {symbol}.", "insider_activity": "N/A", "atr_volatility": "N/A"}
        }
    def __init__(self, total_capital=100000, log_dir="trade_logs"):
        self.total_capital = total_capital
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        self.sector_benchmarks = {
            'Technology': 0.85, 'Financial Services': 0.95, 
            'Healthcare': 0.90, 'Consumer Cyclical': 1.05,
            'Energy': 1.10, 'Utilities': 0.70, 'Default': 0.90
        }

    # --- THE EXPORT & LOGGING MODULE ---
    def save_to_json(self, result_dict):
        """Saves analysis to a daily JSON file for dashboard ingestion."""
        symbol = result_dict.get('metadata', {}).get('symbol', 'UNKNOWN')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.log_dir}/{symbol}_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(result_dict, f, indent=4)
            return filename
        except Exception as e:
            return f"Error saving JSON: {e}"

    # --- CORE ANALYTICS (CONSOLIDATED) ---
    def analyze_ticker(self, symbol):
        ticker = yf.Ticker(symbol)
        price = ticker.info.get('regularMarketPrice', ticker.info.get('previousClose', 0))

        # Fetch analyst targets
        target_low = ticker.info.get('targetLowPrice')
        target_mean = ticker.info.get('targetMeanPrice')
        # Defaults handled in app or if None

        # Handle cases where ticker is invalid or has no price
        if price is None or price == 0:
            analysis_packet = {
                "metadata": {"symbol": symbol, "timestamp": datetime.now().isoformat(), "engine_version": "2.1-Skew-Kelly"},
                "verdict": {"signal": "AVOID", "score": 0, "is_halted": True},
                "trade_parameters": {"entry_price": 0, "position_size_usd": 0, "stop_loss": 0, "take_profit": 0, "risk_reward": 0},
                "logic_breakdown": {"market_sentiment": f"Could not retrieve price for {symbol}.", "insider_activity": "N/A", "atr_volatility": "N/A"}
            }
            log_path = self.save_to_json(analysis_packet)
            analysis_packet['log_path'] = log_path
            return analysis_packet

        # 1. Pillars (Market Skew + Insider Weighting)
        mkt_score, mkt_reason = self._get_market_sentiment(ticker)
        ins_score, ins_reason = self._get_weighted_insider_score(ticker)

        total_score = (mkt_score * 0.6) + (ins_score * 0.4)

        # 2. Risk & Sizing (Calculate exits BEFORE sizing)
        exits = self._get_atr_targets(ticker, price)
        sizing = self._calculate_kelly_sizing(
            total_score,
            exits['risk_reward'],
            price,
            exits['stop_loss']
        )

        # 3. Final Verdict Object
        analysis_packet = {
            "metadata": {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "engine_version": "2.1-Skew-Kelly"
            },
            "analyst_targets": {
                "low": target_low if target_low else price * 0.8,
                "mean": target_mean if target_mean else price * 1.15
            },
            "verdict": {
                "signal": "BUY" if total_score > 60 else "HOLD" if total_score > 45 else "AVOID",
                "score": round(total_score),
                "is_halted": total_score < 45
            },
            "trade_parameters": {
                "entry_price": price,
                "position_size_usd": sizing['suggested_trade_size'],
                "stop_loss": exits['stop_loss'],
                "take_profit": exits['take_profit'],
                "risk_reward": exits['risk_reward']
            },
            "logic_breakdown": {
                "market_sentiment": mkt_reason,
                "market_score": mkt_score,
                "insider_activity": ins_reason,
                "insider_score": ins_score,
                "atr_volatility": exits['atr']
            }
        }

        # Auto-Export
        log_path = self.save_to_json(analysis_packet)
        analysis_packet['log_path'] = log_path

        return analysis_packet

    # --- PILLAR & RISK IMPLEMENTATIONS ---
    def _get_market_sentiment(self, ticker):
        """
        Analyzes market sentiment based on analyst recommendations.
        Placeholder implementation.
        """
        rec = ticker.info.get('recommendationKey', 'hold').lower()
        score_map = {'strong_buy': 95, 'buy': 80, 'hold': 50, 'underperform': 25, 'sell': 10}
        score = score_map.get(rec, 50)
        return score, f"Analyst consensus is '{rec.replace('_', ' ').title()}'."

    def _get_weighted_insider_score(self, ticker):
        """
        Calculates a score based on recent insider trading activity.
        Placeholder implementation.
        """
        # yfinance insider data can be sparse. This is a simplified placeholder.
        net_shares = ticker.info.get('netSharePurchaseActivity')
        if net_shares:
            net_percent = net_shares.get('netPercentInsiderShares')
            if net_percent is not None:
                if net_percent > 0.01: # >1% of shares bought
                    return 80, f"Positive: Insiders bought {net_percent:.2%} of shares recently."
                elif net_percent < -0.01: # >1% of shares sold
                    return 20, f"Negative: Insiders sold {abs(net_percent):.2%} of shares recently."
        return 50, "Neutral insider activity."

    def _get_atr_targets(self, ticker, current_price, atr_period=14, atr_multiplier=2, risk_reward_ratio=1.5):
        """Calculates stop-loss and take-profit targets based on ATR."""
        hist = ticker.history(period="3mo")
        if hist.empty or len(hist) < atr_period:
            atr_val = current_price * 0.05 # Fallback
        else:
            high_low = hist['High'] - hist['Low']
            high_close = np.abs(hist['High'] - hist['Close'].shift())
            low_close = np.abs(hist['Low'] - hist['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            atr_val = true_range.ewm(alpha=1/atr_period, adjust=False).mean().iloc[-1]

        if pd.isna(atr_val) or atr_val == 0:
             atr_val = current_price * 0.05

        stop_loss_price = current_price - (atr_val * atr_multiplier)
        take_profit_price = current_price + (atr_val * atr_multiplier * risk_reward_ratio)
        return {'stop_loss': round(stop_loss_price, 2), 'take_profit': round(take_profit_price, 2), 'risk_reward': risk_reward_ratio, 'atr': round(atr_val, 2)}

    def _calculate_kelly_sizing(self, total_score, risk_reward_ratio, entry_price, stop_loss_price, max_risk_pct=0.02):
        """Calculates suggested position size using a simplified Kelly Criterion."""
        if risk_reward_ratio <= 0: risk_reward_ratio = 1.5
        win_prob = max(0.01, min(0.99, total_score / 100.0))
        kelly_fraction = win_prob - ((1 - win_prob) / risk_reward_ratio)
        risk_fraction = min(max_risk_pct, max(0, kelly_fraction * 0.5))
        if risk_fraction <= 0: return {'suggested_trade_size': 0}
        dollar_risk = self.total_capital * risk_fraction
        risk_per_share = entry_price - stop_loss_price
        if risk_per_share <= 0: return {'suggested_trade_size': 0}
        num_shares = dollar_risk / risk_per_share
        return {'suggested_trade_size': num_shares * entry_price}