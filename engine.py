import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from scipy.stats import norm

class QuantOptionEngine:
    def __init__(self, risk_free_rate=0.045):
        self.r = risk_free_rate # 2026 Average Rate

    def calculate_pitm(self, S, K, T_days, iv):
        """Calculates Probability ITM using Black-Scholes N(d2)."""
        T = T_days / 365.0
        if T <= 0 or iv <= 0: return 0.5
        # d2 is the probability that the option will be exercised
        d2 = (np.log(S / K) + (self.r - 0.5 * iv**2) * T) / (iv * np.sqrt(T))
        return norm.cdf(d2)

    def get_option_recommendations(self, symbol, score, capital):
        ticker = yf.Ticker(symbol)
        price = ticker.info.get('regularMarketPrice', ticker.info.get('previousClose'))
        
        if not ticker.options:
            return {"error": "No options found for this ticker."}

        # 1. Select Expirations (Window: 15 to 45 Days)
        today = datetime.now()
        valid_expirations = []
        for date_str in ticker.options:
            d = datetime.strptime(date_str, "%Y-%m-%d")
            days = (d - today).days
            if 15 <= days <= 45:
                valid_expirations.append(date_str)
        
        if not valid_expirations:
            # Fallback to nearest if no ideal window found
            valid_expirations = [min(ticker.options, key=lambda x: abs((datetime.strptime(x, "%Y-%m-%d") - today).days - 30))]

        # 2. Strategy Logic based on BUY signal strength
        if score > 75: # STRONG BUY
            target_put_itm = 0.45   # Aggressive entry (High Delta)
            target_call_itm = 0.65  # Deep ITM for safety
        elif score > 55: # MODERATE BUY
            target_put_itm = 0.25   # Standard income
            target_call_itm = 0.50  # At-the-money
        else: # WEAK/NEUTRAL
            target_put_itm = 0.15   # High safety
            target_call_itm = 0.35  # OTM Lottery (Low preference)

        # 3. Aggregate Chains & Select Best Match
        all_puts = []
        all_calls = []

        for exp in valid_expirations:
            try:
                chain = ticker.option_chain(exp)
                days = (datetime.strptime(exp, "%Y-%m-%d") - today).days
                
                # Process Puts
                puts = chain.puts.copy()
                puts['expiry'] = exp
                puts['days_to_expiry'] = days
                puts['prob_itm'] = puts.apply(lambda x: 1 - self.calculate_pitm(price, x.strike, days, x.impliedVolatility), axis=1)
                puts['diff'] = (puts['prob_itm'] - target_put_itm).abs()
                puts['annualized_yield'] = (puts['lastPrice'] / puts['strike']) * (365.0 / days)
                
                # Expected Move & Targets
                puts['expected_move'] = price * puts['impliedVolatility'] * np.sqrt(days / 365.0)
                puts['score_target'] = price + (puts['expected_move'] * ((score - 50) / 40.0))
                puts['downside_target'] = price - puts['expected_move']
                all_puts.append(puts)

                # Process Calls
                calls = chain.calls.copy()
                calls['expiry'] = exp
                calls['days_to_expiry'] = days
                calls['prob_itm'] = calls.apply(lambda x: self.calculate_pitm(price, x.strike, days, x.impliedVolatility), axis=1)
                calls['diff'] = (calls['prob_itm'] - target_call_itm).abs()
                
                # Expected Move & Targets
                calls['expected_move'] = price * calls['impliedVolatility'] * np.sqrt(days / 365.0)
                calls['score_target'] = price + (calls['expected_move'] * ((score - 50) / 40.0))
                calls['downside_target'] = price - calls['expected_move']
                all_calls.append(calls)
            except Exception:
                continue
        
        if not all_puts or not all_calls:
             return {"error": "No valid options data parsed."}

        # Select Best (Sort by closeness to target ITM)
        best_put = pd.concat(all_puts).sort_values('diff').iloc[0].to_dict()
        best_call = pd.concat(all_calls).sort_values('diff').iloc[0].to_dict()

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
                "downside_target": best_put['downside_target']
            },
            "long_call": {
                "strike": best_call['strike'],
                "premium": best_call['lastPrice'],
                "prob_itm": f"{best_call['prob_itm']:.1%}",
                "breakeven": best_call['strike'] + best_call['lastPrice'],
                "expiry": best_call['expiry'],
                "score_target": best_call['score_target'],
                "downside_target": best_call['downside_target']
            }
        }

class QuantHedgeEngine:
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