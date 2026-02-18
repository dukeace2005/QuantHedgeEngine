import numpy as np
from scipy.stats import norm

class WheelEngine:
    def __init__(self, S, r, sigma):
        self.S = S          # Current Stock Price
        self.r = r          # Risk-free rate (e.g., 0.036)
        self.sigma = sigma  # Implied Volatility (e.g., 1.27 for USAR)

    def get_greeks(self, K, T):
        # T in years (days/365)
        d1 = (np.log(self.S / K) + (self.r + 0.5 * self.sigma**2) * T) / (self.sigma * np.sqrt(T))
        d2 = d1 - self.sigma * np.sqrt(T)
        
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (self.S * self.sigma * np.sqrt(T))
        theta = -(self.S * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(T)) - self.r * K * np.exp(-self.r * T) * norm.cdf(d2)
        
        return {'delta': delta, 'gamma': gamma, 'theta_daily': abs(theta/365)}

    def check_current_trade(self, K, T, profit_pct):
        greeks = self.get_greeks(K, T)
        tge = greeks['theta_daily'] / greeks['gamma']
        
        # Closing Logic
        if profit_pct >= 0.75 or tge < 0.25 or (T * 365) < 14:
            return "BTC_SIGNAL", tge
        return "HOLD", tge

    def suggest_next_trade(self, target_delta=0.25, next_dte=45):
        # Scans for the strike that matches your target Delta
        # In a real app, you'd iterate through an actual option chain
        potential_strikes = np.linspace(self.S * 1.1, self.S * 1.5, 20)
        best_strike = self.S * 1.3 # Fallback
        min_diff = 1.0
        
        for k in potential_strikes:
            d = self.get_greeks(k, next_dte/365)['delta']
            if abs(d - target_delta) < min_diff:
                min_diff = abs(d - target_delta)
                best_strike = k
        
        return round(best_strike * 2) / 2 # Round to nearest 0.50

# --- LIVE USAR DATA (FEB 17, 2026) ---
usar_wheel = WheelEngine(S=19.25, r=0.0368, sigma=1.27)

# 1. Check current $24 CC
status, tge_val = usar_wheel.check_current_trade(K=24.0, T=17/365, profit_pct=0.73)
print(f"Current Trade Status: {status} (TGE: {tge_val:.2f})")

# 2. If status is BTC, suggest next roll
if status == "BTC_SIGNAL":
    next_strike = usar_wheel.suggest_next_trade(target_delta=0.25, next_dte=45)
    print(f"Next Suggested STO: Sell $ {next_strike} Call (Approx. Apr 3, 2026)")
