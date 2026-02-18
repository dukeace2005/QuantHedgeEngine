import yfinance as yf
import pandas as pd
import numpy as np

class ReportGenerator:
    @staticmethod
    def generate_narrative(results):
        """Generate textual reasoning and strategy guidance for each ticker."""
        narratives = {}
        for ticker, data in results.items():
            signal = data.get("signal", "N/A")
            rsi = data.get("rsi", "N/A")

            if signal == "VERY BULLISH":
                reasoning = (
                    f"{ticker} is in a confirmed uptrend (Price > SMA50) and showing a healthy "
                    f"pullback (RSI {rsi}). This can support higher-probability premium capture."
                )
                recommendation = "Standard Play: Execute at 0.30 delta with high conviction."
            elif signal == "NEUTRAL":
                reasoning = (
                    f"{ticker} is below trend, but fear is elevated (RSI {rsi}). In leveraged ETFs, "
                    "extreme oversold conditions can precede snap-back bounces."
                )
                recommendation = (
                    "Reversal Play: Sell 0.15 delta (deep OTM) to harvest IV while leaving room "
                    "for additional downside."
                )
            else:
                reasoning = (
                    f"{ticker} does not show a strong edge right now. It is either not in a pullback "
                    "from strength or lacks deep oversold reversal conditions."
                )
                recommendation = (
                    "Wait for RSI < 30 for a Reversal Play or for Price > SMA50 for a Bullish Play."
                )

            narratives[ticker] = {
                "audit_analysis": reasoning,
                "strategy_recommendation": recommendation
            }
        return narratives


class QuantAgent:
    def __init__(self, tickers):
        self.tickers = tickers
        self.results = {}

    def get_signal_level(self, price, sma, rsi):
        """
        Refined 4-level logic:
        - VERY BULLISH: Above 50-SMA + RSI dip (<45)
        - BULLISH: Above 50-SMA + RSI >= 45
        - NEUTRAL: Below 50-SMA + RSI < 30
        - CONSERVATIVE: Below 50-SMA + RSI >= 30
        """
        if price > sma:
            return "VERY BULLISH" if rsi < 45 else "BULLISH"
        return "NEUTRAL" if rsi < 30 else "CONSERVATIVE"

    @staticmethod
    def _as_scalar(value):
        """Convert pandas/numpy values to a scalar float for safe comparisons."""
        if isinstance(value, pd.DataFrame):
            if value.empty:
                return np.nan
            value = value.iloc[-1, -1]
        elif isinstance(value, pd.Series):
            if value.empty:
                return np.nan
            value = value.iloc[-1]

        if isinstance(value, (np.ndarray, list, tuple)):
            if len(value) == 0:
                return np.nan
            value = value[-1]

        try:
            return float(value)
        except (TypeError, ValueError):
            return np.nan

    @staticmethod
    def _extract_close(data, ticker, multi_ticker):
        """Get close series from either single- or multi-ticker download output."""
        if multi_ticker:
            if ticker not in data:
                return None
            ticker_data = data[ticker]
            close = ticker_data.get('Close')
        else:
            close = data.get('Close')

        if close is None or (hasattr(close, "empty") and close.empty):
            return None
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return close

    def run_audit(self):
        self.results = {}
        multi_ticker = len(self.tickers) > 1
        all_data = yf.download(
            self.tickers,
            period="1y",
            interval="1d",
            group_by="ticker"
        )

        for ticker in self.tickers:
            close = self._extract_close(all_data, ticker, multi_ticker)
            if close is None:
                continue

            curr = self._as_scalar(close)
            sma50 = self._as_scalar(close.rolling(50).mean())
            
            # RSI Calculation
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rsi_series = 100 - (100 / (1 + (gain / loss)))
            rsi = self._as_scalar(rsi_series)

            if np.isnan(curr) or np.isnan(sma50) or np.isnan(rsi):
                continue
            
            signal = self.get_signal_level(curr, sma50, rsi)
            
            self.results[ticker] = {
                "signal": signal,
                "status": "SKIP" if signal == "CONSERVATIVE" else "SELL",
                "price": round(float(curr), 2),
                "rsi": round(float(rsi), 2),
                "sma_50": round(float(sma50), 2),
                "strategy": {
                    "strike_delta": 0.30 if "BULLISH" in signal else 0.15,
                    "target_profit": "50%",
                    "exit_dte": 21
                }
            }

        narratives = ReportGenerator.generate_narrative(self.results)
        for ticker in self.results:
            self.results[ticker].update(narratives.get(ticker, {}))

        return self.results

if __name__ == "__main__":
    agent = QuantAgent(["XLI", "XLF", "TQQQ", "SPYM", "UPRO", "SOXL"])
    print(agent.run_audit())
