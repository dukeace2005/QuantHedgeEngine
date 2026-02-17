# market_data.py
import yfinance as yf

def fetch_market_data(ticker):
    """Fetch real-time market data from yfinance"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        current_price = info.get('regularMarketPrice', info.get('currentPrice', 0))
        prev_close = info.get('regularMarketPreviousClose', info.get('previousClose', current_price))
        change = current_price - prev_close
        change_pct = (change / prev_close * 100) if prev_close > 0 else 0
        
        bid = info.get('bid', 0)
        ask = info.get('ask', 0)
        bid_size = info.get('bidSize', 0)
        ask_size = info.get('askSize', 0)
        volume = info.get('volume', info.get('regularMarketVolume', 0))
        week_high_52 = info.get('fiftyTwoWeekHigh', 0)
        week_low_52 = info.get('fiftyTwoWeekLow', 0)
        day_high = info.get('dayHigh', current_price * 1.01)
        day_low = info.get('dayLow', current_price * 0.99)
        
        market_cap = info.get('marketCap', 0)
        if market_cap > 1e12:
            market_cap_str = f"${market_cap/1e12:.2f}T"
        elif market_cap > 1e9:
            market_cap_str = f"${market_cap/1e9:.2f}B"
        elif market_cap > 1e6:
            market_cap_str = f"${market_cap/1e6:.2f}M"
        else:
            market_cap_str = f"${market_cap:,.0f}"
        
        company_name = info.get('longName', info.get('shortName', f"{ticker} Inc."))
        exchange = info.get('exchange', 'NASDAQ').upper()
        
        # Simplified S&P 500 check (you might want a proper list)
        is_sp500 = ticker in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'BRK-B', 'JPM', 'V', 'JNJ']
        listed_display = "S&P 500" if is_sp500 else exchange
        
        return {
            'current_price': current_price,
            'change': change,
            'change_pct': change_pct,
            'bid': bid,
            'ask': ask,
            'bid_size': bid_size,
            'ask_size': ask_size,
            'volume': volume,
            'week_high_52': week_high_52,
            'week_low_52': week_low_52,
            'day_high': day_high,
            'day_low': day_low,
            'market_cap': market_cap_str,
            'listed_exchange': listed_display,
            'company_name': company_name
        }
    except Exception as e:
        print(f"Error fetching market data: {e}")
        return None