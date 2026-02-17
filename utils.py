# utils.py
from datetime import datetime

def create_nasdaq_header(ticker, report, base_report, market_data=None):
    """Create Nasdaq-style header HTML with safe null handling"""
    
    # SAFE EXTRACTION with defaults
    if market_data and isinstance(market_data, dict):
        current_price = market_data.get('current_price')
        if current_price is None:
            current_price = report.get('price', 0) if report else 0
        
        change = market_data.get('change', 0)
        if change is None:
            change = 0
            
        change_pct = market_data.get('change_pct', 0)
        if change_pct is None:
            change_pct = 0
            
        bid = market_data.get('bid', 0)
        if bid is None:
            bid = 0
            
        ask = market_data.get('ask', 0)
        if ask is None:
            ask = 0
            
        bid_size = market_data.get('bid_size', 0)
        if bid_size is None:
            bid_size = 0
            
        ask_size = market_data.get('ask_size', 0)
        if ask_size is None:
            ask_size = 0
            
        volume = market_data.get('volume', 0)
        if volume is None:
            volume = 0
            
        week_high_52 = market_data.get('week_high_52', 0)
        if week_high_52 is None:
            week_high_52 = 0
            
        week_low_52 = market_data.get('week_low_52', 0)
        if week_low_52 is None:
            week_low_52 = 0
            
        # CRITICAL FIX: Safe multiplication - don't use default in get()
        day_high = market_data.get('day_high')
        if day_high is None:
            # Only multiply if current_price is valid
            if current_price and isinstance(current_price, (int, float)) and current_price > 0:
                day_high = current_price * 1.01
            else:
                day_high = 0
            
        day_low = market_data.get('day_low')
        if day_low is None:
            if current_price and isinstance(current_price, (int, float)) and current_price > 0:
                day_low = current_price * 0.99
            else:
                day_low = 0
            
        market_cap = market_data.get('market_cap', 'N/A')
        if market_cap is None:
            market_cap = 'N/A'
            
        listed_exchange = market_data.get('listed_exchange', 'NASDAQ')
        if listed_exchange is None:
            listed_exchange = 'NASDAQ'
            
        company_name = market_data.get('company_name', f"{ticker} Inc.")
        if company_name is None:
            company_name = f"{ticker} Inc."
    else:
        # Fallback if no market_data
        current_price = report.get('price', 0) if report else 0
        change = 0
        change_pct = 0
        bid = 0
        ask = 0
        bid_size = 0
        ask_size = 0
        volume = 0
        week_high_52 = 0
        week_low_52 = 0
        
        # Safe multiplication for fallback
        if current_price and isinstance(current_price, (int, float)) and current_price > 0:
            day_high = current_price * 1.01
            day_low = current_price * 0.99
        else:
            day_high = 0
            day_low = 0
            
        market_cap = 'N/A'
        listed_exchange = 'NASDAQ'
        company_name = f"{ticker} Inc."
    
    # Ensure current_price is a number
    try:
        current_price = float(current_price) if current_price is not None else 0
    except (TypeError, ValueError):
        current_price = 0
    
    # Determine change class and sign
    try:
        change_float = float(change) if change is not None else 0
        change_class = 'positive' if change_float > 0 else 'negative' if change_float < 0 else 'neutral'
        change_sign = '+' if change_float > 0 else ''
    except (TypeError, ValueError):
        change_class = 'neutral'
        change_sign = ''
        change_float = 0
    
    # Format volume
    try:
        volume_int = int(volume) if volume is not None else 0
        volume_str = f"{volume_int:,}" if volume_int > 0 else "N/A"
    except (TypeError, ValueError):
        volume_str = "N/A"
    
    # Current time
    current_time_et = datetime.now().strftime('%b %d, %Y %I:%M %p ET')
    
    # Signal from base_report
    signal = base_report.get('verdict', {}).get('signal', 'HOLD') if base_report else 'HOLD'
    score = base_report.get('verdict', {}).get('score', 50) if base_report else 50
    
    signal_color = '#00FF88' if signal == 'BUY' else '#FFA500' if signal == 'HOLD' else '#FF4444'
    
    # SAFE FORMATTING - ensure all values are numbers before formatting
    try:
        price_str = f"${current_price:.2f}" if current_price and current_price > 0 else "$0.00"
        
        change_str = f"{change_sign}{change_float:+.2f}"
        change_pct_float = float(change_pct) if change_pct is not None else 0
        change_pct_str = f"({change_sign}{change_pct_float:+.2f}%)"
        
        bid_float = float(bid) if bid is not None else 0
        ask_float = float(ask) if ask is not None else 0
        bid_str = f"${bid_float:.2f}"
        ask_str = f"${ask_float:.2f}"
        spread_str = f"${ask_float - bid_float:.2f}"
        
        day_high_float = float(day_high) if day_high is not None else 0
        day_low_float = float(day_low) if day_low is not None else 0
        day_high_str = f"${day_high_float:.2f}"
        day_low_str = f"${day_low_float:.2f}"
        
        week_high_float = float(week_high_52) if week_high_52 is not None else 0
        week_low_float = float(week_low_52) if week_low_52 is not None else 0
        week_range_str = f"${week_low_float:.2f} - ${week_high_float:.2f}"
    except (TypeError, ValueError) as e:
        print(f"Formatting error in header: {e}")
        # Fallback if any formatting fails
        price_str = "$0.00"
        change_str = "+0.00"
        change_pct_str = "(+0.00%)"
        bid_str = "$0.00"
        ask_str = "$0.00"
        spread_str = "$0.00"
        day_high_str = "$0.00"
        day_low_str = "$0.00"
        week_range_str = "$0.00 - $0.00"
    
    return f"""
    <div class="nasdaq-header">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
            <div style="flex: 2; min-width: 300px;">
                <div class="ticker-title">{ticker} <span class="exchange-badge">{listed_exchange}</span></div>
                <div class="company-name">{company_name}</div>
                <div style="display: flex; align-items: baseline; margin: 0.5rem 0;">
                    <span class="price-large">{price_str}</span>
                    <span class="price-change {change_class}">
                        {change_str} {change_pct_str}
                    </span>
                </div>
                <div class="market-info">{current_time_et}</div>
                <div style="display: flex; gap: 2rem; margin-top: 1rem; flex-wrap: wrap;">
                    <div><div class="detail-label">Market Cap</div><div class="detail-value small">{market_cap}</div></div>
                    <div><div class="detail-label">Volume</div><div class="detail-value small">{volume_str}</div></div>
                    <div><div class="detail-label">52 Week Range</div><div class="detail-value small">{week_range_str}</div></div>
                </div>
            </div>
            <div style="flex: 1; min-width: 250px;">
                <div class="bid-ask">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <div><div class="detail-label">Bid</div><div class="detail-value">{bid_str}</div><div class="market-info">x {bid_size}</div></div>
                        <div><div class="detail-label">Ask</div><div class="detail-value">{ask_str}</div><div class="market-info">x {ask_size}</div></div>
                        <div><div class="detail-label">Spread</div><div class="detail-value">{spread_str}</div></div>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <div><div class="detail-label">Day High</div><div class="detail-value small">{day_high_str}</div></div>
                        <div><div class="detail-label">Day Low</div><div class="detail-value small">{day_low_str}</div></div>
                        <div><div class="detail-label">Signal</div><div class="detail-value small" style="color: {signal_color};">{signal} ({score})</div></div>
                    </div>
                </div>
                <div style="margin-top: 1rem; display: flex; gap: 0.5rem; justify-content: flex-end;">
                    <span style="background: #2A2A2A; color: #888; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.7rem;">Options</span>
                    <span style="background: #2A2A2A; color: #888; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.7rem;">Greeks</span>
                    <span style="background: #2A2A2A; color: #888; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.7rem;">IV</span>
                </div>
            </div>
        </div>
    </div>
    """