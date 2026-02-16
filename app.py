import streamlit as st
import plotly.graph_objects as go
import numpy as np
from engine import QuantOptionEngine, QuantHedgeEngine # Import both engines

st.set_page_config(layout="wide", page_title="Hedge Fund Options Suite")

# --- CHARTING FUNCTION ---
def create_payoff_chart(strategy_type, strike, premium, current_price):
    """Generates a Plotly payoff diagram for a given option strategy."""
    price_range = np.linspace(current_price * 0.8, current_price * 1.2, 200)
    
    if strategy_type == 'call':
        pnl = np.maximum(0, price_range - strike) - premium
        breakeven = strike + premium
        max_profit = float('inf')
        max_loss = -premium
    elif strategy_type == 'put':
        pnl = premium - np.maximum(0, strike - price_range)
        breakeven = strike - premium
        max_profit = premium
        max_loss = premium - strike # Max loss occurs if stock goes to 0

    fig = go.Figure()

    # P&L Line
    fig.add_trace(go.Scatter(x=price_range, y=pnl, mode='lines', name='Profit/Loss', line=dict(color='#00BFFF', width=3)))

    # Profit/Loss Shading
    fig.add_trace(go.Scatter(x=price_range[pnl >= 0], y=pnl[pnl >= 0], fill='tozeroy', mode='none', fillcolor='rgba(0, 255, 127, 0.2)', name='Profit'))
    fig.add_trace(go.Scatter(x=price_range[pnl < 0], y=pnl[pnl < 0], fill='tozeroy', mode='none', fillcolor='rgba(255, 69, 0, 0.2)', name='Loss'))

    # Key Lines
    fig.add_vline(x=current_price, line_width=2, line_dash="dash", line_color="yellow", annotation_text="Current Price", annotation_position="top right")
    fig.add_vline(x=breakeven, line_width=2, line_dash="dash", line_color="cyan", annotation_text="Breakeven", annotation_position="top left")

    # Styling
    fig.update_layout(
        title=dict(text=f"{'Long Call' if strategy_type == 'call' else 'Cash-Secured Put'} Payoff", x=0.5),
        xaxis_title="Stock Price at Expiration",
        yaxis_title="Profit / Loss per Share",
        showlegend=False,
        template="plotly_dark",
        yaxis=dict(zerolinecolor='grey', zerolinewidth=1),
    )
    return fig, max_profit, max_loss

# --- SIDEBAR & INPUTS ---
with st.sidebar:
    st.header("Analysis Controls")
    ticker_input = st.text_input("Enter Ticker Symbol", value="AAPL").upper()
    run_analysis = st.button("Run Options Analysis")

    
# Instantiate engines
hedge_engine = QuantHedgeEngine()
option_engine = QuantOptionEngine()

if run_analysis:
    # 1. Get base score from the Hedge Engine to inform option strategy
    with st.spinner(f"Running base analysis for {ticker_input}..."):
        base_report = hedge_engine.analyze_ticker(ticker_input)
        total_score = base_report.get('verdict', {}).get('score', 50)
        st.sidebar.metric("Underlying Score", f"{total_score}/100", delta=f"{total_score-50}",
                          help=base_report.get('logic_breakdown', {}).get('market_sentiment'))

    # 2. Fetch Option Recommendations
    with st.spinner(f"Finding optimal option strategies for {ticker_input}..."):
        report = option_engine.get_option_recommendations(ticker_input, total_score, 100000)
    
    st.subheader(f"⚖️ Strategy Comparison: {ticker_input} @ ${report['price']}")
    
    # Create Side-by-Side "Bumper" Cards
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### 🟢 Cash Secured Put (Exp: {report['csp']['expiry']})")
        st.markdown(f"### 🟢 Cash Secured Put (Yield)")
        st.caption(f"Expiration: {report['csp']['expiry']}")
        with st.container(border=True):
            st.metric("Suggested Strike", f"${report['csp']['strike']}")
            st.write(f"**Prob. OTM:** :green[{100 - float(report['csp']['prob_itm'].strip('%')):.1f}%]")
            csp_fig, max_profit, max_loss = create_payoff_chart(
                'put',
                report['csp']['strike'],
                report['csp']['premium'],
                report['price']
            )
            st.plotly_chart(csp_fig, use_container_width=True)
            
            c1, c2 = st.columns(2)
            c1.metric("Est. Premium", f"${report['csp']['premium']}")
            c2.metric("Annualized Yield", report['csp']['annualized_yield'])
            
            st.info(f"💡 {report['csp']['contracts']} Contracts will lock ${report['csp']['strike']*100*report['csp']['contracts']:,} collateral.")
            c1.metric("Max Profit", f"${max_profit*100:,.2f}", help="Premium received per contract.")
            c2.metric("Max Risk", f"${-max_loss*100:,.2f}", help="Collateral at risk if assigned at $0.")
            st.button("Show Strikes", key="csp_strikes", use_container_width=True, disabled=True)

    with col2:
        st.markdown(f"### 🔵 Long Call (Exp: {report['long_call']['expiry']})")
        st.markdown(f"### 🔵 Long Call (Leverage)")
        st.caption(f"Expiration: {report['long_call']['expiry']}")
        with st.container(border=True):
            st.metric("Suggested Strike", f"${report['long_call']['strike']}")
            st.write(f"**Prob. ITM:** :blue[{report['long_call']['prob_itm']}]")
            call_fig, max_profit, max_loss = create_payoff_chart(
                'call',
                report['long_call']['strike'],
                report['long_call']['premium'],
                report['price']
            )
            st.plotly_chart(call_fig, use_container_width=True)
            
            c1, c2 = st.columns(2)
            c1.metric("Cost Basis", f"${report['long_call']['premium']}")
            c2.metric("Breakeven", f"${report['long_call']['breakeven']}")
            
            st.warning(f"⚠️ Requires ${report['price'] * 1.05:.2f} price target to hit 100% ROI.")
            c1.metric("Max Profit", "Unlimited")
            c2.metric("Max Risk", f"${-max_loss*100:,.2f}", help="Premium paid per contract.")
            st.button("Show Strikes", key="call_strikes", use_container_width=True, disabled=True)

    # --- DECISION LOGIC ---
    st.divider()
    if total_score > 75:
        st.success(f"**Engine Consensus:** Preference for **{report['long_call']['strike']} Call**. Alpha score is high enough to justify directional leverage.")
    else:
        st.info(f"**Engine Consensus:** Preference for **{report['csp']['strike']} Put**. Score suggests capturing high-probability premium over directional betting.")