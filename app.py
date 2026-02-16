import streamlit as st
import plotly.graph_objects as go
import numpy as np
from engine import QuantOptionEngine, QuantHedgeEngine # Import both engines

st.set_page_config(layout="wide", page_title="Hedge Fund Options Suite")

# --- CHARTING FUNCTION ---
def create_payoff_chart(strategy_type, strike, premium, current_price, target_price, downside_price):
    """Generates a Plotly payoff diagram for a given option strategy."""
    price_range = np.linspace(current_price * 0.8, current_price * 1.2, 200)
    
    if strategy_type == 'call':
        pnl = np.maximum(0, price_range - strike) - premium
        breakeven = strike + premium
        # Calculate P&L at dynamic targets
        proj_profit = np.maximum(0, target_price - strike) - premium
        proj_loss = np.maximum(0, downside_price - strike) - premium

    elif strategy_type == 'put':
        pnl = premium - np.maximum(0, strike - price_range)
        breakeven = strike - premium
        # Calculate P&L at dynamic targets
        proj_profit = premium - np.maximum(0, strike - target_price)
        proj_loss = premium - np.maximum(0, strike - downside_price)

    fig = go.Figure()

    # P&L Line
    fig.add_trace(go.Scatter(x=price_range, y=pnl, mode='lines', name='Profit/Loss', line=dict(color='#00BFFF', width=3)))

    # Profit/Loss Shading
    fig.add_trace(go.Scatter(x=price_range[pnl >= 0], y=pnl[pnl >= 0], fill='tozeroy', mode='none', fillcolor='rgba(0, 255, 127, 0.2)', name='Profit'))
    fig.add_trace(go.Scatter(x=price_range[pnl < 0], y=pnl[pnl < 0], fill='tozeroy', mode='none', fillcolor='rgba(255, 69, 0, 0.2)', name='Loss'))

    # Key Lines
    # Dynamic label positioning to avoid overlap
    if current_price < breakeven:
        lbl_current, lbl_break = "top left", "top right"
    else:
        lbl_current, lbl_break = "top right", "top left"

    fig.add_vline(x=strike, line_width=1, line_dash="dot", line_color="white", annotation_text=f"Strike: ${strike:.2f}", annotation_position="top")
    fig.add_vline(x=current_price, line_width=2, line_dash="dash", line_color="yellow", annotation_text=f"Current: ${current_price:.2f}", annotation_position=lbl_current)
    fig.add_vline(x=breakeven, line_width=2, line_dash="dash", line_color="cyan", annotation_text=f"Breakeven: ${breakeven:.2f}", annotation_position=lbl_break)
    fig.add_vline(x=target_price, line_width=2, line_dash="dot", line_color="green", annotation_text=f"Target: ${target_price:.2f}", annotation_position="bottom right")

    # Styling
    fig.update_layout(
        title=dict(text=f"{'Long Call' if strategy_type == 'call' else 'Cash-Secured Put'} Payoff", x=0.5),
        xaxis_title="Stock Price at Expiration",
        yaxis_title="Profit / Loss per Share",
        showlegend=False,
        template="plotly_dark",
        yaxis=dict(zerolinecolor='grey', zerolinewidth=1),
    )
    return fig, proj_profit, proj_loss

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
        breakdown = base_report.get('logic_breakdown', {})
        st.sidebar.metric("Underlying Score", f"{total_score}/100", delta=f"{total_score-50}",
                          help=breakdown.get('market_sentiment'))
        st.sidebar.caption(f"Market: {breakdown.get('market_score', '-')}/100 | Insider: {breakdown.get('insider_score', '-')}/100")

    # 2. Fetch Option Recommendations
    with st.spinner(f"Finding optimal option strategies for {ticker_input}..."):
        report = option_engine.get_option_recommendations(ticker_input, total_score, 100000)
    
    st.subheader(f"⚖️ Strategy Comparison: {ticker_input} @ ${report['price']}")
    
    # Create Side-by-Side "Bumper" Cards
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### 🟢 Cash Secured Put (Yield)")
        st.caption(f"Expiration: {report['csp']['expiry']}")
        with st.container(border=True):
            csp_fig, proj_profit, proj_loss = create_payoff_chart(
                'put',
                report['csp']['strike'],
                report['csp']['premium'],
                report['price'],
                report['csp']['score_target'],
                report['csp']['downside_target']
            )
            st.plotly_chart(csp_fig, use_container_width=True)
            
            strike = report['csp']['strike']
            premium = report['csp']['premium']
            prob_itm = float(report['csp']['prob_itm'].strip('%'))

            c1, c2 = st.columns(2)
            c1.metric("Collateral", f"${strike*100:,.0f}", help="Cash required to secure the put.")
            c2.metric("Return on Collateral", f"{(premium/strike):.1%}", help="Return if unassigned.")
            c3, c4 = st.columns(2)
            c3.metric("Premium", f"${premium*100:,.0f}", help="Immediate cash received.")
            c4.metric("Prob. OTM", f"{100-prob_itm:.1f}%", help="Probability of expiring worthless.")
            st.button("Show Strikes", key="csp_strikes", use_container_width=True, disabled=True)

    with col2:
        st.markdown(f"### 🔵 Long Call (Leverage)")
        st.caption(f"Expiration: {report['long_call']['expiry']}")
        with st.container(border=True):
            call_fig, proj_profit, proj_loss = create_payoff_chart(
                'call',
                report['long_call']['strike'],
                report['long_call']['premium'],
                report['price'],
                report['long_call']['score_target'],
                report['long_call']['downside_target']
            )
            st.plotly_chart(call_fig, use_container_width=True)
            
            premium = report['long_call']['premium']
            risk = premium
            ror = proj_profit / risk if risk > 0 else 0

            c1, c2 = st.columns(2)
            c1.metric("Risk", f"${risk*100:,.0f}", help="Maximum capital at risk.")
            c2.metric("Return on Risk", f"{ror:.1%}", help=f"Return if stock hits target (${report['long_call']['score_target']:.2f}).")
            c3, c4 = st.columns(2)
            c3.metric("Est. Profit", f"${proj_profit*100:,.0f}", help=f"P&L at target (${report['long_call']['score_target']:.2f}).")
            c4.metric("Prob. ITM", report['long_call']['prob_itm'], help="Probability of expiring In-The-Money.")
            st.button("Show Strikes", key="call_strikes", use_container_width=True, disabled=True)

    # --- DECISION LOGIC ---
    st.divider()
    if total_score > 75:
        st.success(f"**Engine Consensus:** Preference for **{report['long_call']['strike']} Call**. Alpha score is high enough to justify directional leverage.")
    else:
        st.info(f"**Engine Consensus:** Preference for **{report['csp']['strike']} Put**. Score suggests capturing high-probability premium over directional betting.")