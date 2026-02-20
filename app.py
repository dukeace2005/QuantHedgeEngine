# app.py
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time
import webbrowser
from datetime import datetime
from datetime import datetime, timedelta
from urllib.parse import quote_plus

# Import engines
from engine import QuantOptionEngine, QuantHedgeEngine

# Import modules
from index_funds import QuantAgent
from brokerage_interface import SchwabInterface
from calculations import DataProcessor, OptionsCalculator, track_time
from ui_components import ChartRenderer, MetricsDisplay, OptimizedGrid
from database import init_database, save_to_cache, get_recent_tickers
from market_data import fetch_market_data
from utils import create_nasdaq_header
from covered_calls import CoveredCallEngine, PositionInputs, ReportGenerator

# Page config
st.set_page_config(
    layout="wide", 
    page_title="Hedge Fund Options Suite",
    initial_sidebar_state="expanded",
    page_icon="📈"
)

# Load CSS
try:
    with open('styles/custom.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.markdown("""
    <style>
        section[data-testid="stSidebar"] { min-width: 350px !important; }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()
if 'options_calculator' not in st.session_state:
    st.session_state.options_calculator = OptionsCalculator()
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()
if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = None
if 'market_data' not in st.session_state:
    st.session_state.market_data = None
if 'base_report' not in st.session_state:
    st.session_state.base_report = None
if 'report' not in st.session_state:
    st.session_state.report = None
if 'selected_history_ticker' not in st.session_state:
    st.session_state.selected_history_ticker = None
if 'brokerage_connected' not in st.session_state:
    st.session_state.brokerage_connected = False
if 'show_auth_flow' not in st.session_state:
    st.session_state.show_auth_flow = False
if 'schwab_auth_processed' not in st.session_state:
    st.session_state.schwab_auth_processed = False
if 'schwab_auth_pending' not in st.session_state:
    st.session_state.schwab_auth_pending = False
if 'schwab_auth_expected_state' not in st.session_state:
    st.session_state.schwab_auth_expected_state = None
if 'schwab_auth_launched' not in st.session_state:
    st.session_state.schwab_auth_launched = False
if 'cc_report' not in st.session_state:
    st.session_state.cc_report = None
if 'analysis_mem_cache' not in st.session_state:
    st.session_state.analysis_mem_cache = {}

# Initialize database
conn = init_database()

# Initialize engines
@st.cache_resource(ttl=3600)
def get_engines():
    return QuantHedgeEngine(), QuantOptionEngine(risk_free_rate=0.045)

hedge_engine, option_engine = get_engines()

# --- SIDEBAR ---
with st.sidebar:
    col1, col2 = st.columns([6, 1])
    with col1:
        st.markdown("## **Analysis Controls**")
    
    with st.container():
        st.markdown("**Ticker Symbol**")
        ticker_input = st.text_input(
            "ticker",
            value=st.session_state.current_ticker or "AAPL",
            label_visibility="collapsed",
            help="Enter stock symbol (e.g., AAPL, MSFT, SPY)"
        ).upper().strip()
        
        run_analysis = st.button("Run Options Analysis", type="primary", width="stretch")
    
    # Recently viewed - ticker badges only
    recent_tickers = get_recent_tickers(conn, 5)
    if recent_tickers:
        st.markdown("### 🔍 **Quick View**")
        quick_items = recent_tickers[:3]

        if quick_items:
            cols = st.columns(3)
            for idx, item in enumerate(quick_items):
                with cols[idx]:
                    if st.button(
                        item['ticker'],
                        key=f"quick_{item['ticker']}",
                        help=f"Load {item['ticker']}",
                        width="stretch"
                    ):
                        ticker_input = item['ticker']
                        run_analysis = True
                        st.rerun()
    
    # Market Status
    st.markdown("### 📊 Market Status")
    current_hour = datetime.now().hour
    current_weekday = datetime.now().weekday()
    market_status = "🟢 Open" if (current_hour < 16 and current_weekday < 5) else "🔴 Closed"
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Status:** {market_status}")
    with col2:
        st.markdown(f"**Session:** {'Regular' if market_status == '🟢 Open' else 'Closed'}")
    
    # Advanced settings
    with st.expander("🔧 **Advanced Settings**", expanded=False):
        risk_tolerance = st.select_slider(
            "**Risk Profile**",
            options=["Conservative", "Moderate", "Aggressive"],
            value="Moderate"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            show_greeks = st.toggle("Show Greeks", value=True)
        with col2:
            st.session_state.debug_mode = st.toggle("Debug Mode", value=False)
        
        risk_ranges = {
            "Conservative": (0.70, 0.90), 
            "Moderate": (0.50, 0.80), 
            "Aggressive": (0.30, 0.70)
        }
        put_min_prob, put_max_prob = risk_ranges[risk_tolerance]
        st.caption(f"Put OTM range: {put_min_prob:.0%} - {put_max_prob:.0%}")
    
    # Brokerage Connection
    st.markdown("### 🏦 **Brokerage Connection**")
    brokerage_provider = st.selectbox(
        "Provider", 
        ["Charles Schwab"], 
        key="brokerage_select",
        label_visibility="collapsed"
    )
    
    status_text = "Connected" if st.session_state.brokerage_connected else "Disconnected"
    action_text = "Disconnect" if st.session_state.brokerage_connected else "Connect"
    auth_btn_label = f"🔌 {action_text}"
    if st.button(auth_btn_label, use_container_width=True, key="brokerage_connect_toggle"):
        if st.session_state.brokerage_connected:
            st.session_state.brokerage_connected = False
            st.session_state.show_auth_flow = False
            st.session_state.schwab_auth_pending = False
            st.session_state.schwab_auth_processed = False
            st.session_state.schwab_auth_expected_state = None
            st.session_state.schwab_auth_launched = False
            st.query_params.clear()
            st.rerun()
        elif brokerage_provider == "Charles Schwab":
            interface = SchwabInterface()
            success, msg = interface.authenticate_from_token()
            if success:
                st.session_state.brokerage_connected = True
                st.session_state.show_auth_flow = False
                st.session_state.schwab_auth_pending = False
                st.session_state.schwab_auth_processed = False
                st.session_state.schwab_auth_expected_state = None
                st.session_state.schwab_auth_launched = False
                st.query_params.clear()
                st.rerun()
            else:
                st.session_state.show_auth_flow = True
                st.session_state.schwab_auth_processed = False
                st.session_state.schwab_auth_pending = True
                st.session_state.schwab_auth_expected_state = None
                st.session_state.schwab_auth_launched = False
                st.rerun()

    code = st.query_params.get("code")
    state = st.query_params.get("state")
    has_oauth_callback = bool(code and state)

    if (st.session_state.get("show_auth_flow") or has_oauth_callback) and not st.session_state.brokerage_connected:
        interface = SchwabInterface()

        if (
            code and state
            and not st.session_state.get("schwab_auth_processed", False)
        ):
            expected_state = st.session_state.get("schwab_auth_expected_state")
            if (
                expected_state
                and st.session_state.get("schwab_auth_pending", False)
                and str(state) != expected_state
            ):
                st.warning("Ignoring stale OAuth callback. Please use the current authorize link.")
                st.query_params.clear()
                st.session_state.schwab_auth_processed = False
                st.stop()
            st.session_state.show_auth_flow = True
            st.session_state.schwab_auth_pending = True
            st.session_state.schwab_auth_processed = True
            callback_url = (
                f"{interface.redirect_url}"
                f"?code={quote_plus(str(code))}&state={quote_plus(str(state))}"
            )
            success, msg = interface.authenticate_from_url(callback_url)
            if success:
                st.session_state.brokerage_connected = True
                st.session_state.show_auth_flow = False
                st.session_state.schwab_auth_pending = False
                st.session_state.schwab_auth_expected_state = None
                st.session_state.schwab_auth_launched = False
                st.query_params.clear()
                st.rerun()
            else:
                st.session_state.schwab_auth_pending = False
                st.session_state.schwab_auth_expected_state = None
                st.session_state.schwab_auth_launched = False
                st.query_params.clear()
                st.error(msg)
        elif st.session_state.get("show_auth_flow") and st.session_state.get("schwab_auth_pending", False):
            auth_url = interface.get_auth_url()
            auth_context = st.session_state.get("schwab_auth_context")
            if auth_context is not None:
                expected_state = getattr(auth_context, "state", None)
                st.session_state.schwab_auth_expected_state = str(expected_state) if expected_state is not None else None
            if not st.session_state.get("schwab_auth_launched", False):
                webbrowser.open(auth_url, new=0)
                st.session_state.schwab_auth_launched = True
            st.info("Opened Schwab authorization in your browser. Complete login to continue.")
            st.stop()

    # Footer
    st.markdown("<br>" * 15, unsafe_allow_html=True)
    st.markdown("---")
    cache_size = len(get_recent_tickers(conn, 20))
    st.caption(f"⚙️ Quant Hedge Engine v2.0 | © 2026 | Cache: {cache_size}/20")

# --- MAIN CONTENT ---

# Handle refresh

# Handle history row selection
if st.session_state.selected_history_ticker:
    ticker_input = st.session_state.selected_history_ticker
    run_analysis = True
    st.session_state.selected_history_ticker = None
    st.rerun()

# Load most recent ticker from DB on startup
if not run_analysis and not st.session_state.current_ticker:
    recent = get_recent_tickers(conn, 1)
    if recent:
        st.session_state.current_ticker = recent[0]['ticker']
        ticker_input = recent[0]['ticker']
        run_analysis = True

analysis_requested = bool(run_analysis)
if analysis_requested:
    st.session_state.current_ticker = ticker_input

# Display analysis if we have it
if st.session_state.report and st.session_state.base_report:
        report = st.session_state.report
        base_report = st.session_state.base_report
        market_data = st.session_state.market_data
        
        if isinstance(report, dict) and 'error' in report:
            st.error(f"❌ {report['error']}")
            st.stop()
        
        # Nasdaq-style header
        st.markdown(create_nasdaq_header(
            ticker=st.session_state.current_ticker,
            report=report,
            base_report=base_report,
            market_data=market_data
        ), unsafe_allow_html=True)
        
        # Sidebar metrics update
        with st.sidebar:
            st.markdown("---")
            st.markdown("### 📈 **Current Analysis**")
            total_score = base_report.get('verdict', {}).get('score', 50)
            signal = base_report.get('verdict', {}).get('signal', 'HOLD')
            signal_class = {
                'BUY': 'signal-buy', 
                'HOLD': 'signal-hold', 
                'AVOID': 'signal-avoid'
            }.get(signal, 'signal-hold')
            signal_emoji = {'BUY': '🟢', 'HOLD': '🟡', 'AVOID': '🔴'}.get(signal, '⚪')
            
            st.markdown(
                f'<div class="signal-badge {signal_class}">{signal_emoji} {signal} | Score: {total_score}/100</div>',
                unsafe_allow_html=True
            )
            
            breakdown = base_report.get('logic_breakdown', {})
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Market Score",
                    f"{breakdown.get('market_score', 0)}/100",
                    help=breakdown.get('market_sentiment', '')
                )
            with col2:
                st.metric(
                    "Insider Score",
                    f"{breakdown.get('insider_score', 0)}/100",
                    help=breakdown.get('insider_activity', '')
                )
            
            trade_params = base_report.get('trade_parameters', {})
            if trade_params:
                st.markdown("**Risk Parameters:**")
                st.caption(f"🎯 Stop: ${trade_params.get('stop_loss', 0):.2f}")
                st.caption(f"📈 Target: ${trade_params.get('take_profit', 0):.2f}")
                st.caption(f"⚖️ R/R: {trade_params.get('risk_reward', 1.5):.1f}")
        
        # --- MAIN TABS ---
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 **Strategy Comparison**", 
            "📊 **Options Chain**", 
            "📋 **Risk Analysis**",
            "📜 **History**",
            "💵 **Index Fund CSP**",
            "⚙️ **Covered Calls**"
        ])
        
        with tab1:
            # Get probability ranges
            risk_ranges = {
                "Conservative": (0.70, 0.90),
                "Moderate": (0.50, 0.80),
                "Aggressive": (0.30, 0.70)
            }
            put_min_prob, put_max_prob = risk_ranges[risk_tolerance]

            with st.container():
                put_header = st.empty()
                put_header.markdown("### 💰 **Cash Secured Put**")
                with st.container(border=True):
                    with track_time("Put Processing"):
                        puts_df = pd.DataFrame(report['chain_data']['puts'])
                        
                        if not puts_df.empty:
                            puts_df = st.session_state.data_processor.process_option_subset(
                                report['chain_data']['puts'],
                                report['price'],
                                put_min_prob, 
                                put_max_prob,
                                'put'
                            )
                    
                    if not puts_df.empty:
                        puts_df['expiry_date'] = pd.to_datetime(puts_df['expiry'])
                        puts_df['dte_formatted'] = puts_df.apply(
                            lambda row: f"{row['expiry_date'].strftime('%m/%d')} {int(row['dte'])}d", 
                            axis=1
                        )
                        display_cols = puts_df[[
                                'strike', 'dte_formatted', 'lastPrice', 'prob_otm', 
                                'premium_yield', 'move_to_be', 'volume', 'openInterest'
                            ]].copy()                        

                        display_cols.columns = [
                            'Strike', 'DTE', 'Premium', 'Prob OTM', 
                            'Yield %', 'Move to BE %', 'Vol', 'OI'
                        ]
                        
                        display_cols['Strike'] = display_cols['Strike'].apply(lambda x: f"${x:.2f}")
                        display_cols['Premium'] = display_cols['Premium'].apply(lambda x: f"${x:.2f}")
                        display_cols['Prob OTM'] = display_cols['Prob OTM'].apply(lambda x: f"{x:.1%}")
                        display_cols['Yield %'] = display_cols['Yield %'].apply(lambda x: f"{x:.1f}%")
                        display_cols['Move to BE %'] = display_cols['Move to BE %'].apply(lambda x: f"{x:.1f}%")
                        
                        rec_strike = report['csp']['strike']
                        rec_idx = puts_df[puts_df['strike'] == rec_strike].index[0] if rec_strike in puts_df['strike'].values else 0
                        
                        selected_idx = OptimizedGrid.render(
                            display_cols,
                            rec_idx,
                            height=250,
                            key=f"csp_grid_{st.session_state.current_ticker}"
                        )
                        
                        if selected_idx is not None and 0 <= selected_idx < len(puts_df):
                            selected = puts_df.iloc[selected_idx]
                        else:
                            selected = puts_df.iloc[rec_idx]
                        
                        # Update header with selected option details
                        expiry_str = selected['expiry_date'].strftime('%b%d %y')
                        put_header.markdown(f"### 💰 **Sell @{float(selected['strike']):g} {expiry_str}**")
                        
                        fig = ChartRenderer.create_payoff_chart(
                            'put',
                            selected['strike'],
                            selected['lastPrice'],
                            report['price'],
                            selected.get('score_target', report['price'] * 1.05),
                            selected.get('downside_target', report['price'] * 0.95)
                        )
                        st.plotly_chart(fig, width="stretch")
                        
                        cols = st.columns(4)
                        with cols[0]:
                            st.metric("Credit", f"${selected['lastPrice']*100:,.0f}")
                        with cols[1]:
                            st.metric("Yield", f"{selected['premium_yield']:.1f}%")
                        with cols[2]:
                            st.metric("Prob OTM", f"{selected['prob_otm']:.1%}")
                        with cols[3]:
                            if show_greeks and selected['dte'] > 0:
                                greeks = st.session_state.options_calculator.calculate_greeks_batch(
                                    report['price'], 
                                    selected['strike'], 
                                    selected['dte']/365, 
                                    0.045, 
                                    selected.get('implied_vol', 0.3)
                                )
                                st.metric("Delta", f"{greeks['delta']:.2f}")
                            else:
                                if selected['dte'] > 0:
                                    annual_yield = (selected['premium_yield'] / selected['dte'] * 365)
                                    st.metric("Ann. Yield", f"{annual_yield:.1f}%")
                                else:
                                    st.metric("Ann. Yield", "N/A")
                    else:
                        st.warning("No puts found matching your criteria")
            
            with st.container():
                call_header = st.empty()
                call_header.markdown("### 📈 **Long Call**")
                with st.container(border=True):
                    with track_time("Call Processing"):
                        calls_df = pd.DataFrame(report['chain_data']['calls'])
                        
                        if not calls_df.empty:
                            call_min_prob = 1 - put_max_prob
                            call_max_prob = 1 - put_min_prob
                            
                            calls_df = st.session_state.data_processor.process_option_subset(
                                report['chain_data']['calls'],
                                report['price'],
                                call_min_prob, 
                                call_max_prob,
                                'call'
                            )
                    
                    if not calls_df.empty:
                        calls_df['expiry_date'] = pd.to_datetime(calls_df['expiry'])
                        calls_df['dte_formatted'] = calls_df.apply(
                            lambda row: f"{row['expiry_date'].strftime('%m/%d')} {int(row['dte'])}d", 
                            axis=1
                        )
                        display_cols = calls_df[[
                                'strike', 'dte_formatted', 'lastPrice', 'prob_itm', 
                                'premium_yield', 'move_to_be', 'volume', 'openInterest'
                            ]].copy()

                        display_cols.columns = [
                            'Strike', 'DTE', 'Premium', 'Prob ITM', 
                            'Cost %', 'Move to BE %', 'Vol', 'OI'
                        ]
                        
                        display_cols['Strike'] = display_cols['Strike'].apply(lambda x: f"${x:.2f}")
                        display_cols['Premium'] = display_cols['Premium'].apply(lambda x: f"${x:.2f}")
                        display_cols['Prob ITM'] = display_cols['Prob ITM'].apply(lambda x: f"{x:.1%}")
                        display_cols['Cost %'] = display_cols['Cost %'].apply(lambda x: f"{x:.1f}%")
                        display_cols['Move to BE %'] = display_cols['Move to BE %'].apply(lambda x: f"{x:.1f}%")
                        
                        rec_strike = report['long_call']['strike']
                        rec_idx = calls_df[calls_df['strike'] == rec_strike].index[0] if rec_strike in calls_df['strike'].values else 0
                        
                        selected_idx = OptimizedGrid.render(
                            display_cols,
                            rec_idx,
                            height=250,
                            key=f"call_grid_{st.session_state.current_ticker}"
                        )
                        
                        if selected_idx is not None and 0 <= selected_idx < len(calls_df):
                            selected = calls_df.iloc[selected_idx]
                        else:
                            selected = calls_df.iloc[rec_idx]
                        
                        # Update header with selected option details
                        expiry_str = selected['expiry_date'].strftime('%b%d %y')
                        call_header.markdown(f"### 📈 **Buy @{float(selected['strike']):g} {expiry_str}**")
                        
                        fig = ChartRenderer.create_payoff_chart(
                            'call',
                            selected['strike'],
                            selected['lastPrice'],
                            report['price'],
                            selected.get('score_target', report['price'] * 1.2),
                            selected.get('downside_target', report['price'] * 0.95)
                        )
                        st.plotly_chart(fig, width="stretch")
                        
                        cols = st.columns(4)
                        with cols[0]:
                            st.metric("Debit", f"${selected['lastPrice']*100:,.0f}")
                        with cols[1]:
                            if selected['lastPrice'] > 0:
                                leverage = selected['strike'] / selected['lastPrice']
                                st.metric("Leverage", f"{leverage:.1f}x")
                            else:
                                st.metric("Leverage", "N/A")
                        with cols[2]:
                            st.metric("Prob ITM", f"{selected['prob_itm']:.1%}")
                        with cols[3]:
                            if show_greeks and selected['dte'] > 0:
                                greeks = st.session_state.options_calculator.calculate_greeks_batch(
                                    report['price'], 
                                    selected['strike'], 
                                    selected['dte']/365, 
                                    0.045, 
                                    selected.get('implied_vol', 0.3)
                                )
                                st.metric("Delta", f"{greeks['delta']:.2f}")
                            else:
                                if selected['dte'] > 0:
                                    st.metric("DTE", f"{selected['dte']}")
                                else:
                                    st.metric("DTE", "0")
                    else:
                        st.warning("No calls found matching your criteria")
        
        with tab2:
            st.markdown("### 📊 **Full Options Chain Analysis**")
            
            all_puts = pd.DataFrame(report['chain_data']['puts'])
            all_calls = pd.DataFrame(report['chain_data']['calls'])
            
            if not all_puts.empty and not all_calls.empty:
                all_expiries = sorted(set(all_puts['expiry'].unique()) | set(all_calls['expiry'].unique()))
                
                selected_expiry = st.selectbox(
                    "Select Expiration",
                    all_expiries,
                    index=min(2, len(all_expiries)-1) if len(all_expiries) > 2 else 0
                )
                
                exp_calls = all_calls[all_calls['expiry'] == selected_expiry].copy()
                exp_puts = all_puts[all_puts['expiry'] == selected_expiry].copy()
                
                if not exp_calls.empty and not exp_puts.empty:
                    # Add probability metrics for full-chain view.
                    exp_calls['option_type'] = 'call'
                    exp_puts['option_type'] = 'put'
                    exp_calls = st.session_state.data_processor.calculator.calculate_chain_metrics_vectorized(
                        exp_calls,
                        report['price']
                    )
                    exp_puts = st.session_state.data_processor.calculator.calculate_chain_metrics_vectorized(
                        exp_puts,
                        report['price']
                    )

                    display_calls = exp_calls[['strike', 'lastPrice', 'prob_otm', 'impliedVolatility', 'volume', 'openInterest']].copy()
                    display_puts = exp_puts[['strike', 'lastPrice', 'prob_otm', 'impliedVolatility', 'volume', 'openInterest']].copy()
                    
                    display_calls.columns = ['Strike', 'Call Premium', 'Call Prob.OTM', 'Call IV', 'Call Vol', 'Call OI']
                    display_puts.columns = ['Strike', 'Put Premium', 'Put Prob.OTM', 'Put IV', 'Put Vol', 'Put OI']
                    
                    chain_view = pd.merge(display_calls, display_puts, on='Strike', how='outer')
                    chain_view = chain_view.sort_values('Strike')
                    
                    # Convert probabilities/IV to percentages before rounding to avoid zeroing small values.
                    chain_view['Call Prob.OTM'] = chain_view['Call Prob.OTM'] * 100
                    chain_view['Put Prob.OTM'] = chain_view['Put Prob.OTM'] * 100
                    chain_view['Call IV'] = chain_view['Call IV'] * 100
                    chain_view['Put IV'] = chain_view['Put IV'] * 100
                    chain_view = chain_view.round(2)
                    
                    st.dataframe(
                        chain_view,
                        width="stretch",
                        height=500,
                        column_config={
                            "Strike": st.column_config.NumberColumn("Strike", format="$%.2f"),
                            "Call Premium": st.column_config.NumberColumn("Call Premium", format="$%.2f"),
                            "Put Premium": st.column_config.NumberColumn("Put Premium", format="$%.2f"),
                            "Call Prob.OTM": st.column_config.NumberColumn("Prob.OTM", format="%.1f%%"),
                            "Put Prob.OTM": st.column_config.NumberColumn("Prob.OTM", format="%.1f%%"),
                            "Call IV": st.column_config.NumberColumn("Call IV", format="%.1f%%"),
                            "Put IV": st.column_config.NumberColumn("Put IV", format="%.1f%%"),
                        }
                    )
                    
                    st.markdown("#### Volatility Smile")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=chain_view['Strike'],
                        y=chain_view['Call IV'],
                        mode='lines+markers',
                        name='Call IV',
                        line=dict(color='#00BFFF')
                    ))
                    fig.add_trace(go.Scatter(
                        x=chain_view['Strike'],
                        y=chain_view['Put IV'],
                        mode='lines+markers',
                        name='Put IV',
                        line=dict(color='#FFA500')
                    ))
                    fig.add_vline(x=report['price'], line_dash="dash", line_color="white")
                    fig.update_layout(
                        title="Implied Volatility by Strike",
                        xaxis_title="Strike Price",
                        yaxis_title="Implied Volatility (%)",
                        template="plotly_dark",
                        height=400
                    )
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.warning("No options data for selected expiry")
            else:
                st.warning("No options chain data available")
        
        with tab3:
            st.markdown("### 📋 **Risk Analysis & Position Sizing**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### **Position Sizing Calculator**")
                
                capital = st.number_input(
                    "Trading Capital ($)",
                    min_value=1000,
                    max_value=10000000,
                    value=100000,
                    step=10000,
                    format="%d"
                )
                
                max_risk_pct = st.slider(
                    "Max Risk per Trade (%)",
                    min_value=0.5,
                    max_value=5.0,
                    value=2.0,
                    step=0.5
                ) / 100
                
                call_premium = report['long_call']['premium']
                put_premium = report['csp']['premium']
                
                call_risk = call_premium * 100
                put_risk = put_premium * 100
                
                max_call_contracts = int((capital * max_risk_pct) / call_risk) if call_risk > 0 else 0
                max_put_contracts = int((capital * max_risk_pct) / put_risk) if put_risk > 0 else 0
                
                st.markdown(f"""
                **Long Call:** {max_call_contracts} contracts
                - Cost: ${call_risk * max_call_contracts:,.0f}
                - Notional: ${max_call_contracts * 100 * report['long_call']['strike']:,.0f}
                
                **Cash-Secured Put:** {max_put_contracts} contracts
                - Collateral: ${put_risk * max_put_contracts:,.0f}
                - Premium received: ${put_premium * 100 * max_put_contracts:,.0f}
                """)
            
            with col2:
                st.markdown("#### **Greeks Exposure**")
                
                if show_greeks:
                    call_expiry = datetime.strptime(report['long_call']['expiry'], "%Y-%m-%d")
                    put_expiry = datetime.strptime(report['csp']['expiry'], "%Y-%m-%d")
                    today = datetime.now()
                    
                    call_dte = max((call_expiry - today).days, 1) / 365
                    put_dte = max((put_expiry - today).days, 1) / 365
                    
                    call_greeks = st.session_state.options_calculator.calculate_greeks_batch(
                        report['price'],
                        report['long_call']['strike'],
                        call_dte,
                        0.045,
                        0.3
                    )
                    
                    put_greeks = st.session_state.options_calculator.calculate_greeks_batch(
                        report['price'],
                        report['csp']['strike'],
                        put_dte,
                        0.045,
                        0.3
                    )
                    
                    greeks_df = pd.DataFrame({
                        'Greek': ['Delta', 'Gamma', 'Theta (daily)', 'Vega (1% IV)'],
                        'Long Call': [
                            f"{call_greeks['delta']:.3f}",
                            f"{call_greeks['gamma']:.4f}",
                            f"${call_greeks['theta']*100:.2f}",
                            f"${call_greeks['vega']*100:.2f}"
                        ],
                        'Cash-Secured Put': [
                            f"{-put_greeks['delta']:.3f}",
                            f"{put_greeks['gamma']:.4f}",
                            f"${-put_greeks['theta']*100:.2f}",
                            f"${put_greeks['vega']*100:.2f}"
                        ]
                    })
                    
                    st.dataframe(greeks_df, width="stretch", hide_index=True)
                    
                    with st.expander("📚 **Understanding Greeks**"):
                        st.markdown("""
                        - **Delta**: Rate of change of option price relative to underlying price
                        - **Gamma**: Rate of change of delta relative to underlying price
                        - **Theta**: Time decay (daily erosion of option value)
                        - **Vega**: Sensitivity to 1% change in implied volatility
                        """)
                else:
                    st.info("Enable 'Show Greeks' in settings to see option Greeks")
            
            st.markdown("#### **Engine Risk Assessment**")
            trade_params = base_report.get('trade_parameters', {})
            if trade_params:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Stop Loss", f"${trade_params.get('stop_loss', 0):.2f}")
                with col2:
                    st.metric("Take Profit", f"${trade_params.get('take_profit', 0):.2f}")
                with col3:
                    st.metric("Risk/Reward", f"{trade_params.get('risk_reward', 0):.1f}")
                with col4:
                    st.metric("ATR", f"${trade_params.get('atr', 0):.2f}")
        
        with tab4:
            st.markdown("### 📜 **Analysis History**")
            
            history = get_recent_tickers(conn, 20)
            if history:
                st.caption("Click a ticker badge to load analysis and update the header.")
                cols = st.columns(5)
                for idx, row in enumerate(history):
                    with cols[idx % 5]:
                        if st.button(row['ticker'], key=f"history_badge_{row['ticker']}_{idx}", width="stretch"):
                            st.session_state.selected_history_ticker = row['ticker']
                            st.rerun()
                
                if st.button("Clear Cache", width="stretch"):
                    c = conn.cursor()
                    c.execute('DELETE FROM analysis_cache')
                    conn.commit()
                    st.session_state.current_ticker = None
                    st.session_state.market_data = None
                    st.session_state.base_report = None
                    st.session_state.report = None
                    st.session_state.analysis_mem_cache = {}
                    st.rerun()
            else:
                st.info("No analysis history yet. Run some analyses to see them here.")
        
        with tab5:
            st.markdown("### Index Fund CSP Signal Generator")
            st.caption("Batch audit for index funds/leveraged ETFs with a combined report.")

            default_funds = ["XLI", "XLF", "TQQQ", "SPYM", "UPRO", "SOXL"]
            custom_tickers = st.text_input(
                "Tickers (comma-separated)",
                value=", ".join(default_funds),
                help="Example: XLI, XLF, TQQQ",
                key="index_funds_ticker_input"
            )
            run_index_analysis = st.button("Run Multi-Ticker Audit", key="run_multi_index_signal", width="stretch")

            if run_index_analysis:
                tickers = [t.strip().upper() for t in custom_tickers.split(",") if t.strip()]
                tickers = list(dict.fromkeys(tickers))

                if not tickers:
                    st.warning("Please provide at least one ticker.")
                else:
                    with st.spinner(f"Analyzing {len(tickers)} ticker(s)..."):
                        agent = QuantAgent(tickers)
                        st.session_state.index_fund_multi_result = agent.run_audit()
                        st.session_state.index_fund_multi_tickers = tickers

            if "index_fund_multi_result" in st.session_state:
                results = st.session_state.index_fund_multi_result
                if not results:
                    st.warning("No valid index fund data returned.")
                else:
                    display_data = []
                    for ticker, val in results.items():
                        strategy = val.get("strategy", {})
                        display_data.append({
                            "Ticker": ticker,
                            "Signal": val.get("signal", "N/A"),
                            "Status": val.get("status", "N/A"),
                            "Price": val.get("price", np.nan),
                            "RSI": val.get("rsi", np.nan),
                            "SMA 50": val.get("sma_50", np.nan),
                            "Delta": strategy.get("strike_delta", np.nan)
                        })

                    df = pd.DataFrame(display_data)

                    st.subheader("Combined Market Audit")
                    st.table(df)

                    sell_count = int((df["Status"] == "CSP").sum()) if "Status" in df.columns else 0
                    skip_count = int((df["Status"] == "SKIP").sum()) if "Status" in df.columns else 0
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Tickers Analyzed", len(df))
                    c2.metric("CSP Signals", sell_count)
                    c3.metric("SKIP Signals", skip_count)

                    st.header("Strategic Insights")
                    for ticker, val in results.items():
                        signal = val.get("signal", "N/A")
                        with st.expander(f"Detailed Analysis: {ticker} ({signal})"):
                            c1, c2 = st.columns(2)
                            with c1:
                                st.markdown("**Audit Analysis & Reasoning**")
                                st.info(val.get("audit_analysis", "No analysis available."))
                            with c2:
                                st.markdown("**Strategy Recommendation**")
                                st.success(val.get("strategy_recommendation", "No recommendation available."))

        with tab6:
            st.markdown("### Covered Call Strategy Engine")
            st.caption("Analyze covered call positions for optimal management (roll, close, or hold).")

            st.markdown("#### **Input Position for Analysis**")
            if 'cc_input_df' not in st.session_state:
                st.session_state.cc_input_df = pd.DataFrame({
                    "Parameter": [
                        "Ticker", "Shares Held", "Stock Cost Basis", "Current Stock Price",
                        "Option Strike", "Option Expiration (DTE)", "Option Entry Premium", "Current Option Mark",
                        "Implied Volatility (IV)", "IV Rank"
                    ],
                    "Value": [
                        "USAR", "100", "22.50", "18.30",
                        "24.50", "21",
                        "1.12", "0.66", "1.03", "5.38"
                    ]
                })

            st.session_state.cc_input_df = st.data_editor(
                st.session_state.cc_input_df,
                key="cc_input_editor",
                hide_index=True,
                use_container_width=True,
                num_rows="fixed",
                column_config={
                    "Parameter": st.column_config.TextColumn("Parameter", disabled=True),
                    "Value": st.column_config.TextColumn("Value")
                }
            )

            st.markdown("#### **Backtest Strikes (10 Values)**")
            try:
                _cc_current_price_for_strikes = float(
                    dict(zip(st.session_state.cc_input_df["Parameter"], st.session_state.cc_input_df["Value"]))["Current Stock Price"]
                )
            except (KeyError, TypeError, ValueError):
                _cc_current_price_for_strikes = 18.30

            # Build 10 strikes around current price in $0.50 increments.
            _cc_default_offsets = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5]
            _cc_default_strikes = [round(_cc_current_price_for_strikes + offset, 2) for offset in _cc_default_offsets]

            if 'cc_backtest_strikes_df' not in st.session_state:
                st.session_state.cc_backtest_strikes_df = pd.DataFrame({
                    "Strike": _cc_default_strikes
                })

            st.session_state.cc_backtest_strikes_df = st.data_editor(
                st.session_state.cc_backtest_strikes_df,
                key="cc_backtest_strikes_editor",
                hide_index=True,
                use_container_width=True,
                num_rows="fixed",
                column_config={
                    "Strike": st.column_config.NumberColumn("Strike", min_value=0.01, format="%.2f")
                }
            )

            run_cc_analysis = st.button("Run Covered Call Backtest", key="run_cc_analysis_btn", use_container_width=True)

            if run_cc_analysis:
                with st.spinner("Running Covered Call analysis..."):
                    input_rows = st.session_state.cc_input_df
                    input_map = dict(zip(input_rows["Parameter"], input_rows["Value"]))

                    try:
                        cc_ticker = str(input_map["Ticker"]).upper().strip()
                        cc_shares = int(float(input_map["Shares Held"]))
                        cc_stock_cost_basis = float(input_map["Stock Cost Basis"])
                        cc_current_stock_price = float(input_map["Current Stock Price"])
                        cc_option_strike = float(input_map["Option Strike"])
                        cc_dte_val = int(float(input_map["Option Expiration (DTE)"]))
                        cc_option_entry_premium = float(input_map["Option Entry Premium"])
                        cc_current_option_mark = float(input_map["Current Option Mark"])
                        cc_iv = float(input_map["Implied Volatility (IV)"])
                        cc_iv_rank = float(input_map["IV Rank"])
                    except (KeyError, TypeError, ValueError):
                        st.error("Invalid input values. Please ensure all fields contain valid numbers (except Ticker).")
                        st.stop()

                    if cc_shares <= 0 or cc_dte_val < 0:
                        st.error("Shares must be greater than 0 and DTE cannot be negative.")
                        st.stop()

                    backtest_strikes_raw = st.session_state.cc_backtest_strikes_df["Strike"].dropna().tolist()
                    strikes = sorted({float(x) for x in backtest_strikes_raw if float(x) > 0})
                    if not strikes:
                        st.error("Please provide at least one valid positive strike in Backtest Strikes.")
                        st.stop()

                    cc_option_expiration = datetime.now() + timedelta(days=cc_dte_val)
                    engine = CoveredCallEngine(risk_free_rate=0.05)
                    inputs = PositionInputs(
                        ticker=cc_ticker,
                        shares_held=cc_shares,
                        stock_cost_basis=cc_stock_cost_basis,
                        option_strike=cc_option_strike,
                        option_expiration=cc_option_expiration,
                        option_entry_premium=cc_option_entry_premium,
                        current_option_mark=cc_current_option_mark,
                        current_stock_price=cc_current_stock_price,
                        iv=cc_iv,
                        iv_rank=cc_iv_rank
                    )
                    expirations = [
                        datetime.now() + timedelta(days=30),
                        datetime.now() + timedelta(days=45)
                    ]
                    analysis = engine.analyze_position(inputs, strikes, expirations)
                    report = ReportGenerator().generate_report(analysis)
                    st.session_state.cc_report = report
            
            if st.session_state.cc_report:
                st.markdown("---")
                st.markdown("#### **Analysis Report**")
                st.markdown(st.session_state.cc_report, unsafe_allow_html=True)


# Initial-load tabs (no active analysis yet)
else:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 **Strategy Comparison**",
        "📊 **Options Chain**",
        "📋 **Risk Analysis**",
        "📜 **History**",
        "💵 **Index Fund CSP**",
        "⚙️ **Covered Calls**"
    ])

    with tab1:
        st.info("Run Options Analysis to enable this tab.")

    with tab2:
        st.info("Run Options Analysis to enable this tab.")

    with tab3:
        st.info("Run Options Analysis to enable this tab.")

    with tab4:
        st.markdown("### 📜 **Analysis History**")
        history = get_recent_tickers(conn, 20)
        if history:
            st.caption("Click a ticker badge to load analysis and update the header.")
            cols = st.columns(5)
            for idx, row in enumerate(history):
                with cols[idx % 5]:
                    if st.button(row['ticker'], key=f"history_badge_empty_{row['ticker']}_{idx}", width="stretch"):
                        st.session_state.selected_history_ticker = row['ticker']
                        st.rerun()
            if st.button("Clear Cache", key="clear_cache_empty", width="stretch"):
                c = conn.cursor()
                c.execute('DELETE FROM analysis_cache')
                conn.commit()
                st.session_state.current_ticker = None
                st.session_state.market_data = None
                st.session_state.base_report = None
                st.session_state.report = None
                st.session_state.analysis_mem_cache = {}
                st.rerun()
        else:
            st.info("No analysis history yet. Run some analyses to see them here.")

    with tab5:
        st.info("Run Options Analysis to enable this tab.")

    with tab6:
        st.markdown("### Covered Call Strategy Engine")
        st.caption("Analyze covered call positions for optimal management (roll, close, or hold).")

        st.markdown("#### **Input Position for Analysis**")
        if 'cc_input_df' not in st.session_state:
            st.session_state.cc_input_df = pd.DataFrame({
                "Parameter": [
                    "Ticker", "Shares Held", "Stock Cost Basis", "Current Stock Price",
                    "Option Strike", "Option Expiration (DTE)", "Option Entry Premium", "Current Option Mark",
                    "Implied Volatility (IV)", "IV Rank"
                ],
                "Value": [
                    "USAR", "100", "22.50", "18.30",
                    "24.50", "21",
                    "1.12", "0.66", "1.03", "5.38"
                ]
            })

        st.session_state.cc_input_df = st.data_editor(
            st.session_state.cc_input_df,
            key="cc_input_editor_empty",
            hide_index=True,
            use_container_width=True,
            num_rows="fixed",
            column_config={
                "Parameter": st.column_config.TextColumn("Parameter", disabled=True),
                "Value": st.column_config.TextColumn("Value")
            }
        )

        st.markdown("#### **Backtest Strikes (10 Values)**")
        try:
            _cc_current_price_for_strikes = float(
                dict(zip(st.session_state.cc_input_df["Parameter"], st.session_state.cc_input_df["Value"]))["Current Stock Price"]
            )
        except (KeyError, TypeError, ValueError):
            _cc_current_price_for_strikes = 18.30

        _cc_default_offsets = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5]
        _cc_default_strikes = [round(_cc_current_price_for_strikes + offset, 2) for offset in _cc_default_offsets]

        if 'cc_backtest_strikes_df' not in st.session_state:
            st.session_state.cc_backtest_strikes_df = pd.DataFrame({
                "Strike": _cc_default_strikes
            })

        st.session_state.cc_backtest_strikes_df = st.data_editor(
            st.session_state.cc_backtest_strikes_df,
            key="cc_backtest_strikes_editor_empty",
            hide_index=True,
            use_container_width=True,
            num_rows="fixed",
            column_config={
                "Strike": st.column_config.NumberColumn("Strike", min_value=0.01, format="%.2f")
            }
        )

        run_cc_analysis = st.button("Run Covered Call Backtest", key="run_cc_analysis_btn_empty", use_container_width=True)

        if run_cc_analysis:
            with st.spinner("Running Covered Call analysis..."):
                input_rows = st.session_state.cc_input_df
                input_map = dict(zip(input_rows["Parameter"], input_rows["Value"]))

                try:
                    cc_ticker = str(input_map["Ticker"]).upper().strip()
                    cc_shares = int(float(input_map["Shares Held"]))
                    cc_stock_cost_basis = float(input_map["Stock Cost Basis"])
                    cc_current_stock_price = float(input_map["Current Stock Price"])
                    cc_option_strike = float(input_map["Option Strike"])
                    cc_dte_val = int(float(input_map["Option Expiration (DTE)"]))
                    cc_option_entry_premium = float(input_map["Option Entry Premium"])
                    cc_current_option_mark = float(input_map["Current Option Mark"])
                    cc_iv = float(input_map["Implied Volatility (IV)"])
                    cc_iv_rank = float(input_map["IV Rank"])
                except (KeyError, TypeError, ValueError):
                    st.error("Invalid input values. Please ensure all fields contain valid numbers (except Ticker).")
                    st.stop()

                if cc_shares <= 0 or cc_dte_val < 0:
                    st.error("Shares must be greater than 0 and DTE cannot be negative.")
                    st.stop()

                backtest_strikes_raw = st.session_state.cc_backtest_strikes_df["Strike"].dropna().tolist()
                strikes = sorted({float(x) for x in backtest_strikes_raw if float(x) > 0})
                if not strikes:
                    st.error("Please provide at least one valid positive strike in Backtest Strikes.")
                    st.stop()

                cc_option_expiration = datetime.now() + timedelta(days=cc_dte_val)
                engine = CoveredCallEngine(risk_free_rate=0.05)
                inputs = PositionInputs(
                    ticker=cc_ticker,
                    shares_held=cc_shares,
                    stock_cost_basis=cc_stock_cost_basis,
                    option_strike=cc_option_strike,
                    option_expiration=cc_option_expiration,
                    option_entry_premium=cc_option_entry_premium,
                    current_option_mark=cc_current_option_mark,
                    current_stock_price=cc_current_stock_price,
                    iv=cc_iv,
                    iv_rank=cc_iv_rank
                )
                expirations = [
                    datetime.now() + timedelta(days=30),
                    datetime.now() + timedelta(days=45)
                ]
                analysis = engine.analyze_position(inputs, strikes, expirations)
                report = ReportGenerator().generate_report(analysis)
                st.session_state.cc_report = report
        
        if st.session_state.cc_report:
            st.markdown("---")
            st.markdown("#### **Analysis Report**")
            st.markdown(st.session_state.cc_report, unsafe_allow_html=True)

# Defer analysis execution until after page layout is rendered so tabs stay visible.
if analysis_requested:
    cached = st.session_state.analysis_mem_cache.get(ticker_input)
    cache_age = None
    if cached:
        cache_age = (datetime.now() - cached['timestamp']).seconds / 3600 if cached['timestamp'] else None

    if cached and cache_age and cache_age < 1:
        st.session_state.base_report = cached['base_report']
        st.session_state.report = cached['report']
        st.session_state.last_refresh = cached['timestamp']
        st.session_state.market_data = cached.get('market_data')
        st.success(f"✅ Loaded {ticker_input} from cache ({cache_age:.1f}h old)")
        st.rerun()
    else:
        progress_container = st.empty()
        status_container = st.empty()

        with progress_container.container():
            progress_bar = st.progress(0, text="Initializing...")
            status_container.info(f"🔍 Analyzing {ticker_input}...")

        try:
            progress_bar.progress(10, text="Fetching market data...")
            market_data = fetch_market_data(ticker_input)

            progress_bar.progress(20, text="Running fundamental analysis...")
            base_report = hedge_engine.analyze_ticker(ticker_input)
            if not base_report or 'error' in base_report:
                raise Exception(base_report.get('error', 'Unknown error'))

            progress_bar.progress(50, text="Scanning options chain...")
            total_score = base_report.get('verdict', {}).get('score', 50)
            report = option_engine.get_option_recommendations(ticker_input, total_score, 100000)
            if 'error' in report:
                raise Exception(report['error'])

            progress_bar.progress(80, text="Caching results...")
            st.session_state.market_data = market_data
            save_to_cache(conn, ticker_input)

            st.session_state.base_report = base_report
            st.session_state.report = report
            st.session_state.last_refresh = datetime.now()
            st.session_state.analysis_mem_cache[ticker_input] = {
                'base_report': base_report,
                'report': report,
                'market_data': market_data,
                'timestamp': st.session_state.last_refresh
            }

            progress_bar.progress(100, text="Complete!")
            status_container.success(f"✅ Analysis complete for {ticker_input}")
            time.sleep(1)
            progress_container.empty()
            status_container.empty()
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
