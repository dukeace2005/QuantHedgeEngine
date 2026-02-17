# app.py
"""
Optimized Hedge Fund Options Suite
Works with your existing QuantOptionEngine and QuantHedgeEngine
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time
import sqlite3
import json
from datetime import datetime, timedelta
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

# Import your existing engines (these are YOUR original files)
from engine import QuantOptionEngine, QuantHedgeEngine

# Import our new optimization modules (these are the files I provided)
from calculations import DataProcessor, OptionsCalculator, track_time
from ui_components import ChartRenderer, MetricsDisplay, OptimizedGrid

# Page config
st.set_page_config(
    layout="wide", 
    page_title="Hedge Fund Options Suite",
    initial_sidebar_state="expanded",
    page_icon="📈"
)

# Initialize local database for caching
@st.cache_resource
def init_database():
    """Initialize SQLite database for caching analysis results"""
    conn = sqlite3.connect('analysis_cache.db', check_same_thread=False)
    c = conn.cursor()
    
    # Create tables if they don't exist
    c.execute('''
        CREATE TABLE IF NOT EXISTS analysis_cache (
            ticker TEXT PRIMARY KEY,
            base_report TEXT,
            report TEXT,
            timestamp DATETIME,
            score INTEGER,
            signal TEXT
        )
    ''')
    
    # Create index on timestamp for cleanup
    c.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON analysis_cache(timestamp)')
    
    conn.commit()
    return conn

# Initialize session state
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()
if 'options_calculator' not in st.session_state:
    st.session_state.options_calculator = OptionsCalculator()
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'price_cache' not in st.session_state:
    st.session_state.price_cache = {}
if 'db_initialized' not in st.session_state:
    st.session_state.db_initialized = False
if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = None

# Initialize database
conn = init_database()

# Initialize engines
@st.cache_resource(ttl=3600)
def get_engines():
    return QuantHedgeEngine(), QuantOptionEngine(risk_free_rate=0.045)

hedge_engine, option_engine = get_engines()

# Custom CSS (same as before)
st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        background: linear-gradient(90deg, #1E1E1E 0%, #2D2D2D 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #00BFFF;
    }
    
    /* Metric cards */
    .metric-card {
        background: #1E1E1E;
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #00BFFF;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,191,255,0.2);
    }
    
    /* Signal badges - Improved contrast */
    .signal-badge {
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 1rem;
        text-align: center;
        width: 100%;
        font-size: 1.1rem;
        letter-spacing: 0.5px;
    }
    .signal-buy {
        background: rgba(0, 200, 100, 0.25);
        color: #FFFFFF;
        border: 2px solid #00FF88;
        text-shadow: 0 0 5px rgba(0, 255, 136, 0.5);
        box-shadow: 0 0 10px rgba(0, 255, 136, 0.3);
    }
    .signal-hold {
        background: rgba(255, 165, 0, 0.25);
        color: #FFFFFF;
        border: 2px solid #FFA500;
        text-shadow: 0 0 5px rgba(255, 165, 0, 0.5);
        box-shadow: 0 0 10px rgba(255, 165, 0, 0.3);
    }
    .signal-avoid {
        background: rgba(255, 68, 68, 0.25);
        color: #FFFFFF;
        border: 2px solid #FF4444;
        text-shadow: 0 0 5px rgba(255, 68, 68, 0.5);
        box-shadow: 0 0 10px rgba(255, 68, 68, 0.3);
    }
    
    /* Progress bar customization */
    .stProgress > div > div > div > div {
        background-color: #00BFFF;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
    }
    
    /* Sidebar minimum width */
    section[data-testid="stSidebar"] {
        min-width: 350px !important;
        max-width: 450px !important;
    }
    
    /* Header and refresh button alignment */
    div[data-testid="stSidebar"] div.element-container:has(button[key="refresh_btn"]) {
        margin-top: 0.5rem;
    }
    
    /* Borderless icon button for refresh */
    button[key="refresh_btn"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #00BFFF !important;
        font-size: 1.5rem !important;
        padding: 0 !important;
        height: 2.5rem !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    button[key="refresh_btn"]:hover {
        color: #FFFFFF !important;
        background: transparent !important;
        border: none !important;
        transform: scale(1.1);
        transition: transform 0.2s;
    }
    
    button[key="refresh_btn"]:active {
        transform: scale(0.95);
    }
    
    /* Restore original Run button style */
    button[data-testid="baseButton-primary"] {
        background-color: #00BFFF !important;
        color: white !important;
        border: none !important;
        font-weight: bold !important;
    }
    
    /* Remove any leftover emoji spacing */
    .stMarkdown {
        margin-bottom: 0 !important;
    }
    
    /* History selector styling */
    .history-selector {
        background: #1E1E1E;
        padding: 0.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 3px solid #00BFFF;
    }
</style>
""", unsafe_allow_html=True)

# --- DATABASE FUNCTIONS ---
def cleanup_old_cache(max_entries=20):
    """Keep only the most recent N entries in cache"""
    c = conn.cursor()
    # Get total count
    c.execute('SELECT COUNT(*) FROM analysis_cache')
    count = c.fetchone()[0]
    
    if count > max_entries:
        # Delete oldest entries beyond the limit
        c.execute('''
            DELETE FROM analysis_cache 
            WHERE timestamp IN (
                SELECT timestamp FROM analysis_cache 
                ORDER BY timestamp DESC 
                LIMIT -1 OFFSET ?
            )
        ''', (max_entries,))
        conn.commit()

def save_to_cache(ticker, base_report, report, score, signal):
    """Save analysis results to cache"""
    c = conn.cursor()
    
    # Convert to JSON for storage
    base_report_json = json.dumps(base_report, default=str)
    report_json = json.dumps(report, default=str)
    timestamp = datetime.now()
    
    # Upsert
    c.execute('''
        INSERT OR REPLACE INTO analysis_cache 
        (ticker, base_report, report, timestamp, score, signal)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (ticker, base_report_json, report_json, timestamp, score, signal))
    
    conn.commit()
    cleanup_old_cache(20)  # Keep max 20 entries

def load_from_cache(ticker):
    """Load analysis results from cache"""
    c = conn.cursor()
    c.execute('''
        SELECT base_report, report, timestamp, score, signal 
        FROM analysis_cache 
        WHERE ticker = ?
    ''', (ticker,))
    
    result = c.fetchone()
    if result:
        base_report, report, timestamp, score, signal = result
        return {
            'base_report': json.loads(base_report),
            'report': json.loads(report),
            'timestamp': datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else timestamp,
            'score': score,
            'signal': signal
        }
    return None

def get_recent_tickers(limit=10):
    """Get most recently analyzed tickers"""
    c = conn.cursor()
    c.execute('''
        SELECT ticker, timestamp, score, signal 
        FROM analysis_cache 
        ORDER BY timestamp DESC 
        LIMIT ?
    ''', (limit,))
    
    return [{'ticker': row[0], 'timestamp': row[1], 'score': row[2], 'signal': row[3]} 
            for row in c.fetchall()]

# --- SIDEBAR ---
with st.sidebar:
    # Header with refresh button aligned to the right
    col1, col2 = st.columns([6, 1])
    with col1:
        st.markdown("## **Analysis Controls**")
    with col2:
        st.markdown("")  # Empty markdown to help with alignment
        refresh_key = st.button(
            "🔄", 
            help="Refresh current analysis",
            key="refresh_btn",
            use_container_width=True
        )
    
    # Main input section
    with st.container():
        st.markdown("**Ticker Symbol**")
        ticker_input = st.text_input(
            "ticker",
            value=st.session_state.current_ticker or "AAPL",
            label_visibility="collapsed",
            help="Enter stock symbol (e.g., AAPL, MSFT, SPY)"
        ).upper().strip()
        
        # Run button
        run_analysis = st.button(
            "Run Options Analysis", 
            type="primary", 
            use_container_width=True,
            help="Click to analyze ticker"
        )
    
    # Recent history selector (only show if there's history)
    recent_tickers = get_recent_tickers(5)
    if recent_tickers:
        st.markdown("### 📜 **Recent Analysis**")
        with st.container():
            for item in recent_tickers:
                # Format time
                time_str = pd.to_datetime(item['timestamp']).strftime('%H:%M')
                
                # Signal emoji
                signal_emoji = {'BUY': '🟢', 'HOLD': '🟡', 'AVOID': '🔴'}.get(item['signal'], '⚪')
                
                # Create clickable button for each recent ticker
                if st.button(
                    f"{signal_emoji} {item['ticker']}  |  {item['score']}  |  {time_str}",
                    key=f"hist_{item['ticker']}",
                    use_container_width=True,
                    help=f"Load {item['ticker']} analysis from {time_str}"
                ):
                    ticker_input = item['ticker']
                    run_analysis = True
                    st.rerun()
    
    # Quick stats
    st.markdown("### 📊 Market Status")
    current_hour = datetime.now().hour
    current_weekday = datetime.now().weekday()
    market_status = "🟢 Open" if (current_hour < 16 and current_weekday < 5) else "🔴 Closed"
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Status:** {market_status}")
    with col2:
        st.markdown(f"**Session:** {'Regular' if market_status == '🟢 Open' else 'Closed'}")
    
    # Advanced settings expander
    with st.expander("🔧 **Advanced Settings**", expanded=False):
        risk_tolerance = st.select_slider(
            "**Risk Profile**",
            options=["Conservative", "Moderate", "Aggressive"],
            value="Moderate",
            help="Adjusts option selection and position sizing"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            show_greeks = st.toggle("Show Greeks", value=True, help="Display option Greeks")
        with col2:
            st.session_state.debug_mode = st.toggle("Debug Mode", value=False, help="Show performance metrics")
        
        # Map risk tolerance to probability ranges
        risk_ranges = {
            "Conservative": (0.70, 0.90),
            "Moderate": (0.50, 0.80),
            "Aggressive": (0.30, 0.70)
        }
        put_min_prob, put_max_prob = risk_ranges[risk_tolerance]
        
        st.caption(f"Put OTM range: {put_min_prob:.0%} - {put_max_prob:.0%}")
    
    # Push version to very bottom
    st.markdown("<br>" * 15, unsafe_allow_html=True)
    
    # Footer with version
    st.markdown("---")
    st.caption(f"⚙️ Quant Hedge Engine v2.0 | © 2026 | Cache: {len(recent_tickers)}/20")

# --- MAIN CONTENT ---
# Handle refresh button
if refresh_key:
    if st.session_state.current_ticker:
        # Clear caches
        st.session_state.data_processor._processed_cache.clear()
        st.session_state.price_cache.clear()
        st.cache_data.clear()
        
        # Rerun analysis with same ticker
        ticker_input = st.session_state.current_ticker
        run_analysis = True
        st.rerun()

# Try to load from cache first if no explicit run
if not run_analysis and not st.session_state.current_ticker:
    # Load most recent from cache
    recent = get_recent_tickers(1)
    if recent:
        st.session_state.current_ticker = recent[0]['ticker']
        cached = load_from_cache(recent[0]['ticker'])
        if cached:
            st.session_state.base_report = cached['base_report']
            st.session_state.report = cached['report']
            st.session_state.last_refresh = cached['timestamp']

# Run analysis (either new or from cache)
if run_analysis or st.session_state.current_ticker:
    
    if run_analysis:
        # Store current ticker
        st.session_state.current_ticker = ticker_input
        
        # Check cache first
        cached = load_from_cache(ticker_input)
        cache_age = None
        if cached:
            cache_age = (datetime.now() - cached['timestamp']).seconds / 3600  # hours
        
        # Use cache if less than 1 hour old, otherwise fetch new data
        if cached and cache_age and cache_age < 1 and not refresh_key:
            st.session_state.base_report = cached['base_report']
            st.session_state.report = cached['report']
            st.session_state.last_refresh = cached['timestamp']
            st.success(f"✅ Loaded {ticker_input} from cache ({cache_age:.1f}h old)")
        else:
            # Show progress
            progress_container = st.empty()
            status_container = st.empty()
            
            with progress_container.container():
                progress_bar = st.progress(0, text="Initializing analysis...")
                status_container.info(f"🔍 Analyzing {ticker_input}...")
            
            try:
                with st.spinner(""):
                    # Stage 1: Fundamental analysis
                    progress_bar.progress(20, text="Running fundamental analysis...")
                    with track_time("Fundamental Analysis"):
                        base_report = hedge_engine.analyze_ticker(ticker_input)
                    
                    if not base_report or 'error' in base_report:
                        raise Exception(base_report.get('error', 'Unknown error'))
                    
                    # Stage 2: Options chain analysis
                    progress_bar.progress(50, text="Scanning options chain...")
                    with track_time("Options Analysis"):
                        total_score = base_report.get('verdict', {}).get('score', 50)
                        report = option_engine.get_option_recommendations(
                            ticker_input, 
                            total_score, 
                            100000
                        )
                    
                    if 'error' in report:
                        raise Exception(report['error'])
                    
                    # Stage 3: Cache the results
                    progress_bar.progress(80, text="Caching results...")
                    signal = base_report.get('verdict', {}).get('signal', 'HOLD')
                    save_to_cache(ticker_input, base_report, report, total_score, signal)
                    
                    # Store in session state
                    st.session_state.base_report = base_report
                    st.session_state.report = report
                    st.session_state.last_refresh = datetime.now()
                    
                    # Complete
                    progress_bar.progress(100, text="Analysis complete!")
                    status_container.success(f"✅ Analysis complete for {ticker_input}")
                    
                    time.sleep(1)
                    progress_container.empty()
                    status_container.empty()
                    
            except Exception as e:
                st.error(f"❌ Error analyzing {ticker_input}: {str(e)}")
                st.exception(e)
                st.stop()
    
    # Display analysis if we have it
    if 'report' in st.session_state and 'base_report' in st.session_state:
        report = st.session_state.report
        base_report = st.session_state.base_report
        
        if isinstance(report, dict) and 'error' in report:
            st.error(f"❌ {report['error']}")
            st.stop()
        
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
            
            signal_emoji = {
                'BUY': '🟢',
                'HOLD': '🟡',
                'AVOID': '🔴'
            }.get(signal, '⚪')
            
            st.markdown(
                f'<div class="signal-badge {signal_class}">'
                f'{signal_emoji} {signal} | Score: {total_score}/100'
                f'</div>',
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
        
        # --- MAIN TABS (same as before) ---
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 **Strategy Comparison**", 
            "📊 **Options Chain**", 
            "📋 **Risk Analysis**",
            "📜 **History**"
        ])
        
        with tab1:
            # Header with current price
            st.markdown(f"""
            <h2 style='text-align: center;'>
                {st.session_state.current_ticker} @ ${report['price']:.2f}
                <span style='color: #888; font-size: 0.8rem; margin-left: 1rem;'>
                Updated: {st.session_state.last_refresh.strftime('%H:%M:%S')}
                </span>
            </h2>
            """, unsafe_allow_html=True)
            
            # Get probability ranges based on risk tolerance
            risk_ranges = {
                "Conservative": (0.70, 0.90),
                "Moderate": (0.50, 0.80),
                "Aggressive": (0.30, 0.70)
            }
            put_min_prob, put_max_prob = risk_ranges[risk_tolerance]
            
            # Two-column layout for strategies
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 💰 **Cash Secured Put**")
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
                        
                        fig = ChartRenderer.create_payoff_chart(
                            'put',
                            selected['strike'],
                            selected['lastPrice'],
                            report['price'],
                            selected.get('score_target', report['price'] * 1.05),
                            selected.get('downside_target', report['price'] * 0.95)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
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
            
            with col2:
                st.markdown("### 📈 **Long Call**")
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
                        
                        fig = ChartRenderer.create_payoff_chart(
                            'call',
                            selected['strike'],
                            selected['lastPrice'],
                            report['price'],
                            selected.get('score_target', report['price'] * 1.2),
                            selected.get('downside_target', report['price'] * 0.95)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
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
                    display_calls = exp_calls[['strike', 'lastPrice', 'bid', 'ask', 'impliedVolatility', 'volume', 'openInterest']].copy()
                    display_puts = exp_puts[['strike', 'lastPrice', 'bid', 'ask', 'impliedVolatility', 'volume', 'openInterest']].copy()
                    
                    display_calls.columns = ['Strike', 'Call Price', 'Call Bid', 'Call Ask', 'Call IV', 'Call Vol', 'Call OI']
                    display_puts.columns = ['Strike', 'Put Price', 'Put Bid', 'Put Ask', 'Put IV', 'Put Vol', 'Put OI']
                    
                    chain_view = pd.merge(display_calls, display_puts, on='Strike', how='outer')
                    chain_view = chain_view.sort_values('Strike').round(2)
                    
                    chain_view['Call IV'] = chain_view['Call IV'] * 100
                    chain_view['Put IV'] = chain_view['Put IV'] * 100
                    
                    st.dataframe(
                        chain_view,
                        use_container_width=True,
                        height=500,
                        column_config={
                            "Strike": st.column_config.NumberColumn("Strike", format="$%.2f"),
                            "Call Price": st.column_config.NumberColumn("Call", format="$%.2f"),
                            "Put Price": st.column_config.NumberColumn("Put", format="$%.2f"),
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
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No options data for selected expiry")
            else:
                st.warning("No options chain data available")
        
        with tab3:
            st.markdown("### 📋 **Risk Analysis & Position Sizing**")
            # ... (rest of tab3 remains the same)
            
        with tab4:
            st.markdown("### 📜 **Analysis History**")
            
            history = get_recent_tickers(20)
            if history:
                history_df = pd.DataFrame(history)
                history_df['timestamp'] = pd.to_datetime(history_df['timestamp']).dt.strftime('%H:%M %m/%d')
                
                st.dataframe(
                    history_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "score": st.column_config.ProgressColumn(
                            "Score",
                            format="%d/100",
                            min_value=0,
                            max_value=100
                        )
                    }
                )
                
                if st.button("Clear Cache", use_container_width=True):
                    c = conn.cursor()
                    c.execute('DELETE FROM analysis_cache')
                    conn.commit()
                    st.session_state.current_ticker = None
                    if 'report' in st.session_state:
                        del st.session_state.report
                        del st.session_state.base_report
                    st.rerun()
            else:
                st.info("No analysis history yet. Run some analyses to see them here.")

# No analysis and no history
else:
    st.markdown("""
    <div style='text-align: center; padding: 2rem;'>
        <h2 style='color: #00BFFF;'>👋 Welcome to Quant Hedge Options Suite</h2>
        <p style='color: #888; font-size: 1rem;'>
            Enter a ticker symbol in the sidebar and click <strong>Run</strong> to start analysis
        </p>
    </div>
    """, unsafe_allow_html=True)