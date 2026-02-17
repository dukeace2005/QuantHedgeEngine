import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
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

    fig.add_vline(x=strike, line_width=1, line_dash="dot", line_color="orange", annotation_text=f"Strike: ${strike:.2f}", annotation_position="top")
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

def render_option_selector_grid(grid_df, rec_index, numeric_formatters, key, height=250):
    """Render a single-select AgGrid with click selection and auto-scroll to recommended row."""
    if grid_df.empty:
        return None

    safe_rec_index = max(0, min(int(rec_index), len(grid_df) - 1))
    work_df = grid_df.copy()
    work_df['row_id'] = work_df.index

    gb = GridOptionsBuilder.from_dataframe(work_df)
    gb.configure_default_column(resizable=True, sortable=True, filter=False)
    for col_name, formatter in numeric_formatters.items():
        gb.configure_column(col_name, type=["numericColumn"], valueFormatter=formatter)
    gb.configure_column("row_id", hide=True)
    gb.configure_selection("single", use_checkbox=False, pre_selected_rows=[safe_rec_index])
    gb.configure_grid_options(
        suppressRowClickSelection=False,
        getRowStyle=JsCode(
            """
            function(params) {
                if (params.node && params.node.isSelected && params.node.isSelected()) {
                    return { backgroundColor: 'rgba(46, 204, 113, 0.2)' };
                }
                return null;
            }
            """
        ),
        onFirstDataRendered=JsCode(
            f"""
            function(params) {{
                const idx = {safe_rec_index};
                params.api.ensureIndexVisible(idx, 'middle');
                const node = params.api.getDisplayedRowAtIndex(idx);
                if (node) {{
                    node.setSelected(true);
                }}
            }}
            """
        ),
    )

    grid_response = AgGrid(
        work_df,
        gridOptions=gb.build(),
        height=height,
        fit_columns_on_grid_load=True,
        allow_unsafe_jscode=True,
        update_on=["selectionChanged"],
        key=key,
    )

    selected_rows = grid_response.get("selected_rows", [])
    if isinstance(selected_rows, pd.DataFrame):
        selected_rows = selected_rows.to_dict("records")

    if selected_rows:
        return int(selected_rows[0]["row_id"])
    return safe_rec_index

fragment_decorator = st.fragment if hasattr(st, "fragment") else (lambda fn: fn)

@fragment_decorator
def render_csp_panel(report, ticker_input):
    with st.container(border=True):
        chart_placeholder = st.empty()

        st.markdown("**Strike Selection (50-90% OTM)**")

        puts_data = pd.DataFrame(report['chain_data']['puts'])
        puts_data['prob_otm'] = 1 - puts_data['prob_itm']

        subset = puts_data[(puts_data['prob_otm'] >= 0.50) & (puts_data['prob_otm'] <= 0.90)].copy()
        subset['Return'] = (subset['lastPrice'] / subset['strike']) * 100
        subset['Prob. OTM'] = subset['prob_otm'] * 100
        subset['Move to BE'] = ((report['price'] - (subset['strike'] - subset['lastPrice'])) / report['price']) * 100
        subset = subset.sort_values(['expiry', 'strike'], ascending=[True, True]).reset_index(drop=True)

        display_cols = subset[['strike', 'expiry', 'lastPrice', 'Prob. OTM', 'Return', 'Move to BE', 'volume', 'openInterest']]
        display_cols.columns = ['Strike', 'Expiry', 'Premium', 'Prob. OTM', 'Return', 'Move to BE', 'Vol', 'OI']

        rec_strike = report['csp']['strike']
        rec_expiry = report['csp']['expiry']
        rec_matches = subset[(subset['strike'] == rec_strike) & (subset['expiry'] == rec_expiry)]
        rec_index = int(rec_matches.index[0]) if not rec_matches.empty else 0

        selected_row_id = render_option_selector_grid(
            display_cols,
            rec_index,
            {
                "Strike": "x.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})",
                "Premium": "x.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})",
                "Prob. OTM": "x.toLocaleString(undefined, {minimumFractionDigits: 1, maximumFractionDigits: 1}) + '%'",
                "Return": "x.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}) + '%'",
                "Move to BE": "x.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}) + '%'",
            },
            key=f"csp_grid_{ticker_input}",
        )

        if selected_row_id is not None and 0 <= selected_row_id < len(subset):
            row = subset.iloc[selected_row_id]
            sel_strike, sel_prem, sel_exp = row['strike'], row['lastPrice'], row['expiry']
            sel_target, sel_down = row['score_target'], row['downside_target']
        else:
            sel_strike, sel_prem, sel_exp = report['csp']['strike'], report['csp']['premium'], report['csp']['expiry']
            sel_target, sel_down = report['csp']['score_target'], report['csp']['downside_target']

        with chart_placeholder.container():
            st.caption(f"Expiration: {sel_exp}")
            csp_fig, _, _ = create_payoff_chart(
                'put', sel_strike, sel_prem, report['price'], sel_target, sel_down
            )
            st.plotly_chart(csp_fig, width="stretch")

@fragment_decorator
def render_call_panel(report, ticker_input):
    with st.container(border=True):
        call_chart_placeholder = st.empty()

        st.markdown("**Strike Selection (15-90% ITM)**")

        calls_data = pd.DataFrame(report['chain_data']['calls'])
        call_subset = calls_data[(calls_data['prob_itm'] >= 0.15) & (calls_data['prob_itm'] <= 0.90)].copy()
        call_subset['Prob. ITM'] = call_subset['prob_itm'] * 100
        call_subset['Move to BE'] = (((call_subset['strike'] + call_subset['lastPrice']) / report['price']) - 1) * 100
        call_subset = call_subset.sort_values(['expiry', 'strike'], ascending=[True, True]).reset_index(drop=True)

        call_display_cols = call_subset[['strike', 'expiry', 'lastPrice', 'Prob. ITM', 'Move to BE', 'volume', 'openInterest']]
        call_display_cols.columns = ['Strike', 'Expiry', 'Premium', 'Prob. ITM', 'Move to BE', 'Vol', 'OI']

        rec_call_strike = report['long_call']['strike']
        rec_call_expiry = report['long_call']['expiry']
        rec_call_matches = call_subset[(call_subset['strike'] == rec_call_strike) & (call_subset['expiry'] == rec_call_expiry)]
        rec_call_index = int(rec_call_matches.index[0]) if not rec_call_matches.empty else 0

        selected_call_row_id = render_option_selector_grid(
            call_display_cols,
            rec_call_index,
            {
                "Strike": "x.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})",
                "Premium": "x.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})",
                "Prob. ITM": "x.toLocaleString(undefined, {minimumFractionDigits: 1, maximumFractionDigits: 1}) + '%'",
                "Move to BE": "x.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}) + '%'",
            },
            key=f"call_grid_{ticker_input}",
        )

        if selected_call_row_id is not None and 0 <= selected_call_row_id < len(call_subset):
            selected_call_row = call_subset.iloc[selected_call_row_id]
            sel_call_strike = selected_call_row['strike']
            sel_call_premium = selected_call_row['lastPrice']
            sel_call_expiry = selected_call_row['expiry']
            sel_call_target = selected_call_row['score_target']
            sel_call_downside = selected_call_row['downside_target']
            sel_call_prob_itm = selected_call_row['prob_itm']
        else:
            sel_call_strike = report['long_call']['strike']
            sel_call_premium = report['long_call']['premium']
            sel_call_expiry = report['long_call']['expiry']
            sel_call_target = report['long_call']['score_target']
            sel_call_downside = report['long_call']['downside_target']
            sel_call_prob_itm = float(str(report['long_call']['prob_itm']).strip('%')) / 100.0

        with call_chart_placeholder.container():
            st.caption(f"Expiration: {sel_call_expiry}")
            call_fig, proj_profit, _ = create_payoff_chart(
                'call',
                sel_call_strike,
                sel_call_premium,
                report['price'],
                sel_call_target,
                sel_call_downside
            )
            st.plotly_chart(call_fig, width="stretch")

        risk = sel_call_premium
        ror = proj_profit / risk if risk > 0 else 0

        c1, c2 = st.columns(2)
        c1.metric("Risk", f"${risk*100:,.0f}", help="Maximum capital at risk.")
        c2.metric("Return on Risk", f"{ror:.1%}", help=f"Return if stock hits target (${sel_call_target:.2f}).")
        c3, c4 = st.columns(2)
        c3.metric("Est. Profit", f"${proj_profit*100:,.0f}", help=f"P&L at target (${sel_call_target:.2f}).")
        c4.metric("Prob. ITM", f"{sel_call_prob_itm:.1%}", help="Probability of expiring In-The-Money.")

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
        st.session_state['base_report'] = hedge_engine.analyze_ticker(ticker_input)

    # 2. Fetch Option Recommendations
    with st.spinner(f"Finding optimal option strategies for {ticker_input}..."):
        total_score = st.session_state['base_report'].get('verdict', {}).get('score', 50)
        st.session_state['report'] = option_engine.get_option_recommendations(ticker_input, total_score, 100000)

if 'report' in st.session_state:
    report = st.session_state['report']
    base_report = st.session_state['base_report']
    
    # Sidebar Metrics (Re-render)
    total_score = base_report.get('verdict', {}).get('score', 50)
    breakdown = base_report.get('logic_breakdown', {})
    st.sidebar.metric("Underlying Score", f"{total_score}/100", delta=f"{total_score-50}",
                        help=breakdown.get('market_sentiment'))
    st.sidebar.caption(f"Market: {breakdown.get('market_score', '-')}/100 | Insider: {breakdown.get('insider_score', '-')}/100")
    
    st.subheader(f"⚖️ Strategy Comparison: {ticker_input} @ ${report['price']}")
    
    # Create Side-by-Side "Bumper" Cards
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Cash Secured Put (Yield)")
        render_csp_panel(report, ticker_input)

    with col2:
        st.markdown("### Long Call (Leverage)")
        render_call_panel(report, ticker_input)

    # --- DECISION LOGIC ---
    st.divider()
    if total_score > 75:
        st.success(f"**Engine Consensus:** Preference for **{report['long_call']['strike']} Call**. Alpha score is high enough to justify directional leverage.")
    else:
        st.info(f"**Engine Consensus:** Preference for **{report['csp']['strike']} Put**. Score suggests capturing high-probability premium over directional betting.")
