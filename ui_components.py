# ui_components.py
"""
Reusable UI components with optimized rendering
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

class ChartRenderer:
    """Optimized chart rendering"""
    
    @staticmethod
    @st.cache_data(ttl=300, max_entries=50)
    def create_payoff_chart(strategy_type, strike, premium, current_price, target_price, downside_price):
        """Cached chart generation with improved label placement"""
        
        # Optimize point count based on range
        price_range = np.linspace(current_price * 0.7, current_price * 1.3, 120)
        
        if strategy_type == 'call':
            pnl = np.maximum(0, price_range - strike) - premium
            breakeven = strike + premium
            strategy_name = "Long Call"
        else:
            pnl = premium - np.maximum(0, strike - price_range)
            breakeven = strike - premium
            strategy_name = "Cash-Secured Put"
        
        # Create optimized figure
        fig = go.Figure()
        
        # Single trace with gradient fill
        fig.add_trace(go.Scatter(
            x=price_range,
            y=pnl,
            mode='lines',
            line=dict(color='#00BFFF', width=2.5),
            fill='tozeroy',
            fillcolor='rgba(0, 191, 255, 0.1)',
            name='P&L',
            hovertemplate='Price: $%{x:.2f}<br>P&L: $%{y:.2f}<extra></extra>'
        ))
        
        # Add key levels with intelligent label placement
        # Always show strike (most important)
        fig.add_vline(
            x=strike,
            line_width=1.5,
            line_dash="dot",
            line_color="orange",
            annotation_text=f"Strike: ${strike:.2f}",
            annotation_position="top",
            annotation_font=dict(size=10, color="orange")
        )
        
        # Show breakeven (second most important)
        # Calculate position to avoid overlap
        price_range_width = price_range.max() - price_range.min()
        if abs(strike - breakeven) < price_range_width * 0.1:
            # If strike and breakeven are close, put breakeven label at bottom
            be_position = "bottom"
        else:
            be_position = "top"
            
        fig.add_vline(
            x=breakeven,
            line_width=2,
            line_dash="dash",
            line_color="cyan",
            annotation_text=f"B/E: ${breakeven:.2f}",
            annotation_position=be_position,
            annotation_font=dict(size=10, color="cyan")
        )
        
        # Only show target price if it's significantly different from breakeven and strike
        target_diff_from_be = abs(target_price - breakeven) / current_price
        target_diff_from_strike = abs(target_price - strike) / current_price
        
        if target_diff_from_be > 0.02 and target_diff_from_strike > 0.02:
            # Place target label at bottom right to avoid top clutter
            fig.add_vline(
                x=target_price,
                line_width=1.5,
                line_dash="dot",
                line_color="#00FF88",
                annotation_text=f"Target: ${target_price:.2f}",
                annotation_position="bottom right",
                annotation_font=dict(size=10, color="#00FF88")
            )
        
        # Only show downside price for puts and if significantly different
        if strategy_type == 'put':
            downside_diff_from_be = abs(downside_price - breakeven) / current_price
            downside_diff_from_strike = abs(downside_price - strike) / current_price
            
            if downside_diff_from_be > 0.02 and downside_diff_from_strike > 0.02:
                fig.add_vline(
                    x=downside_price,
                    line_width=1.5,
                    line_dash="dot",
                    line_color="#FF4444",
                    annotation_text=f"Stop: ${downside_price:.2f}",
                    annotation_position="top left" if be_position == "bottom" else "bottom left",
                    annotation_font=dict(size=10, color="#FF4444")
                )
        
        # Add horizontal zero line for reference
        fig.add_hline(
            y=0,
            line_width=1,
            line_color="rgba(255, 255, 255, 0.3)",
            line_dash="solid"
        )
        
        # Optimize layout
        fig.update_layout(
            title={
                'text': f"{strategy_name} Payoff Diagram",
                'x': 0.5,
                'xanchor': 'center',
                'font': dict(size=14, color='white')
            },
            xaxis_title="Stock Price at Expiry",
            yaxis_title="Profit / Loss ($)",
            template="plotly_dark",
            showlegend=False,
            height=350,
            margin=dict(l=50, r=50, t=50, b=50),
            hovermode='x unified',
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                tickformat='$.2f'
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                zeroline=True,
                zerolinecolor='rgba(255, 255, 255, 0.3)',
                tickformat='$.2f'
            )
        )
        
        return fig

class MetricsDisplay:
    """Optimized metrics display with HTML"""
    
    @staticmethod
    def metric_card(title, value, delta=None, help_text=None, color=None, size='medium'):
        """HTML metric card for better performance"""
        size_class = {
            'small': '0.9rem',
            'medium': '1.2rem',
            'large': '1.8rem'
        }
        
        color_style = f"color: {color};" if color else ""
        
        delta_html = f'<div style="color: #00FF88; font-size: 0.7rem;">{delta}</div>' if delta else ''
        help_html = f'<div style="color: #666; font-size: 0.65rem; margin-top: 0.3rem;">{help_text}</div>' if help_text else ''
        
        return f"""
        <div style="background: #1E1E1E; padding: 1rem; border-radius: 8px; 
                    border-left: 3px solid #00BFFF; margin: 0.3rem 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
            <div style="color: #888; font-size: 0.75rem; text-transform: uppercase; 
                        letter-spacing: 0.5px;">{title}</div>
            <div style="font-size: {size_class[size]}; font-weight: bold; 
                        {color_style} margin: 0.2rem 0;">{value}</div>
            {delta_html}
            {help_html}
        </div>
        """

class OptimizedGrid:
    """Optimized AgGrid renderer"""
    
    @staticmethod
    @st.cache_data(ttl=60, max_entries=20)
    def prepare_grid_data(df, formatters):
        """Prepare and cache grid data"""
        if df.empty:
            return df
        
        work_df = df.copy()
        return work_df
    
    @staticmethod
    def render(df, rec_index, height=250, key=None):
        """Render optimized grid with proper highlighting"""
        if df.empty:
            return None
        
        # Add row_id for selection
        work_df = df.copy()
        work_df['row_id'] = range(len(work_df))
        
        # Safe rec_index
        safe_idx = max(0, min(rec_index, len(work_df) - 1))
        
        # Configure grid
        gb = GridOptionsBuilder.from_dataframe(work_df)
        gb.configure_default_column(
            resizable=True,
            sortable=True,
            filter=False,
            cellStyle={'color': '#111111', 'borderColor': '#dcdcdc'}
        )
        
        # Hide row_id
        gb.configure_column("row_id", hide=True)
        
        # Configure selection with pre-selected row
        gb.configure_selection(
            "single",
            use_checkbox=False,
            pre_selected_rows=[safe_idx]
        )
        
        # Grid options with row styling
        gb.configure_grid_options(
            suppressRowClickSelection=False,
            rowHeight=35,
            headerHeight=40,
            getRowStyle=JsCode("""
                function(params) {
                    if (params.node && params.node.selected) {
                        return { 
                            backgroundColor: 'rgba(0, 191, 255, 0.3) !important',
                            color: '#111111',
                            fontWeight: 'bold',
                            borderLeft: '3px solid #00BFFF'
                        };
                    }
                    if (params.data && params.data.row_id === params.context.selectedRow) {
                        return { 
                            backgroundColor: 'rgba(0, 191, 255, 0.3) !important',
                            color: '#111111',
                            fontWeight: 'bold',
                            borderLeft: '3px solid #00BFFF'
                        };
                    }
                    if (params.rowIndex % 2 === 0) {
                        return { backgroundColor: '#ffffff', color: '#111111' };
                    }
                    return { backgroundColor: '#f2f2f2', color: '#111111' };
                }
            """),
            onFirstDataRendered=JsCode(
                f"""
                function(params) {{
                    setTimeout(function() {{
                        if (params && params.api) {{
                            const idx = {safe_idx};
                            params.api.forEachNode(function(node) {{
                                if (node.rowIndex === idx) {{
                                    node.setSelected(true);
                                    params.api.ensureIndexVisible(idx, 'middle');
                                }}
                            }});
                        }}
                    }}, 100);
                }}
                """
            )
        )
        
        # Set context with selected row
        grid_options = gb.build()
        grid_options['context'] = {'selectedRow': safe_idx}
        
        # Render
        grid_response = AgGrid(
            work_df,
            gridOptions=grid_options,
            height=height,
            fit_columns_on_grid_load=True,
            allow_unsafe_jscode=True,
            update_on=['selectionChanged'],
            key=key,
            theme='streamlit',
            custom_css={
                ".ag-header": {
                    "background-color": "#e6e6e6 !important",
                    "border-bottom": "1px solid #d0d0d0 !important"
                },
                ".ag-header-cell-text": {
                    "color": "#111111 !important",
                    "font-weight": "600 !important"
                },
                ".ag-cell": {
                    "color": "#111111 !important",
                    "border-color": "#dcdcdc !important"
                }
            },
            reload_data=True  # Force reload to ensure highlighting
        )
        
        # Get selection
        selected = grid_response.get('selected_rows', [])
        if isinstance(selected, pd.DataFrame) and not selected.empty:
            return int(selected.iloc[0]['row_id'])
        elif isinstance(selected, list) and selected:
            return int(selected[0]['row_id'])
        
        return safe_idx

class UIOptimizer:
    """Optimize UI rendering and state management"""
    
    @staticmethod
    def lazy_load_component(component_func, key, *args, **kwargs):
        """Only render component when needed"""
        if key not in st.session_state:
            st.session_state[key] = False
        
        if st.session_state[key] or kwargs.get('force_render', False):
            result = component_func(*args, **kwargs)
            st.session_state[key] = True
            return result
        return None
    
    @staticmethod
    def debounce_input(wait=0.5):
        """Debounce input changes"""
        def decorator(func):
            def wrapped(*args, **kwargs):
                key = f"debounce_{func.__name__}"
                if key not in st.session_state:
                    st.session_state[key] = 0
                
                import time
                current_time = time.time()
                if current_time - st.session_state[key] > wait:
                    st.session_state[key] = current_time
                    return func(*args, **kwargs)
            return wrapped
        return decorator
