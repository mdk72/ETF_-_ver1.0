import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# Charting Utilities
# -----------------------------------------------------------------------------

def plot_candle_chart(df, ticker, name, ma_list=[20, 60], ref_date=None, height=450): # Height 조정
    """
    Plot interactive candlestick chart with MA lines and Momentum Score (Dual Axis).
    Includes Volume panel and Buying Zone threshold.
    """
    if df.empty:
        return go.Figure()

    # Create Subplots: Row 1 = Returns/Price (Left) & Momentum (Right), Row 2 = Volume
    # Secondary_y=True allows dual axis on the first row
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.04, 
        row_heights=[0.85, 0.15],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )

    # --- 1. Candlestick (Price) ---
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Price',
        increasing_line_color='red', decreasing_line_color='blue'
    ), row=1, col=1, secondary_y=False)

    # --- 2. Moving Averages ---
    if ma_list is None:
        ma_list = []
    
    colors = ['orange', 'purple', 'green']
    for i, ma in enumerate(ma_list):
        col_name = f'MA_{ma}'
        if col_name not in df.columns:
            df[col_name] = df['Close'].rolling(window=ma).mean()
        
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col_name],
            mode='lines', name=f'MA {ma}',
            line=dict(width=1, color=colors[i % len(colors)])
        ), row=1, col=1, secondary_y=False)

    # --- 3. Momentum Score (Right Axis) ---
    # Need to calculate it on the fly if not present, but usually it is passed in the DF or pre-calc
    # If not in DF, valid logic: Score = (R3M*0.5 + R1M*0.3 + R1W*0.2) / Vol
    if 'MomentumScore' in df.columns:
         fig.add_trace(go.Scatter(
            x=df.index, y=df['MomentumScore'],
            mode='lines', name='Mom Score',
            line=dict(width=2, color='rgba(255, 165, 0, 0.7)', dash='dot'), # Orange dashed
            opacity=0.8
        ), row=1, col=1, secondary_y=True)
         
         # Buy Zone Threshold
         fig.add_hline(y=0.7, line_dash="dash", line_color="red", opacity=0.5, row=1, col=1, secondary_y=True, annotation_text="Buy Zone (0.7)")

    # --- 4. Volume (Bar Chart) ---
    colors_vol = ['red' if o < c else 'blue' for o, c in zip(df['Open'], df['Close'])]
    fig.add_trace(go.Bar(
        x=df.index, y=df['Volume'],
        name='Volume',
        marker_color=colors_vol,
        opacity=0.5
    ), row=2, col=1)

    # --- Layout Updates ---
    fig.update_layout(
        title=f'{name} ({ticker})',
        yaxis_title='Price',
        yaxis2_title='Score', # Right axis title
        height=height,
        margin=dict(l=20, r=20, t=40, b=80),  # Bottom margin for vertical dates
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    # 세로 날짜 표시 (Tick Angle -90)
    fig.update_xaxes(tickangle=-90, row=2, col=1) 

    # Add Reference Line (Analyze Date)
    if ref_date:
        fig.add_vline(x=ref_date, line_width=1, line_dash="dash", line_color="green", opacity=0.8)

    return fig

def render_market_breadth_chart(market_stats, height=350):
    """Render Market Breadth (Mean/Median Return) Chart"""
    if market_stats is None:
        return None
    
    # Handle DataFrame case
    if isinstance(market_stats, pd.DataFrame):
        if market_stats.empty:
            return None
        stats_df = market_stats
    else:
        stats_df = pd.DataFrame(market_stats)
        
        # Melt for multi-line plot
        melted = stats_df.melt(id_vars=['Date'], value_vars=['Mean Return', 'Median Return'], var_name='Metric', value_name='Return')
        
        fig = px.line(
            melted, x='Date', y='Return', color='Metric',
            title="시장 온도계 (최근 6개월 평균/중위 수익률)",
            color_discrete_map={'Mean Return': 'red', 'Median Return': 'blue'}
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        # 세로 날짜 표시
        fig.update_xaxes(tickangle=-90)
        
        fig.update_layout(
            height=height,
            margin=dict(l=20, r=20, t=40, b=80),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig
    return None
