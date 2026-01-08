import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import concurrent.futures
import io

from utils_config import load_user_config, on_config_change, ETF_UNIVERSE, save_user_config, load_etf_holdings
from data_manager import get_stock_data_cached, fetch_market_data
from db_manager import log_report_generation, get_report_history, delete_report
from analysis_engine import (
    get_top_etfs, analyze_overlapping_stocks_report, 
    run_simulation, run_advanced_simulation, calculate_post_stats,
    add_momentum_columns, calculate_momentum_score, check_market_defense
)
from chart_utils import plot_candle_chart, render_market_breadth_chart

# -----------------------------------------------------------------------------
# 1. Momentum Ranking UI (Tab 1)
# -----------------------------------------------------------------------------
def render_momentum_ranking(rank_df, data_map, min_score=None):
    """모멘텀 랭킹 상세 화면 (메인 탭 1)"""
    # [Fix] Manager Filter for Ranking View
    config = load_user_config()
    with st.expander("랭킹 필터 설정", expanded=False):
        c1, c2 = st.columns([1, 1])
        with c1:
            # Filter managers based on current rank_df (which is already filtered by category)
            if not rank_df.empty:
                available_tickers = set(rank_df['Ticker'])
                available_managers = sorted(list(set(
                    ETF_UNIVERSE[t]['manager'] for t in available_tickers 
                    if t in ETF_UNIVERSE and 'manager' in ETF_UNIVERSE[t]
                )))
                all_managers = ["전체"] + available_managers
            else:
                all_managers = ["전체"]

            saved_manager = config.get('rank_manager', ["전체"])
            def_man = [saved_manager] if isinstance(saved_manager, str) else saved_manager
            
            # [Fix] Streamlit API Exception prevent: Ensure default values are in options
            valid_defaults = [m for m in def_man if m in all_managers]
            if not valid_defaults: valid_defaults = ["전체"]
            
            sel_manager = st.multiselect("운용사 필터", all_managers, default=valid_defaults, key="rank_man", on_change=on_config_change)
            if not sel_manager or "전체" in sel_manager: sel_manager = "전체"
            
    # Apply Manager Filter
    if sel_manager != "전체" and not rank_df.empty:
        # Get list of allowed managers
        target_managers = set(sel_manager)
        # Filter rank_df
        # We need to map Ticker -> Manager to filter
        # optimization: create local map
        ticker_to_man = {t: ETF_UNIVERSE[t]['manager'] for t in rank_df['Ticker'] if t in ETF_UNIVERSE}
        rank_df = rank_df[rank_df['Ticker'].map(ticker_to_man).isin(target_managers)]

    # [Fix] 최소 점수 필터링
    if min_score is not None and not rank_df.empty:
        rank_df = rank_df[rank_df['Score'] >= min_score]
        
    # 1. 상단 섹션: 랭킹 테이블 & 캔들 차트
    col1, col2 = st.columns([4.8, 5.2])
    selected_ticker = _render_ranking_table_section(col1, rank_df)
    _render_ranking_chart_section(col2, selected_ticker, rank_df, data_map)
    
    st.divider()

    # 2. 하단 섹션: ETF 보유종목 & 개별 종목 차트
    col3, col4 = st.columns([4.8, 5.2])
    selected_stock = _render_holdings_table_section(col3, selected_ticker)
    _render_stock_chart_section(col4, selected_stock)

def _render_ranking_table_section(col, rank_df):
    """모멘텀 랭킹 테이블 렌더링"""
    with col:
        st.markdown("### 모멘텀 랭킹")
        if rank_df.empty:
            st.warning(f"조건을 만족하는 ETF가 없습니다.")
            return None
            
        display_df = rank_df[['ShortName', 'Close', 'Score', 'R_1w', 'R_1m', 'R_3m', 'Vol_20d']].copy()
        display_df['Close'] = display_df['Close'].apply(lambda x: f"{x:,.0f}")
        display_df['Score'] = display_df['Score'].apply(lambda x: f"{x:.2f}")
        for c in ['R_1w', 'R_1m', 'R_3m']:
            display_df[c] = display_df[c].apply(lambda x: f"{x*100:.1f}%")
        display_df['Vol_20d'] = display_df['Vol_20d'].apply(lambda x: f"{x:.4f}")
        display_df.columns = ['Name', 'Price', 'Score', '1W', '1M', '3M', 'Vol']
        
        event = st.dataframe(
            display_df, width="stretch", height=420,
            on_select="rerun", selection_mode="single-row", hide_index=False,
            key="ranking_table",
            column_config={
                "Name": st.column_config.TextColumn("이름", width="medium"),
                "Price": st.column_config.TextColumn("현재가", width="small"),
                "Score": st.column_config.NumberColumn("점수", format="%.2f", width="small"),
                "1W": st.column_config.TextColumn("1주", width="small"),
                "1M": st.column_config.TextColumn("1달", width="small"),
                "3M": st.column_config.TextColumn("3달", width="small"),
            }
        )
        
        selected_ticker = rank_df.iloc[0]['Ticker']
        if len(event.selection.rows) > 0:
            selected_ticker = rank_df.iloc[event.selection.rows[0]]['Ticker']
        return selected_ticker

def _render_ranking_chart_section(col, ticker, rank_df, data_map):
    """선택된 ETF의 상세 차트 분석 렌더링"""
    with col:
        st.markdown("### 차트 분석")
        if not ticker or ticker not in ETF_UNIVERSE: return
        
        info = ETF_UNIVERSE[ticker]
        st.caption(f"{info['theme']} | {info['manager']}")
        
        sel_df = data_map.get(ticker, pd.DataFrame()).copy()
        if not sel_df.empty:
            # 보조지표 계산 (1년치 시각화 최적화)
            sel_df['MA_20'] = sel_df['Close'].rolling(20).mean()
            sel_df['MA_60'] = sel_df['Close'].rolling(60).mean()
            if len(sel_df) > 252: sel_df = sel_df.iloc[-252:]
            
            fig = plot_candle_chart(sel_df, ticker, info['name'])
            fig.update_layout(
                title_font_size=15, height=420, margin=dict(t=40, b=40, l=20, r=20),
                xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#f1f5f9')
            )
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{ticker}")

def _render_holdings_table_section(col, ticker):
    """선택된 ETF의 보유 종목 Top 10 테이블 렌더링"""
    with col:
        st.markdown("### 개별종목 Top10")
        if not ticker:
            st.info("ETF를 선택하세요.")
            return None
            
        holdings = load_etf_holdings(ticker)
        if not holdings:
            st.info("보유종목 데이터가 없습니다.")
            return None

        # 성과 지표 포함 가공
        processed = []
        # [Fix] Date precision stability for caching & dataframe consistency
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=130)
        
        for h in holdings:
            h_ticker = h['ticker']
            rets = {'ret_1w': 0, 'ret_1m': 0, 'ret_3m': 0}
            try:
                # Pass date objects or string to ensure stable cache keys
                df = get_stock_data_cached(h_ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
                if not df.empty and len(df) > 10:
                    last_p = df['Close'].iloc[-1]
                    if len(df) >= 6: rets['ret_1w'] = (last_p / df['Close'].iloc[-6] - 1) * 100
                    if len(df) >= 21: rets['ret_1m'] = (last_p / df['Close'].iloc[-21] - 1) * 100
                    if len(df) >= 61: rets['ret_3m'] = (last_p / df['Close'].iloc[-61] - 1) * 100
            except: pass
            processed.append({**h, **rets})
        
        h_df = pd.DataFrame(processed)
        for c in ['ret_1w', 'ret_1m', 'ret_3m']:
            h_df[c] = h_df[c].apply(lambda x: f"{x:.1f}%")
            
        h_event = st.dataframe(
            h_df, width="stretch", height=420, on_select="rerun", 
            selection_mode="single-row", hide_index=True,
            # [Fix] Unique key per ETF to prevent state collisions, ensuring selection state persists for the active table
            key=f"holdings_table_{ticker}",
            column_config={"name": st.column_config.TextColumn("종목명", width="medium"), "pct": st.column_config.TextColumn("비중", width="small")}
        )
        
        st.session_state['current_holdings'] = holdings
        st.session_state['holdings_event'] = h_event
        
        selected_stock = holdings[0]
        if h_event and len(h_event.selection.rows) > 0:
            idx = h_event.selection.rows[0]
            if idx < len(holdings): selected_stock = holdings[idx]
        return selected_stock

def _render_stock_chart_section(col, stock):
    """개별 종목의 상세 분석 차트 렌더링"""
    with col:
        st.markdown("### 개별 종목")
        if not stock: return
        
        with st.spinner(f"{stock['name']} 로딩..."):
            df = get_stock_data_cached(stock['ticker'], datetime.now()-timedelta(days=365), datetime.now())
            if not df.empty:
                df = add_momentum_columns(df)
                fig = plot_candle_chart(df, stock['ticker'], stock['name'])
                fig.update_layout(height=420, margin=dict(t=30, b=20, l=10, r=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("데이터 없음")

# -----------------------------------------------------------------------------
# 2. Backtest History UI (Tab 2)
# -----------------------------------------------------------------------------
def render_backtest_ui(current_rank_df, data_map):
    """백테스트 과거 시점 분석 화면 (메인 탭 2)"""
    st.markdown("### 백테스트 (과거 시점 및 기간 분석)")
    
    config = load_user_config()
    config = load_user_config()
    sel_manager, min_score, top_n_etf, exclude_risky = _render_backtest_settings_section(config, data_map)
    
    # 1. 히스토리 테이블 섹션
    target_date = _render_backtest_history_section(config, min_score, sel_manager, data_map, exclude_risky, top_n_etf)
    
    # 2. 상세 결과 섹션 (버튼 클릭 시)
    _render_backtest_detail_section(target_date, data_map, sel_manager, min_score, top_n_etf, exclude_risky)
    
    # 3. 시장 온도계 섹션
    _render_market_breadth_section()

def _render_backtest_settings_section(config, data_map):
    """백테스트 상단 필터 설정 영역"""
    with st.expander("분석 설정", expanded=False):
        st.markdown("---")
        c1, c2 = st.columns([1, 1])
        with c1:
            # [Fix] Filter managers based on current data_map (which is already filtered by category)
            # data_map keys are tickers. We need to look up their manager in ETF_UNIVERSE
            if data_map:
                available_tickers = set(data_map.keys())
                available_managers = sorted(list(set(
                    ETF_UNIVERSE[t]['manager'] for t in available_tickers 
                    if t in ETF_UNIVERSE and 'manager' in ETF_UNIVERSE[t]
                )))
                all_managers = ["전체"] + available_managers
            else:
                all_managers = ["전체"]

            saved_manager = config.get('manager', '전체')
            def_man = saved_manager if isinstance(saved_manager, list) else ([saved_manager] if saved_manager != "전체" else ["전체"])
            sel_manager = st.multiselect("운용사 필터", all_managers, default=def_man, key="bt_man", on_change=on_config_change)
            if not sel_manager or "전체" in sel_manager: sel_manager = "전체"
            
        with c2:
            min_score = st.slider("최소 모멘텀 점수", 0.0, 3.0, float(config.get('min_score', 0.5)), 0.1, key="bt_score", on_change=on_config_change)
            top_n_etf = st.number_input("분석 대상 ETF 수 (Top N)", 1, 30, int(config.get('bt_top_n', 5)), key="bt_top_n", on_change=on_config_change)
            exclude_risky = st.checkbox("위험 ETF 제외", value=config.get('exclude_risky', True), key="bt_risk", on_change=on_config_change)
    return sel_manager, min_score, top_n_etf, exclude_risky

def _render_backtest_history_section(config, min_score, sel_manager, data_map, exclude_risky, top_n_etf):
    """백테스트 기간 조회 및 결과 테이블 렌더링"""
    st.markdown("### 기간별 선정 내역 (Trend)")
    rh1, rh2, rh3 = st.columns([1, 1, 1])
    
    def_h_start = datetime.strptime(config.get('bt_h_start_str', "2025-04-01"), "%Y-%m-%d")
    def_h_end = datetime.strptime(config.get('bt_h_end_str', "2025-05-31"), "%Y-%m-%d")
    
    with rh1: h_start = st.date_input("시작일", value=def_h_start, key="bt_h_start", on_change=on_config_change)
    with rh2: h_end = st.date_input("종료일", value=def_h_end, key="bt_h_end", on_change=on_config_change)
    
    progress_placeholder = st.empty()
    with rh3:
        st.markdown('<div style="margin-top: 28px;"></div>', unsafe_allow_html=True)
        if st.button("기간 내역 조회", key="bt_btn_range", type="primary", width="stretch"):
            with progress_placeholder.container():
                with st.spinner("기간 분석 중..."):
                    stats_df, log_df = run_simulation(h_start, h_end, None, min_score, sel_manager, data_map, freq='B', exclude_risky=exclude_risky, top_n_etf=top_n_etf)
                    st.session_state['bt_history_log_v2'], st.session_state['bt_market_stats_v2'] = log_df, stats_df

    target_date = None
    log_df = st.session_state.get('bt_history_log_v2')
    if log_df is not None and not log_df.empty:
        event_hist = st.dataframe(
            log_df, width="stretch", height=450, on_select="rerun", selection_mode="single-row", hide_index=True, key="bt_hist_table_v2",
            column_order=["날짜", "운용사", "선정 ETF", "대표 주도주 (Top 5)", "ETF 수", "평균 점수"],
            column_config={"날짜": st.column_config.DateColumn("날짜", format="YYYY-MM-DD"), "ETF 수": st.column_config.NumberColumn("추천수", format="%d개"), "평균 점수": st.column_config.NumberColumn("강도", format="%.2f")}
        )
        if len(event_hist.selection.rows) > 0:
            target_date = log_df.iloc[event_hist.selection.rows[0]]['날짜']
    return target_date

def _render_backtest_detail_section(target_date, data_map, sel_manager, min_score, top_n_etf, exclude_risky):
    """특정 시점 클릭 또는 버튼 클릭 시 상세 분석 결과 렌더링"""
    st.markdown("### 특정 시점 상세 분석")
    if 'bt_date' not in st.session_state: st.session_state['bt_date'] = datetime.now().date()
    
    # [Fix] Date Picker Callback to prevent Rerun Loop
    def on_date_change():
        st.session_state['bt_date'] = st.session_state['bt_date_picker']
        st.session_state['bt_detail_cache'] = None
        st.session_state['bt_selected_stock'] = None

    if target_date and st.session_state['bt_date'] != target_date:
        st.session_state['bt_date'] = target_date
        # Sync widget state directly
        st.session_state['bt_date_picker'] = target_date
        st.session_state['bt_run_detail'] = True
        st.session_state['bt_detail_cache'] = None 
        st.session_state['bt_selected_stock'] = None

    col_d1, col_d2, col_d3 = st.columns([1, 1, 1])
    with col_d1: 
        # Remove manual rerun logic, use callback
        st.date_input("분석 시점", key="bt_date_picker", value=st.session_state['bt_date'], on_change=on_date_change)

    with col_d3:
        st.markdown('<div style="margin-top: 28px;"></div>', unsafe_allow_html=True)
        if st.button("상세 분석 실행", key="bt_btn_detail", type="primary", width="stretch"):
            st.session_state['bt_run_detail'] = True

    if st.session_state.get('bt_run_detail', False):
        if st.session_state.get('bt_detail_cache') is None:
            top_list = get_top_etfs(st.session_state['bt_date'], data_map, sel_manager, min_score, top_n_etf=top_n_etf, exclude_risky=exclude_risky)
            if not top_list:
                st.warning("해당 시점에 조건에 맞는 ETF가 없습니다.")
                st.session_state['bt_detail_cache'] = {'empty': True}
                return

            sel_tickers = [e['ticker'] for e in top_list]
            overlap_list = analyze_overlapping_stocks_report(sel_tickers, top_n=10, ref_date=st.session_state['bt_date'])
            
            st.session_state['bt_detail_cache'] = {
                'empty': False,
                'top_list': top_list,
                'overlap_list': overlap_list
            }
        
        cache = st.session_state['bt_detail_cache']
        if cache.get('empty'):
            st.warning("조건에 맞는 ETF가 없습니다.")
            return

        top_list = cache['top_list']
        overlap_list = cache['overlap_list']
        if not top_list:
            st.warning("해당 시점에 조건에 맞는 ETF가 없습니다.")
            return

        st.subheader(f"{st.session_state['bt_date']} 선정 ETF ({len(top_list)}개)")
        etf_df = pd.DataFrame(top_list)
        etf_df['Score'] = etf_df['momentum_score'].apply(lambda x: f"{x:.3f}")
        st.dataframe(etf_df[['ticker', 'name', 'manager', 'Score']], hide_index=True)
        

        if overlap_list:
            st.subheader("대표 주도주 (Top 10 Overlap)")
            ov_df = pd.DataFrame(overlap_list)
            cols = [c for c in ['순위', '종목명', '중복횟수', '중복비율(%)', '당시가', '수익률', '최고%', '최저%'] if c in ov_df.columns]
            evt = st.dataframe(
                ov_df[cols], 
                hide_index=True, 
                width="stretch", 
                on_select="rerun", 
                selection_mode="single-row", 
                key="bt_detail_ov_table_v2", 
                column_config={"중복횟수": st.column_config.NumberColumn("중복횟수", format="%d회")}
            )
            
            if len(evt.selection.rows) > 0:
                idx = evt.selection.rows[0]
                selected_row = ov_df.iloc[idx]
                st.session_state['bt_selected_stock'] = {
                    'ticker': selected_row['티커'],
                    'name': selected_row['종목명']
                }
            
            if st.session_state.get('bt_selected_stock'):
                sel = st.session_state['bt_selected_stock']
                _render_detail_stock_chart(sel['ticker'], sel['name'], st.session_state['bt_date'])

def _render_detail_stock_chart(ticker, name, ref_date):
    """상세 분석 섹션 내의 개별 종목 차트 렌더링"""
    st.divider()
    st.markdown(f"#### 📈 {name} ({ticker}) 상세 차트")
    p_date = pd.Timestamp(ref_date)
    sdf = get_stock_data_cached(ticker, p_date - timedelta(days=180), datetime.now())
    if not sdf.empty:
        sdf = add_momentum_columns(sdf)
        fig = plot_candle_chart(sdf, ticker, name, ref_date=p_date)
        fig.update_layout(height=500, margin=dict(t=30, b=20, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True, key=f"bt_chart_{ticker}_{ref_date}")
    else:
        st.error(f"{name} 데이터를 불러올 수 없습니다.")

def _render_market_breadth_section():
    """시장 활성도(온도계) 차트 섹션 렌더링"""
    stats_df = st.session_state.get('bt_market_stats_v2')
    if stats_df is not None and not stats_df.empty:
        st.markdown("---")
        stats_df['DateStr'] = stats_df['Date'].astype(str)
        fig = go.Figure()
        colors = ['#FF4B4B' if c >= 3 else '#1F77B4' for c in stats_df['Qualified_Count']]
        fig.add_trace(go.Bar(x=stats_df['DateStr'], y=stats_df['Qualified_Count'], name="Count", marker_color=colors, yaxis='y'))
        fig.add_trace(go.Scatter(x=stats_df['DateStr'], y=stats_df['Avg_Score'], name="Score", mode='lines', line=dict(color='black'), yaxis='y2'))
        fig.update_layout(title="시장 활성도 vs 주도주 강도", yaxis=dict(title="Count", side='left'), yaxis2=dict(title="Score", side='right', overlaying='y', showgrid=False), xaxis=dict(tickangle=-90, type='category'), legend=dict(orientation="h", x=0, y=1.1))
        st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# 3. Current Overlapping Report UI (Tab 3)
# -----------------------------------------------------------------------------
def render_overlapping_report(current_rank_df, data_map):
    """주도주 전략 분석 화면 (메인 탭 3)"""
    st.markdown("### 주도주 전략 분석 (현재)")
    config = load_user_config()
    
    # 1. 필터 설정 섹션
    # 1. 필터 설정 섹션
    sel_manager, min_score, top_n, exclude_risky = _render_report_filters(config, data_map)
    
    # 2. 분석 실행 버튼 및 로직
    if st.button("분석 실행", type="primary", key="curr_btn_run"):
        _run_current_overlap_analysis(sel_manager, min_score, top_n, exclude_risky, data_map)

    # 3. 결과 렌더링 섹션
    _render_report_results()

def _render_report_filters(config, data_map):
    """보고서 탭의 상단 필터 설정"""
    with st.expander("필터 설정", expanded=False):
        c1, c2 = st.columns([1, 1])
        with c1:
            if data_map:
                available_tickers = set(data_map.keys())
                available_managers = sorted(list(set(
                    ETF_UNIVERSE[t]['manager'] for t in available_tickers 
                    if t in ETF_UNIVERSE and 'manager' in ETF_UNIVERSE[t]
                )))
                all_managers = ["전체"] + available_managers
            else:
                all_managers = ["전체"]
                
            saved_manager = config.get('curr_manager', ["전체"])
        with c2:
            min_score = st.slider("최소 점수", 0.0, 3.0, config.get('curr_score', 0.5), 0.1, key="curr_score", on_change=on_config_change)
            top_n = st.number_input("Top N", 1, 30, config.get('curr_top_n', 5), key="curr_top_n", on_change=on_config_change)
            exclude_risky = st.checkbox("위험 제외", value=config.get('curr_risk', True), key="curr_risk", on_change=on_config_change)
    return sel_manager, min_score, top_n, exclude_risky

def _run_current_overlap_analysis(sel_manager, min_score, top_n, exclude_risky, data_map):
    """현재 시점 기준 주도주 중복 분석 실행"""
    ref_date = datetime.now().date()
    with st.spinner("분석 중..."):
        top_etfs = get_top_etfs(ref_date, data_map, sel_manager, min_score, exclude_risky=exclude_risky, top_n_etf=top_n)
        overlap_list = []
        if top_etfs:
            tickers = [e['ticker'] for e in top_etfs]
            overlap_list = analyze_overlapping_stocks_report(tickers, top_n=10, ref_date=ref_date)
        st.session_state['curr_top_etfs'] = top_etfs
        st.session_state['curr_overlap_list'] = overlap_list

def _render_report_results():
    """분석 결과(테이블 및 분포 차트) 렌더링"""
    top_etfs = st.session_state.get('curr_top_etfs', [])
    overlap_list = st.session_state.get('curr_overlap_list', [])
    
    if not top_etfs:
        if st.session_state.get('curr_btn_run'): st.warning("조건 만족 ETF 없음")
        return

    st.subheader(f"선정 ETF {len(top_etfs)}개")
    df_etf = pd.DataFrame(top_etfs)
    df_etf['Score'] = df_etf['momentum_score'].apply(lambda x: f"{x:.3f}")
    st.dataframe(df_etf[['ticker', 'name', 'manager', 'Score']], hide_index=True)
    
    if overlap_list:
        st.subheader("Top 10 Overlap")
        of = pd.DataFrame(overlap_list)
        cols = ['순위', '종목명', '중복횟수', '중복비율(%)'] + (['당시가', '수익률', '최고%', '최저%'] if '수익률' in of.columns else [])
        
        c1, c2 = st.columns([3, 2])
        with c1:
            evt = st.dataframe(of[cols], hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row", key="curr_ov_table", column_config={"중복횟수": st.column_config.NumberColumn("중복횟수", format="%d회")})
        with c2:
            fig = px.bar(of, y='종목명', x='중복횟수', orientation='h', title="중복 분포")
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
        if len(evt.selection.rows) > 0:
            idx = evt.selection.rows[0]
            _render_detail_stock_chart(of.iloc[idx]['티커'], of.iloc[idx]['종목명'], datetime.now().date())

# -----------------------------------------------------------------------------
# 4. Advanced Simulation UI (Tab 4)
# -----------------------------------------------------------------------------
def render_advanced_backtest_ui():
    st.markdown("### 고급 전략 시뮬레이션")
    config = load_user_config()
    
    with st.expander("파라미터", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            # Inputs (Simplified)
            all_man = ["전체"] + sorted(list(set(i['manager'] for i in ETF_UNIVERSE.values())))
            saved_man = config.get('adv_manager', ["전체"])
            if isinstance(saved_man, str): saved_man = [saved_man]
            sel_man = st.multiselect("운용사", all_man, default=saved_man, key="adv_man", on_change=on_config_change)
            if not sel_man or "전체" in sel_man: sel_man = "전체"
            
            s_date = st.date_input("Start", value=datetime(2025,1,1), key="adv_start", on_change=on_config_change)
            e_date = st.date_input("End", value=datetime.now(), key="adv_end", on_change=on_config_change)
        
        with c2:
            min_score = st.number_input("ETF Min Score", 0.7, step=0.1, key="adv_score", on_change=on_config_change)
            etf_ma = st.selectbox("ETF MA Filter", [5, 20, 60], index=1, key="adv_ma_etf", on_change=on_config_change)
            stk_ma = st.selectbox("Stock MA Exit", [5, 10, 20, 60], index=2, key="adv_ma_stock", on_change=on_config_change)
            overlap = st.number_input("Overlap Threshold", 2, 10, int(config.get('adv_overlap', 2)), key="adv_overlap", on_change=on_config_change)
            cap = st.number_input("Init Capital", value=int(config.get('adv_cap', 10000000)), step=1000000, key="adv_cap", on_change=on_config_change)
            amt = st.number_input("Trade Amt", value=int(config.get('adv_amt', 2000000)), step=500000, key="adv_amt", on_change=on_config_change)

    if st.button("시뮬레이션 시작", type="primary"):
        params = {
            'start_date': s_date, 'end_date': e_date, 'sel_managers': sel_man,
            'etf_min_score': min_score, 'etf_ma_period': etf_ma, 'stock_ma_period': stk_ma,
            'overlap_threshold': overlap, 'initial_capital': cap, 'per_trade_amt': amt
        }
        with st.spinner("Processing..."):
            res = run_advanced_simulation(params)
            if res:
                st.session_state['adv_sim_results'] = res

    if 'adv_sim_results' in st.session_state:
        res = st.session_state['adv_sim_results']
        eq = res['equity_curve']
        tr = res['trades']
        
        final = eq[-1]['Equity']
        ret = (final / res['initial_capital'] - 1) * 100
        
        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("End Capital", f"{int(final):,} KRW")
        m2.metric("Return", f"{ret:+.2f}%")
        m3.metric("Trades", len(tr))
        
        fig = px.line(pd.DataFrame(eq), x='Date', y='Equity', title="Equity Curve")
        st.plotly_chart(fig, width="stretch")
        
        if tr:
            st.markdown("### Trade Log")
            tdf = pd.DataFrame(tr)
            st.dataframe(tdf, hide_index=True, width="stretch")

# -----------------------------------------------------------------------------
# 5. Individual Stock UI (Tab 5)
# -----------------------------------------------------------------------------
def render_individual_backtest_ui():
    """개별 종목 심층 분석 화면 (메인 탭 5)"""
    st.markdown("### 개별 종목 심층 분석")
    c1, c2, c3 = st.columns(3)
    with c1: tick = st.text_input("Ticker", "005930")
    with c2: s_d = st.date_input("Start", datetime(2025,1,1))
    with c3: e_d = st.date_input("End", datetime.now())

    if st.button("분석", type="primary"):
        if tick: _run_individual_analysis(tick, s_d, e_d)

    if 'ind_res' in st.session_state:
        _render_individual_analysis_results()

def _run_individual_analysis(ticker, s_date, e_date):
    """개별 종목 상세 지표 및 모멘텀 계산"""
    with st.spinner("Loading..."):
        df = get_stock_data_cached(ticker, s_date - timedelta(days=200), e_date)
        if not df.empty:
            df = add_momentum_columns(df)
            sub = df.loc[s_date:e_date]
            if not sub.empty:
                st.session_state['ind_res'] = {'df': sub, 'ticker': ticker, 'dt': datetime.now()}

def _render_individual_analysis_results():
    """개별 종목 물리적 결과 및 지표 차트 렌더링"""
    res = st.session_state['ind_res']
    df, t = res['df'], res['ticker']
    st.divider()
    
    last = df.iloc[-1]
    m1, m2 = st.columns(2)
    m1.metric("Score", f"{last['MomentumScore']:.4f}")
    m2.metric("3M Ret", f"{last['R_3m']*100:.1f}%")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['MomentumScore'], name='Score', line=dict(color='orange'), yaxis='y2'))
    fig.add_hline(y=0.7, line_dash='dash', line_color='red', yref='y2')
    
    fig.update_layout(title=f"{t} Analysis", yaxis=dict(title="Price"), yaxis2=dict(title="Score", overlaying='y', side='right', showgrid=False), height=500)
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# 6. Deep Analysis Report Board (Enhanced)
# -----------------------------------------------------------------------------
def render_deep_report_board(data_map):
    """Gemini 심층 분석용 리포트 게시판 (과거 특정 시점 분석 기능 추가)"""
    st.markdown("### 📊 AI 심층 분석 리포트 매니저 (v2.6)")
    st.caption("과거 특정 일자를 선택하여 당시의 시장 상황과 주도주 리포트를 즉시 생성하고 누적 보관할 수 있습니다.")
    
    st.markdown("---")
    
    # [NEW] 분석 기준일 선택
    st.markdown("#### 🔍 분석 기준일 설정 (Snapshot Date)")
    col_date, col_info = st.columns([1, 2])
    with col_date:
        snapshot_date = st.date_input("날짜 선택", value=datetime.now())
    with col_info:
        st.info(f"선택된 날짜 **({snapshot_date.strftime('%Y-%m-%d')})** 시점의 데이터를 기준으로 리포트를 생성합니다.")
    
    st.markdown("---")
    
    # 1. Periodical Analysis Selection (Pass snapshot_date)
    col1, col2, col3 = st.columns(3)
    with col1:
        _render_report_card("📋 일간 브리핑", "선택일 기준 데일리 변동성", "daily", data_map, snapshot_date)
    with col2:
        _render_report_card("🗓️ 주간 트렌드", "선택일 기준 최근 5일 추세", "weekly", data_map, snapshot_date)
    with col3:
        _render_report_card("📈 월간 전략", "선택일 기준 최근 20일 모멘텀", "monthly", data_map, snapshot_date)

    # 2. History Billboard (Billboard Style with Downloads)
    st.markdown("---")
    st.markdown("#### 📜 누적 리포트 저장소 (Cumulative History)")
    st.caption("과거에 생성된 모든 리포트가 DB에 보관되며, 언제든 다시 다운로드할 수 있습니다.")
    
    history = get_report_history(limit=20) # 더 많이 조회
    
    if history:
        tab_all, tab_d, tab_w, tab_m = st.tabs(["📂 전체 (All)", "📋 일간 (Daily)", "🗓️ 주간 (Weekly)", "📈 월간 (Monthly)"])
        
        def _render_history_list(filtered_history, key_suffix):
            if not filtered_history:
                st.info("해당 주기 리포트가 없습니다.")
                return
            for item in filtered_history:
                report_type = item['report_type'].lower()
                report_label = f"[{item['report_date']}] {report_type.upper()} - {item['top_etf']}"
                with st.expander(report_label):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.write(f"🔹 **평균 점수**: `{item['avg_score']:.2f}`")
                        st.write(f"🔹 **생성 시각**: `{item['created_at']}`")
                        st.caption(f"파일명: {item['file_name']}")
                    
                    with c2:
                        # 키 충돌 방지를 위해 suffix 추가
                        dl_key = f"hist_dl_{item['id']}_{key_suffix}"
                        del_btn_key = f"btn_del_{item['id']}_{key_suffix}"
                        del_confirm_key = f"del_confirm_{item['id']}_{key_suffix}"
                        pw_key = f"pw_{item['id']}_{key_suffix}"
                        cancel_key = f"cancel_{item['id']}_{key_suffix}"

                        if item.get('report_data'):
                            st.download_button(
                                label="📥 재다운로드",
                                data=item['report_data'],
                                file_name=item['file_name'],
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key=dl_key,
                                use_container_width=True
                            )
                        
                        # 삭제 버튼
                        if st.button("🗑️ 삭제", key=del_btn_key, use_container_width=True, type="secondary"):
                            st.session_state[del_confirm_key] = True
                        
                        if st.session_state.get(del_confirm_key):
                            pw = st.text_input("비밀번호 입력", type="password", key=pw_key)
                            if pw == "8051":
                                delete_report(item['id'])
                                st.success("삭제되었습니다.")
                                del st.session_state[del_confirm_key]
                                st.rerun()
                            elif pw != "":
                                st.error("잘못된 비밀번호입니다.")
                                if st.button("취소", key=cancel_key):
                                    del st.session_state[del_confirm_key]
                                    st.rerun()

        with tab_all:
             _render_history_list(history, "all")
        with tab_d:
            _render_history_list([h for h in history if h['report_type'].lower() == 'daily'], "daily")
        with tab_w:
            _render_history_list([h for h in history if h['report_type'].lower() == 'weekly'], "weekly")
        with tab_m:
            _render_history_list([h for h in history if h['report_type'].lower() == 'monthly'], "monthly")
    else:
        st.info("아직 저장된 리포트가 없습니다.")

    st.markdown("---")
    st.info("💡 **Gemini 활용 팁**: 엑셀의 `Trend_20D` 시트에는 최근 20일간의 점수 변화가 담겨 있습니다. '최근 주도주 중 상승 탄력이 가장 가파른 종목 3개를 뽑고 이유를 설명해줘'라고 물어보세요.")

def _render_report_card(title, subtitle, period, data_map, snapshot_date):
    """리포트 카드 UI (Snapshot Date 지원)"""
    with st.container(border=True):
        st.subheader(title)
        st.caption(subtitle)
        
        # 메트릭 계산 (snapshot_date 기준)
        rank_df = calculate_momentum_score(data_map, ref_date=snapshot_date)
        if not rank_df.empty:
            top_name = rank_df.iloc[0]['ShortName']
            avg_score = rank_df.head(5)['Score'].mean()
            st.write(f"🔹 **TOP**: `{top_name}`")
            st.write(f"🔹 **Avg Score**: `{avg_score:.2f}`")
        else:
            top_name, avg_score = "N/A", 0.0

        filename = f"ETF_Strategy_Report_{period}_{snapshot_date.strftime('%Y%m%d')}_{datetime.now().strftime('%H%M%S')}.xlsx"
        excel_data = generate_report_excel(data_map, period, snapshot_date)
        
        # 버튼을 누르면 즉시 DB에 기록하고 다운로드
        if st.download_button(
            label=f"Excel 생성 및 다운로드",
            data=excel_data,
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key=f"dl_{period}_{snapshot_date.strftime('%Y%m%d')}"
        ):
            # DB 저장 시 report_date는 snapshot_date로 기록하여 관리 용이하게 함
            log_report_generation(period, top_name, avg_score, filename, excel_data)
            st.rerun() 

def generate_report_excel(data_map, period='daily', snapshot_date=None):
    """애플리케이션의 모든 가용 데이터를 포함하는 마스터 엑셀 리포트 생성"""
    output = io.BytesIO()
    ref_dt = pd.Timestamp(snapshot_date) if snapshot_date else pd.Timestamp(datetime.now().date())
    
    # 0. AI 분석 가이드 (데이터 설명 포함)
    ai_guidance = pd.DataFrame([
        ["분석 목적", "한국 시장 ETF 주도주 전략 마스터 데이터입니다. 전략 수립에 필요한 모든 로데이터가 포함되어 있습니다."],
        ["시트: Summary", "현재 시장 온도 및 방어 지표"],
        ["시트: Full_Universe_Rankings", "전체 ETF(약 90개 이상)의 모멘텀 상 상세 랭킹 및 지표"],
        ["시트: Theme_Analysis", "섹터/테마별 점수 분포"],
        ["시트: Leading_Stocks_Overlap", "상위 ETF들이 공통 보유한 핵심 주도 종목 분석"],
        ["시트: All_ETF_Constituents", "모든 ETF의 구성 종목 및 비중(Weight) 데이터"],
        ["시트: Historical_Scores_60D", "최근 60거래일간의 모멘텀 스코어 시계열 (추세 가속도 분석용)"],
        ["시트: Price_History_60D", "최근 60거래일간의 종가 데이터 (수익률 분석용)"],
        ["추천 프롬프트", "모든 데이터를 참조하여, 1. 현재 시장 지배력이 가장 높은 테마, 2. 최근 60일간 스코어가 우상향하고 있는 주도 ETF 5개, 3. 그 ETF들의 공통 종목 중 실질적 수혜가 예상되는 종목 10개를 선정해줘."]
    ], columns=["항목", "내용"])

    # 1. 전체 유니버스 랭킹
    rank_df = calculate_momentum_score(data_map, ref_date=ref_dt)
    all_tickers = rank_df['Ticker'].tolist()

    # 2. 전체 ETF 구성 종목 (비중 포함)
    constituents_list = []
    for ticker in all_tickers:
        holdings = load_etf_holdings(ticker)
        etf_name = rank_df[rank_df['Ticker'] == ticker]['ShortName'].iloc[0] if ticker in rank_df['Ticker'].values else ticker
        for h in holdings:
            constituents_list.append({
                'ETF_Ticker': ticker,
                'ETF_Name': etf_name,
                'Stock_Name': h.get('name'),
                'Stock_Ticker': h.get('ticker'),
                'Weight': h.get('pct', '0%')
            })
    constituents_df = pd.DataFrame(constituents_list)

    # 3. 60일 시계열 데이터 (Score & Price)
    score_history = []
    price_history = []
    
    for ticker in all_tickers:
        if ticker in data_map:
            df = data_map[ticker]
            df_pre = df[df.index <= ref_dt].tail(60)
            if not df_pre.empty:
                s_row = {'Ticker': ticker, 'Name': df_pre['Name'].iloc[-1]}
                p_row = {'Ticker': ticker, 'Name': df_pre['Name'].iloc[-1]}
                for i, (idx, row) in enumerate(reversed(list(df_pre.iterrows()))):
                    s_row[f'D-{i}({idx.strftime("%m%d")})'] = round(row.get('MomentumScore', 0), 4)
                    p_row[f'D-{i}({idx.strftime("%m%d")})'] = row.get('Close', 0)
                score_history.append(s_row)
                price_history.append(p_row)
                
    score_df = pd.DataFrame(score_history)
    price_df = pd.DataFrame(price_history)

    # 4. 테마/시장/중복 분석 (확장형)
    theme_summary = rank_df.groupby('Theme').agg({
        'Score': 'mean',
        'Ticker': 'count',
        'R_1w': 'mean',
        'R_1m': 'mean'
    }).rename(columns={'Ticker': 'ETF_Count'}).sort_values(by='Score', ascending=False).reset_index()

    is_bull, k_now, k_ma60 = check_market_defense(data_map, ref_date=ref_dt)
    summary_df = pd.DataFrame([{
        'Snapshot_Date': ref_dt.strftime('%Y-%m-%d'),
        'Market_Status': 'Bull' if is_bull else 'Bear',
        'KOSDAQ': k_now,
        'KOSDAQ_MA60': k_ma60,
        'Universe_Size': len(rank_df)
    }])
    
    overlap_list = analyze_overlapping_stocks_report(all_tickers[:15], top_n=30, ref_date=ref_dt)
    overlap_df = pd.DataFrame(overlap_list) if overlap_list else pd.DataFrame()

    # 엑셀 저장
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        ai_guidance.to_excel(writer, sheet_name='AI_Master_Prompt', index=False)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        rank_df.to_excel(writer, sheet_name='Full_Universe_Rankings', index=False)
        theme_summary.to_excel(writer, sheet_name='Theme_Analysis', index=False)
        if not overlap_df.empty:
            overlap_df.to_excel(writer, sheet_name='Leading_Stocks_Overlap', index=False)
        if not constituents_df.empty:
            constituents_df.to_excel(writer, sheet_name='All_ETF_Constituents', index=False)
        if not score_df.empty:
            score_df.to_excel(writer, sheet_name='Historical_Scores_60D', index=False)
        if not price_df.empty:
            price_df.to_excel(writer, sheet_name='Price_History_60D', index=False)
        
    return output.getvalue()
