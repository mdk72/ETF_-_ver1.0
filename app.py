import logging
import warnings
import streamlit as st
logging.getLogger("streamlit").setLevel(logging.ERROR)

class ScriptRunContextFilter(logging.Filter):
    def filter(self, record): return "ScriptRunContext" not in record.getMessage()

for name in logging.root.manager.loggerDict:
    logger = logging.getLogger(name)
    logger.addFilter(ScriptRunContextFilter())
    if "streamlit" in name: logger.setLevel(logging.ERROR)

warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")

import streamlit as st
import pandas as pd
from datetime import datetime

from utils_config import load_user_config, ETF_UNIVERSE, on_config_change, load_etf_universe
from data_manager import fetch_market_data, run_data_update
from analysis_engine import calculate_momentum_score, check_market_defense
from ui_views import (
    render_momentum_ranking, render_backtest_ui, 
    render_overlapping_report, render_advanced_backtest_ui, 
    render_individual_backtest_ui, render_deep_report_board
)
from db_manager import init_db

# DB 초기화 (테이블 생성 등)
init_db()

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ETF 주도주 전략 (Refactored)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# [UI Optimization] 심플 & 클린 (가독성 보완 버전)
st.markdown("""
    <style>
    /* 기본 폰트 설정 */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    html, body, [data-testid="stAppViewContainer"], .main {
        font-family: 'Inter', 'Malgun Gothic', sans-serif !important;
        color: #1e293b;
    }

    /* 메인 컨테이너 - 여백 축소 및 가로폭 확장 */
    .block-container {
        padding-top: 3.5rem !important; /* 상단바 가림 방지 */
        padding-bottom: 2rem !important;
        padding-left: 1.5rem !important; /* 좌측 여백 축소하여 더 왼쪽으로 붙임 */
        padding-right: 1.5rem !important;
        max-width: 99% !important; /* 화면 가득 채우기 */
    }
    
    /* 요소 간 간격 */
    div.stVerticalBlock {
        gap: 1.0rem !important;
    }
    
    /* 사이드바 - 상단 정렬 및 배경 */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #f1f5f9;
    }
    [data-testid="stSidebarUserContent"] {
        padding-top: 1.2rem !important; /* 사이드바 타이틀 위치 보정 */
    }
    
    /* 제목(Headers) - 잘림 방지 스타일 */
    h1, h2, h3 {
        color: #0f172a !important;
        font-weight: 600 !important;
        margin-top: 0.8rem !important; /* 제목 위에 최소한의 공간 부여 */
        margin-bottom: 0.5rem !important;
        padding-top: 0rem !important;
        line-height: 1.4 !important; /* 텍스트가 잘리지 않도록 높이 확보 */
    }
    h1 { font-size: 1.6rem !important; }
    h2 { font-size: 1.3rem !important; }
    h3 { 
        font-size: 0.95rem !important; 
        text-transform: uppercase; 
        letter-spacing: 0.03em; 
        color: #475569 !important; 
        border-bottom: 1px solid #f1f5f9; /* 제목 구분감 부여 */
        padding-bottom: 0.3rem; 
    }
    
    /* 메트릭 */
    div[data-testid="stMetricValue"] {
        font-family: 'Inter', monospace !important;
        font-size: 1.2rem !important;
        font-weight: 500 !important;
    }
    
    /* 버튼 */
    .stButton>button {
        border-radius: 6px !important;
        border: 1px solid #e2e8f0 !important;
        padding: 0.3rem 0.8rem !important;
    }
    
    /* 구분선 */
    hr {
        margin: 1.2rem 0 !important;
        border-top: 1px solid #f1f5f9 !important;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    # [Fix] Reload Universe on every rerun explicitly to update cached references
    from utils_config import update_etf_universe
    update_etf_universe()
    
    # 1. Sidebar - Navigation & Global Controls
    st.sidebar.markdown("### ETF Strategy")
    st.sidebar.caption("Momentum & Overlap Analysis")
    
    menu = st.sidebar.radio("메뉴 선택", ["현재 랭킹 분석", "시뮬레이션 (Backtest)", "현재 주도주 분석", "심층 분석 리포트", "고급 전략 (Advanced)", "개별 종목 분석"])
    st.sidebar.markdown("---")

    # [Category Toggle]
    st.sidebar.markdown("### 분석 대상 (Universe)")
    category_mode = st.sidebar.radio("Category", ["테마 (Theme)", "섹터 (Sector)"], label_visibility="collapsed")
    target_category = 'Theme' if '테마' in category_mode else 'Sector'
    
    # Filter Universe
    if ETF_UNIVERSE:
        filtered_universe = {k: v for k, v in ETF_UNIVERSE.items() if v.get('category', 'Theme') == target_category}
    else:
        filtered_universe = {}

    with st.sidebar.expander("가중치 설정 (Weights)"):
        w3m = st.slider("3개월 수익률", 0.0, 1.0, 0.5, step=0.1)
        w1m = st.slider("1개월 수익률", 0.0, 1.0, 0.3, step=0.1)
        w1w = st.slider("1주 수익률", 0.0, 1.0, 0.2, step=0.1)
        st.markdown("---")
        min_score = st.slider("최소 모멘텀 점수 (Threshold)", -1.0, 3.0, 0.7, step=0.1)

    if st.sidebar.button("데이터 업데이트 (전체)", use_container_width=True):
        run_data_update()
        st.rerun()

    st.sidebar.markdown("---")

    # 2. Shared Data Loading (Only for relevant tabs)
    data_map = {}
    if menu != "개별 종목 분석":
        # Fetch data only for filtered universe
        # fetch_market_data generally fetches everything in DB or cached. 
        # We need to filter the returned map to match the category.
        full_data_map = fetch_market_data()
        data_map = {k: v for k, v in full_data_map.items() if k in filtered_universe or k == 'KOSDAQ'}
        
        if not data_map:
            st.error("보유한 ETF 데이터가 없습니다. 업데이트 버튼을 눌러주세요.")
            return
        
        # Market Check Section
        is_bull, k_now, k_ma60 = check_market_defense(data_map)
        status_color, status_text = ("green", "Bull (상승장)") if is_bull else ("red", "Bear (하락장)")
        
        # [NEW] 데이터 기준일 확인
        last_date_str = "N/A"
        if data_map:
            # KOSDAQ 혹은 첫 번째 데이터의 마지막 날짜 확인
            sample_df = data_map.get('KOSDAQ')
            if sample_df is None or sample_df.empty:
                # KOSDAQ 없으면 아무거나
                if len(data_map) > 0:
                    sample_df = next(iter(data_map.values()))
            
            if sample_df is not None and not sample_df.empty:
                last_dt = sample_df.index[-1]
                last_date_str = last_dt.strftime('%Y-%m-%d')

        st.sidebar.markdown("### Market Status")
        st.sidebar.caption(f"Data Date: {last_date_str}")
        st.sidebar.markdown(f"KOSDAQ: <span style='color:{status_color}; font-weight:bold'>{status_text}</span>", unsafe_allow_html=True)
        st.sidebar.caption(f"{k_now:,.0f} / MA60: {k_ma60:,.0f}")
        st.sidebar.markdown("---")

    # Sidebar Links
    st.sidebar.markdown("### ETF 자료실")
    st.sidebar.markdown("- [네이버 증권 ETF](https://finance.naver.com/sise/etf.naver)\n- [SEIBRO ETF](https://seibro.or.kr/websquare/control.jsp?w2xPath=/IPORTAL/user/index.xml)\n- [ETF CHECK](https://www.etfcheck.co.kr/)")
    st.sidebar.markdown("---")
    st.sidebar.text("Ver 2.5 (Clean Optimized)")

    # 3. Content Routing
    if menu == "현재 랭킹 분석":
        # [Fix] 최소 모멘텀 점수 필터 적용
        render_momentum_ranking(calculate_momentum_score(data_map, w3m=w3m, w1m=w1m, w1w=w1w), data_map, min_score=min_score)

    elif menu == "시뮬레이션 (Backtest)":
        render_backtest_ui(calculate_momentum_score(data_map), data_map)

    elif menu == "현재 주도주 분석":
        render_overlapping_report(calculate_momentum_score(data_map), data_map)
        
    elif menu == "고급 전략 (Advanced)":
        render_advanced_backtest_ui(data_map)
        
    elif menu == "심층 분석 리포트":
        render_deep_report_board(data_map)
        
    elif menu == "개별 종목 분석":
        render_individual_backtest_ui()

if __name__ == "__main__":
    main()
