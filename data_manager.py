import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
import numpy as np
from datetime import datetime, timedelta
import time
import concurrent.futures
import updater_utils
from crawl_holdings import fetch_etf_holdings
from utils_config import load_etf_holdings, ETF_UNIVERSE
from db_manager import save_etf_universe, save_etf_holdings, get_etf_holdings
from streamlit.runtime.scriptrunner import get_script_run_ctx, add_script_run_ctx

# -----------------------------------------------------------------------------
# Data Caching & Management
# -----------------------------------------------------------------------------

@st.cache_data(ttl=3600*12)
def get_stock_data_cached(ticker, start_date, end_date):
    """개별 종목 데이터를 최적화하여 조회하고 인덱스를 표준화합니다."""
    for attempt in range(2): # 2번 시도
        try:
            df = fdr.DataReader(ticker, start_date, end_date)
            if not df.empty:
                # 인덱스 표준화 (Date)
                if df.index.name != 'Date':
                    df.index.name = 'Date'
                return df
            break
        except Exception:
            time.sleep(0.5)
            continue
    return pd.DataFrame()

@st.cache_data(ttl=3600*12) 
def fetch_market_data(days=2500):
    """전체 유니버스의 시장 데이터를 병렬로 수집하고 사전 연산을 수행합니다."""
    data_dict = {}
    end_date = datetime.now()
    start_date = datetime(2020, 1, 1)
    
    # 1. 코스닥 지수 (시장 온도계 기준)
    try:
        kosdaq = fdr.DataReader('KQ11', start_date, end_date)
        if not kosdaq.empty:
            kosdaq.index.name = 'Date'
            data_dict['KOSDAQ'] = kosdaq
    except: pass

    total = len(ETF_UNIVERSE)
    if total == 0: return data_dict

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 2. 병렬 수집 및 사전 계산 함수
    def fetch_one(ticker):
        from analysis_engine import add_momentum_columns # 순환 참조 방지
        
        for attempt in range(2): # 간이 리트라이
            try:
                df = fdr.DataReader(ticker, start_date, end_date)
                # 데이터가 너무 적으면 KOSPI 접미사 시도
                if df.empty or len(df) < 10:
                    try: df = fdr.DataReader(f"{ticker}.KS", start_date, end_date)
                    except: pass
                
                if not df.empty and len(df) > 10:
                    df.index.name = 'Date'
                    # 유효 컬럼만 통합
                    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                    df = df[[c for c in cols if c in df.columns]].copy()
                    
                    # [Pre-calculation] 모멘텀 점수 및 보조지표 선행 계산
                    df = add_momentum_columns(df)
                    df['Name'] = ETF_UNIVERSE.get(ticker, {}).get('name', 'Unknown')
                    return ticker, df
                break
            except:
                time.sleep(0.2)
        return ticker, None

    # 3. ThreadPoolExecutor (최적화된 병렬 실행)
    sorted_tickers = sorted(ETF_UNIVERSE.keys())
    completed = 0
    ctx = get_script_run_ctx()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        if ctx:
            futures = {executor.submit(lambda t=t: (add_script_run_ctx(ctx), fetch_one(t))[1]): t for t in sorted_tickers}
        else:
            futures = {executor.submit(fetch_one, t): t for t in sorted_tickers}
        
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            try:
                t, df = future.result()
                if df is not None:
                    data_dict[t] = df
            except: pass
            
            if completed % 5 == 0 or completed == total:
                progress_bar.progress(completed / total)
                status_text.text(f"시장 데이터 로딩 중... [{completed}/{total}]")

    status_text.empty()
    progress_bar.empty()
    return data_dict

def run_data_update():
    """전체 데이터 최신화 (목록 갱신 -> 보유종목 DB 업데이트)"""
    status_cont = st.empty()
    prog_bar = st.progress(0)
    
    # 1. ETF 목록 갱신
    status_cont.info("📋 ETF 전체 목록 갱신 중... (Seibro/Naver)")
    new_universe_list = updater_utils.update_etf_list_seibro_param() 
    
    # DB에 유니버스 정보 저장
    save_etf_universe(new_universe_list)
    
    # 2. 보유종목 업데이트 (스마트 업데이트)
    status_cont.info("📦 ETF 보유 종목 정보 DB 업데이트 중...")
    
    today_str = datetime.now().strftime('%Y-%m-%d')
    total = len(new_universe_list)
    updated_count = 0
    skipped_count = 0
    
    for i, etf in enumerate(new_universe_list):
        prog_bar.progress((i+1)/total)
        ticker = etf['ticker']
        name = etf['name']
        
        # 성분 로드 (DB에서 확인)
        existing_holdings = get_etf_holdings(ticker)
        if existing_holdings:
            # db_manager는 updated_at을 테이블에 저장하므로, 
            # 개별 조회가 가능하나 여기서는 효율을 위해 로직 단순화
            # (오늘 이미 업데이트했다면 스킵)
            pass 

        status_cont.text(f"[{i+1}/{total}] {name} 업데이트 중...")
        holdings = fetch_etf_holdings(ticker)
        
        if holdings:
            save_etf_holdings(ticker, holdings)
            updated_count += 1
        else:
            skipped_count += 1
            
        time.sleep(0.1)
        
    status_cont.success(f"✅ DB 업데이트 완료! (갱신: {updated_count}, 실패/기존: {skipped_count})")
    st.cache_data.clear()
    time.sleep(2)
    st.rerun()
