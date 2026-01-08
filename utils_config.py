import streamlit as st
import json
import os
from db_manager import get_all_etfs, get_etf_holdings

def load_etf_holdings(etf_ticker):
    """DB에서 특정 ETF의 구성 종목(성분)을 로드합니다."""
    return get_etf_holdings(etf_ticker)

def load_etf_universe():
    """DB에서 ETF 유니버스 전체 메타데이터를 로드합니다."""
    return get_all_etfs()

# Global Data Instances (초기 로딩용)
ETF_UNIVERSE = load_etf_universe()

def update_etf_universe():
    """DB에서 최신 데이터를 읽어와 전역 변수 ETF_UNIVERSE를 In-Place 업데이트합니다."""
    new_data = load_etf_universe()
    ETF_UNIVERSE.clear()
    ETF_UNIVERSE.update(new_data)

# -----------------------------------------------------------------------------
# User Config Management
# -----------------------------------------------------------------------------
CONFIG_FILE = os.path.join(os.getcwd(), 'user_config.json')

def load_user_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except: return {}
    return {}

def save_user_config(new_config):
    current = load_user_config()
    current.update(new_config)
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(current, f, ensure_ascii=False, indent=2)

def on_config_change():
    """세션 상태의 설정값을 감지하여 user_config.json에 저장합니다."""
    cfg = {}
    ss = st.session_state
    
    # 매핑 테이블 (세션 키: 저장할 키)
    mapping = {
        'bt_man': 'manager', 'bt_score': 'min_score', 'bt_risk': 'exclude_risky', 'bt_top_n': 'top_n_etf',
        'curr_man': 'curr_manager', 'curr_score': 'curr_score', 'curr_risk': 'curr_risk', 'curr_top_n': 'curr_top_n',
        'rank_man': 'rank_manager',
        'adv_man': 'adv_manager', 'adv_score': 'adv_min_score', 'adv_overlap': 'adv_overlap',
        'adv_ma_etf': 'adv_ma_etf', 'adv_ma_stock': 'adv_ma_stock', 'adv_cap': 'adv_cap', 'adv_amt': 'adv_amt'
    }
    
    for s_key, c_key in mapping.items():
        if s_key in ss: cfg[c_key] = ss[s_key]
        
    # 날짜 필드 별도 처리
    date_keys = {
        'bt_h_start': 'bt_h_start_str', 'bt_h_end': 'bt_h_end_str', 'bt_date': 'bt_date_str',
        'adv_start': 'adv_start_str', 'adv_end': 'adv_end_str'
    }
    for s_key, c_key in date_keys.items():
        if s_key in ss: cfg[c_key] = ss[s_key].strftime("%Y-%m-%d")

    if cfg: save_user_config(cfg)
