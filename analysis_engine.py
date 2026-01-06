import pandas as pd
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import concurrent.futures
import numpy as np
import FinanceDataReader as fdr
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx, add_script_run_ctx

from utils_config import ETF_UNIVERSE, load_etf_holdings
from data_manager import get_stock_data_cached

# -----------------------------------------------------------------------------
# Core Analysis Logic
# -----------------------------------------------------------------------------

def add_momentum_columns(df):
    """DataFrame에 모멘텀 점수 및 관련 보조 지표들을 벡터 연산으로 계산해 추가합니다."""
    if df is None or df.empty or len(df) < 5:
        return df
    
    df = df.copy()
    try:
        # Close 가격 기반 수익률 계산 (벡터 연산 활용)
        close = df['Close']
        r_1w = close.pct_change(5)
        r_1m = close.pct_change(20)
        r_3m = close.pct_change(60)
        
        # 20일 이동 변동성 (연화)
        vol_20d = close.pct_change().rolling(20).std() * np.sqrt(252)
        # 변동성 0 방지 및 결측치 처리
        vol_20d = vol_20d.replace(0, 0.001).fillna(0.001)
        
        # 주도주 전략 공식: (R3M * 0.5 + R1M * 0.3 + R1W * 0.2) / Volatility
        df['MomentumScore'] = (r_3m * 0.5 + r_1m * 0.3 + r_1w * 0.2) / vol_20d
        df['MomentumScore'] = df['MomentumScore'].fillna(0)
        
        # UI 보조 지표 저장
        df['R_1w'] = r_1w
        df['R_1m'] = r_1m
        df['R_3m'] = r_3m
        df['Vol_20d'] = vol_20d
        
        # 이동평균선 (트렌드 필터용)
        df['MA_20'] = close.rolling(20).mean()
        df['MA_60'] = close.rolling(60).mean()
    except Exception as e:
        print(f"Error in add_momentum_columns: {e}")
    return df

def calculate_momentum_score(data_dict, ref_date=None):
    """지정된 기준일(ref_date) 시점의 전 종목 모멘텀 점수를 한 번에 산출합니다."""
    metrics = []
    target_dt = pd.Timestamp(ref_date) if ref_date else None
    
    for ticker, df in data_dict.items():
        if ticker == 'KOSDAQ' or df.empty:
            continue

        try:
            # 1. 기준일 행 찾기
            if target_dt:
                if target_dt in df.index:
                    row = df.loc[target_dt]
                else:
                    asof_idx = df.index.asof(target_dt)
                    if pd.isna(asof_idx): continue 
                    row = df.loc[asof_idx]
            else:
                row = df.iloc[-1]
            
            # 2. 값 추출
            curr_price = row.get('Close', 0)
            ma_20 = row.get('MA_20', 0)
            
            metrics.append({
                'Ticker': ticker,
                'Name': row.get('Name', ticker),
                'Score': row.get('MomentumScore', 0),
                'Trend': 'Pass' if curr_price >= ma_20 > 0 else 'Fail',
                'Manager': ETF_UNIVERSE.get(ticker, {}).get('manager', 'Unknown'),
                'Theme': ETF_UNIVERSE.get(ticker, {}).get('theme', '자산배분'),
                'ShortName': row.get('Name', ticker),
                'Price': curr_price,
                'Close': curr_price,
                'R_1w': row.get('R_1w', 0),
                'R_1m': row.get('R_1m', 0),
                'R_3m': row.get('R_3m', 0),
                'Vol_20d': row.get('Vol_20d', 0),
                'MA_20': ma_20
            })
        except Exception:
            continue
            
    if not metrics:
        return pd.DataFrame()
        
    return pd.DataFrame(metrics).sort_values(by='Score', ascending=False).reset_index(drop=True)

def get_top_etfs(target_date, data_map, manager, min_score, top_n_etf=5, exclude_risky=False):
    # Pass string date if needed, but timestamp is better handled inside calculate_momentum_score
    rank_df = calculate_momentum_score(data_map, ref_date=target_date)
    if rank_df.empty: return []
    
    cond = (rank_df['Score'] >= min_score)
    
    # Manager Filter (Support List or String)
    if isinstance(manager, list):
        clean_managers = [m for m in manager if m != "전체"]
        if clean_managers:
            cond &= (rank_df['Manager'].isin(clean_managers))
    elif isinstance(manager, str):
        if manager != "전체": 
            cond &= (rank_df['Manager'] == manager)
    
    # [위험 종목 필터링]
    if exclude_risky:
        if 'ShortName' in rank_df.columns:
            risky_keywords = ['레버리지', '인버스', '2X', '합성', '선물']
            pattern = '|'.join(risky_keywords)
            cond &= ~rank_df['ShortName'].str.contains(pattern, na=False)
            
    filtered = rank_df[cond].copy()
    filtered = filtered.sort_values(by='Score', ascending=False)
    
    top_list = []
    for _, row in filtered.head(top_n_etf).iterrows():
        top_list.append({
            'ticker': row['Ticker'], 'name': row['ShortName'],
            'manager': row['Manager'], 'momentum_score': row['Score']
        })
    return top_list

# Removed unused calculate_period_return function

def calculate_post_stats(df, ref_date):
    """기준일 이후의 성과(현재가, 최고가, 최저가 수익률) 분석"""
    try:
        ts = pd.Timestamp(ref_date)
        
        # 1. 기준일 이후 데이터 필터링
        sub = df[df.index >= ts].copy()
        
        if sub.empty: return None
        
        # [FIX] 거래정지 구간(가격 = 0) 제외
        valid_sub = sub[(sub['Close'] > 0) & (sub['High'] > 0) & (sub['Low'] > 0)].copy()
        
        if valid_sub.empty: return None
        
        base_price = valid_sub['Close'].iloc[0]
        curr_price = valid_sub['Close'].iloc[-1]
        highest = valid_sub['High'].max()
        lowest = valid_sub['Low'].min()
        days = (valid_sub.index[-1] - valid_sub.index[0]).days
        
        if base_price <= 0 or lowest <= 0: return None
        
        return {
            'base_price': base_price,
            'curr_price': curr_price,
            'ret_curr': (curr_price / base_price - 1) * 100,
            'ret_high': (highest / base_price - 1) * 100,
            'ret_low': (lowest / base_price - 1) * 100,
            'days': days
        }
    except Exception:
        return None

def analyze_overlapping_stocks_report(selected_etfs, top_n=10, ref_date=None, etf_name_map=None):
    """ETF들의 보유 종목 중 중복도가 높은 종목들을 분석하여 리포트용 데이터를 생성합니다."""
    all_stocks = []
    
    # 1. 모든 ETF의 보유 종목 수집 (Top 20)
    for ticker in selected_etfs:
        ticker_str = str(ticker).strip()
        holdings_raw = load_etf_holdings(ticker_str)
        if holdings_raw:
            holdings = holdings_raw[:20]
            for h in holdings:
                t = h.get('ticker')
                if t and t not in ('N/A', 'None', None):
                    all_stocks.append({
                        'ticker': t,
                        'name': h.get('name', t),
                        'etf_ticker': ticker 
                    })
    
    if not all_stocks:
        return []
    
    # 중복도 집계
    ticker_counter = Counter([s['ticker'] for s in all_stocks])
    ticker_to_name = {s['ticker']: s['name'] for s in all_stocks}
    
    # 종목별 소스 ETF 집계
    ticker_to_etfs = defaultdict(set)
    for s in all_stocks:
        t_etf = s['etf_ticker']
        display_name = str(t_etf)
        if etf_name_map and t_etf in etf_name_map:
            display_name = etf_name_map[t_etf].replace('TIGER ', '').replace('KODEX ', '').replace('KBSTAR ', '').replace('ACE ', '').replace('SOL ', '')
        ticker_to_etfs[s['ticker']].add(display_name)
    
    unique_tickers = list(ticker_counter.keys())
    
    def get_stock_metrics(ticker):
        count = ticker_counter[ticker]
        name = ticker_to_name[ticker]
        etf_str = ", ".join(sorted(ticker_to_etfs[ticker]))
        total_etfs = len(selected_etfs)
        ratio = (count / total_etfs * 100) if total_etfs > 0 else 0

        res_row = {
            '순위': 0,
            '종목명': name, 
            '티커': ticker, 
            'name': name,    # [FIX] Internal logic alias
            'ticker': ticker, # [FIX] Internal logic alias
            '중복횟수': count, 
            '중복비율(%)': f"{ratio:.1f}%",
            '포함 ETF': etf_str, 
            '1M_모멘텀(%)': 0.0, # 정렬용 점수 (차후 할당)
            '당시가': '-', '수익률': '-', '최고%': '-', '최저%': '-'
        }
        
        # 내부 정렬용 필드 (Excel 노출 안함)
        res_row['_sort_val'] = 0.0

        if ref_date:
            try:
                # 데이터 캐싱 조회 최적화 (범위 축소)
                start_dt = pd.Timestamp(ref_date) - timedelta(days=150) 
                df = get_stock_data_cached(ticker, start_dt, datetime.now())
                
                if not df.empty:
                    # 기준일 시점 모멘텀 점수 (정렬용)
                    df_pre = df[df.index <= pd.Timestamp(ref_date)]
                    if len(df_pre) > 20:
                        curr_p = df_pre['Close'].iloc[-1]
                        prev_p = df_pre['Close'].iloc[-20] # 대략 1개월 모멘텀
                        if prev_p > 0: 
                            m_val = (curr_p / prev_p - 1)
                            res_row['1M_모멘텀(%)'] = round(m_val * 100, 2)
                            res_row['_sort_val'] = m_val
                    
                    # 성과 지표 (기준일 이후)
                    stats = calculate_post_stats(df, ref_date)
                    if stats:
                        res_row.update({
                            '당시가': f"{stats['base_price']:,.0f}",
                            '수익률': f"{stats['ret_curr']:+.1f}%",
                            '최고%': f"{stats['ret_high']:+.1f}%",
                            '최저%': f"{stats['ret_low']:+.1f}%"
                        })
            except Exception: pass
        return res_row

    # 병렬 처리
    if ref_date:
        ctx = get_script_run_ctx()
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(unique_tickers), 15)) as executor:
            if ctx:
                futures = [executor.submit(lambda t=t: (add_script_run_ctx(ctx), get_stock_metrics(t))[1]) for t in unique_tickers]
                full_results = [f.result() for f in futures]
            else:
                full_results = list(executor.map(get_stock_metrics, unique_tickers))
    else:
        full_results = [get_stock_metrics(t) for t in unique_tickers]
            
    full_results.sort(key=lambda x: (x['중복횟수'], x['_sort_val']), reverse=True)
    final_res = full_results[:top_n]
    for i, r in enumerate(final_res): 
        r['순위'] = i + 1
        if '_sort_val' in r: del r['_sort_val'] # 내부 필드 삭제
    
    return final_res
    
    return final_res

def run_simulation(start_date, end_date, universe, min_score, selected_manager, data_map, freq='W-FRI', exclude_risky=False, top_n_etf=5):
    """지정된 기간 동안 주별/월별 시뮬레이션을 수행하여 성과 로그를 생성합니다."""
    # 기준 날짜 선택 (KODEX 200 또는 사용 가능한 첫 번째 데이터)
    ref_ticker = '069500' if '069500' in data_map else (next(iter(data_map)) if data_map else None)
    if not ref_ticker: return pd.DataFrame(), pd.DataFrame()
    
    ref_df = data_map[ref_ticker]
    dates = ref_df.index[(ref_df.index.date >= start_date) & (ref_df.index.date <= end_date)]
    if len(dates) == 0: return pd.DataFrame(), pd.DataFrame()

    history_log = []
    daily_stats = []
    
    # 운용사 필터 텍스트 미리 생성
    manager_str = "전체"
    if isinstance(selected_manager, list):
        clean = [m for m in selected_manager if m != "전체"]
        if clean: manager_str = ", ".join(clean[:2]) + (f" 외 {len(clean)-2}개" if len(clean) > 2 else "")
    elif isinstance(selected_manager, str) and selected_manager != "전체":
        manager_str = selected_manager

    progress_bar = st.progress(0)
    status_text = st.empty()
    total_steps = len(dates)
    
    # 맵 전역 이름을 위해 복사 (lookup 속도 향상)
    # 맵 전역 이름을 위해 복사 (lookup 속도 향상) - Name 컬럼 부재 시 안전 처리
    etf_name_map = {}
    for t, df in data_map.items():
        if df.empty: continue
        if 'Name' in df.columns:
            etf_name_map[t] = df['Name'].iloc[-1]
        else:
            etf_name_map[t] = ETF_UNIVERSE.get(t, {}).get('name', t)
    
    for i, curr_dt in enumerate(dates):
        curr_date = curr_dt.date()
        progress_bar.progress((i + 1) / total_steps)
        status_text.text(f"시뮬레이션 진행 중: {curr_date} ({i+1}/{total_steps})")
        
        # 1. 해당 시점 랭킹 산출
        top_etfs = get_top_etfs(curr_date, data_map, selected_manager, min_score, exclude_risky=exclude_risky, top_n_etf=top_n_etf)
        
        q_count = len(top_etfs)
        avg_score = sum([e['momentum_score'] for e in top_etfs[:3]]) / min(q_count, 3) if q_count > 0 else 0
        
        etf_summary = "(없음)"
        stock_summary = "-"
        
        if top_etfs:
            # 2. 대표 주도주 분석 (중복 종목)
            sel_tickers = [e['ticker'] for e in top_etfs]
            overlap_list = analyze_overlapping_stocks_report(sel_tickers, top_n=5, ref_date=curr_date, etf_name_map=etf_name_map)
            
            # 요약 메시지 생성
            etf_summary = ", ".join([e['name'] for e in top_etfs[:3]]) + (f" 외 {q_count-3}개" if q_count > 3 else "")
            if overlap_list:
                stock_summary = ", ".join([f"{s.get('name', s.get('종목명'))}({s['중복횟수']}회)" for s in overlap_list[:3]])
        
        daily_stats.append({'Date': curr_date, 'Qualified_Count': q_count, 'Avg_Score': avg_score})
        history_log.append({
            '날짜': curr_date, '운용사': manager_str, '선정 ETF': etf_summary,
            '대표 주도주 (Top 5)': stock_summary, 'ETF 수': q_count, '평균 점수': avg_score
        })
        
    progress_bar.empty()
    status_text.empty()
    return pd.DataFrame(daily_stats), pd.DataFrame(history_log)

def check_market_defense(data_dict, ref_date=None):
    """코스닥 지수의 60일 이동평균선 상향 돌파 여부를 체크하여 시장 방어 상태를 산출합니다."""
    if 'KOSDAQ' not in data_dict: return True, 0, 0
    df = data_dict['KOSDAQ']
    if len(df) < 60: return True, 0, 0
    
    if ref_date:
        target_dt = pd.Timestamp(ref_date)
        if target_dt in df.index:
            close = df.loc[:target_dt, 'Close']
        else:
            asof_idx = df.index.asof(target_dt)
            if pd.isna(asof_idx): return True, 0, 0
            close = df.loc[:asof_idx, 'Close']
    else:
        close = df['Close']
        
    if len(close) < 60: return True, 0, 0
    curr_idx = close.iloc[-1]
    ma_60 = close.rolling(window=60).mean().iloc[-1]
    return curr_idx >= ma_60, curr_idx, ma_60

def run_advanced_simulation(params):
    """
    params: dict with keys:
      start_date, end_date, sel_managers, etf_min_score, etf_ma_period, 
      stock_ma_period, overlap_threshold, initial_capital, per_trade_amt
    """
    start_date = params['start_date']
    end_date = params['end_date']
    sel_managers = params['sel_managers']
    
    status_text = st.empty()
    status_bar = st.progress(0)
    
    try:
        # A. Prepare ETF Data (Universe)
        target_etf_tickers = [t for t, info in ETF_UNIVERSE.items() if info['manager'] in sel_managers]
        
        status_text.text(f"1. 대상 ETF 데이터 로딩 중... ({len(target_etf_tickers)}개)")
        
        etf_data_pool = {}
        fetch_start = start_date - timedelta(days=200)
        
        def fetch_single(t):
            try:
                return t, fdr.DataReader(t, fetch_start, end_date)
            except Exception:
                return t, pd.DataFrame()
                
        ctx = get_script_run_ctx()
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            if ctx:
                futures = [executor.submit(lambda t=t: (add_script_run_ctx(ctx), fetch_single(t))[1]) for t in target_etf_tickers]
                executor_results = [f.result() for f in futures]
            else:
                executor_results = list(executor.map(fetch_single, target_etf_tickers))
            
        for t, df in executor_results:
            if not df.empty and len(df) > 60:
                # Pre-calc Indicators
                df['MA_Filter'] = df['Close'].rolling(params['etf_ma_period']).mean()
                
                df['R_1W'] = df['Close'].pct_change(5)
                df['R_1M'] = df['Close'].pct_change(20)
                df['R_3M'] = df['Close'].pct_change(60)
                vol = df['Close'].pct_change().rolling(20).std() * np.sqrt(252)
                df['Vol'] = vol.replace(0, 0.01).fillna(0.01)
                
                df['Score'] = (df['R_3M'] * 0.5 + df['R_1M'] * 0.3 + df['R_1W'] * 0.2) / df['Vol']
                etf_data_pool[t] = df
        
        # B. Prepare Stock Data
        potential_stocks = set()
        for t in target_etf_tickers:
            h_list = load_etf_holdings(t)
            if h_list:
                for h in h_list[:10]:
                    if h.get('ticker'): potential_stocks.add(h['ticker'])
        
        status_text.text(f"2. 잠재적 구성종목 데이터 로딩 중... ({len(potential_stocks)}개)")
        
        stock_data_pool = {}
        stock_tickers_list = list(potential_stocks)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            if ctx:
                futures = [executor.submit(lambda t=t: (add_script_run_ctx(ctx), fetch_single(t))[1]) for t in stock_tickers_list]
                s_executor_results = [f.result() for f in futures]
            else:
                s_executor_results = list(executor.map(fetch_single, stock_tickers_list))
            
        for t, df in s_executor_results:
            if not df.empty and len(df) > 60:
                df['MA_Exit'] = df['Close'].rolling(params['stock_ma_period']).mean()
                stock_data_pool[t] = df
        
        # C. Run Simulation
        date_range = pd.date_range(start_date, end_date, freq='B') 
        
        cash = params['initial_capital']
        portfolio = {} 
        trades = []
        equity_curve = []
        
        last_known_prices = {}
        
        total_days = len(date_range)
        status_text.text("3. 시뮬레이션 진행 중...")
        
        for i, today in enumerate(date_range):
            status_bar.progress((i+1)/total_days)
            today_str = today.strftime('%Y-%m-%d')
            
            # Mark-to-Market
            curr_stock_val = 0
            for stk_t, p_info in portfolio.items():
                qty = p_info['qty']
                curr_p = 0
                if stk_t in stock_data_pool:
                        s_df = stock_data_pool[stk_t]
                        if today in s_df.index:
                            p = s_df.loc[today]['Close']
                            if p > 0: 
                                curr_p = p
                                last_known_prices[stk_t] = p
                
                if curr_p == 0:
                    curr_p = last_known_prices.get(stk_t, p_info['buy_price'])
                
                curr_stock_val += qty * curr_p
            
            # Sell Logic
            stocks_to_sell = []
            current_holdings = list(portfolio.keys())
            
            for stk_t in current_holdings:
                if stk_t in stock_data_pool:
                    s_df = stock_data_pool[stk_t]
                    if today in s_df.index:
                        curr_price = s_df.loc[today]['Close']
                        exit_price = s_df.loc[today]['MA_Exit']
                        
                        if curr_price > 0 and exit_price > 0 and curr_price < exit_price:
                            stocks_to_sell.append(stk_t)
            
            for stk_t in stocks_to_sell:
                p_info = portfolio[stk_t]
                qty = p_info['qty']
                buy_price = p_info['buy_price']
                stk_name = p_info['name']
                
                curr_price = 0
                if stk_t in stock_data_pool:
                        s_df = stock_data_pool[stk_t]
                        if today in s_df.index:
                            curr_price = s_df.loc[today]['Close']
                if curr_price <= 0: curr_price = last_known_prices.get(stk_t, buy_price)

                sell_amt = qty * curr_price
                cash += sell_amt
                
                curr_stock_val -= (qty * curr_price) 
                
                ret_pct = (curr_price - buy_price) / buy_price * 100
                del portfolio[stk_t]
                
                trades.append({
                    'Date': today_str, 'Type': 'SELL', 'Ticker': stk_t, 
                    'Name': stk_name, # Simplified name logic
                    'Price': curr_price, 
                    'Return': f"{ret_pct:+.2f}%",
                    'Note': 'MA 이탈',
                    'Total Asset': int(cash + curr_stock_val),
                    'Stock Asset': int(curr_stock_val),
                    'Cash Asset': int(cash)
                })

            # Buy Logic
            qualified_etfs = []
            for etf_t, e_df in etf_data_pool.items():
                if today in e_df.index:
                    row = e_df.loc[today]
                    if row['Score'] >= params['etf_min_score'] and row['Close'] >= row['MA_Filter']:
                        qualified_etfs.append(etf_t)
            
            if qualified_etfs:
                    candidate_stocks = []
                    candidate_maps = {} 
                    
                    for etf_t in qualified_etfs:
                        h_list = load_etf_holdings(etf_t)
                        if h_list:
                            for h in h_list:
                                t_code = h.get('ticker')
                                if t_code:
                                    candidate_stocks.append(t_code)
                                    candidate_maps[t_code] = h.get('name', t_code)
                    
                    counts = Counter(candidate_stocks)
                    targets = [t for t, c in counts.items() if c >= params['overlap_threshold']]
                    
                    for stk_t in targets:
                        if stk_t not in portfolio and stk_t in stock_data_pool:
                            s_df = stock_data_pool[stk_t]
                            if today in s_df.index:
                                curr_price = s_df.loc[today]['Close']
                                ma_val = s_df.loc[today]['MA_Exit']
                                
                                if curr_price > 0 and curr_price > ma_val:
                                    if cash >= params['per_trade_amt']: 
                                        buy_amt = params['per_trade_amt']
                                        
                                        qty = buy_amt / curr_price
                                        if qty > 0:
                                            stk_name = candidate_maps.get(stk_t, stk_t)
                                            portfolio[stk_t] = {
                                                'qty': qty, 'name': stk_name, 'buy_price': curr_price
                                            }
                                            last_known_prices[stk_t] = curr_price
                                            
                                            cash -= buy_amt
                                            curr_stock_val += buy_amt # Add new stock value
                                            
                                            trades.append({
                                            'Date': today_str, 'Type': 'BUY', 'Ticker': stk_t, 
                                            'Name': stk_name, 'Price': curr_price, 
                                            'Return': '-',
                                            'Note': f"Overlap({counts[stk_t]})",
                                            'Total Asset': int(cash + curr_stock_val),
                                            'Stock Asset': int(curr_stock_val),
                                            'Cash Asset': int(cash)
                                            })

            # Valuation (Final for Day)
            final_stock_val = 0
            for stk_t, p_info in portfolio.items():
                qty = p_info['qty']
                curr_p = 0
                if stk_t in stock_data_pool:
                        s_df = stock_data_pool[stk_t]
                        if today in s_df.index:
                            p = s_df.loc[today]['Close']
                            if p > 0: curr_p = p
                if curr_p == 0: curr_p = last_known_prices.get(stk_t, p_info['buy_price'])
                final_stock_val += qty * curr_p
            
            total_equity = cash + final_stock_val
            equity_curve.append({'Date': today, 'Equity': total_equity})
        
        status_text.text("분석 완료!")
        status_bar.empty()
        
        return {
            'equity_curve': equity_curve,
            'trades': trades,
            'initial_capital': params['initial_capital'],
            'run_date': datetime.now().strftime('%Y-%m-%d %H:%M')
        }
    
    except Exception as e:
        st.error(f"시뮬레이션 중 오류 발생: {e}")
        return None
