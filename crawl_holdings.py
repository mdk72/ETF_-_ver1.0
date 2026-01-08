import pandas as pd
import requests
import json
import os
import time

def fetch_etf_holdings(ticker):
    """
    Fetches Top 10 constituents for a given KOSPI/KOSDAQ ETF from Naver Finance.
    Returns a list of dicts: [{'name': 'Samsung Elec', 'ticker': '005930', 'pct': '25.0%'}, ...]
    """
    url = f"https://finance.naver.com/item/main.naver?code={ticker}"
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        resp = requests.get(url, headers=headers, timeout=5)
        
        # pd.read_html will auto-detect encoding from HTTP headers
        from io import StringIO
        dfs = pd.read_html(StringIO(resp.text))
        
        # Look for the holdings table
        # Actual column names from Naver: '종목명(코드명)', '보유주식수(좌)', '비중', etc.
        holding_df = None
        for df in dfs:
            cols_str = ' '.join([str(c) for c in df.columns])
            # Match if contains '종목' AND ('비중' OR '구성')
            if '종목' in cols_str and ('비중' in cols_str or '구성' in cols_str):
                holding_df = df
                break
        
        if holding_df is None:
            return []

        # Find the correct column indices
        name_col = None
        pct_col = None
        
        for col in holding_df.columns:
            col_str = str(col)
            if '종목' in col_str and name_col is None:
                name_col = col
            if '비중' in col_str and pct_col is None:
                pct_col = col
        
        if name_col is None or pct_col is None:
            return []
        
        # 추가: HTML에서 종목코드 추출
        import re
        ticker_pattern = r'<a href="/item/main\.naver\?code=(\d{6})">([^<]+)</a>'
        ticker_map = {}  # {name: ticker}
        for code, name in re.findall(ticker_pattern, resp.text):
            ticker_map[name.strip()] = code
        
        # Extract holdings
        holdings = []
        for _, row in holding_df.iterrows():
            name = row[name_col]
            if pd.isna(name) or name == '':
                continue
            
            name_str = str(name).strip()
            pct = row[pct_col]
            pct_str = str(pct) if not pd.isna(pct) else ""
            
            # Manual Map for recent listings or special cases
            MANUAL_TICKER_MAP = {
                '삼성에피스홀딩스': '0126Z0'
            }
            
            # Get ticker from map
            stock_ticker = ticker_map.get(name_str, None)
            
            # [Fallback 1] Manual Map
            if not stock_ticker:
                stock_ticker = MANUAL_TICKER_MAP.get(name_str, None)
            
            # [Fallback] If ticker is None (e.g. regex failed), try FDR listing map
            if not stock_ticker:
                try:
                    import FinanceDataReader as fdr
                    # Memoization could be better, but for now let's use a simple static map if possible or fetch
                    # To avoid fetching every time, let's assume we can fetch once per run or rely on DB
                    # For minimal change, let's fetch KRX list if not cached in global
                    if 'KRX_TICKER_MAP' not in globals():
                        global KRX_TICKER_MAP
                        df_krx = fdr.StockListing('KRX')
                        KRX_TICKER_MAP = dict(zip(df_krx['Name'], df_krx['Code']))
                    
                    stock_ticker = KRX_TICKER_MAP.get(name_str, None)
                except:
                    pass
            
            holdings.append({
                'name': name_str, 
                'ticker': stock_ticker,
                'pct': pct_str
            })
            
        return holdings[:10]  # Top 10
        
    except Exception as e:
        # Silently fail (too verbose otherwise)
        return []

def update_holdings_db():
    from db_manager import get_all_etfs, save_etf_holdings
    print("Loading Universe from DB...")
    universe = get_all_etfs()
    
    if not universe:
        print("DB Universe is empty.")
        return

    print(f"Start crawling for {len(universe)} ETFs...")
    for i, (ticker, info) in enumerate(universe.items()):
        print(f"[{i+1}/{len(universe)}] Crawling {info['name']} ({ticker})...")
        holdings = fetch_etf_holdings(ticker)
        
        if holdings:
            save_etf_holdings(ticker, holdings)
        else:
            print(f"  -> No holdings found or error.")
        
        time.sleep(0.3)
    print("Done. DB updated.")

if __name__ == "__main__":
    update_holdings_db()
