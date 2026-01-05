
import os
import shutil
import json
import datetime
import FinanceDataReader as fdr

# 주요 운용사 브랜드 매핑
BRAND_TO_MANAGER = {
    'KODEX': '삼성자산운용',
    'TIGER': '미래에셋자산운용',
    'KBSTAR': 'KB자산운용',
    'ACE': '한국투자신탁운용',
    'SOL': '신한자산운용',
    'HANARO': 'NH-Amundi자산운용',
    'KOSEF': '키움투자자산운용',
    'ARIRANG': '한화자산운용',
    'TIMEFOLIO': '타임폴리오자산운용',
    'WON': '우리자산운용',
    'HK': '흥국자산운용',
    'FOCUS': '브이아이자산운용',
    'MASTER': 'KCGI자산운용',
    'MIGHTY': 'DB자산운용',
    'WOORI': '우리자산운용'
}

def backup_data_files():
    """백업 폴더 생성 후 DB 파일 백업"""
    backup_dir = "backup"
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
        
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    db_file = 'etf_strategy.db'
    
    if os.path.exists(db_file):
        dest = os.path.join(backup_dir, f"etf_strategy_{timestamp}.db")
        shutil.copy2(db_file, dest)
        return [dest]
            
    return []

def fetch_latest_etf_list():
    """FDR을 사용하여 최신 ETF 목록 수집 및 포맷 변환"""
    from db_manager import get_all_etfs
    existing_universe = get_all_etfs()
    
    try:
        df = fdr.StockListing('ETF/KR')
    except Exception as e:
        print(f"FDR Error: {e}")
        return []
        
    etf_list = []
    for _, row in df.iterrows():
        ticker = str(row['Symbol'])
        name = row['Name']
        brand = name.split()[0].upper()
        manager = BRAND_TO_MANAGER.get(brand, brand)
        net_assets = row['MarCap'] if 'MarCap' in row else 0
        
        # 기존 테마 정보 유지
        theme = existing_universe.get(ticker, {}).get('theme', '')
        
        etf_list.append({
            'ticker': ticker,
            'name': name,
            'theme': theme,
            'net_assets': net_assets,
            'manager': manager
        })
    return etf_list

