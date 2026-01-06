
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

def update_etf_list_seibro_param():
    """
    네이버 금융 ETF > 테마/섹터 탭(etfType=2)의 데이터를 직접 크롤링합니다 (API 사용).
    Seibro의 테마 분류와 가장 유사하며, 순자산 50억 이상만 필터링합니다.
    """
    import requests
    
    # Naver Finance Internal API URL
    url = "https://finance.naver.com/api/sise/etfItemList.nhn?etfType=2&targetColumn=market_sum&sortOrder=desc"
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': 'https://finance.naver.com/sise/etf.naver'
        }
        resp = requests.get(url, headers=headers, timeout=5)
        data = resp.json()
        
        if data['resultCode'] != 'success':
            print("Naver API Error")
            return []
            
        items = data['result']['etfItemList']
        
    except Exception as e:
        print(f"Naver Crawl Error: {e}")
        return []
        
    filtered_list = []
    
    # 제외 키워드 (순수 테마가 아닌 것들)
    # 200 IT, 200 Energy 등 KOSPI 200 섹터 지수 상품은 '테마'라기보다 '섹터'임.
    # 하지만 사용자는 '반도체', '2차전지' 등을 원함.
    # 너무 엄격하게 자르면 TIGER 200 IT(반도체 포함) 같은게 날아감.
    # 일단 '테마/섹터' 탭에 있는 것들은 다 포함하되, 너무 작은 것만 자름.
    
    for item in items:
        name = item['itemname']
        ticker = item['itemcode']
        net_assets = int(item['marketSum']) # 억원 단위
        
        # [User Request] "아무런 조건을 필터링 할 필요가 없어"
        # 사이트에 있는 그대로 다 받습니다.
        
        brand = name.split()[0].upper()
        manager = BRAND_TO_MANAGER.get(brand, brand)
        
        filtered_list.append({
            'ticker': ticker,
            'name': name,
            'theme': 'Theme/Sector', 
            'net_assets': net_assets,
            'manager': manager
        })
        
        filtered_list.append({
            'ticker': ticker,
            'name': name,
            'theme': 'Theme/Sector', 
            'net_assets': net_assets,
            'manager': manager
        })
            
    # 정렬: 순자산 순
    filtered_list.sort(key=lambda x: x['net_assets'], reverse=True)
    
    return filtered_list

# Alias for backward compatibility if needed
fetch_latest_etf_list = update_etf_list_seibro_param

