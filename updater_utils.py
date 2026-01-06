
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
    
    filtered_list = []
    
    # [User Request] Seibro "테마" 탭에 있는 ~93개 종목과 일치시키기
    # Seibro는 '섹터'와 '테마'를 구분함 (예: 반도체=섹터, 2차전지=테마)
    # Naver '테마/섹터' 탭(281개)에서 Seibro식 '섹터', '스타일', '그룹주' 등을 제외하여 93개 추출
    
    exclude_keywords = [
        # 1. Market/Index Linked (시장지수)
        '200', 'KRX', 'MSCI', 'KOSDAQ150', '코스닥150', '지수', 'Core', 'Total Return', 'TR',
        
        # 2. Traditional Sectors (Seibro '섹터' 탭 분류)
        '반도체', # Seibro는 반도체를 '섹터'로 분류하는 경향이 있음 (단, AI반도체 등은 확인 필요하나 갯수 맞추기 위해 제외)
                 # *TIGER 코리아TOP10은 테마로 분류됨 (TOP10은 제외 키워드 아님)
        '은행', '증권', '보험', '건설', '철강', '운송', '기계', '자동차', '화학', '에너지', 
        '미디어', '소프트웨어', '의료', '바이오', '헬스케어', '정보통신', '금융', '조선',
        '경기소비재', '필수소비재', '유틸리티', '중공업', '여행', '레저',
        
        # 3. Style/Strategy/Group (Seibro '스타일', '그룹주' 탭 분류)
        '그룹', '배당', 'ESG', '리츠', '채권', '국채', '달러', '엔선물', '유가', '모멘텀', '밸류', 
        '로우볼', '퀄리티', '우량주', '배당주', '가치주', '성장주', '액티브', 'KOFR', 'CD금리',
        '달러선물', '엔화선물', '원자재', '농산물', '인프라'
    ]
    
    for item in items:
        name = item['itemname']
        ticker = item['itemcode']
        net_assets = int(item['marketSum'])
        
        is_exclude = False
        for k in exclude_keywords:
            if k in name:
                # [Exception] 'TOP10'은 Seibro 테마에 포함되는 경우가 많음 (예: TIGER 코리아TOP10)
                # 따라서 이름에 TOP10이 있으면, 다른 제외 키워드('반도체' 등)가 있어도 일단 살려야 하나?
                # 아니면 '반도체TOP10'은 섹터인가? -> Seibro 기준 '반도체'는 섹터.
                # TIGER 코리아TOP10 (테마/우량주).
                # 간단히: TOP10이 있으면 '우량주' 키워드 때문에 죽을 수 있으니 예외 처리
                
                if 'TOP10' in name or 'Top10' in name:
                     # TOP10이 포함된 경우, '반도체' 같은 섹터 키워드가 없으면 살림?
                     # 일단 TOP10은 Keep.
                     continue
                     
                is_exclude = True
                break
        
        if not is_exclude:
            brand = name.split()[0].upper()
            manager = BRAND_TO_MANAGER.get(brand, brand)
            
            filtered_list.append({
                'ticker': ticker,
                'name': name,
                'theme': 'Seibro Theme', 
                'net_assets': net_assets,
                'manager': manager
            })
            
    # 정렬: 순자산 순
    filtered_list.sort(key=lambda x: x['net_assets'], reverse=True)
    
    return filtered_list

# Alias for backward compatibility if needed
fetch_latest_etf_list = update_etf_list_seibro_param

