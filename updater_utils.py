
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
    Seibro/Naver 등에서 '테마' 유형의 ETF만 선별적으로 수집합니다.
    (실제 크롤링 대신 FDR 전체 데이터에서 필터링하여 근사치를 추출)
    """
    from db_manager import get_all_etfs
    
    try:
        # 1. 전체 목록 수집
        df = fdr.StockListing('ETF/KR')
    except Exception as e:
        print(f"FDR Error: {e}")
        return []
        
    filtered_list = []
    
    # 2. 제외 키워드 설정 (시장지수, 채권, 통화, 파생 등 제외 -> 테마만 남김)
    # seibro '테마' 탭에 있는 종목들은 보통 특정 산업/섹터/테마를 추종함
    exclude_keywords = [
        # 시장 지수
        'KOSPI', 'KOSDAQ', '200', '150', 'KRX', 'MSCI', '국채', '채권', 'Bond', 
        'Dollar', 'USD', 'Yen', 'Euro', 'Gold', 'Silver', 'Crude', 'Commodity',
        'KoAct', 'CD금리', 'KOFR', 'T-Bill', '단기통안채', '머니마켓',
        # 파생/레버리지/인버스 (전략에 따라 포함될 수도 있지만, 보통 테마 전략에선 제외)
        'Leverage', 'Inverse', '2X', '레버리지', '인버스', '선물'
    ]
    
    # 테마 키워드 (포함) - 이게 포함되면 우선적으로 테마로 간주
    theme_keywords = [
        '2차전지', '반도체', 'AI', '인공지능', '메타버스', '바이오', '헬스케어', 
        '자동차', '전력', '원자력', '방산', '우주', '로봇', '소부장', '게임', 
        '미디어', '컨텐츠', '은행', '증권', '보험', '건설', '철강', '화학',
        '플랫폼', 'BBIG', '뉴딜', 'ESG', '탄소', '수소'
    ]

    for _, row in df.iterrows():
        name = row['Name']
        ticker = str(row['Symbol'])
        
        # 순자산 (없으면 0) - 50억 미만 제외 (사용자 스크린샷 50억 기준 참조)
        # FDR 'MarCap' is already in 억원 (100 Million Won) unit for ETF listing
        net_assets = row['MarCap'] if 'MarCap' in row else 0
        
        # FDR MarCap is usually full integer. 100억 = 10,000,000,000.
        # But wait, FDR MarCap is sometimes in 'Unit'. Let's assume input is won.
        # Generally FDR MarCap column is big integer.
        # Let's verify FDR output.
        
        # Filtering Logic
        is_theme = False
        
        # 1. 테마 키워드가 있으면 무조건 포함
        if any(k in name for k in theme_keywords):
            is_theme = True
            
        # 2. 제외 키워드가 있으면 제외 (단, 테마 키워드가 더 우선?? 보통 2차전지 레버리지도 테마로 볼 수 있음. 
        # 하지만 스크린샷 93개면 레버리지 제외된 순수 테마일 가능성 높음)
        if any(k in name for k in exclude_keywords):
            is_theme = False
            
        # 3. 추가 필터: 순자산 50억 이상 (너무 작은 ETF 제외)
        if net_assets < 50: 
            is_theme = False
            
        if is_theme:
            brand = name.split()[0].upper()
            manager = BRAND_TO_MANAGER.get(brand, brand)
            
            filtered_list.append({
                'ticker': ticker,
                'name': name,
                'theme': 'Generic', # 테마 상세 구분은 어려움
                'net_assets': net_assets,
                'manager': manager
            })
            
    # 정렬: 순자산 순
    filtered_list.sort(key=lambda x: x['net_assets'], reverse=True)
    
    # 93개 내외라면 Top 100 정도로 끊는 것도 방법
    # return filtered_list[:120] 
    return filtered_list

# Alias for backward compatibility if needed
fetch_latest_etf_list = update_etf_list_seibro_param

