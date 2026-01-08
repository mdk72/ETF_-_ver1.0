
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


# [User Request Update] 
# 사용자가 제공한 'etf_list.xlsx' (93종목)이 Single Source of Truth입니다.
# 더 이상 크롤링/필터링 로직을 사용하지 않고, 파일 내용을 그대로 DB에 반영합니다.

def update_etf_list_seibro_param():
    """
    etf_list.xlsx 파일에서 종목명을 읽어와 FDR 마스터 정보와 매핑하여 반환합니다.
    '테마etf_list' 시트는 'Theme', '섹터etf_list' 시트는 'Sector' 카테고리로 분류합니다.
    """
    import pandas as pd
    import FinanceDataReader as fdr
    import os
    
    file_path = "etf_list.xlsx"
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return []
        
    try:
        # [Step 1] Read Excel Sheets
        xls = pd.ExcelFile(file_path)
        all_etfs = []
        
        # FDR Master List for Ticker Mapping
        df_master = fdr.StockListing('ETF/KR')
        name_to_row = {row['Name']: row for _, row in df_master.iterrows()}
        
        # Define sheets and their categories mapping
        # Try to find sheets that close match the user provided names
        sheet_map = {}
        for sheet in xls.sheet_names:
            if '테마' in sheet: sheet_map[sheet] = 'Theme'
            elif '섹터' in sheet: sheet_map[sheet] = 'Sector'
            else: print(f"Skipping unknown sheet: {sheet}")
        
        if not sheet_map:
            # Fallback for legacy format (one sheet or different names)
            print("Warning: Could not identify Theme/Sector sheets. Treating first sheet as Theme.")
            sheet_map[xls.sheet_names[0]] = 'Theme'

        print(f"Found sheets: {sheet_map}")

        for sheet_name, category in sheet_map.items():
            df_target = pd.read_excel(xls, sheet_name=sheet_name)
            target_names = df_target.iloc[:, 0].dropna().astype(str).str.strip().tolist()
            
            for name in target_names:
                if name in name_to_row:
                    row = name_to_row[name]
                    ticker = str(row['Symbol'])
                    net_assets = row['MarCap'] if 'MarCap' in row else 0
                    brand = name.split()[0].upper()
                    manager = BRAND_TO_MANAGER.get(brand, brand)
                    
                    all_etfs.append({
                        'ticker': ticker,
                        'name': name,
                        'theme': 'Selected Theme' if category == 'Theme' else 'Selected Sector', 
                        'net_assets': net_assets,
                        'manager': manager,
                        'category': category
                    })
                else:
                    print(f"Warning: ETF Name '{name}' (Sheet: {sheet_name}) not found in FDR listing.")
        
        # Sort by Net Assets
        all_etfs.sort(key=lambda x: x['net_assets'], reverse=True)
        
        print(f"Loaded {len(all_etfs)} ETFs from {file_path}")
        return all_etfs

    except Exception as e:
        print(f"Error loading Excel list: {e}")
        return []


# Alias for backward compatibility if needed
fetch_latest_etf_list = update_etf_list_seibro_param
