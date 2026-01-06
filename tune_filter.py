
import requests

def tune_filter():
    url = "https://finance.naver.com/api/sise/etfItemList.nhn?etfType=2&targetColumn=market_sum&sortOrder=desc"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    resp = requests.get(url, headers=headers)
    items = resp.json()['result']['etfItemList']
    
    print(f"Total Naver Items: {len(items)}")
    
    # Exclusion List to match Seibro 'Theme' (Removes Sectors, Groups, Strategies)
    exclude_keywords = [
        # Market/Index Linked
        '200', 'KRX', 'MSCI', 'KOSDAQ150', '코스닥150', '지수', 'Core', 'Total Return', 'TR',
        'Top10', 'TOP10', # Seibro 'Theme' usually doesn't include generic Top10 unless it has a theme prefix.
                           # But 'TIGER 코리아TOP10' was in user screenshot! So TOP10 is ALLOWED?
                           # Wait, 'TIGER 코리아TOP10' is type '테마/우량주'.
                           # So 'TOP10' might be okay.
        
        # Traditional Sectors (Seibro puts these in 'Sector' tab)
        '반도체', # Wait! Is 'Semiconductor' a Theme or Sector?
                 # Naver has 'KODEX 반도체'. Seibro probably has it in 'Sector'.
                 # User screenshot has '2차전지' (Secondary Battery), '방산' (Defense). 
                 # Usually 'Semiconductor' is Sector, 'AI Semiconductor' is Theme.
                 # Let's try excluding '반도체' first, but keeping 'AI반도체'?
                 # No, excluding '반도체' is risky.
                 
        '은행', '증권', '보험', '건설', '철강', '운송', '기계', '자동차', '화학', '에너지', 
        '미디어', '소프트웨어', '의료', '바이오', '헬스케어', '정보통신', '금융', '조선',
        '경기소비재', '필수소비재', '유틸리티', '중공업', '여행', '레저',
        
        # Style/Strategy/Group (Seibro tabs)
        '그룹', '배당', 'ESG', '리츠', '채권', '국채', '달러', '엔선물', '유가', '모멘텀', '밸류', 
        '로우볼', '퀄리티', '우량주', '배당주', '가치주', '성장주', '액티브', 'KOFR', 'CD금리',
        '달러선물', '엔화선물', '원자재', '농산물', '리츠', '인프라'
    ]
    
    # Refined Logic
    filtered = []
    excluded_samples = []
    
    for item in items:
        name = item['itemname']
        is_exclude = False
        
        # If TOP10 is in name, be careful. 
        # But generally standard exclusions apply.
        
        # Check standard exclusions
        for k in exclude_keywords:
            if k in name and k not in ['TOP10']: # Exception for TOP10 for now
                is_exclude = True
                break
        
        # Special check for '반도체': 
        # If name is just 'KODEX 반도체', exclude (Sector).
        # If 'KODEX AI반도체', keep (Theme).
        # How to distinguish?
        
        if not is_exclude:
            filtered.append(item)
        else:
            excluded_samples.append(name)
            
    print(f"Filtered Count: {len(filtered)}")
    print("\nSample Kept:")
    for x in filtered[:10]:
        print(x['itemname'])
        
    print(f"\nSample Excluded (Total {len(excluded_samples)}):")
    for x in excluded_samples[:10]:
        print(x)

if __name__ == "__main__":
    tune_filter()
