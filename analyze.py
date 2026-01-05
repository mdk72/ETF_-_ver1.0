import sqlite3
import pandas as pd
import io
import os
import sys
from datetime import datetime

# Windows에서 출력 인코딩 강제 설정 (이모지 출력 오류 방지)
sys.stdout.reconfigure(encoding='utf-8')

# -----------------------------------------------------------------------------
# 1. Import App Modules for Automation
# -----------------------------------------------------------------------------
try:
    from data_manager import fetch_market_data
    from ui_views import generate_report_excel
    from db_manager import log_report_generation, get_report_history, delete_report
    from analysis_engine import calculate_momentum_score
except ImportError as e:
    print(f"❌ 모듈 임포트 실패: {e}")
    print("   이 스크립트는 프로젝트 루트 폴더에서 실행해야 합니다.")
    sys.exit(1)

# DB 경로 설정 (프로젝트 루트 기준)
DB_PATH = 'etf_strategy.db'

def get_latest_report_blob(period):
    """DB에서 가장 최근의 해당 주기 리포트 BLOB을 가져옵니다."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        query = "SELECT report_data, file_name FROM report_logs WHERE report_type = ? ORDER BY created_at DESC LIMIT 1"
        cursor.execute(query, (period,))
        row = cursor.fetchone()
        conn.close()
        return (row[0], row[1]) if row else (None, None)
    except Exception as e:
        print(f"❌ DB Error: {e}")
        return None, None

def automated_report_generation():
    """모든 주기(Daily/Weekly/Monthly) 리포트를 자동으로 생성하고 DB에 저장"""
    print("🚀 [Step 1] 최신 시장 데이터 가져오는 중 (Market Data Fetching)...")
    data_map = fetch_market_data()
    if not data_map:
        print("❌ 데이터 가져오기 실패. 인터넷 연결을 확인하세요.")
        return False
    
    print("\n🚀 [Step 2] 리포트 자동 생성 및 저장 (Generating Reports)...")
    current_date = datetime.now().date() # 실제 오늘 날짜 기준
    
    generated_files = []
    
    for period in ['daily', 'weekly', 'monthly']:
        print(f"  generating {period.capitalize()} report...", end=" ")
        try:
            # 1. 데이터 계산
            rank_df = calculate_momentum_score(data_map, ref_date=pd.Timestamp(current_date))
            if rank_df.empty:
                print("Skipped (Empty Data)")
                continue

            top_name = rank_df.iloc[0]['ShortName']
            avg_score = rank_df.head(5)['Score'].mean()
            
            # 2. 엑셀 바이트 생성 (analyze.py에서는 Snapshot Date를 '오늘'로 고정)
            excel_bytes = generate_report_excel(data_map, period=period, snapshot_date=pd.Timestamp(current_date))
            
            # 3. 파일명 생성 & DB 저장
            filename = f"ETF_{period}_{current_date.strftime('%Y%m%d')}_AUTO.xlsx"
            log_report_generation(period, top_name, avg_score, filename, excel_bytes)
            
            print(f"✅ Success ({len(excel_bytes)/1024:.1f} KB)")
            generated_files.append(filename)
        except Exception as e:
            print(f"❌ Error: {e}")

    print(f"\n✅ 리포트 생성 완료: {len(generated_files)}개 생성됨.\n")
    return True

class QuantDeepResearch:
    def __init__(self):
        self.data = {}
        self.periods = ['daily', 'weekly', 'monthly']
        self.load_data()

    def load_data(self):
        """DB에서 엑셀 바이트를 읽어 데이터프레임으로 변환"""
        print("🔄 [Step 3] 분석용 데이터 로딩 중 (From DB)...")
        
        sheet_map = {
            'summary': 'Summary',
            'rankings': 'Full_Universe_Rankings',
            'stocks': 'Leading_Stocks_Overlap',
            'themes': 'Theme_Analysis'
        }

        for period in self.periods:
            self.data[period] = {}
            blob, filename = get_latest_report_blob(period)
            
            if blob:
                # print(f"  Reading {period}: {filename}")
                try:
                    excel_file = pd.ExcelFile(io.BytesIO(blob))
                    for key, sheet_name in sheet_map.items():
                        if sheet_name in excel_file.sheet_names:
                            self.data[period][key] = pd.read_excel(excel_file, sheet_name=sheet_name)
                        else:
                            self.data[period][key] = pd.DataFrame() 
                except Exception as e:
                    print(f"     ❌ Parsing Error: {e}")
            else:
                print(f"  ⚠️ {period.capitalize()} Report Not Found in DB")
                for key in sheet_map.keys():
                    self.data[period][key] = pd.DataFrame()

    def check_market_confluence(self):
        """Research: Market Confluence"""
        print("📊 [Analysis 1] Market Pulse (Triple Bull Check)")
        statuses = []
        for p in self.periods:
            df = self.data[p].get('summary', pd.DataFrame())
            status = df.iloc[0]['Market_Status'] if not df.empty and 'Market_Status' in df.columns else "N/A"
            statuses.append(status)
            print(f"  - {p.capitalize()}: {status}")
        
        if statuses and all('Bull' in s for s in statuses):
            print("  🚀 TRIPLY BULLISH: Strong Buy Signal")
        else:
            print("  ⚠️ Mixed Signals: Caution Required")
        print("-" * 40)

    def find_triple_crown_stocks(self, top_n=5):
        """Research: Triple Crown Stocks"""
        print(f"👑 [Analysis 2] Triple Crown Stocks (Top {top_n} Intersection)")
        
        top_sets = []
        for p in self.periods:
            df = self.data[p].get('stocks', pd.DataFrame())
            if not df.empty and '종목명' in df.columns:
                top_sets.append(set(df.head(top_n)['종목명']))
            else:
                top_sets.append(set())

        triple_crown = set.intersection(*[s for s in top_sets if s]) if any(top_sets) else set()
        
        if triple_crown:
            print(f"  🏆 Winners ({len(triple_crown)}):")
            for stock in triple_crown:
                # Get detail from daily
                daily_df = self.data['daily'].get('stocks', pd.DataFrame())
                cnt = daily_df[daily_df['종목명']==stock]['중복횟수'].values[0] if not daily_df.empty and stock in daily_df['종목명'].values else "?"
                print(f"    🌟 {stock} (Freq: {cnt})")
        else:
            print("    (No common stocks found across all timeframes)")
        print("-" * 40)

    def analyze_theme_persistence(self):
        """Research: Theme Persistence"""
        print("🌊 [Analysis 3] Dominant Theme Persistence")
        leaders = []
        for p in self.periods:
            df = self.data[p].get('themes', pd.DataFrame())
            if not df.empty:
                theme = df.iloc[0]['Theme']
                leaders.append(theme)
                print(f"  - {p.capitalize()}: {theme}")
            else:
                print(f"  - {p.capitalize()}: N/A")
        
        if len(leaders) == 3 and len(set(leaders)) == 1:
            print(f"  💎 Mega Trend Identified: '{leaders[0]}'")
        else:
            print("  🔄 Sector Rotation Detected")
        print("-" * 40)

    def calculate_etf_momentum_correlation(self):
        """Research: ETF Correlation"""
        print("📈 [Analysis 4] Top Aligned ETFs")
        try:
            dfs = [self.data[p].get('rankings', pd.DataFrame()) for p in self.periods]
            if any(d.empty for d in dfs):
                print("  ⚠️ Insufficient data for correlation analysis")
                return

            # Merge strategies
            cols = ['Name', 'Score']
            merged = dfs[0][cols].rename(columns={'Score':'S_D'})
            merged = merged.merge(dfs[1][cols].rename(columns={'Score':'S_W'}), on='Name')
            merged = merged.merge(dfs[2][cols].rename(columns={'Score':'S_M'}), on='Name')
            
            merged['Avg'] = (merged['S_D'] + merged['S_W'] + merged['S_M']) / 3
            top5 = merged.sort_values('Avg', ascending=False).head(5)
            
            for i, r in top5.iterrows():
                print(f"    {i+1}. {r['Name']} (Avg: {r['Avg']:.2f})")
        except Exception:
            print("  Analyis Error")

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🤖 AUTO-QUANT SYSTEM: GEN & ANALYZE")
    print("="*50 + "\n")
    
    # 1. 자동 생성 실행
    success = automated_report_generation()
    
    if success:
        # 2. 분석 실행
        researcher = QuantDeepResearch()
        researcher.check_market_confluence()
        researcher.analyze_theme_persistence()
        researcher.find_triple_crown_stocks(top_n=5)
        researcher.calculate_etf_momentum_correlation()
    
    print("\n" + "="*50)
    print("🏁 SYSTEM COMPLETE")
    print("="*50)
