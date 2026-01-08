import sqlite3
import os
from datetime import datetime
import pandas as pd

DB_PATH = os.path.join(os.getcwd(), 'etf_strategy.db')

def get_connection():
    """SQLite 데이터베이스 연결을 반환합니다."""
    return sqlite3.connect(DB_PATH)

def init_db():
    """데이터베이스 테이블을 초기화합니다."""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # ETF 유니버스 테이블 (메타데이터)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS etf_universe (
                ticker TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                theme TEXT,
                manager TEXT,
                net_assets INTEGER,
                category TEXT,
                updated_at TEXT
            )
        ''')
        
        # [Migration] 'category' 컬럼 추가 (Schema Evolution)
        try:
            cursor.execute("ALTER TABLE etf_universe ADD COLUMN category TEXT")
        except:
            pass 

        
        # ETF 구성 종목 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS etf_holdings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                etf_ticker TEXT,
                stock_ticker TEXT,
                stock_name TEXT,
                weight REAL,
                updated_at TEXT,
                FOREIGN KEY (etf_ticker) REFERENCES etf_universe (ticker)
            )
        ''')
        
        # 검색 최적화를 위한 인덱스 추가
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_holdings_etf ON etf_holdings(etf_ticker)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_holdings_stock ON etf_holdings(stock_name)')
        
        # 리포트 생성 로그 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS report_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_date TEXT,
                report_type TEXT,
                top_etf TEXT,
                avg_score REAL,
                file_name TEXT,
                report_data BLOB,
                created_at TEXT
            )
        ''')
        
        conn.commit()

def save_etf_universe(data_list):
    """ETF 유니버스 데이터를 저장합니다. (List of dicts)"""
    with get_connection() as conn:
        cursor = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # [Fix] 기존 목록 삭제 후 재저장 (필터링된 목록만 유지하기 위함)
        cursor.execute('DELETE FROM etf_universe')
        
        for item in data_list:
            cursor.execute('''
                INSERT OR REPLACE INTO etf_universe (ticker, name, theme, manager, net_assets, category, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                item['ticker'], 
                item['name'], 
                item.get('type', item.get('theme', 'Unknown')), 
                item.get('manager', 'Unknown'), 
                item.get('net_assets', 0),
                item.get('category', 'Theme'),
                now
            ))
        conn.commit()

def save_etf_holdings(etf_ticker, holdings_list):
    """특정 ETF의 구성 종목을 저장합니다."""
    with get_connection() as conn:
        cursor = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 기존 데이터 삭제 후 재삽입
        cursor.execute('DELETE FROM etf_holdings WHERE etf_ticker = ?', (etf_ticker,))
        
        for h in holdings_list:
            # weight 세척 (예: "15.05%" -> 15.05)
            weight_val = h.get('pct', '0').replace('%', '').strip()
            try: weight_val = float(weight_val)
            except: weight_val = 0.0
            
            cursor.execute('''
                INSERT INTO etf_holdings (etf_ticker, stock_ticker, stock_name, weight, updated_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (etf_ticker, h['ticker'], h['name'], weight_val, h.get('updated_at', now)))
        conn.commit()

def get_all_etfs():
    """모든 ETF 메타데이터를 딕셔너리 형태로 반환합니다."""
    try:
        with get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM etf_universe')
            rows = cursor.fetchall()
            
            universe = {}
            for row in rows:
                universe[row['ticker']] = {
                    'name': row['name'],
                    'theme': row['theme'],
                    'manager': row['manager'],
                    'manager': row['manager'],
                    'net_assets': row['net_assets'],
                    'category': row['category'],
                    'desc': f"{row['manager']} | {row['net_assets']}억"
                }
            return universe
    except:
        return {}

def get_etf_holdings(etf_ticker):
    """특정 ETF의 구성 종목 리스트를 반환합니다."""
    try:
        with get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM etf_holdings WHERE etf_ticker = ? ORDER BY weight DESC', (etf_ticker,))
            rows = cursor.fetchall()
            
            holdings = []
            for row in rows:
                holdings.append({
                    'name': row['stock_name'],
                    'ticker': row['stock_ticker'],
                    'pct': f"{row['weight']}%"
                })
            return holdings
    except:
        return []

def get_etfs_by_stock(stock_name):
    """특정 종목을 포함하고 있는 모든 ETF를 검색합니다."""
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT u.name as etf_name, h.weight 
            FROM etf_holdings h
            JOIN etf_universe u ON h.etf_ticker = u.ticker
            WHERE h.stock_name LIKE ?
            ORDER BY h.weight DESC
        ''', (f'%{stock_name}%',))
        return [dict(row) for row in cursor.fetchall()]

def log_report_generation(report_type, top_etf, avg_score, file_name, file_binary):
    """리포트 생성 이력과 파일을 DB에 저장합니다."""
    with get_connection() as conn:
        cursor = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('''
            INSERT INTO report_logs (report_date, report_type, top_etf, avg_score, file_name, report_data, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (datetime.now().strftime("%Y-%m-%d"), report_type, top_etf, avg_score, file_name, sqlite3.Binary(file_binary), now))
        conn.commit()

def get_report_history(limit=10):
    """최근 생성된 리포트 이력을 가져옵니다."""
    try:
        with get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM report_logs ORDER BY created_at DESC LIMIT ?', (limit,))
            return [dict(row) for row in cursor.fetchall()]
    except:
        return []

def delete_report(report_id):
    """특정 ID의 리포트를 삭제합니다."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM report_logs WHERE id = ?', (report_id,))
        conn.commit()
