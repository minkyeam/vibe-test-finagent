import sqlite3
import os

db_path = os.path.join(os.path.dirname(__file__), "portfolio.db")

def init_db():
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Create table
    c.execute('''
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            name TEXT,
            quantity REAL,
            buy_price REAL,
            current_price REAL,
            profit_krw INTEGER
        )
    ''')
    
    # Clear existing data if any
    c.execute('DELETE FROM portfolio')
    
    # Data from image
    data = [
        ("SOXL", "디렉시온 데일리 세미컨덕터 불 3X ETF", 297.463371, 31.91, 16.48, -6547192),
        ("TSLA", "테슬라", 10.999607, 342.33, 285.53, -719578),
        ("NVDA", "엔비디아", 19.864072, 84.97, 115.00, 700569),
        ("COIN", "코인베이스 글로벌", 7.399676, 267.89, 182.88, -882604),
        ("TQQQ", "프로셰어즈 울트라프로 QQQ ETF", 19.999818, 80.89, 57.40, -649067),
        ("MSTR", "마이크로스트레티지", 3.199929, 385.49, 246.03, -622465),
        ("GOOGL", "알파벳 A", 3.499694, 191.60, 167.06, -118529),
        ("MSFT", "마이크로소프트", 1.349942, 361.64, 411.05, 91010),
        ("QQQ", "인베스코 QQQ 트러스트 ETF", 0.499887, 501.21, 498.42, -3392),
        ("XOM", "엑슨 모빌", 1.499927, 122.84, 106.32, -36812),
        ("AAPL", "애플", 0.499940, 206.37, 237.33, 26149),
        ("AMZN", "아마존닷컴", 0.499933, 236.21, 192.14, -33267),
        ("DELL", "델 테크놀로지스 C", 0.499961, 133.72, 123.42, -7543),
        ("META", "메타 플랫폼스", 0.099987, 593.41, 519.89, -10131),
        ("LLY", "일라이 릴리", 0.049993, 874.52, 730.06, -12238),
        ("V", "비자", 0.149996, 323.51, 279.79, -8619),
        ("JPM", "JPMorgan Chase & Co", 0.149994, 251.27, 224.26, -5934),
        ("MA", "마스터카드", 0.079991, 578.89, 495.91, -8220),
        ("ABBV", "애브비", 0.149994, 167.92, 179.99, 3083),
    ]
    
    c.executemany('''
        INSERT INTO portfolio (ticker, name, quantity, buy_price, current_price, profit_krw)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', data)
    
    conn.commit()
    conn.close()
    print("DB successfully initialized and populated.")

if __name__ == "__main__":
    init_db()
