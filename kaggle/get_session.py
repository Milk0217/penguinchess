"""Extract Edge Kaggle cookies and fetch notebook output."""
import os, sqlite3, shutil, tempfile, json, requests
from pathlib import Path

EDGE_COOKIES = Path(os.environ['LOCALAPPDATA']) / 'Microsoft' / 'Edge' / 'User Data' / 'Default' / 'Network' / 'Cookies'

def get_kaggle_cookies():
    """Extract Kaggle session cookies from Edge's encrypted database."""
    if not EDGE_COOKIES.exists():
        print(f'Edge cookies not found at: {EDGE_COOKIES}')
        return {}
    
    # Copy to temp to avoid DB lock
    tmp = Path(tempfile.mkdtemp()) / 'Cookies'
    shutil.copy2(EDGE_COOKIES, tmp)
    
    cookies = {}
    try:
        conn = sqlite3.connect(str(tmp))
        cursor = conn.cursor()
        # Browser cookies table (Chromium-based)
        try:
            cursor.execute("SELECT host_key, name, encrypted_value, path, is_secure, is_httponly FROM cookies WHERE host_key LIKE '%kaggle%'")
        except:
            cursor.execute("SELECT host_key, name, value, path, is_secure, is_httponly FROM cookies WHERE host_key LIKE '%kaggle%'")
        
        for row in cursor.fetchall():
            host_key, name = row[0], row[1]
            # Try decryption with win32crypt
            try:
                import win32crypt
                encrypted_value = row[2]
                value = win32crypt.CryptUnprotectData(encrypted_value, None, None, None, 0)[1].decode('utf-8')
            except:
                value = str(row[2]) if isinstance(row[2], str) else row[2].decode('utf-8', errors='replace')
            cookies[name] = value
            print(f'  {name}: {value[:30]}...')
        conn.close()
    except Exception as e:
        print(f'DB error: {e}')
    
    shutil.rmtree(tmp.parent)
    return cookies

cookies = get_kaggle_cookies()
print(f'\nFound {len(cookies)} Kaggle cookies')

if cookies:
    # Try fetching notebook page
    jar = requests.cookies.RequestsCookieJar()
    for name, value in cookies.items():
        jar.set(name, value, domain='.kaggle.com', path='/')
    
    r = requests.get('https://www.kaggle.com/code/milkyblack/penguinchess312m', cookies=jar, timeout=30,
                     headers={'User-Agent': 'Mozilla/5.0'})
    print(f'Notebook page: {r.status_code}')
    if r.status_code == 200:
        html = r.text
        for kw in ['Iter', 'GPU:', 'Rust engine', 'Self-play', 'vs Random', 'Training']:
            if kw in html:
                idx = html.find(kw)
                s = max(0, idx-50)
                e = min(len(html), idx+200)
                print(f'Found "{kw}": ...{html[s:e]}...')
                break
        else:
            print('Page is JS-rendered (no training text in raw HTML)')
            print(f'Page size: {len(html)} bytes')
