"""Try to fetch Kaggle notebook page."""
import requests

TOKEN = 'CfDJ8KXhJwB4GRRPqJWFH0ywOelFPOohdSWnqZSkq_gBXucV91q4MFjMGsOWgzBDUCLx7tU4itPOwhSVv2-kpmlsBg5vfh8Elveh1MWtDCF2gjCht776LaGLy66PinO_pbgWS3p9Fo5iPyxFyJY'
H = {'Authorization': f'Bearer {TOKEN}', 'User-Agent': 'Mozilla/5.0'}

r = requests.get('https://www.kaggle.com/code/milkyblack/penguinchess312m', headers=H, timeout=30)
print(f'HTML page: {r.status_code}')
if r.status_code == 200:
    html = r.text
    for keyword in ['Iter', 'GPU:', 'Rust engine', 'Training', 'Self-play', 'vs Random']:
        if keyword in html:
            idx = html.find(keyword)
            start = max(0, idx-80)
            end = min(len(html), idx+200)
            snippet = html[start:end]
            print(f'Found "{keyword}": ...{snippet}...')
            break
    else:
        print('No training keywords - JS-rendered page')
        print(f'Page length: {len(html)} bytes')
else:
    print(r.text[:300])
