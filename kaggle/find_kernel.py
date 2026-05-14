"""Find penguinchess kernel in Kaggle listings."""
import requests

TOKEN = 'CfDJ8KXhJwB4GRRPqJWFH0ywOelFPOohdSWnqZSkq_gBXucV91q4MFjMGsOWgzBDUCLx7tU4itPOwhSVv2-kpmlsBg5vfh8Elveh1MWtDCF2gjCht776LaGLy66PinO_pbgWS3p9Fo5iPyxFyJY'
H = {'Authorization': f'Bearer {TOKEN}'}

r = requests.get('https://www.kaggle.com/api/v1/kernels/list', headers=H, timeout=15)
if r.status_code != 200:
    print(f'Error: {r.status_code}')
    exit()

count = 0
for k in r.json():
    ref = k.get('ref', '')
    title = k.get('title', '')
    if 'penguin' in ref.lower() or 'penguin' in title.lower() or '312' in title:
        count += 1
        print(f'Ref: {ref}')
        print(f'Title: {title}')
        print(f'GPU: {k.get("enableGpu")}')
        print(f'Last run: {k.get("lastRunTime", "never")}')
        print()

if count == 0:
    print("No penguinchess kernel found")
    print("All kernel refs:")
    for k in r.json()[:5]:
        print(f'  {k.get("ref", "?")}')
