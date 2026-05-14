"""Test Kaggle API access with Bearer token and CLI."""
import subprocess, requests, os

TOKEN = 'CfDJ8KXhJwB4GRRPqJWFH0ywOelFPOohdSWnqZSkq_gBXucV91q4MFjMGsOWgzBDUCLx7tU4itPOwhSVv2-kpmlsBg5vfh8Elveh1MWtDCF2gjCht776LaGLy66PinO_pbgWS3p9Fo5iPyxFyJY'
H = {'Authorization': f'Bearer {TOKEN}'}

print("=== Bearer Token API ===")
for label, url in [
    ('kernels list', 'https://www.kaggle.com/api/v1/kernels/list'),
    ('datasets list', 'https://www.kaggle.com/api/v1/datasets/list'),
]:
    r = requests.get(url, headers=H, timeout=15)
    count = len(r.json()) if r.status_code == 200 else 'N/A'
    print(f'  {label}: {r.status_code} ({count})')

env = {**os.environ, 'KAGGLE_USERNAME': 'milkyblack', 'KAGGLE_KEY': 'KGAT_ee1b0f76b33a92c2f2ec4502765dccff'}
print("\n=== Kaggle CLI ===")
for cmd in [
    ['kaggle', 'kernels', 'list', '--mine'],
    ['kaggle', 'datasets', 'list', '--search', 'penguinchess'],
]:
    label = ' '.join(cmd[1:])
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=15)
    out = (r.stdout.strip()[:200] if r.stdout else r.stderr.strip()[:200])
    print(f'  {label}: RC={r.returncode} | {out}')
