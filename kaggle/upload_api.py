"""Upload project to Kaggle dataset using direct API with Bearer token."""
import os, sys, json, shutil, tempfile, time, zipfile, requests
from pathlib import Path

KAGGLE_KEY = 'KGAT_ee1b0f76b33a92c2f2ec4502765dccff'
DATASET_SLUG = 'milkyblack/penguinchess-training'
HEADERS = {'Authorization': f'Bearer {KAGGLE_KEY}'}

def _api_get(path):
    return requests.get(f'https://www.kaggle.com/api/v1/{path}', headers=HEADERS)

def _api_post(path, data=None, json_data=None):
    return requests.post(f'https://www.kaggle.com/api/v1/{path}', headers=HEADERS, data=data, json=json_data)

def upload():
    """Package and upload to Kaggle dataset."""
    tmp = Path(tempfile.mkdtemp())
    root = Path('.')

    # Package files
    (tmp / 'game_engine' / 'src').mkdir(parents=True)
    for f in (root / 'game_engine' / 'src').glob('*.rs'):
        shutil.copy2(f, tmp / 'game_engine' / 'src')
    shutil.copy2(root / 'game_engine' / 'Cargo.toml', tmp / 'game_engine')
    shutil.copy2(root / 'kaggle' / 'kaggle_train_xl.py', tmp)
    shutil.copy2(root / 'pyproject.toml', tmp)
    shutil.copy2(root / 'kaggle' / 'sync.py', tmp / 'sync.py')

    # Create zip
    zip_path = tmp / 'upload.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in tmp.rglob('*'):
            if f != zip_path and f.is_file():
                zf.write(f, f.relative_to(tmp))

    zip_size = zip_path.stat().st_size
    print(f'Package: {zip_size/1024:.0f}KB', flush=True)

    # Check if dataset exists
    r = _api_get(f'datasets/{DATASET_SLUG}')
    exists = r.status_code == 200
    print(f'Dataset exists: {exists}', flush=True)

    if not exists:
        # Create dataset
        r = _api_post('datasets/create/new', json_data={
            'title': 'PenguinChess Training',
            'id': DATASET_SLUG,
            'isPrivate': True,
            'licenseName': 'MIT',
        })
        print(f'Create: {r.status_code}', flush=True)
        if r.status_code not in (200, 201):
            print(f'  {r.text[:200]}', flush=True)

    # Initiate blob upload
    r = _api_post('blobs/upload/init', json_data={'contentLength': zip_size})
    print(f'Blob init: {r.status_code}', flush=True)
    if r.status_code != 200:
        print(f'  {r.text[:300]}', flush=True)
        # Try alternative endpoint
        r = _api_post('blobs/start-upload', json_data={'contentLength': zip_size})
        print(f'Blob init alt: {r.status_code}', flush=True)
        if r.status_code != 200:
            print(f'  {r.text[:300]}', flush=True)

    # Upload file
    with open(zip_path, 'rb') as f:
        r = requests.put(r.json()['url'], data=f) if r.status_code == 200 else \
            _api_post('blobs/upload', data=open(zip_path, 'rb').read())
    
    if r.status_code in (200, 201):
        print(f'Upload success!', flush=True)
    else:
        print(f'Upload failed: {r.status_code} {r.text[:200]}', flush=True)

    shutil.rmtree(tmp)
    print('Done', flush=True)

if __name__ == '__main__':
    upload()
