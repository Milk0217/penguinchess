"""Push Kaggle Notebook via kagglehub gRPC API."""
import json, tempfile, shutil, zipfile, requests
from pathlib import Path
from kagglehub.clients import KaggleClient
from kagglesdk.blobs.types.blob_api_service import ApiStartBlobUploadRequest

client = KaggleClient()
print('Auth OK', flush=True)

tmp = Path(tempfile.mkdtemp())
meta = {
    'id': 'milkyblack/penguinchess-xl-training',
    'title': 'PenguinChess XL Training',
    'code_file': 'kaggle_train_xl.py',
    'language': 'python',
    'kernel_type': 'notebook',
    'is_private': True,
    'enable_gpu': True,
    'enable_internet': True,
}
(tmp / 'kernel-metadata.json').write_text(json.dumps(meta))
src = Path('kaggle') / 'kaggle_train_xl.py'
if src.exists():
    shutil.copy2(src, tmp / 'kaggle_train_xl.py')

zip_path = tmp / 'kernel.zip'
with zipfile.ZipFile(zip_path, 'w') as zf:
    zf.write(tmp / 'kernel-metadata.json', 'kernel-metadata.json')
    zf.write(tmp / 'kaggle_train_xl.py', 'kaggle_train_xl.py')
print(f'Package: {zip_path.stat().st_size/1024:.0f}KB', flush=True)

# Blob upload
req = ApiStartBlobUploadRequest()
req.content_length = zip_path.stat().st_size
blob = client.blobs.blob_api_client.start_blob_upload(req)
token = blob.blob_token if hasattr(blob, 'blob_token') else None
url = blob.upload_url if hasattr(blob, 'upload_url') else None
print(f'Blob token: {token[:20] if token else "N/A"}', flush=True)

if url:
    with open(zip_path, 'rb') as f:
        r = requests.put(url, data=f)
    print(f'Upload: {r.status_code}', flush=True)
    if r.status_code in (200, 201) and token:
        result = client.kernels.kernels_api_client.save_kernel(blob_token=token)
        print(f'Kernel saved: {result}', flush=True)
else:
    print(f'No upload URL in response', flush=True)

shutil.rmtree(tmp)
print('Done', flush=True)
