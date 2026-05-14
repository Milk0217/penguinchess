"""Push the training script as a Kaggle Notebook."""
import os, json, tempfile, shutil, subprocess
from pathlib import Path

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

tmp = Path(tempfile.mkdtemp())
(tmp / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
shutil.copy2(Path('kaggle') / 'kaggle_train_xl.py', tmp / 'kaggle_train_xl.py')

env = {**os.environ, 'KAGGLE_USERNAME': 'milkyblack',
       'KAGGLE_KEY': 'KGAT_ee1b0f76b33a92c2f2ec4502765dccff',
       'PYTHONIOENCODING': 'utf-8'}

result = subprocess.run(
    ['kaggle', 'kernels', 'push', '-p', str(tmp)],
    capture_output=True, text=True, timeout=120, env=env)

print(f'RC={result.returncode}')
stdout = result.stdout[:800]
stderr = result.stderr[:500]
if stdout:
    print(f'OUT: {stdout}')
if stderr:
    print(f'ERR: {stderr}')

shutil.rmtree(tmp)
