"""Upload project to Kaggle dataset via kagglehub."""
import os, shutil, tempfile, inspect
from pathlib import Path
from kagglehub.datasets import dataset_upload

# Print signature
sig = inspect.signature(dataset_upload)
print(f"dataset_upload params: {list(sig.parameters.keys())}")

# Package files
tmp = Path(tempfile.mkdtemp())
root = Path('.')

(tmp / 'game_engine' / 'src').mkdir(parents=True)
for f in (root / 'game_engine' / 'src').glob('*.rs'):
    shutil.copy2(f, tmp / 'game_engine' / 'src')
shutil.copy2(root / 'game_engine' / 'Cargo.toml', tmp / 'game_engine')
shutil.copy2(root / 'kaggle' / 'kaggle_train_xl.py', tmp)

os.environ['KAGGLE_USERNAME'] = 'milkyblack'
os.environ['KAGGLE_KEY'] = 'KGAT_ee1b0f76b33a92c2f2ec4502765dccff'

try:
    result = dataset_upload(filepath=str(tmp), dataset_name='penguinchess-training')
    print(f"Success: {result}")
except TypeError as e:
    print(f"TypeError: {e}")
    # Try different arg combinations
    try:
        from kagglehub.datasets import create_dataset_or_version
        sig2 = inspect.signature(create_dataset_or_version)
        print(f"create_dataset_or_version params: {list(sig2.parameters.keys())}")
    except Exception as e2:
        print(f"Also failed: {e2}")

shutil.rmtree(tmp)
