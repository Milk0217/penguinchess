"""
Kaggle ↔ Local data sync for PenguinChess training.
Usage:
    python kaggle/sync.py upload    — Upload code + data to Kaggle dataset
    python kaggle/sync.py download  — Download trained models from Kaggle
    python kaggle/sync.py status    — Check Kaggle dataset status
"""
import sys, os, json, pickle, time, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

DATASET_NAME = "penguinchess-training"
PROJECT_ROOT = Path(__file__).parent.parent

def _ensure_kaggle_auth():
    """Check kagglehub is configured."""
    try:
        import kagglehub
        kagglehub.login()  # Will prompt if not configured
        return kagglehub
    except Exception as e:
        print(f"⚠ Kaggle auth failed: {e}")
        print("  Configure via: kagglehub.login() or set KAGGLE_USERNAME/KAGGLE_KEY")
        sys.exit(1)


def upload():
    """Upload project code + data to Kaggle dataset."""
    kh = _ensure_kaggle_auth()
    
    # Create temp directory with only what's needed
    tmp = PROJECT_ROOT / '.kaggle_upload'
    if tmp.exists():
        shutil.rmtree(tmp)
    
    # Copy source code (Python + Rust)
    for src_dir in ['penguinchess', 'alphazero', 'game_engine/src', 'game_engine/Cargo.toml']:
        dst = tmp / src_dir
        dst.parent.mkdir(parents=True, exist_ok=True)
        if Path(src_dir).is_dir():
            shutil.copytree(src_dir, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src_dir, dst)
    
    # Copy training scripts
    for f in ['kaggle/kaggle_train_xl.py', 'pyproject.toml', 'AGENTS.md']:
        dst = tmp / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
    
    # Copy best model and data
    model_dir = tmp / 'models' / 'alphazero'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    for model_file in ['alphazero_resnet_xl_best.pth', 'alphazero_resnet_xl_checkpoint.pth',
                       'alphazero_resnet_xl_final.pth']:
        src = PROJECT_ROOT / 'models' / 'alphazero' / model_file
        if src.exists():
            shutil.copy2(src, model_dir)
    
    # Create dataset version
    print(f"Uploading {tmp} to Kaggle dataset '{DATASET_NAME}'...")
    
    try:
        dataset_path = kh.dataset_upload(
            filepath=str(tmp),
            dataset_name=DATASET_NAME,
            license='mit',
            is_private=True,
        )
        print(f"✅ Uploaded: {dataset_path}")
    except Exception as e:
        print(f"Upload failed: {e}")
        print("Creating dataset instead...")
        path = kh.dataset_create(
            folder=str(tmp),
            dataset_name=DATASET_NAME,
            license='mit',
            is_private=True,
        )
        print(f"✅ Created: {path}")
    
    # Cleanup
    shutil.rmtree(tmp)
    print("Temp files cleaned")


def download():
    """Download trained models from Kaggle."""
    kh = _ensure_kaggle_auth()
    
    print(f"Downloading dataset '{DATASET_NAME}'...")
    model_dir = PROJECT_ROOT / 'models' / 'alphazero'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        path = kh.dataset_download(dataset_name=DATASET_NAME)
        src = Path(path) / 'models' / 'alphazero'
        if src.exists():
            for f in src.iterdir():
                if f.suffix == '.pth':
                    shutil.copy2(f, model_dir)
                    print(f"  Downloaded: {f.name} ({f.stat().st_size / 1024**2:.1f}MB)")
        else:
            print(f"  No model files found in dataset")
    except Exception as e:
        print(f"Download failed: {e}")


def status():
    """Check what's available on Kaggle and locally."""
    import kagglehub
    try:
        path = kagglehub.dataset_download(dataset_name=DATASET_NAME)
        print(f"Kaggle dataset: {path}")
        if path:
            for f in Path(path).rglob('*.pth'):
                print(f"  {f.relative_to(path)} ({f.stat().st_size / 1024**2:.1f}MB)")
    except:
        print("Kaggle dataset: not found (run 'upload' first)")
    
    print()
    model_dir = PROJECT_ROOT / 'models' / 'alphazero'
    if model_dir.exists():
        print(f"Local models ({model_dir}):")
        for f in sorted(model_dir.glob('*.pth'), key=lambda x: x.stat().st_size, reverse=True)[:10]:
            print(f"  {f.name} ({f.stat().st_size / 1024**2:.1f}MB)")
    
    print()
    print(f"Kaggle GPU quota: ~30h/week (T4 16GB)")
    print(f"Local GPU: RTX 4060 (8GB)")


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'status'
    if cmd == 'upload':
        upload()
    elif cmd == 'download':
        download()
    elif cmd == 'status':
        status()
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python kaggle/sync.py [upload|download|status]")
