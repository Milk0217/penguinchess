"""
Kaggle to Local sync.
Push to GitHub, Kaggle Notebook clones from GitHub, train XL, download .pth.

Usage:
    python kaggle/sync.py status
"""
import sys
from pathlib import Path

def status():
    xl_dir = Path('models') / 'alphazero'
    print("Local XL checkpoints:")
    if xl_dir.exists():
        for f in sorted(xl_dir.glob('*.pth'), key=lambda x: x.stat().st_size):
            sz = f.stat().st_size / 1048576
            print("  {} ({:.1f}MB)".format(f.name, sz))
    print()
    print("Workflow:")
    print("  1. jj git push (local)")
    print("  2. Create Kaggle Notebook from kaggle_train_xl.py")
    print("  3. Notebook clones from GitHub, installs Rust, trains XL")
    print("  4. Download .pth from Kaggle output")

if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'status'
    funcs = {'status': status}
    funcs.get(cmd, lambda: print("Unknown command"))()
