"""
Kaggle ↔ Local 数据同步脚本 for PenguinChess XL 训练。

工作流:
  1. 在本地修改代码并 jj git push
  2. 在 Kaggle Notebook 里 !git pull && python kaggle/kaggle_train_xl.py
  3. 训练完成后下载 checkpoint

Usage:
    python kaggle/sync.py status    — 查看本地 XL 模型状态
    python kaggle/sync.py download  — 从 Kaggle 输出下载模型（需要 kaggle API）
"""
import sys, os, subprocess, json
from pathlib import Path

DATASET_SLUG = "milkyblack/penguinchess-training"
KAGGLE_NOTEBOOK_SLUG = "milkyblack/penguinchess312m"

def status():
    """显示本地 XL 模型文件和 Kaggle Notebook 状态。"""
    xl_dir = Path('models/alphazero')
    print("Local XL checkpoints:")
    if xl_dir.exists():
        for f in sorted(xl_dir.glob('*.pth'), key=lambda x: x.stat().st_size):
            age_h = (time.time() - f.stat().st_mtime) / 3600
            print(f"  {f.name} ({f.stat().st_size/1048576:.1f}MB, {age_h:.0f}h)")
    else:
        print("  (none)")
    print(f"\nKaggle Notebook: https://kaggle.com/{KAGGLE_NOTEBOOK_SLUG}/edit")
    print(f"Workflow: jj git push → Notebook: !git pull && python kaggle/kaggle_train_xl.py")
    print(f"Checkpoints saved to /kaggle/output/penguinchess_checkpoints/ (跨 session 持久化)")

def download():
    """从 Kaggle 下载训练好的模型文件。需要 kaggle CLI + API token。"""
    tmp = Path(os.environ.get('TMPDIR', '/tmp'))
    print(f"Downloading notebook output...")
    result = subprocess.run(
        ['kaggle', 'kernels', 'output', KAGGLE_NOTEBOOK_SLUG,
         '-p', str(tmp)],
        capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"Download failed: {result.stderr[:200]}")
        print("Make sure kaggle CLI is configured: kaggle auth login")
        return
    model_dir = Path('models/alphazero')
    model_dir.mkdir(parents=True, exist_ok=True)
    for f in tmp.rglob('*.pth'):
        shutil.copy2(f, model_dir / f.name)
        print(f"  Downloaded: {f.name} ({f.stat().st_size/1048576:.1f}MB)")
    print("Done")

if __name__ == '__main__':
    import time, shutil
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'status'
    {'status': status, 'download': download}.get(cmd, lambda: print(f"Usage: python kaggle/sync.py [status|download]"))()
