"""
NNUE 训练脚本 — 符合训练规范的参考实现。

特性:
  - GPU 饱和: AMP + 大 batch + pin_memory + TF32 + num_workers
  - Checkpoint: 每 N 轮保存 + 自动续训
  - 自动导入前代最优模型
  - 配置化参数 + 可复现种子
  - 早停 + 日志 + 注册

Usage:
    # 从随机权重开始
    uv run python examples/train_nnue_random.py --data backend_data/nnue_random_50k.bin
    
    # 续训
    uv run python examples/train_nnue_random.py --resume models/nnue/nnue_gen_3.pt
    
    # 自动迭代（gen 0 → gen 1 → gen 2 ...）
    uv run python examples/train_nnue_random.py --iterations 5
"""

import os, sys, time, struct, json, pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler


# ─── 配置 ─────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # 数据
    data_path: str = "backend_data/nnue_random_50k.bin"
    max_samples: int = 55000        # 总样本数（含验证集）
    train_split: float = 0.9        # 训练集比例

    # 训练
    batch_size: int = 8192          # 尽量大以饱和 GPU
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0          # 梯度裁剪
    use_amp: bool = True            # 自动混合精度
    seed: int = 42

    # 硬件
    num_workers: int = 0            # Windows 上用 0, Linux 用 4+ (实际测试: Windows dataloader 多进程有 bug)
    pin_memory: bool = True

    # Checkpoint
    save_dir: str = "models/nnue"
    save_every: int = 5             # 每 N 轮保存 checkpoint
    resume: Optional[str] = None    # 续训用 checkpoint 路径

    # 迭代自动化
    iterations: int = 1             # 迭代次数
    auto_import_best: bool = True   # 自动导入前代最优

    # 早停
    early_stop_patience: int = 10   # 验证损失不改善后停止的轮数
    early_stop_delta: float = 1e-4

    # 日志
    log_interval: int = 10          # 每 N batch 打印一次


# ─── 数据加载 ─────────────────────────────────────────────────

def load_binary_data(path: str, max_samples: int = None):
    """Load Rust-generated binary NNUE training data."""
    raw = np.fromfile(path, dtype=np.uint8)
    count = int.from_bytes(raw[:8].tobytes(), 'little')
    if max_samples:
        count = min(count, max_samples)
    records = raw[8:8 + count * 292].reshape(-1, 292)

    sparse_raw = records[:, 0:24].view(np.int32).reshape(-1, 6)
    dense_raw = records[:, 24:24 + 66 * 4].view(np.float32).reshape(-1, 66)
    outcome_raw = records[:, 288:292].view(np.float32).flatten()

    sparse_list = []
    for row in sparse_raw:
        sparse_list.append([int(x) for x in row if x >= 0])

    return sparse_list, dense_raw, outcome_raw, count


class NNUERandomDataset(Dataset):
    """Memory-mapped dataset for speed. No Python list operations in __getitem__."""

    def __init__(self, sparse, dense, values):
        # Pre-convert to tensors so __getitem__ is zero-copy
        self.sparse_np = np.full((len(sparse), 6), -1, dtype=np.int32)
        for i, s in enumerate(sparse):
            for j, v in enumerate(s):
                if j >= 6: break
                self.sparse_np[i, j] = np.int32(v)
        self.dense = torch.from_numpy(np.ascontiguousarray(dense))
        self.values = torch.from_numpy(np.ascontiguousarray(values))

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        # Return tuple of (sparse_list, dense, value, stm=0)
        row = self.sparse_np[idx]
        # Filter -1: create list comprehension (fast for 6 elements)
        sparse = [int(row[0]), int(row[1]), int(row[2]),
                  int(row[3]), int(row[4]), int(row[5])]
        sparse = [x for x in sparse if x >= 0]
        return sparse, self.dense[idx], self.values[idx], 0


def collate_nnue(batch):
    sb = [b[0] for b in batch]
    db = torch.stack([b[1] for b in batch])
    vb = torch.stack([b[2] for b in batch])
    return sb, db, vb, [0] * len(batch)


# ─── Checkpoint ───────────────────────────────────────────────

def save_checkpoint(state: dict, path: str):
    """Save checkpoint atomically: write temp file, then rename."""
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str, model: nn.Module, optimizer: optim.Optimizer,
                    scheduler=None, scaler=None):
    """Load checkpoint and return epoch, best_val_loss."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"])
    if optimizer and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    if scaler and "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    return state.get("epoch", 0), state.get("best_val_loss", float("inf"))


# ─── 设备设置 ─────────────────────────────────────────────────

def setup_device(cfg: TrainConfig) -> torch.device:
    """Configure GPU with optimal settings."""
    if not torch.cuda.is_available():
        return torch.device("cpu")

    # Enable TF32 on Ampere+ GPUs (RTX 30xx/40xx series)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Fastest convolution algorithm selection
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
    print(f"  GPU: {gpu_name}  {mem:.0f}GB VRAM")
    print(f"  TF32: enabled  AMP: {cfg.use_amp}")

    return device


# ─── 迭代自动化 ────────────────────────────────────────────────

def find_previous_best(save_dir: str) -> Optional[str]:
    """Find the best model from the previous generation."""
    save_path = Path(save_dir)
    if not save_path.exists():
        return None
    # Find nnue_gen_N.pt with largest N
    best = None
    best_n = -1
    for p in save_path.glob("nnue_gen_*.pt"):
        try:
            n = int(p.stem.split("_")[-1])
            if n > best_n:
                best = str(p)
                best_n = n
        except (ValueError, IndexError):
            pass
    return best


def get_next_generation(save_dir: str) -> int:
    """Determine next generation number."""
    save_path = Path(save_dir)
    if not save_path.exists():
        return 0
    max_n = -1
    for p in save_path.glob("nnue_gen_*.pt"):
        try:
            n = int(p.stem.split("_")[-1])
            max_n = max(max_n, n)
        except (ValueError, IndexError):
            pass
    return max_n + 1


# ─── 训练循环 ─────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, cfg, scaler=None):
    model.train()
    total_loss = 0.0
    total_samples = 0
    t0 = time.time()

    for batch_idx, (sb, db, vb, stm_b) in enumerate(loader):
        db, vb = db.to(device, non_blocking=cfg.pin_memory), vb.to(device, non_blocking=cfg.pin_memory)

        if cfg.use_amp and scaler:
            with autocast():
                pred = model(sb, db, stm_players=stm_b)
                loss = criterion(pred, vb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(sb, db, stm_players=stm_b)
            loss = criterion(pred, vb)
            optimizer.zero_grad()
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        optimizer.zero_grad()
        batch_size = len(vb)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(1, total_samples), time.time() - t0


@torch.no_grad()
def eval_model(model, loader, criterion, device, cfg):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    for sb, db, vb, stm_b in loader:
        db, vb = db.to(device, non_blocking=cfg.pin_memory), vb.to(device, non_blocking=cfg.pin_memory)
        if cfg.use_amp:
            with autocast():
                pred = model(sb, db, stm_players=stm_b)
                loss = criterion(pred, vb)
        else:
            pred = model(sb, db, stm_players=stm_b)
            loss = criterion(pred, vb)
        total_loss += loss.item() * len(vb)
        total_samples += len(vb)
    return total_loss / max(1, total_samples)


# ─── 主入口 ───────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="NNUE training with full standards")
    parser.add_argument("--data", default="backend_data/nnue_random_50k.bin")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--iterations", type=int, default=1, help="Number of auto-iterations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP")

    args = parser.parse_args()
    cfg = TrainConfig(
        data_path=args.data,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        resume=args.resume,
        iterations=args.iterations,
        seed=args.seed,
        use_amp=not args.no_amp,
    )

    # ── 固定种子 ──
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ── 设备 ──
    device = setup_device(cfg)
    print(f"  Batch: {cfg.batch_size}  Epochs: {cfg.epochs}  AMP: {cfg.use_amp}")
    print()

    # ── 数据 ──
    data_path = Path(cfg.data_path)
    if not data_path.exists():
        print(f"[ERROR] Data not found: {data_path}")
        sys.exit(1)

    print(f"[1/4] Loading data...")
    t0 = time.time()
    sparse, dense, values, total = load_binary_data(str(data_path), cfg.max_samples)
    print(f"      {total} positions loaded in {time.time()-t0:.0f}s")

    n_train = int(len(sparse) * cfg.train_split)
    train_ds = NNUERandomDataset(sparse[:n_train], dense[:n_train], values[:n_train])
    val_ds = NNUERandomDataset(sparse[n_train:], dense[n_train:], values[n_train:])

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate_nnue, num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size * 2, shuffle=False,
        collate_fn=collate_nnue, num_workers=0,
        pin_memory=cfg.pin_memory,
    )

    print(f"      Train: {len(train_ds)}  Val: {len(val_ds)}  ({len(train_loader)} batches/epoch)")

    # ── 迭代循环 ──
    out_dir = Path(cfg.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_gen = get_next_generation(cfg.save_dir)

    for iteration in range(cfg.iterations):
        gen = start_gen + iteration
        print(f"\n{'='*50}")
        print(f"[2/4] Generation {gen}")
        print(f"{'='*50}")

        # ── 模型 ──
        model = NNUE().to(device)
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, threshold=cfg.early_stop_delta)
        scaler = GradScaler(enabled=cfg.use_amp) if cfg.use_amp else None
        criterion = nn.MSELoss()

        start_epoch = 0
        best_val_loss = float("inf")
        best_state = model.state_dict().copy()

        # ── 续训 ──
        if cfg.resume:
            cp_path = cfg.resume
        else:
            cp_path = str(out_dir / f"checkpoint_gen_{gen}.pt")

        if os.path.exists(cp_path):
            print(f"  Resuming from checkpoint: {cp_path}")
            start_epoch, best_val_loss = load_checkpoint(
                cp_path, model, optimizer, scheduler, scaler)
            print(f"  Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

        # ── 前代最优导入 ──
        if cfg.auto_import_best and start_epoch == 0:
            prev_best = find_previous_best(str(out_dir))
            if prev_best and Path(prev_best).exists():
                try:
                    prev_state = torch.load(prev_best, map_location="cpu", weights_only=False)
                    model.load_state_dict(prev_state["model_state"])
                    lr = optimizer.param_groups[0]["lr"]
                    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
                    print(f"  Imported previous best: {prev_best}")
                except Exception as e:
                    print(f"  Import failed (will train from scratch): {e}")

        # ── 训练 ──
        print(f"  Training {cfg.epochs} epochs...")
        t0 = time.time()
        no_improve = 0

        for epoch in range(start_epoch, cfg.epochs):
            train_loss, epoch_time = train_epoch(
                model, train_loader, optimizer, criterion, device, cfg, scaler)
            val_loss = eval_model(model, val_loader, criterion, device, cfg)
            scheduler.step(val_loss)

            if val_loss < best_val_loss - cfg.early_stop_delta:
                best_val_loss = val_loss
                best_state = model.state_dict().copy()
                no_improve = 0
            else:
                no_improve += 1

            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  [{epoch+1:>2d}/{cfg.epochs}]  train={train_loss:.4f}  "
                  f"val={val_loss:.4f}  LR={lr_now:.1e}  {epoch_time:.0f}s")

            # Save checkpoint
            if (epoch + 1) % cfg.save_every == 0:
                save_checkpoint({
                    "epoch": epoch + 1,
                    "model_state": best_state,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict() if scaler else None,
                    "best_val_loss": best_val_loss,
                    "config": json.dumps(asdict(cfg)),
                }, cp_path)

            # Early stop
            if no_improve >= cfg.early_stop_patience:
                print(f"  Early stopping at epoch {epoch+1} (no improvement for {no_improve} epochs)")
                break

            if lr_now < 1e-6:
                print(f"  LR too small, stopping")
                break

        # ── 保存最优 ──
        model.load_state_dict(best_state)
        out_path = str(out_dir / f"nnue_gen_{gen}.pt")
        torch.save({"model_state": best_state, "val_loss": best_val_loss}, out_path)
        print(f"\n  Saved: {out_path}  (val_loss={best_val_loss:.4f})")

        # ── 注册 ──
        try:
            from penguinchess.model_registry import register_model
            register_model(f"nnue_gen_{gen}", "nnue", f"nnue/nnue_gen_{gen}.pt",
                           generation=gen, arch="nnue")
        except Exception as e:
            print(f"  Register: {e}")

    print(f"\nDone! Total: {time.time()-t0:.0f}s")


from dataclasses import dataclass as _dc
def asdict(obj):
    return {f.name: getattr(obj, f.name) for f in obj.__dataclass_fields__.values()}


if __name__ == "__main__":
    main()
