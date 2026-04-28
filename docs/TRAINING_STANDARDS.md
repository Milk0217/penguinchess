# PenguinChess 训练规范

## 核心原则

**Python 只管训练，Rust 负责其余。**

训练脚本是项目中唯一会运行 Python 推理代码的地方。所有训练产出的模型最终必须能导出到 Rust 进行推理。

---

## 1. 硬件利用 — 饱和 GPU/CPU

### GPU 设置

```python
# 启用 TF32（Ampere+ 架构，RTX 30xx/40xx）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# cuDNN 自动调优（首次运行会 benchmark，后续加速）
torch.backends.cudnn.benchmark = True
```

### 批次大小

| GPU 显存 | 推荐 batch_size |
|----------|----------------|
| 8GB (RTX 4060) | 4096-8192 |
| 12GB (RTX 4070) | 8192-16384 |
| 24GB (RTX 4090) | 32768+ |

batch_size 越小，GPU 利用率越低。目标：batch_size 能刚好塞满显存而不 OOM。

### 自动混合精度 (AMP)

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler(enabled=use_amp)
with autocast():
    pred = model(x)
    loss = criterion(pred, y)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

AMP 约 1.5-2x 加速，且几乎不影响 NNUE/AlphaZero 这种小模型的精度。

### DataLoader

**Windows 限制**：`num_workers > 0` 在 Windows 上可能导致死锁或极慢。原因：Python 多进程序列化开销大。建议：

| 平台 | num_workers |
|------|-------------|
| Windows | **0**（实测更快） |
| Linux | 4-8（CPU 核数的一半） |

```python
DataLoader(..., pin_memory=True, num_workers=0)  # Windows
DataLoader(..., pin_memory=True, num_workers=4)  # Linux
```

`pin_memory=True` 让 CPU 张量直接固定在页锁定内存，GPU 异步拷贝更快。

### 数据预处理

不要在 `__getitem__` 或 `collate` 里做重量级操作。应该在 `Dataset.__init__` 里把所有数据预处理好（转为 tensor / numpy array）。

```python
class GoodDataset(Dataset):
    def __init__(self, data):
        # 预处理一次，之后 __getitem__ 零开销
        self.tensor = torch.from_numpy(data)

    def __getitem__(self, idx):
        return self.tensor[idx]  # O(1)，无 Python 循环
```

---

## 2. Checkpoint — 训练中断恢复

### 保存频率

每 N 个 epoch 保存一次 checkpoint（推荐 N=5）。

### Checkpoint 内容

```python
checkpoint = {
    "epoch": epoch + 1,
    "model_state": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "scaler": scaler.state_dict(),       # AMP scaler（如果使用）
    "best_val_loss": best_val_loss,
    "config": config_dict,               # 训练配置，用于复现
}
```

### 原子保存

```python
# 先写临时文件，再 rename，防止写入中断导致文件损坏
tmp = path + ".tmp"
torch.save(state, tmp)
os.replace(tmp, path)
```

### 自动续训

```python
if os.path.exists(checkpoint_path):
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer"])
    start_epoch = state["epoch"]
    print(f"Resumed from epoch {start_epoch}")
```

---

## 3. 迭代自动化

### 前代最优导入

```python
def find_previous_best(save_dir: str) -> Optional[str]:
    """Find nnue_gen_N.pt with largest N."""
    files = sorted(Path(save_dir).glob("nnue_gen_*.pt"))
    return str(files[-1]) if files else None
```

在每轮训练开始前，自动加载前代最优模型权重。

### Generation 编号

```python
def get_next_generation(save_dir: str) -> int:
    """Auto-increment generation number."""
    exist = list(Path(save_dir).glob("nnue_gen_*.pt"))
    return max((int(p.stem.split("_")[-1]) for p in exist), default=-1) + 1
```

### 自动 Pipeline

```python
for gen in range(start_gen, start_gen + iterations):
    model = NNUE()
    if auto_import_best and gen > 0:
        prev = find_previous_best(save_dir)
        if prev: model.load_state_dict(torch.load(prev)["model_state"])
    # 训练...
    torch.save({"model_state": model.state_dict()}, f"nnue_gen_{gen}.pt")
    register_model(f"nnue_gen_{gen}", "nnue", ...)
```

---

## 4. 可复现性

```python
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 以下会降低性能，仅调试用：
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
```

---

## 5. 早停

```python
no_improve = 0
patience = 10
delta = 1e-4

for epoch in range(epochs):
    val_loss = evaluate(model, val_loader)
    if val_loss < best_val_loss - delta:
        best_val_loss = val_loss
        no_improve = 0
    else:
        no_improve += 1

    if no_improve >= patience:
        print(f"Early stop at epoch {epoch}")
        break
```

---

## 6. 模型注册与导出

训练完成后必须：

1. **保存模型**：`torch.save({"model_state": state_dict}, path)`
2. **注册**：`register_model(id, type, file_path, generation=N, arch="nnue")`
3. **导出到 Rust**：权重扁平化为 float32 数组，与 Rust 端 `NNUEWeights::total_floats()` 校验

```python
# 导出到 Rust 的权重总数必须匹配
from penguinchess.rust_ffi import ffi_ab_create
handle = ffi_ab_create(config_json)
ok = handle.set_weights(model.state_dict())  # 内部校验 total_floats
assert ok, "Weight size mismatch with Rust"
```

---

## 7. 损失函数与指标

| 模型 | 损失 | 指标 |
|------|------|------|
| NNUE (value) | MSELoss | val_loss, vs_random 胜率 |
| AlphaZero (policy+value) | CrossEntropy + MSE | policy_loss, value_loss, 总 loss |
| PPO (actor-critic) | SB3 内部 | 奖励曲线, ELO |

NNUE 的 val_loss 在 0.97 左右时 vs Random 可达 70%+。不需要追求 loss 降到 0.5 以下——AB 搜索会放大微弱信号。

---

## 8. CLI 参数规范

所有训练脚本应支持以下参数：

```
--data PATH        训练数据路径
--epochs N         训练轮数
--batch-size N     batch size
--lr FLOAT         学习率
--resume PATH      checkpoint 续训
--iterations N     自动迭代次数
--seed N           随机种子
--no-amp           禁用 AMP
--log-interval N   日志频率
```

---

## 9. 代码结构模板

```
examples/train_*.py
├── 配置 (dataclass / argparse)
├── 数据加载 (load_* + Dataset)
├── Checkpoint (save_checkpoint / load_checkpoint)
├── 设备设置 (setup_device)
├── 迭代自动化 (find_previous_best / get_next_generation)
├── 训练循环 (train_epoch / eval_model)
├── 主入口 (main)
│   ├── 固定种子
│   ├── 加载数据
│   ├── 迭代循环
│   │   ├── 创建模型
│   │   ├── 续训检查
│   │   ├── 前代导入
│   │   ├── 训练
│   │   ├── 保存最优
│   │   └── 注册
│   └── 完成
```
