# Kaggle 训练指南

## 概述

本项目使用 Kaggle 的免费 T4 GPU（16GB）进行 AlphaZeroResNetXL（3.13 亿参数）的分布式训练。训练代码以单文件形式存在于 `kaggle/kaggle_train_xl.py`，可直接粘贴到 Kaggle Notebook 中运行。

## 架构

```
┌── 本地 (Windows, RTX 4060 8GB) ─────────────────────┐
│                                                       │
│  alphazero/train.py --model 2m                        │
│    └─ Rust MCTS + AVX2 推理 + PyTorch 训练            │
│    └─ 训练 2M 参数模型 (~10s/iter)                    │
│                                                       │
│  jj git push (代码 + 模型同步到 GitHub)               │
└───────────────────────────────────────────────────────┘
                         │
                         ▼
┌── Kaggle (Tesla T4 16GB) ───────────────────────────┐
│                                                       │
│  kaggle/kaggle_train_xl.py                            │
│    └─ 自动安装 Rust + cargo build (2min)              │
│    └─ Rust MCTS + PyTorch GPU 推理                    │
│    └─ AMP 混合精度训练                                 │
│    └─ 每 5 代 checkpoint 到 /kaggle/output/            │
│                                                       │
│  一次运行 50 迭代 ≈ 12min                             │
│  9h GPU 限制内 ≈ 2000+ 迭代                           │
└───────────────────────────────────────────────────────┘
```

## 快速开始

### 在 Kaggle 上创建 Notebook

1. 打开 https://www.kaggle.com/notebooks → `New Notebook`
2. 删掉默认 cell，在第一个 cell 输入：

```python
import os, shutil
os.chdir('/kaggle/working')
if os.path.exists('/kaggle/working/penguinchess'):
    shutil.rmtree('/kaggle/working/penguinchess')
os.system('git clone https://github.com/Milk0217/penguinchess.git /kaggle/working/penguinchess')
%cd /kaggle/working/penguinchess
!python kaggle/kaggle_train_xl.py
```

3. 右侧 Settings → Accelerator → `GPU T4 x2`
4. 点击 `Run All`

### 恢复中断的训练

如果 Notebook 超时或被中断，重新运行**同样的 cell** 即可自动从 checkpoint 恢复。

### 检查当前训练状态

打开 Notebook URL（`/edit` 或 `/` 都可以看到实时输出）。

## 文件说明

| 文件 | 用途 |
|------|------|
| `kaggle/kaggle_train_xl.py` | Kaggle 训练主脚本（单文件，可直接粘贴） |
| `kaggle/sync.py` | 本地 ↔ Kaggle 同步工具 |
| `/kaggle/output/penguinchess_checkpoints/` | 跨 session 持久化 checkpoint（Kaggle 侧） |
| `models/alphazero/alphazero_resnet_xl_best.pth` | 最佳模型（存入 GitHub） |

## kaggle_train_xl.py 详解

### 训练流程

```
start ── 安装 Rust (首次) ── cargo build --release (2min) ── 循环:
  ├── 自对弈: RustCore × 8 workers × 200 游戏
  │    各游戏内: Rust MCTS (handle-based) + GPU batch 推理
  │    输出: 272-dim 观测 + t=1.0 soft 策略 + 结果价值
  ├── 训练: AMP (autocast + GradScaler), AdamW, CosineAnnealing
  │    batch_size=2048, epochs=10
  ├── Checkpoint: 每 5 迭代存到 /kaggle/output/
  └── 评估: 30 局 vs Random (裸 forward, 非 MCTS)
```

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `iterations` | 50 | 总迭代数 |
| `games_per_iter` | 200 | 每迭代自对弈局数 |
| `simulations` | 200 | MCTS 模拟次数 |
| `c_puct` | 3.0 | MCTS 探索常数 |
| `batch_size` | 2048 | 训练 batch size |
| `epochs` | 10 | 训练 epoch 数 |
| `lr` | 1e-3 | 学习率 |
| `max_buffer` | 100000 | 经验池上限 |
| `mcts_batch_size` | 128 | MCTS 批量评估大小 |
| `workers` | 8 | 游戏并行线程数 |

### 训练正确性设计

| 特性 | 实现 | 位置 |
|------|------|------|
| Dirichlet 噪声 | Rust MCTS 内建 (通过 `mcts_rs.rs`) | Rust 侧 |
| 训练目标 t=1.0 | `policy[a] = c / total` (soft counts) | `self_play()` |
| 行为温度 | `temp=1.0(前30步)→0.1(后)` | `self_play()` |
| 随机开局 | 前 10 步 uniform random (不存入 MCTS 数据) | `self_play()` |
| AMP | `torch.amp.autocast("cuda")` + `GradScaler` | `train()` |
| 梯度裁剪 | `clip_grad_norm_(1.0)` | `train()` |
| Checkpoint 原子保存 | `write to .tmp → rename` | `atomic_save()` |

## Rust MCTS + GPU 推理机制

Kaggle 脚本使用 Rust MCTS（单线程 handle 版）+ PyTorch GPU 推理：

1. **树遍历**：Rust 侧做纯 CPU 工作（SELECT/EXPAND/BACKUP），比 Python MCTS **快 ~100x**
2. **NN 推理**：Rust 通过 FFI 回调 Python → PyTorch GPU (`evaluate_flat_batch`)
3. **批量评估**：每 `mcts_batch_size=128` 个叶子节点触发一次 GPU forward
4. **游戏级并行**：8 个 `ThreadPoolExecutor` worker 同时运行不同游戏

```
每个游戏: Rust MCTS ─(FFI callback)─→ PyTorch GPU ─(结果)─→ Rust
               ↑                                    │
               └── 8 workers 并行 ──────────────────┘
```

这种设计比纯 Python MCTS（每步 ~7ms）快得多：Rust MCTS 的树遍历只需 <0.1ms/步，GPU 推理是唯一的瓶颈。

## 观测编码

观测空间为 272 维 float32：

| 维度范围 | 内容 | 大小 |
|----------|------|------|
| 0-179 | 棋盘格 (60 hexes × 3 features) | 180 |
| 180-203 | 棋子 (6 pieces × 4 features) | 24 |
| 204 | 当前玩家 | 1 |
| 205 | 游戏阶段 (placement=0, movement=1) | 1 |
| 206-271 | 零填充 (为 Rust encode_obs 预留) | 66 |

`evaluate_flat_batch` 在推理时会自动将 206→272 补零填充，与 `encode_obs` 生成的数据格式一致。

## 常见问题

### Q: Rust 编译失败
检查 `game_engine/Cargo.toml` 和 `src/` 是否完整。Kaggle 默认有 Rust 工具链，不需要额外安装。

### Q: GPU 显存不足 OOM
XL 模型训练需要约 5GB：
- fp32 权重: 1.25GB
- Adam 状态: 2.5GB  
- 梯度: 1.25GB
- 激活值: ~100MB (B=2048)

T4 16GB 完全够用。如果遇到 OOM，降低 `batch_size` 或 `mcts_batch_size`。

### Q: 训练速度太慢
检查以下项：
- AMP 是否启用（应该有 `autocast` + `GradScaler` 日志）
- Rust MCTS 是否正确（日志应显示 Rust engine loaded）
- workers 是否充分利用（8 workers）

预期每迭代 ~10-15s（自对弈 200 局 + 训练 10 epochs）。

### Q: Notebook 断开后训练是否继续
是的。Kaggle Notebook 关闭浏览器后 Kernel 继续运行直到完成或超时（9h GPU 限制）。
训练使用 `/kaggle/output/`（跨 session 持久化），重新打开 Notebook 会看到最新输出。

### Q: 如何续训中断的训练
用同样的 Notebook cell 重新运行即可。脚本自动检测 `/kaggle/output/penguinchess_checkpoints/checkpoint.pt`，加载模型权重、迭代号和 replay buffer 后继续训练。

## GitHub 同步流程

1. 本地修改代码后: `jj describe -m "msg"; jj bookmark set main; jj git push`
2. 在 Kaggle Notebook 执行: `!git pull`
3. 训练完成后将模型文件 `.pth` 从 Kaggle 下载到本地 `models/alphazero/`
4. 本地提交: `jj describe -m "xl: iter 50 model"; jj bookmark set main; jj git push`
