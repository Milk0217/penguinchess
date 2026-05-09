# AlphaZero → AB+NNUE 知识蒸馏训练计划

## 一、动机

当前 AB+NNUE 管线的天花板是 **71% vs Random**，原因是训练标签来自 NNUE（106K 参数）自身的深度 6 搜索，标签质量受限于模型容量。

**AlphaZero ResNet**（550K 参数）+ **MCTS（800 模拟）** 已可达 1200+ ELO。若能利用 AZ 产生的质量生成数据来训练 NNUE，预期可将 AB+NNUE 提升到 80%+。

## 二、架构

```
                           ┌──────────────────────┐
                           │  AZ ResNet (550K)     │
                           │  + MCTS (800 sims)    │
                           └──────────┬───────────┘
                                      │ 自对弈 2000 局
                                      │ 每步记录 (sparse, dense, value)
                                      ▼
                           ┌──────────────────────┐
                           │  训练数据 (AZ标签)     │
                           │  ~60K 位置            │
                           └──────────┬───────────┘
                                      │
                    ┌─────────────────────────────────┐
                    │  训练 NNUE (106K)                │
                    │  - 从 scratch 训练（buggy Adam）  │
                    │  - 50 epoch, batch_size=4096     │
                    └────────────────┬────────────────┘
                                     │
                        ┌────────────┴────────────┐
                        ▼                         ▼
               ┌──────────────┐          ┌──────────────────┐
               │ AB+NNUE d6   │          │ AZ + MCTS 评估    │
               │ 推理评估      │          │ (基准线)           │
               └──────────────┘          └──────────────────┘
```

## 三、文件架构

```
penguinchess/
├── penguinchess/ai/          # AI 模型定义
│   ├── nnue.py               # NNUE 架构
│   ├── nnue_train.py         # NNUE 训练
│   ├── sparse_features.py    # 稀疏特征
│   └── alphazero_net.py      # AlphaZero 网络
│
├── game_engine/src/          # Rust 引擎
│   ├── nnue_rs.rs            # NNUE 推理 (AVX2)
│   ├── nnue_train.rs         # Rust 训练
│   ├── alphabeta_rs.rs       # AB 搜索
│   ├── mcts_rs.rs            # MCTS 搜索
│   └── ffi.rs                # FFI 接口
│
├── examples/
│   └── train_alphazero.py    # AZ 训练
│
├── scripts/
│   ├── az_distill.py         # AZ → NNUE 蒸馏训练
│   ├── vs_ppo.py             # AB+NNUE 对 PPO 评估
│   └── ensemble_train.py     # 集成训练
│
├── models/
│   ├── nnue_trained/         # NNUE 模型
│   ├── alphazero/            # AZ 模型
│   ├── nnue/                 # NNUE (gen_x)
│   └── ppo_*.zip             # PPO 模型
│
└── docs/
    ├── AZ_NNUE_PLAN.md       # 本文档
    └── TECHNICAL_REFERENCE.md  # 技术参考
```

## 四、实施步骤

### Step 1: 训练高质量 AZ 模型

```bash
# 从 az_resnet_best 续训 50 迭代
uv run python examples/train_alphazero.py \
    --resume models/alphazero/alphazero_resnet_best.pth \
    --iterations 50 --games 1000 --simulations 800
```

### Step 2: AZ 生成训练数据

使用 AZ ResNet + MCTS 自对弈，每步记录 NNUE 训练所需特征：

```bash
uv run python scripts/az_distill.py \
    --teacher models/alphazero/alphazero_resnet_best.pth \
    --games 2000 --sims 800 --output data/az_teacher.bin
```

### Step 3: 从 scratch 训练 NNUE

```bash
# Rust 训练（buggy Adam, 50 epoch）
uv run python -c "
from penguinchess.rust_ffi import get_engine, ...
eng._lib.ffi_nnue_train(ptr, len(flat), b'data/az_teacher.bin', ...)
"
```

### Step 4: 评估 vs 基准

| 模型 | AB+NNUE d4 | vs PPO |
|------|-----------|--------|
| 当前最佳 (gen_2 数据) | 71% | 56% |
| AZ 蒸馏后 (预期) | **80%+** | **65%+** |

## 五、预期瓶颈分析

| 瓶颈 | 风险 | 缓解方案 |
|------|------|---------|
| AZ 标签分布与 NNUE 不匹配 | 中 | 先用 depth 4 AB 搜索初步学习 |
| Buggy Adam 无法正确学习 | 低 | 已验证 71% 基线 |
| 数据量不足 | 中 | 2000 局 AZ 自对弈 ≈ 60K 位置 |
| 评估噪声 | 低 | 100 局评估 + 和棋统计 |

## 六、时间估算

| 阶段 | 耗时 | 说明 |
|------|------|------|
| AZ 续训 50 迭代 | ~16h | 后台运行 |
| AZ 生成 2000 局数据 | ~4h | Rust MCTS, 8 线程 |
| NNUE 训练 | ~3min | Rust buggy Adam |
| 评估 | ~5min | 200 局 vs Random + PPO |
| **合计** | **~20h** | 主要为 AZ 训练时间 |
